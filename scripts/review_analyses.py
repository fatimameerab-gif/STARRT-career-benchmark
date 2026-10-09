"""Extra analyses added after internal review (all CPU, TF-IDF where a model must be retrained).

1. Near-duplicate leakage: for every test row, the highest TF-IDF cosine similarity to any training row of
   the same fold. All released predictions are re-scored on the test rows without a near-duplicate.
2. Djinni ablations (TF-IDF + SVM, same folds): drop the "Looking For" field; mask role and keyword words.
3. Controlled text-type test: Djinni profiles reduced to the tools they mention (Stack Overflow tool
   vocabulary, Stack Overflow sentence format), same people, same labels, same folds.
4. Zero-shot LLM: Wilson intervals and exact McNemar against each trained model on the same rows.
5. Holm correction for the DistilBERT McNemar family.

Usage: python3 -I review_analyses.py <project_dir>
Writes <project_dir>/processed/analysis/review_*.csv|json and kaggle/data/djinni_{nolook,masked,tools}_folds.csv
"""
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score

P = Path(sys.argv[1]); RAW = P / "raw"; KD = P / "kaggle" / "data"; R = P / "kaggle_results"
OUT = P / "processed" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "MiniLM + LR",
          "sbert_all-mpnet-base-v2": "mpnet + LR", "distilbert": "DistilBERT"}
report = {}


def norm(t):
    t = str(t).replace("•", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", t).strip()


def load(ds):
    return pd.read_csv(KD / f"{ds}_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)


def pooled(df):
    """rows in the order every prediction file uses: fold 0 test rows, fold 1 test rows, ..."""
    return pd.concat([df[df.fold == k] for k in range(5)]).reset_index(drop=True)


def svm_cv(df, col="text"):
    preds = []
    for k in range(5):
        tr, te = df[df.fold != k], df[df.fold == k]
        m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
                          LinearSVC(class_weight="balanced")).fit(tr[col], tr.label)
        preds.append(pd.Series(m.predict(te[col]), index=te.index))
    p = pd.concat(preds).loc[df.index]
    accs = [(p[df.fold == k] == df.label[df.fold == k]).mean() for k in range(5)]
    return p, float(np.mean(accs)), float(np.std(accs))


def mcnemar(a, b):
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    return n01, n10, (binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0)


def wilson(k, n, z=1.96):
    p_ = k / n; d = 1 + z * z / n; c = (p_ + z * z / (2 * n)) / d
    h = z * np.sqrt(p_ * (1 - p_) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


# ---------------- 1. near-duplicate leakage ----------------
nd_rows = []
for ds in ["djinni", "faruque", "stackoverflow"]:
    df = load(ds)
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(df.text)
    X = vec.transform(df.text)
    maxsim = np.zeros(len(df))
    for k in range(5):
        te = np.where(df.fold == k)[0]; tr = np.where(df.fold != k)[0]
        Xtr = X[tr].T.tocsc()
        for s in range(0, len(te), 2000):
            blk = te[s:s + 2000]
            maxsim[blk] = (X[blk] @ Xtr).max(axis=1).toarray().ravel()
    df["maxsim"] = maxsim
    pool = pooled(df)
    for thr in [0.8, 0.9]:
        clean = (pool.maxsim < thr).values
        row = {"dataset": ds, "threshold": thr, "n_test": len(pool), "near_dup_rows": int((~clean).sum()),
               "near_dup_share": float((~clean).mean())}
        for m, name in MODELS.items():
            pr = pd.read_csv(R / f"{m}__{ds}__predictions.csv")
            assert (pr.id.values == pool.id.values).all() and (pr.true.values == pool.label.values).all(), (m, ds)
            ok = (pr.true == pr.pred).values
            row[f"{name} | all"] = float(ok.mean()); row[f"{name} | clean"] = float(ok[clean].mean())
        nd_rows.append(row)
nd = pd.DataFrame(nd_rows); nd.to_csv(OUT / "review_near_duplicates.csv", index=False)
print("\nNear-duplicates:\n", nd.round(4).T.to_string())

# ---------------- 2 & 3. Djinni ablations ----------------
dj = load("djinni")
raw = pd.read_parquet(RAW / "djinni/candidates_en.parquet")
raw["id"] = "dj_" + raw["id"].astype(str)
raw = raw.drop_duplicates("id").set_index("id").loc[dj.id]
assert len(raw) == len(dj)


def join(r, cols):
    return " ".join(norm(r[c]) for c in cols if pd.notna(r[c]) and norm(r[c]))


dj["text_nolook"] = [join(r, ["Moreinfo", "Highlights"]) for _, r in raw.iterrows()]
# role and keyword words that name the label directly (titles, field names and the platform keywords)
ROLE_TERMS = [
    "full[- ]?stack", "front[- ]?end", "back[- ]?end", "web developer", "web development", "javascript", "php", "node\\.?js", "ruby",
    "mobile", "android", "ios", "flutter", "unity", "game ?dev\\w*", "game developer", "game development", "games?",
    "data scien\\w*", "data analy\\w*", "analyst", "analytics", "data engineer\\w*", "sql", "dba", "database administrator",
    "machine learning", "deep learning", "ml", "ai", "artificial intelligence", "nlp", "computer vision", "llm", "mlops",
    "devops", "sre", "site reliability", "cloud engineer", "security", "cyber ?security", "pentest\\w*", "penetration test\\w*", "soc",
    "sys ?admin\\w*", "system administrator", "network engineer", "network administrator", "networking",
    "qa", "quality assurance", "tester", "testing", "test automation", "automation qa",
    "designer", "ui", "ux", "ui/ux", "product designer", "product manager", "project manager", "scrum master", "product owner",
    "delivery manager", "pm"]
MASK = re.compile(r"(?i)(?<![\w])(" + "|".join(ROLE_TERMS) + r")(?![\w])")
dj["text_masked"] = dj.text.map(lambda t: re.sub(r"\s+", " ", MASK.sub(" ", t)).strip())
report["masked_share_of_words_removed"] = float(1 - dj.text_masked.str.split().str.len().sum() / dj.text.str.split().str.len().sum())

# Stack Overflow tool vocabulary, by field, from the three survey years
FIELDS = [("LanguageHaveWorkedWith", "Languages"), ("DatabaseHaveWorkedWith", "Databases"),
          ("PlatformHaveWorkedWith", "Platforms"), ("WebframeHaveWorkedWith", "Frameworks"),
          ("OfficeStackAsyncHaveWorkedWith", "Collaboration tools")]
AMBIG = {"go", "c", "r", "d", "ada", "apex", "julia", "dart", "crystal", "zig", "nim", "mojo", "solidity", "elixir", "erlang",
         "jira", "trello", "asana", "notion", "clickup", "linear", "smartsheet", "redmine", "wrike", "basecamp", "monday.com",
         "markdown file", "doxygen", "airtable", "shortcut", "youtrack", "dynamodb",
         "make", "flow", "express"}            # ordinary English words: too many false matches
CASE = {"swift", "ruby", "rust", "spring", "oracle", "zephyr"}   # matched only with their capital letter
vocab = {}
for y in ["2023", "2024", "2025"]:
    so = pd.read_csv(RAW / f"stackoverflow/{y}/results.csv", usecols=lambda c: c in [f for f, _ in FIELDS], low_memory=False)
    for c, n in FIELDS:
        if c not in so.columns: continue
        for v in so[c].dropna().str.split(";").explode().unique():
            vocab.setdefault(v.strip(), n)
pats = []
for tool, field in vocab.items():
    names = {re.sub(r"\s*\(.*?\)", "", tool).strip().lower()} | {x.strip().lower() for x in re.findall(r"\((.*?)\)", tool)}
    names = {x for x in names if x and (len(x) > 2 or x in {"c#", "c++"}) and x not in AMBIG}
    if names:
        cs = names & CASE; ci = names - CASE
        alts = [re.escape(x) for x in sorted(ci, key=len, reverse=True)] + ["(?-i:" + re.escape(x.capitalize()) + ")" for x in cs]
        pats.append((tool, field, re.compile(r"(?i)(?<![\w#+])(" + "|".join(alts) + r")(?![\w#+])")))
report["tool_vocabulary_size"] = len(pats)


def tools_text(t):
    found = {}
    for tool, field, rx in pats:
        if rx.search(t): found.setdefault(field, []).append(tool)
    n = sum(len(v) for v in found.values())
    return " ".join(f"{f}: {', '.join(v)}." for _, f in FIELDS for f_, v in found.items() if f_ == f), n


tt = dj.text.map(tools_text)
dj["text_tools"] = [a for a, _ in tt]; dj["n_tools"] = [b for _, b in tt]

abl = []
p_full, a_full, s_full = svm_cv(dj, "text")
abl.append({"variant": "full text (as in Table 4)", "rows": len(dj), "acc": a_full, "sd": s_full})
for col, name in [("text_nolook", "without Looking For"), ("text_masked", "role and keyword words masked")]:
    sub = dj[dj[col].str.len() > 0]
    p, a, s = svm_cv(sub, col)
    abl.append({"variant": name, "rows": len(sub), "acc": a, "sd": s})
both = dj.copy(); both["text_both"] = both.text_nolook.map(lambda t: re.sub(r"\s+", " ", MASK.sub(" ", t)).strip())
sub = both[both.text_both.str.len() > 0]; p, a, s = svm_cv(sub, "text_both")
abl.append({"variant": "without Looking For and masked", "rows": len(sub), "acc": a, "sd": s})
# controlled text type: same people, only the tools they mention, Stack Overflow format
sub = dj[dj.n_tools >= 2].copy()
p_t, a_t, s_t = svm_cv(sub, "text_tools")
p_f, a_f, s_f = svm_cv(sub, "text")
abl.append({"variant": "same profiles (>= 2 tools), full text", "rows": len(sub), "acc": a_f, "sd": s_f})
abl.append({"variant": "same profiles (>= 2 tools), tool list only", "rows": len(sub), "acc": a_t, "sd": s_t})
report["tool_coverage_share_ge2"] = float((dj.n_tools >= 2).mean())
report["tools_median_per_profile"] = float(sub.n_tools.median())
report["tool_text_median_words"] = float(sub.text_tools.str.split().str.len().median())
report["tool_vs_full_macro_f1"] = [float(f1_score(sub.label, p_t, average="macro")), float(f1_score(sub.label, p_f, average="macro"))]
abl = pd.DataFrame(abl); abl.to_csv(OUT / "review_djinni_ablation.csv", index=False)
print("\nDjinni ablations (TF-IDF + SVM):\n", abl.round(4).to_string(index=False))
# fold files for the GPU check of the same ablations
for col, name in [("text_nolook", "nolook"), ("text_masked", "masked"), ("text_tools", "tools")]:
    s = dj if name != "tools" else dj[dj.n_tools >= 2]
    s = s[s[col].str.len() > 0]
    s[["id", "source", col, "label", "fold"]].rename(columns={col: "text"}).to_csv(KD / f"djinni_{name}_folds.csv", index=False)

# ---------------- 4. LLM: intervals and paired tests ----------------
FOLDS = KD
llm_rows = []
for ds, llm_ds in [("djinni", "djinni_fold0_sample"), ("faruque", "faruque_fold0_sample")]:
    L = pd.read_csv(R / f"llm_qwen2.5-3b_zeroshot__{llm_ds}__predictions.csv")
    full = pd.read_csv(FOLDS / f"{ds}_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    t = full[full.fold == 0]
    samp = pd.concat([g.sample(min(len(g), 40), random_state=0) for _, g in t.groupby("label")])
    assert (L.id.values == samp.id.values).all()
    pos = [list(t.index).index(i) for i in samp.index]
    lok = (L.true == L.pred).astype(int).values
    lo, hi = wilson(lok.sum(), len(lok))
    llm_rows.append({"dataset": ds, "model": "Qwen2.5-3B zero-shot", "n": len(lok), "acc": lok.mean(), "ci_low": lo, "ci_high": hi,
                     "llm_only_right": None, "other_only_right": None, "p": None})
    for m, name in MODELS.items():
        p0 = pd.read_csv(R / f"{m}__{ds}__predictions.csv"); p0 = p0[p0.fold == 0].reset_index(drop=True).iloc[pos]
        ok = (p0.true == p0.pred).astype(int).values
        lo, hi = wilson(ok.sum(), len(ok)); a, b, pv = mcnemar(lok, ok)
        llm_rows.append({"dataset": ds, "model": name, "n": len(ok), "acc": ok.mean(), "ci_low": lo, "ci_high": hi,
                         "llm_only_right": a, "other_only_right": b, "p": pv})
llm = pd.DataFrame(llm_rows)
llm["p_holm"] = np.nan
mask = llm.p.notna(); llm.loc[mask, "p_holm"] = holm(llm.loc[mask, "p"].astype(float).values)
llm.to_csv(OUT / "review_llm_tests.csv", index=False)
print("\nLLM vs trained models (same rows):\n", llm.round(4).to_string(index=False))

# ---------------- 5. Holm correction, DistilBERT family ----------------
sig = pd.read_csv(OUT / "significance.csv")
sig["p_holm"] = holm(sig.p_value.values)
sig.to_csv(OUT / "review_significance_holm.csv", index=False)
print("\nHolm-adjusted McNemar:\n", sig.round(4).to_string(index=False))

# exact means for consistent rounding in table and figure
report["exact_means"] = {f"{m}|{ds}": json.load(open(R / f"{m}__{ds}.json"))["summary"]["acc"] for m in MODELS for ds in ["djinni", "faruque", "stackoverflow"]}
report["exact_fold_means"] = {f"{m}|{ds}": float(np.mean([f["acc"] for f in json.load(open(R / f"{m}__{ds}.json"))["folds"]]))
                              for m in MODELS for ds in ["djinni", "faruque", "stackoverflow"]}
json.dump(report, open(OUT / "review_extra.json", "w"), indent=2)
print(json.dumps({k: v for k, v in report.items() if not k.startswith("exact")}, indent=1))
