"""Combine Kaggle results into paper tables: main table, same-sample LLM comparison,
paired significance tests, per-class F1, and parent-group (hierarchical) accuracy.

Usage: python3 -I analyze_results.py <results_dir> <project_dir>
"""
import sys, json, glob, os
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, classification_report

R = Path(sys.argv[1]); P = Path(sys.argv[2]); OUT = P / "processed" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(P / "scripts"))
from labels import CLASSES  # noqa: E402
PARENT = {c: v[0] for c, v in CLASSES.items()}

MODELS = ["tfidf_lr", "tfidf_svm", "sbert_all-MiniLM-L6-v2", "sbert_all-mpnet-base-v2", "distilbert", "llm_qwen2.5-3b_zeroshot"]
NAMES = {"tfidf_lr": "TF-IDF + LR", "tfidf_svm": "TF-IDF + SVM", "sbert_all-MiniLM-L6-v2": "SBERT MiniLM + LR (original STARRT.ai)",
         "sbert_all-mpnet-base-v2": "SBERT mpnet + LR", "distilbert": "DistilBERT fine-tuned", "llm_qwen2.5-3b_zeroshot": "Qwen2.5-3B zero-shot"}


def pred(model, ds):
    f = R / f"{model}__{ds}__predictions.csv"
    return pd.read_csv(f) if f.exists() else None


def mcnemar(a, b):
    """exact McNemar on paired correctness vectors"""
    n01 = int(((a == 1) & (b == 0)).sum()); n10 = int(((a == 0) & (b == 1)).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


report = {}
# 1. main table (5-fold CV)
rows = []
for m in MODELS[:-1]:
    for ds in ["djinni", "faruque", "stackoverflow", "forms_from_djinni"]:
        f = R / f"{m}__{ds}.json"
        if not f.exists():
            continue
        s = json.load(open(f))["summary"]
        rows.append({"model": NAMES[m], "dataset": ds, "acc": s["acc"], "acc_sd": s.get("acc_sd"), "macro_f1": s["macro_f1"], "top3": s.get("top3"), "n": s.get("n")})
main = pd.DataFrame(rows); main.to_csv(OUT / "main_table.csv", index=False)
print(main.pivot(index="model", columns="dataset", values="acc").round(4).to_string())

# 2. LLM vs supervised on the exact same sampled rows
same = []
FOLDS = P / "kaggle" / "data"
for ds, llm_ds in [("djinni", "djinni_fold0_sample"), ("faruque", "faruque_fold0_sample")]:
    L = pred("llm_qwen2.5-3b_zeroshot", llm_ds)
    # rebuild the exact LLM sample (same code as run_llm_zeroshot.sample_fold0); ids are not unique in Faruque
    full = pd.read_csv(FOLDS / f"{ds}_folds.csv").dropna(subset=["text", "label"]).reset_index(drop=True)
    t = full[full.fold == 0]
    samp = pd.concat([g.sample(min(len(g), 40), random_state=0) for _, g in t.groupby("label")])
    assert (L.id.values == samp.id.values).all() and (L.true.values == samp.label.values).all()
    pos = [list(t.index).index(i) for i in samp.index]
    same.append({"dataset": ds, "model": NAMES["llm_qwen2.5-3b_zeroshot"], "n": len(L), "acc": (L.true == L.pred).mean(),
                 "macro_f1": f1_score(L.true, L.pred, average="macro")})
    for m in MODELS[:-1]:
        p0 = pred(m, ds); p0 = p0[p0.fold == 0].reset_index(drop=True)
        assert (p0.id.values == t.id.values).all()
        p = p0.iloc[pos]; assert (p.true.values == samp.label.values).all()
        same.append({"dataset": ds, "model": NAMES[m], "n": len(p), "acc": (p.true == p.pred).mean(), "macro_f1": f1_score(p.true, p.pred, average="macro")})
same = pd.DataFrame(same); same.to_csv(OUT / "llm_same_sample.csv", index=False)
print("\nSame-sample comparison (fold 0, 40 per class):"); print(same.pivot(index="model", columns="dataset", values="acc").round(4).to_string())

# 3. Pakistani forms: every model on the same 33 rows (LLM also on all 34)
L = pred("llm_qwen2.5-3b_zeroshot", "forms_zeroshot")
common = set(pred("distilbert", "forms_from_djinni").id)
forms = []
for m in MODELS:
    p = L if m.startswith("llm") else pred(m, "forms_from_djinni")
    q = p[p.id.isin(common)]
    forms.append({"model": NAMES[m], "n": len(q), "correct": int((q.true == q.pred).sum()), "acc": (q.true == q.pred).mean()})
forms = pd.DataFrame(forms); forms.to_csv(OUT / "forms_33.csv", index=False)
print("\nPakistani forms, same 33 rows:"); print(forms.to_string(index=False))
lc = L.set_index("id"); tc = pred("tfidf_lr", "forms_from_djinni").set_index("id").loc[list(common)]
a = (lc.loc[list(common)].true == lc.loc[list(common)].pred).astype(int); b = (tc.true == tc.pred).astype(int)
report["forms_llm_vs_tfidf_mcnemar"] = mcnemar(a.values, b.values)

# 4. paired significance on full CV predictions
sig = []
for ds in ["djinni", "faruque", "stackoverflow"]:
    base = pred("distilbert", ds).reset_index(drop=True)
    for m in ["tfidf_svm", "tfidf_lr", "sbert_all-mpnet-base-v2", "sbert_all-MiniLM-L6-v2"]:
        o = pred(m, ds).reset_index(drop=True)        # every script writes rows in the same fold order
        assert (o.id.values == base.id.values).all() and (o.true.values == base.true.values).all(), (m, ds)
        n01, n10, p = mcnemar((base.true == base.pred).astype(int).values, (o.true == o.pred).astype(int).values)
        sig.append({"dataset": ds, "comparison": f"DistilBERT vs {NAMES[m]}", "distil_only_right": n01, "other_only_right": n10, "p_value": p})
sig = pd.DataFrame(sig); sig.to_csv(OUT / "significance.csv", index=False)
print("\nMcNemar tests:"); print(sig.to_string(index=False))

# 5. per-class F1 and parent-group accuracy (RQ3)
pc, grp = [], []
for m in ["tfidf_svm", "sbert_all-MiniLM-L6-v2", "distilbert"]:
    for ds in ["djinni", "stackoverflow"]:
        p = pred(m, ds)
        f1s = f1_score(p.true, p.pred, average=None, labels=sorted(p.true.unique()))
        for c, v in zip(sorted(p.true.unique()), f1s):
            pc.append({"model": NAMES[m], "dataset": ds, "class": c, "f1": v})
        g_acc = (p.true.map(PARENT) == p.pred.map(PARENT)).mean()
        grp.append({"model": NAMES[m], "dataset": ds, "fine_acc": (p.true == p.pred).mean(), "parent_group_acc": g_acc})
pc = pd.DataFrame(pc); pc.to_csv(OUT / "per_class_f1.csv", index=False)
grp = pd.DataFrame(grp); grp.to_csv(OUT / "hierarchical.csv", index=False)
print("\nPer-class F1 (Djinni):"); print(pc[pc.dataset == "djinni"].pivot(index="class", columns="model", values="f1").round(3).to_string())
print("\nFine vs parent-group accuracy:"); print(grp.round(4).to_string(index=False))

# 6. confusions of the best model
for ds in ["djinni", "stackoverflow"]:
    report[f"distilbert_{ds}_confusions"] = json.load(open(R / f"distilbert__{ds}.json"))["top_confusions"][:6]
json.dump(report, open(OUT / "extra.json", "w"), indent=2, default=str)
print("\n", json.dumps(report, indent=1, default=str))
