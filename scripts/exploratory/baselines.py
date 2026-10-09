"""Leakage-free TF-IDF baselines on each processed dataset.

Usage: python3 -I baselines.py <project_dir>
Identical texts are grouped so a text never appears in both train and test.
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

P = Path(sys.argv[1]); D = P / "processed"
results = {}


def run(name, df, label_col, models, seed=42):
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    X, y = df.text.values, df[label_col].values
    groups = pd.factorize(df.text.str.lower())[0]
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    labels = np.unique(y)
    out = {"n": len(df), "classes": len(labels), "majority_acc": round(pd.Series(y).value_counts(normalize=True).iloc[0], 3)}
    for mname, make in models.items():
        acc, f1, top3 = [], [], []
        for tr, te in cv.split(X, y, groups):
            m = make(); m.fit(X[tr], y[tr]); pred = m.predict(X[te])
            acc.append(accuracy_score(y[te], pred)); f1.append(f1_score(y[te], pred, average="macro"))
            if hasattr(m, "predict_proba") and len(labels) > 3:
                top3.append(top_k_accuracy_score(y[te], m.predict_proba(X[te]), k=3, labels=m.classes_))
        out[mname] = {"acc": round(np.mean(acc), 3), "acc_sd": round(np.std(acc), 3),
                      "macro_f1": round(np.mean(f1), 3), "top3": round(np.mean(top3), 3) if top3 else None}
    results[name] = out
    print(name, json.dumps(out))


models = {
    "tfidf_logreg": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
                                          LogisticRegression(max_iter=4000, class_weight="balanced")),
    "tfidf_linearsvm": lambda: make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=200000),
                                             LinearSVC(class_weight="balanced")),
}

fa = pd.read_csv(D / "faruque_clean.csv")
run("faruque_6class_replication", fa, "orig_label", models)

so = pd.read_csv(D / "stackoverflow_2023_2025.csv")
# cap big classes so one class does not swamp the rest
so_bal = pd.concat([g.sample(min(len(g), 3000), random_state=0) for _, g in so.groupby("label")])
run("stackoverflow_13class_cv", so_bal, "label", models)

# temporal hold-out: train on 2023+2024, test on 2025 (closer to a real deployment)
tr = so_bal[so_bal.source != "stackoverflow2025"]; te = so_bal[so_bal.source == "stackoverflow2025"]
te = te[~te.text.str.lower().isin(set(tr.text.str.lower()))]
m = models["tfidf_logreg"](); m.fit(tr.text, tr.label)
p_ = m.predict(te.text)
results["stackoverflow_train2023-24_test2025"] = {
    "n_train": len(tr), "n_test": len(te), "acc": round(accuracy_score(te.label, p_), 3),
    "macro_f1": round(f1_score(te.label, p_, average="macro"), 3),
    "top3": round(top_k_accuracy_score(te.label, m.predict_proba(te.text), k=3, labels=m.classes_), 3)}
print("stackoverflow_train2023-24_test2025", results["stackoverflow_train2023-24_test2025"])

dj = pd.read_csv(D / "djinni_clean.csv")
dj_bal = pd.concat([g.sample(min(len(g), 3000), random_state=0) for _, g in dj.groupby("label")])
run("djinni_12class_cv", dj_bal, "label", {"tfidf_logreg": models["tfidf_logreg"]})

# cross-source checks on the 34 real Pakistani form answers
fo = pd.read_csv(D / "starrt_forms_pilot.csv").dropna(subset=["label"])
fa_m = fa.dropna(subset=["label"])
for name, train in [("forms_from_stackoverflow", so_bal), ("forms_from_djinni", dj_bal),
                    ("forms_from_all_sources", pd.concat([so_bal, dj_bal, fa_m]))]:
    m = models["tfidf_logreg"](); m.fit(train.text, train.label)
    ft = fo[fo.label.isin(m.classes_)]            # a class missing from training cannot be predicted
    pred = m.predict(ft.text)
    top3 = top_k_accuracy_score(ft.label, m.predict_proba(ft.text), k=3, labels=m.classes_)
    results[name] = {"n_train": len(train), "n_test": len(ft), "acc": round(accuracy_score(ft.label, pred), 3), "top3": round(top3, 3)}
    print(name, results[name])
json.dump(results, open(D / "baseline_results.json", "w"), indent=2)
