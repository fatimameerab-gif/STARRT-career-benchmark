"""Create fixed, leakage-free 5-fold splits shared by every model (TF-IDF here, transformers on Kaggle).

Usage: python3 -I make_folds.py <project_dir>
Writes <project_dir>/kaggle/data/{djinni,stackoverflow,faruque}_folds.csv, forms_test.csv, label_set.csv
"""
import sys, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

P = Path(sys.argv[1]); D = P / "processed"; OUT = P / "kaggle" / "data"; OUT.mkdir(parents=True, exist_ok=True)
CAP, SEED = 3000, 42


def folds(df, label_col):
    df = df.dropna(subset=[label_col]).reset_index(drop=True)
    groups = pd.factorize(df.text.str.lower())[0]          # identical text never split across folds
    df["fold"] = -1
    for k, (_, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df.text, df[label_col], groups)):
        df.loc[te, "fold"] = k
    return df


def cap(df):
    return pd.concat([g.sample(min(len(g), CAP), random_state=0) for _, g in df.groupby("label")])


dj = folds(cap(pd.read_csv(D / "djinni_clean.csv")), "label")
so = folds(cap(pd.read_csv(D / "stackoverflow_2023_2025.csv")), "label")
fa = pd.read_csv(D / "faruque_clean.csv").rename(columns={"label": "label13"}).rename(columns={"orig_label": "label"})
fa = folds(fa, "label")                                     # Faruque benchmark uses its own 6 labels
FO = D / "starrt_forms_pilot.csv"                          # private pilot; skipped if absent
fo = pd.read_csv(FO).dropna(subset=["label"]) if FO.exists() and len(pd.read_csv(FO)) else None

keep = ["id", "source", "text", "label", "fold"]
dj[keep].to_csv(OUT / "djinni_folds.csv", index=False)
so[keep].to_csv(OUT / "stackoverflow_folds.csv", index=False)
fa[keep + ["label13"]].to_csv(OUT / "faruque_folds.csv", index=False)
if fo is not None:
    fo[["id", "source", "text", "label"]].to_csv(OUT / "forms_test.csv", index=False)
shutil.copy(D / "label_set.csv", OUT / "label_set.csv")
for n, d in [("djinni", dj), ("stackoverflow", so), ("faruque", fa)]:
    print(n, len(d), "rows; per fold:", d.fold.value_counts().sort_index().tolist())
print("forms", 0 if fo is None else len(fo))
