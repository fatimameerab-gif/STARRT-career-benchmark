"""Build cleaned, labelled datasets from each source.

Usage: python3 -I build_datasets.py <project_dir>
Writes to <project_dir>/processed/.
"""
import sys, re, hashlib
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from labels import CLASSES, SO_DEVTYPE, SO_DEVTYPE_2023_2024, FARUQUE, FORMS, DJINNI, DJINNI_ML_TITLE  # noqa: E402

P = Path(sys.argv[1]); RAW = P / "raw"; OUT = P / "processed"; OUT.mkdir(exist_ok=True)


def norm(t):
    t = str(t).replace("•", " ").replace("\t", " ")
    return re.sub(r"\s+", " ", t).strip()


def uid(source, text):
    return source + "_" + hashlib.md5(text.lower().encode()).hexdigest()[:12]


# ---------- 1. Label set with official definitions ----------
onet = pd.read_csv(RAW / "onet/occupation_data.csv")
esco = pd.read_csv(RAW / "esco/occupations_en.csv")
rows = []
for cls, (parent, socs, esco_labels) in CLASSES.items():
    o = onet[onet["O*NET-SOC Code"].isin(socs)]
    e = esco[esco.preferredLabel.isin(esco_labels)]
    missing = set(esco_labels) - set(e.preferredLabel)
    if missing:
        print("WARNING ESCO label not found:", missing)
    rows.append({
        "class": cls, "parent": parent,
        "onet_codes": "; ".join(socs), "onet_titles": "; ".join(o.Title),
        "esco_labels": "; ".join(e.preferredLabel), "esco_codes": "; ".join(e.code.astype(str)),
        "esco_definition": norm(e.description.iloc[0]) if len(e) else "",
        "onet_definition": norm(o.Description.iloc[0]) if len(o) else "",
    })
pd.DataFrame(rows).to_csv(OUT / "label_set.csv", index=False)
print("label_set:", len(rows), "classes")

# ---------- 2. Stack Overflow 2023, 2024, 2025 ----------
# only fields asked in all three years, so text format does not reveal the survey year
FIELDS = [("LanguageHaveWorkedWith", "Languages"), ("DatabaseHaveWorkedWith", "Databases"),
          ("PlatformHaveWorkedWith", "Platforms"), ("WebframeHaveWorkedWith", "Frameworks"),
          ("OfficeStackAsyncHaveWorkedWith", "Collaboration tools")]
KEEP = ["I am a developer by profession", "I am not primarily a developer, but I write code sometimes as part of my work/studies",
        "I work with developers or my work supports developers but am not a developer by profession"]
parts_all = []
for year, devmap in [("2025", SO_DEVTYPE), ("2024", SO_DEVTYPE_2023_2024), ("2023", SO_DEVTYPE_2023_2024)]:
    so = pd.read_csv(RAW / f"stackoverflow/{year}/results.csv", low_memory=False)
    fields = [(c, n) for c, n in FIELDS if c in so.columns]
    so = so[so.MainBranch.isin(KEEP)].copy()
    so["label"] = so.DevType.map(devmap)          # single-role answers only; multi-role strings do not match
    so = so[so.label.notna()].copy()
    so["text"] = so.apply(lambda r: " ".join(f"{n}: {', '.join(str(r[c]).split(';'))}." for c, n in fields if pd.notna(r[c])), axis=1)
    so["n_fields"] = so[[c for c, _ in fields]].notna().sum(axis=1)
    so = so[so.n_fields >= 2]
    so["source"] = f"stackoverflow{year}"; so["orig_label"] = so.DevType
    so["id"] = f"so{year}_" + so.ResponseId.astype(str); so["country"] = so.Country
    parts_all.append(so[["id", "source", "text", "label", "orig_label", "country"]])
    print(f"stackoverflow {year}:", len(so), "rows")
so = pd.concat(parts_all, ignore_index=True)
so.to_csv(OUT / "stackoverflow_2023_2025.csv", index=False)
print("stackoverflow all years:", len(so), "rows;", so.label.value_counts().to_dict())

# ---------- 3. Faruque et al. student career dataset ----------
fa = pd.read_excel(RAW / "faruque/Career_Dataset.xlsx").dropna()
fa["text"] = fa.Skill.map(norm); fa["orig_label"] = fa.Career.map(norm)
n0 = len(fa)
fa = fa[fa.text.str.len() > 0].drop_duplicates(["text", "orig_label"])
conflict = fa.groupby("text").orig_label.nunique()
fa = fa[~fa.text.isin(conflict[conflict > 1].index)]      # same text with two labels: drop
fa["label"] = fa.orig_label.map(FARUQUE)
fa["source"] = "faruque2024"; fa["id"] = [uid("fa", t) for t in fa.text]; fa["country"] = "Bangladesh"
fa[["id", "source", "text", "label", "orig_label", "country"]].to_csv(OUT / "faruque_clean.csv", index=False)
print(f"faruque: {n0} -> {len(fa)} rows after dedup/conflict removal;", fa.orig_label.value_counts().to_dict())

# ---------- 4. Old STARRT.ai Google Form (46 real answers) ----------
FORMS_FILE = RAW / "starrt_old/forms_data.csv"     # private pilot data, not in the public release
fo = pd.read_csv(FORMS_FILE) if FORMS_FILE.exists() else pd.DataFrame(columns=["timestamp", "field", "text"])
fo.columns = ["timestamp", "field", "text"]
fo["field"] = fo.field.map(norm); fo["text"] = fo.text.map(norm)
fo["label"] = fo.field.map(FORMS); fo["orig_label"] = fo.field
fo["source"] = "starrt_forms2025"; fo["id"] = [uid("fo", t) for t in fo.text]; fo["country"] = "Pakistan"
fo = fo[fo.text.str.len() > 0].drop_duplicates("text")
fo[["id", "source", "text", "label", "orig_label", "country"]].to_csv(OUT / "starrt_forms_pilot.csv", index=False)
print("forms:", len(fo), "rows,", fo.label.notna().sum(), "mapped to a class")

# ---------- 5. Djinni candidate profiles (English) ----------
dj = pd.read_parquet(RAW / "djinni/candidates_en.parquet")
n0 = len(dj)
pos = dj.Position.fillna("")
dj["label"] = dj["Primary Keyword"].map(DJINNI)
is_ml = pos.str.contains(DJINNI_ML_TITLE, regex=True)
dj.loc[dj["Primary Keyword"].isin(["Data Science", "Python"]) & is_ml, "label"] = "AI/ML Engineering"
# Design keyword also holds graphic / 3D / brand designers; keep only UI/UX/product/web designers
design_other = pos.str.contains(r"(?i)graphic|illustrat|motion|3d|brand|print|interior|fashion", regex=True)
dj.loc[(dj["Primary Keyword"] == "Design") & design_other, "label"] = None
dj = dj[dj.label.notna()].copy()


def dj_text(r):
    parts = [norm(r[c]) for c in ["Moreinfo", "Highlights", "Looking For"] if pd.notna(r[c]) and norm(r[c])]
    return " ".join(parts)


dj["text"] = dj.apply(dj_text, axis=1)
nonascii = dj.text.map(lambda t: sum(ord(c) > 127 for c in t) / max(len(t), 1))
dj = dj[(nonascii < 0.1) & (dj.text.str.split().str.len() >= 15)]      # English, enough content
dj = dj.drop_duplicates("text")
dj["orig_label"] = dj["Primary Keyword"] + " | " + pos.loc[dj.index]
dj["source"] = "djinni2024"; dj["id"] = "dj_" + dj["id"].astype(str); dj["country"] = "Ukraine (platform)"
dj["experience_years"] = dj["Experience Years"]
dj[["id", "source", "text", "label", "orig_label", "country", "experience_years"]].to_csv(OUT / "djinni_clean.csv", index=False)
print(f"djinni: {n0} -> {len(dj)} rows;", dj.label.value_counts().to_dict())
