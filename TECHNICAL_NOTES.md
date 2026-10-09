# Technical notes

These notes give the exact settings behind every number in the paper. Where the notes and the code ever disagree, the code is correct.

## 1. Pipeline

```
raw/ ──build_datasets.py──> processed/*.csv ──make_folds.py──> kaggle/data/*_folds.csv
                                                                     │
                     run_tfidf / run_sbert / run_distilbert / run_llm_zeroshot (kaggle/)
                                                                     │
                                                       kaggle_results/*.json, *__predictions.csv
                                                                     │
                                   analyze_results.py ──> processed/analysis/*.csv
                                   make_figures.py    ──> paper/Figures/Figure_1-5.pdf
```

## 2. Label set (`scripts/labels.py`)

- **Classes.** There are 13 fields in 4 parent groups (listed in `DATA_STATEMENT.md`). Each field has ESCO v1.2.1 occupation labels and, where one exists, O\*NET-SOC codes. O\*NET has no separate AI/ML engineering occupation. `build_datasets.py` copies the official titles and descriptions into `label_set.csv`.
- **Djinni** (`DJINNI`).
  - JavaScript, PHP, Node.js and Ruby → Web.
  - iOS, Android and Flutter → Mobile.
  - Unity → Game.
  - Data Science and Data Analyst → Data Science & Analytics.
  - Data Engineer and SQL → Data Engineering.
  - DevOps → DevOps & Cloud.
  - Security → Cybersecurity.
  - Sysadmin → Networking & SysAdmin.
  - QA and QA Automation → QA & Testing.
  - Design → UI/UX.
  - Project Manager, Product Manager and Scrum Master → Product & Project Management.
  - Data Science or Python profiles whose title matches `(?i)\b(machine learning|deep learning|ml|ai|computer vision|cv engineer|nlp|llm|mlops)\b` → AI/ML Engineering.
  - Design profiles whose title matches graphic, illustration, motion, 3D, brand, print, interior or fashion are removed.
  - All other keywords are dropped.
- **Stack Overflow.** `SO_DEVTYPE` is used for 2025 and `SO_DEVTYPE_2023_2024` for the two earlier years. Role names changed between years, which is why there are two maps. Generic roles (for example "Developer, desktop or enterprise applications", students, researchers, executives) are not mapped.
- **Faruque.** The original 6 labels are used for the benchmark. `label13` maps 4 of them to our fields. "Development" and "Software Development and Engineering" have no single field, so their `label13` is empty.

## 3. Cleaning (`scripts/build_datasets.py`)

**All sources**
- Whitespace is normalised and bullets are removed.
- Ids are deterministic: the source id, or the md5 of the lower-cased text for sources without ids.

**Djinni**
- Text = `Moreinfo` + `Highlights` + `Looking For`. `Position` (the job title) and `CV` are never in the text.
- A row is kept if it has less than 10% non-ASCII characters and at least 15 words.
- Exact duplicate texts are removed.
- 210,250 → 128,891 rows.

**Stack Overflow**
- `MainBranch` must be one of the three working-developer answers.
- The text has the form `Languages: a, b. Databases: c. Platforms: d. Frameworks: e. Collaboration tools: f.`. Only these five fields are used because they were asked in all three years, so the format does not reveal the survey year.
- At least 2 fields must be non-empty.
- Result: 114,879 rows.

**Faruque**
- 4,076 rows → 2,135 after removing 1,900 exact duplicates and 41 texts that carry two different labels.
- `scripts/exploratory/eval_faruque.py` compares cross-validation with and without the duplicates.

**Pilot**
- 46 answers → 44 after removing empty and duplicate answers. 34 map to our fields and 33 fall in the 12 Djinni classes.

## 4. Folds (`scripts/make_folds.py`)

- Djinni and Stack Overflow are capped at 3,000 rows per class with `groupby(label).sample(random_state=0)`. Faruque is not capped.
- `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, grouped by the lower-cased text, so identical texts are always in the same fold.
- The fold files are the benchmark. Every model reads the same files, which makes paired tests possible.

## 5. Models (`kaggle/`)

| Model | Settings |
|---|---|
| TF-IDF + LR | word 1–2-grams, `sublinear_tf`, `min_df=2`, `max_features=200,000`. `LogisticRegression(max_iter=4000, class_weight="balanced")` |
| TF-IDF + SVM | same features. `LinearSVC(class_weight="balanced")` |
| SBERT + LR | `all-MiniLM-L6-v2` (original STARRT.ai design) or `all-mpnet-base-v2`. Normalised embeddings, default max sequence length. `LogisticRegression(C=4, max_iter=4000, class_weight="balanced")` |
| DistilBERT | `distilbert-base-uncased`, 3 epochs, AdamW lr 5e-5, weight decay 0.01, batch 32, max length 256, linear schedule with 6% warm-up, gradient clip 1.0, fp16, seed 42 for every fold. No validation split and no early stopping, so the test fold is never used for tuning. |
| Qwen2.5-3B zero-shot | `Qwen/Qwen2.5-3B-Instruct`, fp16, no training. Details in the list below. |

How the zero-shot model works:
- The prompt lists every field with the first sentence (at most 22 words) of its ESCO definition, or of its O\*NET definition if there is no ESCO one. The six Faruque labels use short hand-written definitions (`FARUQUE_DEF`).
- The person's text is cut to 250 words.
- Each class name followed by `<|im_end|>` is scored by its summed log-probability, and the highest score wins. The output is therefore always a valid label.
- The model is run on a stratified sample of fold 0: at most 40 rows per class, `random_state=0`. That gives 480 rows for Djinni and 212 for Faruque.
- On the pilot, all 13 classes are allowed.

Hyper-parameters were set once, before the results were seen, and never tuned on test folds.

Compute:
- TF-IDF ran on CPU.
- All other models ran on one Kaggle T4 GPU: about 1.5 h for DistilBERT, 3 datasets × 5 folds, and about 30 min for the LLM.

## 6. Metrics and statistics (`kaggle/common.py`, `scripts/analyze_results.py`)

- Accuracy and macro-F1 are computed per fold. Tables report the mean ± SD (population SD over 5 folds).
- Top-3 accuracy is reported for models that give probabilities.
- **Paired tests:** exact McNemar (two-sided binomial on discordant pairs) on the pooled out-of-fold predictions, which share the same rows in the same order (checked by asserts).
- **LLM comparison:** the trained models' fold-0 predictions are taken on exactly the LLM's sampled rows. Faruque ids are not unique, so rows are matched by position (checked by asserts).
- **Pilot:** models are trained on all of Djinni and tested on 33 answers. Accuracy is given with 95% Wilson intervals.
- **Hierarchical accuracy:** a prediction counts as correct if it is in the same parent group as the true label.

## 7. Reproducibility checks done

- `reproduce.sh` run from the raw downloads gave fold files identical to the released ones (same rows, labels and folds), with and without the private pilot file.
- `analyze_results.py` on the released predictions gives exactly the numbers in the paper.
- GPU results can vary slightly between GPU types and library versions (non-deterministic CUDA kernels), so a re-run of DistilBERT may differ in the third decimal. TF-IDF and SBERT + LR results are deterministic given the same embeddings.

## 8. Known limitations

- Labels are self-reported or platform categories, not expert-annotated. There is no inter-annotator agreement.
- Djinni has no Embedded & IoT class, so the Djinni models predict 12 classes.
- The pilot has only 33 answers, so its Wilson intervals are wide (for example 0.727–0.952 for 29/33).
- The LLM was tested on a sample, not on all folds, because of GPU time.
- DistilBERT used three seeds on Djinni and Faruque and one on Stack Overflow. The other models are deterministic given the folds.

## 9. Leakage and validity checks (`scripts/review_analyses.py`)

- **Near-duplicates.** For each test row, the highest TF-IDF cosine similarity to the training rows of the same fold. At a threshold of 0.9:
  - Djinni: 0 rows
  - Faruque: 273 rows (12.8%)
  - Stack Overflow: 1,044 rows (4.3%)

  Re-scoring every model without these rows cost at most 0.011 on Faruque and 0.008 on Stack Overflow. DistilBERT stayed the best model on all three datasets.
- **Djinni ablations** (TF-IDF + SVM, same folds):

  | Text | Accuracy |
  |---|---|
  | Full | 0.886 |
  | Without "Looking For" | 0.882 |
  | Role and keyword words masked | 0.854 |
  | Both | 0.852 |

- **Controlled text type.** Djinni profiles were reduced to the Stack Overflow tools they mention, keeping only profiles with at least 2 tools (12,133). On these same profiles, TF-IDF + SVM scored 0.885 on the full text and 0.676 on the tool list alone.
- **Zero-shot LLM.** Wilson intervals and exact McNemar tests against every trained model on the same rows, with Holm adjustment.
- **Holm correction.** Applied to all 12 DistilBERT McNemar tests.
- **Extra seeds.** DistilBERT was re-run on Djinni and Faruque with seeds 13 and 7 (`scripts/seed_analysis.py`, `kaggle/STARRT_review_checks.ipynb`).

  | Dataset | Seed 42 | Seed 13 | Seed 7 | Mean ± SD | McNemar p vs TF-IDF + SVM (42 / 13 / 7) |
  |---|---|---|---|---|---|
  | Djinni | 0.889 | 0.890 | 0.890 | 0.890 ± 0.001 | 0.060 / 0.002 / 0.010 |
  | Faruque | 0.931 | 0.932 | 0.935 | 0.933 ± 0.002 | 0.13 / 0.062 / 0.017 |

  The DistilBERT versions of the Djinni ablations (`djinni_nolook`, `djinni_masked`, `djinni_tools`) can be run with the same notebook. They were not run for the paper, which reports these ablations with TF-IDF + SVM only.
