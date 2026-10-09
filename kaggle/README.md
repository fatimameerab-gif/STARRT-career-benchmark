# Running the experiments on Kaggle

1. Zip this `kaggle/` folder and upload it as a Kaggle dataset: **Create → New Dataset**. `common.py` finds `label_set.csv` anywhere under `/kaggle/input/`.
2. Open a new notebook, choose **File → Import Notebook**, and select `STARRT_experiments.ipynb`. Then add the dataset as input.
3. Under **Settings**, set **Accelerator** to **GPU T4** and turn **Internet** on, so the models can download.
4. Run the smoke-test cell first, then the rest. On a T4, approximate times are:
   - TF-IDF: about 15 min
   - Sentence-BERT + LR: about 15 min
   - DistilBERT, 3 datasets × 5 folds: about 1.5 h
   - Qwen2.5-3B zero-shot: about 30 min
5. Results go to `/kaggle/working/results/`. The last cell zips them.

To run locally, use `python run_tfidf.py` (CPU). For the others (`run_sbert.py`, `run_distilbert.py`, `run_llm_zeroshot.py`) a GPU is recommended. Every script takes `--datasets` and `--folds`, and DistilBERT also takes `--limit` for a quick test.

The private Pakistani pilot set (`data/forms_test.csv`) is not part of the public release. Every pilot step is skipped when the file is absent.
