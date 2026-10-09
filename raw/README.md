# Raw downloads

The raw files are not redistributed here because of their size, and because each source should be downloaded under its own terms. To rebuild everything with `reproduce.sh`, put the files at exactly these paths. The benchmark files in `kaggle/data/` are already included, so you only need this step if you want to check or change the cleaning.

| Path | Source | Licence | Notes |
|---|---|---|---|
| `djinni/candidates_en.parquet` | Djinni Recruitment Dataset, English candidate profiles: <https://huggingface.co/datasets/lang-uk/recruitment-dataset-candidate-profiles-english> (paper: Drushchak & Romanyshyn, UNLP 2024, <https://aclanthology.org/2024.unlp-1.2>) | MIT | 210,250 rows. Rename the parquet file from "Files and versions" to `candidates_en.parquet`. |
| `stackoverflow/2023/results.csv`, `stackoverflow/2023/schema.csv` | Stack Overflow Developer Survey 2023: <https://survey.stackoverflow.co/> | ODbL 1.0 (database), DbCL 1.0 (contents) | Download the year's zip, then rename `survey_results_public.csv` to `results.csv` and `survey_results_schema.csv` to `schema.csv`. |
| `stackoverflow/2024/...` | Developer Survey 2024 | same | same |
| `stackoverflow/2025/...` | Developer Survey 2025 | same | same |
| `faruque/Career_Dataset.xlsx` | Student Career Dataset, Faruque (2024), Mendeley Data v2: <https://doi.org/10.17632/4spj4mbpjr.2> | CC BY 4.0 | Paper: Faruque, Khushbu & Akter, arXiv:2405.18139. |
| `onet/occupation_data.csv` | O\*NET 31.0 Database, "Occupation Data": <https://www.onetcenter.org/database.html> | CC BY 4.0 | Only this file is used by the pipeline. |
| `esco/occupations_en.csv` | ESCO v1.2.1, English, CSV: <https://esco.ec.europa.eu/en/use-esco/download> | European Commission (free of charge; accept the terms shown at download) | Only `occupations_en.csv` is used. |
| `starrt_old/forms_data.csv` | STARRT.ai Google Form answers (Nov 2025) | private, not released | Optional. Without it, every pilot step is skipped. |

All licences were checked on 8 October 2026. Please check them again before you redistribute anything.
