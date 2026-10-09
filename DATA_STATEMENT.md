# Data statement

This statement follows the schema of Bender & Friedman (2018), *Data Statements for Natural Language Processing*, TACL 6:587–604. It covers the three benchmark files in `kaggle/data/` and the private pilot set used in the paper.

## A. Curation rationale

We wanted a benchmark for predicting a computing career field from text written by the person, using only data with a stated public licence. Earlier career-prediction studies often used small or synthetic data, and sometimes test sets that contained duplicates of the training rows. We picked three sources whose text and labels both come from real people:

| Source | Text | Label comes from | Rows after cleaning | In benchmark |
|---|---|---|---|---|
| Djinni (candidate profiles) | the candidate's own "about me", highlights and "looking for" fields | the platform category the candidate chose (`Primary Keyword`), plus job title for AI/ML | 128,891 | 27,861 |
| Stack Overflow Survey 2023–25 | languages, databases, platforms, web frameworks and collaboration tools the respondent uses, written as one sentence | the respondent's single self-reported role (`DevType`) | 114,879 | 24,380 |
| Faruque Student Career Dataset | comma-separated skills and interests | the student's chosen career field (6 classes) | 2,135 | 2,135 |
| STARRT.ai pilot (private) | answer to "list the five most important actions to become an expert in your field" | the respondent's self-reported field | 34 mapped (33 in Djinni classes) | not released |

Texts were never chosen by how well a model could classify them.

## B. Language variety

The data is English only. The Djinni texts are written by people from a Ukrainian IT platform, mostly non-native writers. Non-English texts were removed with a simple rule: more than 10% non-ASCII characters. The Stack Overflow texts are a fixed template filled with technology names. The Faruque texts were written by Bangladeshi students (en-BD), and the pilot answers by Pakistani students and professionals (en-PK).

## C. Speaker demographics

- **Djinni:** job seekers in IT, mainly from Ukraine and neighbouring countries. The data was anonymised by its publishers. Years of experience are available in the source, but gender and age are not.
- **Stack Overflow:** professional developers and people who code or work with developers as part of their job, from many countries. Pakistan is included. The survey's own country and age fields are not used in the benchmark texts.
- **Faruque:** computer science and software engineering students in Bangladesh.
- **Pilot:** computing students and professionals in Pakistan, collected 14–18 November 2025 through the STARRT.ai Google Form. No names or contact details were collected.

## D. Annotator demographics

There was no new manual annotation. Every label is the person's own choice: a platform category, a survey role, or a form field. The mapping from source categories to our 13 fields (`scripts/labels.py`) was designed by the first author, a computer science graduate in Pakistan, and anchored to ESCO and O\*NET occupation definitions. The mapping is published so that it can be checked and changed.

## E. Speech situation

- Djinni profiles were written by job seekers for recruiters. They are self-promotional and can be long, with a median of 111 words.
- Stack Overflow answers are survey multiple-choice answers, written as a list of tools with a median of 22 words.
- Faruque texts are short survey answers with a median of 17 words.
- The pilot answers are short free text with a median of 64 words. They describe steps to expertise, not the person.

Publication years: Djinni 2024, Stack Overflow 2023–2025, Faruque 2024. The pilot was collected in 2025.

## F. Text characteristics and label distribution

The 13 fields belong to 4 parent groups:

- **Software Development:** Web, Mobile, Game, Embedded & IoT
- **Data & AI:** Data Science & Analytics, Data Engineering & Databases, AI/ML Engineering
- **Infrastructure & Security:** DevOps & Cloud, Cybersecurity, Networking & SysAdmin
- **Quality, Design & Management:** QA & Testing, UI/UX Design, Product & Project Management

Benchmark rows per class (after the 3,000-per-class cap):

| Class | Djinni | Stack Overflow |
|---|---|---|
| Web Development | 3,000 | 3,000 |
| Mobile Development | 3,000 | 3,000 |
| Game Development | 1,790 | 1,491 |
| Embedded & IoT | – | 3,000 |
| Data Science & Analytics | 3,000 | 1,670 |
| Data Engineering & Databases | 1,896 | 3,000 |
| AI/ML Engineering | 895 | 1,011 |
| DevOps & Cloud | 3,000 | 3,000 |
| Cybersecurity | 772 | 892 |
| Networking & SysAdmin | 1,508 | 1,253 |
| QA & Testing | 3,000 | 1,236 |
| UI/UX Design | 3,000 | 356 |
| Product & Project Management | 3,000 | 1,471 |
| **Total** | **27,861** | **24,380** |

Faruque keeps its original 6 labels:

| Label | Rows |
|---|---|
| Software Development and Engineering | 555 |
| Development | 457 |
| Artificial Intelligence | 426 |
| Data Science | 333 |
| Security | 307 |
| UX/UI Design | 57 |

## G. Cleaning (summary; details in `TECHNICAL_NOTES.md`)

- **Djinni:**
  - 21 of the 41 categories are mapped. Language-only categories (Java, .NET, Python, C++ and others) and non-computing categories (HR, sales, marketing) are dropped.
  - Data Science and Python profiles whose job title names ML or AI are relabelled AI/ML Engineering.
  - Graphic, 3D and print designers are removed from Design.
  - Non-English texts, texts under 15 words and exact duplicates are removed.
  - The job title and the CV are never part of the input text.
- **Stack Overflow:**
  - Only professional respondents are kept, and only roles that map to one of our classes.
  - The 2023–2024 role "data scientist or machine learning specialist" is dropped because it mixes two classes.
  - At least 2 of the 5 tool fields must be answered.
- **Faruque:** 1,900 exact duplicates and 41 texts carrying two labels are removed.

## H. Known biases and limits

- **Djinni labels:** these are platform categories, not verified job roles. AI/ML is under-represented because the platform has no AI category; AI/ML profiles are recovered from job titles only.
- **Stack Overflow:** a tool list carries less information about the role than free text, so its accuracy ceiling is low (about 0.5 for every model).
- **Geography:** no source is a random sample of computing students. Pakistan is represented only by Stack Overflow respondents and by the 34 private pilot answers.
- **Gender and age:** not available, so fairness across these groups could not be tested.
- **Use:** the data supports research on career-field classification. It should not be used to make decisions about individual people without human review.

## I. Provenance and licences

Each benchmark file keeps the licence of its source. See `DATA_LICENSES.md`.
