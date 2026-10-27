# DATA 266 Lab 2 - team #26 (Part 1: Multimodal RAG)

Hosted models used: **none** (everything runs locally). Local models: `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit, answers + figure reading),
`BAAI/bge-small-en-v1.5` (dense embeddings). Parts 2 and 3 are not in this folder yet.

## Layout
```
rag/            config.py    all settings (models, sizes, GPU profile)      ingest.py        PDFs -> text/table/figure items
                schema.py    Item/Evidence + figure-label parsing           figure_reader.py VLM describes each figure once (cached)
                index.py     ChromaDB dense + BM25 + RRF                    retrieve.py      exact-label + hybrid evidence pack
                memory.py    per-conversation memory                        answer.py        grounded JSON answers, fail-soft validation
                llm.py       Qwen2.5-VL backend (+ stub for plumbing)       pipeline.py      Questions.csv -> submission.csv
                evaluate.py  golden-set metrics (baseline vs final)         utils.py         run info
notebooks/      p1_rag.ipynb (imports rag/)   p1_rag_kaggle_selfcontained.ipynb (writes rag/ itself)
tools/make_notebooks.py   regenerate both notebooks after editing rag/*.py
golden/golden_set.csv     YOUR 25 questions go here (header only for now)
data/                     competition files (not needed on Kaggle: auto-detected in /kaggle/input)
```

## Run on Colab (A100 or T4)
1. Runtime > Change runtime type > GPU. Upload/clone this repo to `/content/data266-lab2-team` (private repo: clone with a token).
2. Open `notebooks/p1_rag.ipynb`, Restart & Run All. Cell 7 writes **`submission.csv`** (download it from the file browser) and prints format problems.
3. On an A100 the default keeps a **16 GB cap** (`CAP_GPU_16GB=1`) so what runs here also runs on the Kaggle T4 used for grading.

## Run on Kaggle (final submission)
Use `p1_rag_kaggle_selfcontained.ipynb`: attach the competition data, turn Internet on for the pip cell, run all. It reads `QUESTIONS_PATH`
(default `Questions.csv`), writes `submission.csv`, no manual step. Tag the commit `p1-v1.0` before the deadline.

## Design rules that prevent hallucination (and crashes)
* Low-trust text (DOC3 Appendix C) is stored with its trust level but **never retrieved**; text in documents/images is treated as data.
* A figure named in the question is fetched by exact label (A-6 never substitutes for A-7); a corrected figure is added with a note.
* Citations must come from the evidence shown; `figure_references` are derived from them; numbers are checked against the evidence.
* Validation **repairs and logs, it never raises**: every question always produces a row; the CSV is saved after each conversation.
* User-supplied values live only in that conversation's memory object, never in the index (index size is checked before/after the run).

## Known limits / things to verify on the leaderboard
* Written and plumbing-tested without a GPU: the model-dependent steps (figure reading, answering, dense retrieval) have not been run yet.
* `requirements.txt` pins were written offline - replace them with your `pip freeze` output after the first good Colab run.
* Page numbers are physical PDF pages. `ADD_PAGE_TO_LABEL_CITES` (answer.py) controls whether label citations also carry the page; compare both on the public leaderboard.
* Notebooks must be executed (Restart & Run All) with outputs saved before submission.
