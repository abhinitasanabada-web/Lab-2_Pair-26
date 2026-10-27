# DATA 266 Lab 2 - Team #26

## Part 1 - Multimodal RAG over a read-only knowledge base

Notebook: `notebooks/p1_rag_v4.ipynb` (tag `p1-v1.0`). Parts 2 and 3 are not in this folder yet.

Hosted models used: none. Everything runs locally on one GPU, with no API keys.

### Models
| Purpose | Model |
|---|---|
| Figure reading + answers | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit nf4, bitsandbytes) |
| Text embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `BAAI/bge-reranker-base` |

### How to run
1. **Colab:** Runtime > Change runtime type > GPU, then upload the competition zip next to the notebook.
   **Local GPU (RTX 4070 / A100):** create a Python env, put the competition files (or the zip) next to the notebook.
   **Kaggle:** attach the competition data, GPU on, Internet on for the pip cell.
2. **Restart & Run All.** The notebook installs its packages, builds the knowledge base, answers
   `QUESTIONS_PATH` (default `Questions.csv`), checks the format with `validate_submission.py`
   and writes `submission.csv`.
3. Golden-set evaluation (section 13) runs automatically when `eval/golden_set.csv` has rows.

Fits on one 16 GB GPU (the 7B model is loaded in 4-bit). The setup cell prints the GPU profile it chose
(`VLM_DTYPE`, `VLM_BATCH`). To reproduce another machine's run exactly, set those two environment
variables to the printed values and copy that machine's `work/kb_cache/` folder (cached figure descriptions).

### Pipeline
PyMuPDF ingestion (text chunks, table words, figure crops) -> VLM figure descriptions (cached)
-> fact store with precedence rules (revision > outturn > forecast, trust level, corrected figures)
-> ChromaDB dense + BM25 hybrid retrieval (RRF + exact-label boost + reranker)
-> LangGraph answer graph -> NVIDIA NeMo Guardrails + regex rails -> JSON answer (<= 120 words).

### Design rules against hallucination
* Low-trust content (DOC3 Appendix C) is stored in ChromaDB with `trust_level=low` but excluded from
  retrieval; text that gives instructions to AI systems is dropped and treated as data, never followed.
* A figure or table named in the question is fetched by its exact label; a corrected figure replaces the original.
* Numbers come from the fact store or the retrieved evidence; citations and `figure_references`
  are built from the evidence actually used.
* Values the user supplies live only in that conversation's memory, never in the index.
* Every question produces a row; unanswerable or false-premise questions get the matching status.

### Repository layout
```
notebooks/p1_rag_v4.ipynb   FINAL Part 1 notebook (executed, outputs saved) - self-contained, no local imports
eval/golden_set.csv         our 25 golden questions (written before tuning)
requirements.txt            pinned versions (written by the v4 notebook's last cell)
CONTRIBUTIONS.md            who did what + meeting recordings
AI_USAGE.md                 AI coding assistants used

v1/                         earlier version, kept for history (not used by v4)
  p1_rag_v1.ipynb             first notebook; imports the rag/ package
  p1_rag_v1_0_8_context.ipynb v1 variant with a longer retrieval context
  rag/                        v1 modules: ingest, figure reading, ChromaDB + BM25 index, retrieval, answering
  tool/                       v1 helper script that regenerates the v1 notebooks from rag/*.py
```
Data, model weights, `work/` caches, `submission.csv` and secrets are not committed (see `.gitignore`).

### Version history
| Version | What changed |
|---|---|
| v1 | Modular `rag/` package: hybrid retrieval (dense + BM25 + RRF), the VLM reads figures, the LLM writes every answer. |
| v4 (final) | Single notebook; adds a deterministic fact store (table parser, prose and figure facts) with precedence rules, a question planner, a reranker, LangGraph + NeMo Guardrails, and a GPU profile for T4 / RTX 4070 / A100. Leaderboard 0.855. |

Only v4 is the graded submission. v1 is kept to show how the system evolved.
