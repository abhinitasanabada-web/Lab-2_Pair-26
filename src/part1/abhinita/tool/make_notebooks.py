"""Generate the two Part 1 notebooks from one source of truth (no nbformat dependency).

  python tools/make_notebooks.py

Creates:
  notebooks/p1_rag.ipynb                  - imports the `rag` package (Colab: clone/upload the repo)
  notebooks/p1_rag_kaggle_selfcontained.ipynb - same, but first writes every rag/*.py file out with
                                             %%writefile, so it runs on Kaggle with no repo access.
Re-run this script whenever rag/*.py changes, then Restart & Run All to save the outputs.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "config", "schema", "ingest", "figure_reader", "index", "memory", "retrieve",
           "llm", "answer", "pipeline", "evaluate", "utils"]


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


CELL1 = '''
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try:
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
except Exception:
    commit = ""
print({"commit": commit or "n/a (not a git checkout)", "python": sys.version.split()[0], "torch": torch.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})
'''

INSTALL = '''
# Cell 2 - install dependencies (Colab/Kaggle). torch/numpy/pandas are already present.
# Versions are pinned in requirements.txt; on Kaggle, turn "Internet" ON for the install, or attach the packages as a dataset.
!pip -q install transformers==4.49.0 accelerate==1.3.0 bitsandbytes==0.45.2 sentence-transformers==3.4.1 chromadb==0.6.3 pdfplumber==0.11.5
'''

BOOTSTRAP = '''
# Cell 3 - make the `rag` package importable.
# Colab:  upload/clone the repo so that /content/data266-lab2-team/rag exists (private repo: use a token), or
#         upload the competition data folder and set DATA_DIR below.
import os, sys, pathlib
for cand in ["/content/data266-lab2-team", "/kaggle/working", "."]:
    if (pathlib.Path(cand) / "rag" / "config.py").exists():
        sys.path.insert(0, cand); break
# os.environ["DATA_DIR"] = "/content/data"          # folder holding kb_manifest.json + the 3 PDFs (auto-detected on Kaggle)
# os.environ["CAP_GPU_16GB"] = "1"                  # default: on an A100, behave like the 16 GB grading GPU
# os.environ["RAG_BACKEND"] = "stub"                # ONLY to test plumbing without a GPU; answers are not real
'''

LOAD = '''
# Cell 4 - load the vision-language model (reads figures AND writes answers). 4-bit Qwen2.5-VL-7B fits a 16 GB T4.
import logging; logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
from rag import config
from rag.llm import load_backend
backend = load_backend()
print("backend:", backend.name)
'''

BUILD = '''
# Cell 5 - Task 1.1 ingestion + indexing: text chunks, tables, figure crops (with labels), figure descriptions,
# then a ChromaDB dense index + BM25 index. Low-trust material (DOC3 Appendix C) is stored but never retrieved.
from rag.pipeline import build_system
system = build_system(backend)
system.stats
'''

SMOKE = '''
# Cell 6 - smoke test on the first question. Never fatal: problems are shown, not raised.
from rag.pipeline import read_questions
from rag.memory import ConversationMemory
from rag.retrieve import retrieve
from rag.answer import answer_question
import json
q = read_questions()
first = q.iloc[0]
ev = retrieve(system.index, first["question"], ConversationMemory("smoke"))
ans, dbg = answer_question(first["question"], ev, ConversationMemory("smoke"), backend)
print(json.dumps(ans, indent=1)); print("issues:", dbg["issues"]); print("evidence:", [i.item_id for i in ev.items])
'''

RUN = '''
# Cell 7 - answer every question (conversation by conversation) and write submission.csv with no manual step.
# QUESTIONS_PATH is honoured so graders can swap in the hidden set.
import time, pathlib
from rag.pipeline import run_questions, validate_submission
from rag.config import questions_path, find_data_dir, work_dir
q = read_questions(questions_path(find_data_dir()))
t0 = time.time()
sub = run_questions(system, q, "submission.csv", work_dir() / "trace_p1.jsonl")
print(f"answered {len(sub)} questions in {time.time()-t0:.0f}s")
problems = validate_submission(sub, q)
print("format problems:", problems or "none")
sub.head(10)
'''

EVAL = '''
# Cell 8 - Task 1.4 evaluation on YOUR golden set (golden/golden_set.csv): baseline (dense only) vs final (hybrid).
import pandas as pd, pathlib
from rag.evaluate import evaluate
gpath = next((p for p in ["golden/golden_set.csv", "../golden/golden_set.csv"] if pathlib.Path(p).exists()), None)
golden = pd.read_csv(gpath, dtype=str, keep_default_na=False) if gpath else pd.DataFrame()
if len(golden):
    results = [evaluate(system, golden, mode=m)[0] for m in ("dense", "hybrid")]
    display(pd.DataFrame(results))
else:
    print("golden/golden_set.csv is empty - write your 25 questions first (see README).")
'''


def build(selfcontained: bool) -> dict:
    cells = [md("# DATA 266 Lab 2 - Part 1: Multimodal RAG over a read-only knowledge base\n"
                "Modular pipeline in `rag/` (ingest -> figure descriptions -> hybrid index -> grounded answers). "
                "Run all cells top to bottom; `submission.csv` is written by Cell 7."),
             code(CELL1), code(INSTALL)]
    if selfcontained:
        cells.append(md("### Package source (written to disk so this notebook runs without the repo)"))
        for m in MODULES:
            src = (ROOT / "rag" / f"{m}.py").read_text()
            cells.append(code(f"%%writefile rag/{m}.py\n{src}"))
    cells += [code(BOOTSTRAP), code(LOAD), code(BUILD), code(SMOKE), code(RUN), code(EVAL)]
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "accelerator": "GPU"}}


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    (out / "p1_rag.ipynb").write_text(json.dumps(build(False), indent=1))
    (out / "p1_rag_kaggle_selfcontained.ipynb").write_text(json.dumps(build(True), indent=1))
    print("wrote", [p.name for p in out.glob("*.ipynb")])
