"""Central configuration for the Part 1 multimodal RAG system.

Everything that you may want to change (model names, paths, retrieval sizes) lives here,
so the notebook and the other modules never hard-code it. Values can be overridden with
environment variables, which is handy on Kaggle/Colab.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

SEED = 266  # Lab requirement: fixed seed wherever randomness is involved

# ---------------------------------------------------------------- models
# One vision-language model does everything (figure reading, answering). Using a single
# model keeps us inside a 16 GB GPU (T4) when loaded in 4-bit.
#   * Colab/Kaggle T4 (16 GB): Qwen2.5-VL-7B-Instruct in 4-bit  (~6 GB)  <- default
#   * If bitsandbytes is a problem or memory is tight: Qwen/Qwen2.5-VL-3B-Instruct in fp16
#
# GPU PROFILES. The lab says all required work MUST run on one 16 GB GPU, and graders re-run the
# tagged notebook on a Kaggle T4/P100 within 90 minutes. So the DEFAULTS below are T4-safe.
#   * Developing on an A100 is fine: keep the defaults (same behaviour as the final run) and set
#     CAP_GPU_16GB=1 so PyTorch refuses to use more than 16 GB - if it runs, it will run on a T4.
#   * To experiment with the larger-precision model on the A100 (LOAD_4BIT=0), re-run the final
#     numbers with the defaults before tagging p1-v1.0.
VLM_NAME = os.environ.get("VLM_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
LOAD_4BIT = os.environ.get("LOAD_4BIT", "1") == "1"
CAP_GPU_16GB = os.environ.get("CAP_GPU_16GB", "1") == "1"  # simulate a 16 GB card on bigger GPUs
EMBED_NAME = os.environ.get("EMBED_NAME", "BAAI/bge-small-en-v1.5")
BACKEND = os.environ.get("RAG_BACKEND", "hf")  # "hf" = real models, "stub" = no models (plumbing test only)

# ---------------------------------------------------------------- generation
MAX_NEW_TOKENS_ANSWER = 320
MAX_NEW_TOKENS_FIGURE = 700
FIG_MAX_PIXELS = 1000 * 28 * 28  # caps image tokens per figure so two images fit comfortably on a T4
ANSWER_MAX_WORDS = 120           # competition rule

# ---------------------------------------------------------------- chunking / retrieval
CHUNK_WORDS = 140
CHUNK_OVERLAP = 30
TOP_K_DENSE = 20
TOP_K_BM25 = 20
RRF_K = 60                 # reciprocal-rank-fusion constant
FINAL_EVIDENCE_ITEMS = 6   # items handed to the answering model (besides exact-label hits)
MAX_IMAGES_PER_QUESTION = 2

# ---------------------------------------------------------------- answer validation
# If True, an "answered" reply whose numbers cannot be traced to the evidence (even after one
# retry) is downgraded to not_found. Default False because multi-step calculations can produce
# legitimately new numbers; the count of ungrounded answers is always reported.
STRICT_NUMBERS = os.environ.get("STRICT_NUMBERS", "0") == "1"
NUMBER_TOLERANCE = 0.011   # relative/absolute tolerance when matching numbers (about +-1%)

STATUSES = {"answered", "not_found", "conflict", "false_premise", "session_override"}


def find_data_dir() -> Path:
    """Locate the folder that holds kb_manifest.json (local repo, Colab, or Kaggle input)."""
    candidates = []
    if os.environ.get("DATA_DIR"):
        candidates.append(os.environ["DATA_DIR"])
    candidates += ["data", "../data", "."]
    candidates += glob.glob("/kaggle/input/*")          # Kaggle competition data
    candidates += glob.glob("/content/**/", recursive=False)  # Colab uploads
    for c in candidates:
        if (Path(c) / "kb_manifest.json").exists():
            return Path(c)
    raise FileNotFoundError("kb_manifest.json not found - set DATA_DIR to the competition data folder")


def questions_path(data_dir: Path) -> str:
    """Graders swap in a hidden file through QUESTIONS_PATH, so honour it first."""
    return os.environ.get("QUESTIONS_PATH", str(data_dir / "Questions.csv"))


def work_dir() -> Path:
    """Writable scratch area (figure crops, caches, traces)."""
    base = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("outputs")
    base.mkdir(parents=True, exist_ok=True)
    return base
