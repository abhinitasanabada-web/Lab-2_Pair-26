"""End-to-end pipeline: build the knowledge base, answer Questions.csv, write submission.csv.

Fail-soft guarantees (a crash late in a 90-minute Kaggle run costs everything, so we avoid it):
  * every question produces exactly one row, even if its answer step errors;
  * the CSV is re-written after every conversation, so partial progress is never lost;
  * validation of the final file only REPORTS problems and repairs what it can.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .answer import answer_question
from .config import BACKEND, find_data_dir, questions_path, work_dir
from .figure_reader import describe_figures
from .index import Embedder, HybridIndex
from .ingest import ingest_all
from .memory import ConversationMemory
from .retrieve import retrieve
from .schema import LABEL_PATTERN

log = logging.getLogger("rag.pipeline")
COLS = ["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]
NOT_FOUND_ROW = "The knowledge base does not contain this information."


@dataclass
class System:
    """Everything needed to answer questions: the (read-only) index and the model backend."""
    index: HybridIndex
    backend: object
    stats: dict


def build_system(backend, data_dir: Path | None = None, use_embedder: bool | None = None) -> System:
    """Tasks 1.1 + 1.2 set-up: ingest PDFs, describe figures, build dense + BM25 indexes."""
    data_dir = data_dir or find_data_dir()
    wd = work_dir()
    t0 = time.time()
    items = ingest_all(data_dir, wd / "figs")
    fig_stats = describe_figures(items, backend, wd / "figure_descriptions.json")
    if use_embedder is None:
        use_embedder = BACKEND != "stub"
    embedder = None
    if use_embedder:
        try:
            embedder = Embedder()
        except Exception as e:  # no sentence-transformers: BM25-only still works
            log.warning("Embedder unavailable (%s); dense retrieval disabled", e)
    index = HybridIndex(items, embedder)
    kinds = pd.Series([f"{i.doc_id}/{i.kind}" for i in items]).value_counts().to_dict()
    stats = {"items": len(items), "by_doc_kind": kinds, "figure_descriptions": fig_stats,
             "dense_enabled": embedder is not None, "build_seconds": round(time.time() - t0, 1)}
    return System(index, backend, stats)


def _row(qrow: pd.Series, ans: dict) -> dict:
    """One submission row in the exact format the competition expects."""
    return {"id": qrow["id"], "conversation_id": qrow["conversation_id"], "question_id": qrow["question_id"],
            "answer": " ".join(str(ans["answer"]).split()) or NOT_FOUND_ROW,
            "figure_references": ",".join(ans["figure_references"]) or "NONE",
            "citations": ",".join(ans["citations"]) or "NONE", "status": ans["status"]}


def run_questions(system: System, q: pd.DataFrame, out_csv: str = "submission.csv",
                  trace_path: Path | None = None, mode: str = "hybrid") -> pd.DataFrame:
    """Answer every conversation separately, turn by turn, each with a fresh empty memory."""
    rows, traces = [], []
    base_count = system.index.count()
    for conv_id, turns in q.groupby("conversation_id", sort=False):
        memory = ConversationMemory(str(conv_id))  # new, empty memory per conversation
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            t0 = time.time()
            try:
                ev = retrieve(system.index, r["question"], memory, mode=mode)
                ans, dbg = answer_question(r["question"], ev, memory, system.backend)
            except Exception as e:  # absolute last line of defence: still emit a valid row
                ev, dbg = None, {"issues": [f"pipeline_exception:{type(e).__name__}:{str(e)[:120]}"], "raw": ""}
                ans = {"answer": NOT_FOUND_ROW, "status": "not_found", "citations": [], "figure_references": []}
            memory.add(r["question"], ans)
            secs = time.time() - t0
            rows.append(_row(r, ans))
            traces.append({"id": r["id"], "conversation_id": conv_id, "turn": r["turn"], "seconds": round(secs, 2),
                           "evidence": [i.item_id for i in ev.items] if ev else [],
                           "images": ev.images if ev else [], "status": ans["status"], **dbg})
        # progress is saved after each conversation; a later crash cannot lose finished rows
        pd.DataFrame(rows, columns=COLS).to_csv(out_csv + ".partial", index=False)
    if system.index.count() != base_count:
        log.error("base index size changed during the run - conversations must never write to it")
    sub = pd.DataFrame(rows, columns=COLS).set_index("id").loc[q["id"]].reset_index()
    sub.to_csv(out_csv, index=False)
    if trace_path is not None:
        trace_path.write_text("\n".join(json.dumps(t) for t in traces))
    return sub


# ------------------------------------------------------------------ validation (mirrors validate_submission.py)
_LAB = LABEL_PATTERN
_FIG = re.compile(rf"^(NONE|{_LAB}(,{_LAB})*)$")
_CIT = re.compile(rf"^(NONE|DOC[1-3]:(p\d+|{_LAB})(,DOC[1-3]:(p\d+|{_LAB}))*)$")
_STATUS = {"answered", "not_found", "conflict", "false_premise", "session_override"}


def validate_submission(sub: pd.DataFrame, q: pd.DataFrame) -> list[str]:
    """Return a list of problems (empty list = valid). Same checks as the official validator."""
    problems = []
    if list(sub.columns) != COLS:
        problems.append("wrong columns/order")
    if list(sub["id"]) != list(q["id"]):
        problems.append("ids missing/extra/out of order")
    for i, r in sub.iterrows():
        if r["status"] not in _STATUS:
            problems.append(f"row {r['id']}: bad status")
        if not str(r["answer"]).strip():
            problems.append(f"row {r['id']}: empty answer")
        if not _FIG.match(str(r["figure_references"])):
            problems.append(f"row {r['id']}: bad figure_references {r['figure_references']!r}")
        if not _CIT.match(str(r["citations"])):
            problems.append(f"row {r['id']}: bad citations {r['citations']!r}")
    return problems


def read_questions(path: str | None = None) -> pd.DataFrame:
    """Read the questions file as text exactly as the lab specifies (never edit/reorder rows)."""
    path = path or questions_path(find_data_dir())
    return pd.read_csv(path, dtype=str, keep_default_na=False)
