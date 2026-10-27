"""Task 1.4 - evaluation on YOUR golden set (25 questions you write yourselves, before tuning).

golden/golden_set.csv columns (read as text):
  id, conversation_id, turn, question, category, gold_answer, gold_status, gold_sources
  gold_sources = the pages/figures that really contain the answer, e.g. "DOC2:p4|DOC2:Figure A-3"
Questions copied or paraphrased from Questions.csv do not count - write new ones from the PDFs.

Primary:   answer correctness (every number in gold_answer found within +-1% in the answer AND status right)
Secondary: Retrieval Recall@5 (a gold source is among the top-5 evidence items) and citation accuracy
Efficiency: median seconds per question
Run twice - mode='dense' (baseline) and mode='hybrid' (final) - to fill the results table.
"""
from __future__ import annotations

import statistics
import time

import pandas as pd

from .answer import answer_question, numbers_in
from .memory import ConversationMemory
from .retrieve import retrieve


def _within(x: float, pool: list[float], tol: float = 0.01) -> bool:
    return any(abs(x - y) <= tol * max(1.0, abs(y)) for y in pool)


def _sources(it) -> set[str]:
    return set(it.cite_ids())


def evaluate(system, golden: pd.DataFrame, mode: str = "hybrid") -> tuple[dict, pd.DataFrame]:
    rows = []
    for conv, turns in golden.groupby("conversation_id", sort=False):
        memory = ConversationMemory(str(conv))
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            t0 = time.time()
            ev = retrieve(system.index, r["question"], memory, mode=mode)
            ans, dbg = answer_question(r["question"], ev, memory, system.backend)
            secs = time.time() - t0
            memory.add(r["question"], ans)

            gold_src = {s.strip() for s in r["gold_sources"].split("|") if s.strip()}
            top5 = ev.items[:5]
            recall = any(_sources(it) & gold_src for it in top5)
            gold_nums = numbers_in(r["gold_answer"])
            num_ok = all(_within(n, numbers_in(ans["answer"])) for n in gold_nums)
            correct = num_ok and ans["status"] == r["gold_status"]
            cited = ans["citations"]
            cit_acc = (sum(c in gold_src for c in cited) / len(cited)) if cited else float(not gold_src)
            rows.append({"id": r["id"], "category": r.get("category", ""), "correct": correct, "recall@5": recall,
                         "citation_accuracy": cit_acc, "seconds": secs, "status": ans["status"],
                         "gold_status": r["gold_status"], "answer": ans["answer"], "issues": ";".join(dbg["issues"])})
    df = pd.DataFrame(rows)
    summary = {"mode": mode, "n": len(df), "answer_correctness": f"{int(df.correct.sum())}/{len(df)}",
               "retrieval_recall@5": round(df["recall@5"].mean(), 3),
               "citation_accuracy": round(df.citation_accuracy.mean(), 3),
               "median_seconds_per_question": round(statistics.median(df.seconds), 2)}
    return summary, df
