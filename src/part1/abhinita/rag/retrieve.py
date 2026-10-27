"""Task 1.2 - build the evidence pack for one question.

Order of evidence:
  1. Exact label hits. If the question names "Figure A-6" we fetch exactly that figure - never a
     look-alike (A-7). If a corrected version of a named label exists we also fetch it and add a
     note, so the answer can say the correction supersedes the original.
  2. Hybrid (dense + BM25) results, low-trust items already excluded by the index.
  3. One revision/erratum passage if any appears among the strong candidates, so revisions are
     never missed.
Follow-up questions ("And in 2013?") are enriched with earlier turns of the SAME conversation.
"""
from __future__ import annotations

from .config import FINAL_EVIDENCE_ITEMS, MAX_IMAGES_PER_QUESTION
from .index import HybridIndex
from .memory import ConversationMemory
from .schema import Evidence, Item, base_label, labels_in_question


def build_query(question: str, memory: ConversationMemory | None) -> str:
    """Retrieval query = current message + earlier user messages of this conversation."""
    if memory is None or not memory.turns:
        return question
    return question + " " + " ".join(memory.previous_questions(2))


def exact_label_items(index: HybridIndex, labels: list[str]) -> list[Item]:
    """Items whose label equals a requested label, plus corrected variants of a requested base label."""
    hits: list[Item] = []
    for lab in labels:
        for it in index.items:
            if not it.label or it.trust == "low":
                continue
            same = it.label == lab
            corrected_variant = (not lab.endswith("(corrected)")) and it.label == f"{lab} (corrected)"
            if (same or corrected_variant) and it not in hits:
                hits.append(it)
    # corrected versions first: they supersede the printed original
    hits.sort(key=lambda it: 0 if it.label.endswith("(corrected)") else 1)
    return hits


def retrieve(index: HybridIndex, question: str, memory: ConversationMemory | None = None,
             mode: str = "hybrid", k: int = FINAL_EVIDENCE_ITEMS) -> Evidence:
    labels = labels_in_question(question)
    inherited = False
    if not labels and memory is not None and memory.turns and len(question.split()) <= 12:
        labels = memory.previous_labels()[:1]  # short follow-up with no label: keep the earlier one
        inherited = bool(labels)

    items: list[Item] = exact_label_items(index, labels)
    notes: list[str] = []
    for it in items:
        if it.label.endswith("(corrected)") and base_label(it.label) in labels + [base_label(l) for l in labels]:
            notes.append(f"{it.label} is a corrected version that supersedes the originally printed "
                         f"{base_label(it.label)}.")
    if inherited:
        notes.append("The label(s) above were mentioned earlier in this conversation.")

    n_exact = len(items)
    ranked = index.search(build_query(question, memory), mode=mode)
    for it, _ in ranked:
        if len(items) >= n_exact + k:  # k extra items on top of the exact-label hits
            break
        if it not in items:
            items.append(it)

    # make sure a strong revision/erratum passage is not missed
    if mode == "hybrid" and not any(i.is_revision for i in items):
        for it, _ in ranked[:10]:
            if it.is_revision and it not in items:
                items.append(it)
                break

    images = [it.image_path for it in items if it.kind == "figure" and it.label in labels_with_variants(labels)
              and it.image_path][:MAX_IMAGES_PER_QUESTION]
    return Evidence(items=items, images=images, requested_labels=labels, notes=notes)


def labels_with_variants(labels: list[str]) -> set[str]:
    out = set(labels)
    out.update(f"{base_label(l)} (corrected)" for l in labels)
    return out
