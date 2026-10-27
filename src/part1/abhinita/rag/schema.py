"""Shared data structures and label helpers (no heavy imports, safe to import anywhere)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Same label grammar as validate_submission.py: "Figure A-3", "Table 1-2", "Figure B-5 (corrected)".
LABEL_PATTERN = r"(Figure|Table) ([12]|[A-C])-(\d+)( \(corrected\))?"
LABEL_RE = re.compile(LABEL_PATTERN)
# Looser matcher for user questions ("fig. a-3", "FIGURE B-5 corrected").
QUESTION_LABEL_RE = re.compile(r"\b(Fig(?:ure)?\.?|Table)\s+([12]|[A-Ca-c])\s*-\s*(\d+)(\s*\(?corrected\)?)?", re.I)


def normalise_label(kind: str, group: str, num: str, corrected: bool = False) -> str:
    kind = "Table" if kind.lower().startswith("t") else "Figure"
    return f"{kind} {group.upper()}-{num}" + (" (corrected)" if corrected else "")


# "Figures A-6 and A-7", "Tables 1-1, 1-2 and 1-3": a plural keyword followed by a list of numbers
PLURAL_LIST_RE = re.compile(
    r"\b(Fig(?:ure)?s\.?|Tables)\s+((?:[12]|[A-Ca-c])\s*-\s*\d+(?:\s*\(?corrected\)?)?"
    r"(?:\s*(?:,|and|&|or)\s*(?:[12]|[A-Ca-c])\s*-\s*\d+(?:\s*\(?corrected\)?)?)*)", re.I)
_ITEM_RE = re.compile(r"([12]|[A-Ca-c])\s*-\s*(\d+)(\s*\(?corrected\)?)?", re.I)


def labels_in_question(question: str) -> list[str]:
    """Every figure/table label explicitly named in a question, in order, de-duplicated."""
    found: list[tuple[int, str]] = []
    for m in QUESTION_LABEL_RE.finditer(question):
        found.append((m.start(), normalise_label(m.group(1), m.group(2), m.group(3), bool(m.group(4)))))
    for m in PLURAL_LIST_RE.finditer(question):
        for im in _ITEM_RE.finditer(m.group(2)):
            found.append((m.start() + im.start(), normalise_label(m.group(1), im.group(1), im.group(2),
                                                                 bool(im.group(3)))))
    out: list[str] = []
    for _, lab in sorted(found):
        if lab not in out:
            out.append(lab)
    return out


def base_label(label: str) -> str:
    return label.replace(" (corrected)", "")


@dataclass
class Item:
    """One retrievable unit: a text chunk, a table, or a figure."""
    item_id: str
    kind: str            # "text" | "table" | "figure"
    doc_id: str          # DOC1 | DOC2 | DOC3
    page: int            # 1-based physical PDF page
    trust: str           # high | medium | low (low is stored but never retrieved)
    text: str            # what gets embedded / shown to the LLM
    label: str = ""      # e.g. "Figure A-3" (tables and figures only)
    section: str = ""
    image_path: str = ""  # PNG crop for figures
    is_revision: bool = False  # revision notice / erratum / correction text
    meta: dict = field(default_factory=dict)

    def cite_ids(self) -> list[str]:
        """Citation strings this item can legitimately support."""
        ids = [f"{self.doc_id}:p{self.page}"]
        if self.label:
            ids.insert(0, f"{self.doc_id}:{self.label}")
        return ids


@dataclass
class Evidence:
    """What the answering step receives for one question."""
    items: list[Item]
    images: list[str]          # image paths passed to the VLM
    requested_labels: list[str]
    notes: list[str] = field(default_factory=list)  # e.g. "Figure B-5 has a corrected version"

    def allowed_citations(self) -> set[str]:
        out: set[str] = set()
        for it in self.items:
            out.update(it.cite_ids())
        return out
