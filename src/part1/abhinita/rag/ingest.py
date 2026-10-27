"""Task 1.1 - Ingestion: PDFs -> text chunks, tables and figure crops, all with provenance.

Every produced `Item` keeps doc_id, page, label, trust_level (read from kb_manifest.json), so
retrieval can filter by trust and answers can cite exactly. Only pdfplumber (+ its pypdfium2
renderer) is used, which is preinstalled on Colab/Kaggle or one `pip install pdfplumber` away.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdfplumber

from .config import CHUNK_OVERLAP, CHUNK_WORDS
from .schema import Item, LABEL_PATTERN

# A *caption* line starts with the label and is followed by nothing, a digit (printed page number
# in DOC1) or a Capitalised title. A *reference* in body text ("Figure 1-4 shows ...") continues with
# a lowercase word, so it is not matched. Optional prefixes cover "1.6 Figure 1-1" style layouts.
CAPTION_RE = re.compile(rf"^(?:[A-Z]\w+\s+|\d+\.\d+\s+)?{LABEL_PATTERN}(?:\s+(?:[A-Z0-9].*)?)?$")
REVISION_RE = re.compile(r"revision notice|erratum|correction to|supersede|has been revised|revised from", re.I)
SECTION_RE = re.compile(r"^(Appendix [A-Z]\b.*|[AB]\.\d\s.*|\d\.\d\s.*)$")
CID_RE = re.compile(r"\(cid:\d+\)")


def load_manifest(data_dir: Path) -> dict:
    return json.loads((Path(data_dir) / "kb_manifest.json").read_text())


def _trust_for_page(doc_meta: dict, page: int) -> str:
    """Document trust, overridden per page for sections such as DOC3's untrusted Appendix C."""
    for ov in doc_meta.get("section_trust_overrides", []):
        if page in ov.get("pages", []):
            return ov["trust_level"]
    return doc_meta["trust_level"]


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", CID_RE.sub("", text)).strip()


def _chunk_words(text: str) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max(1, CHUNK_WORDS - CHUNK_OVERLAP)
    return [" ".join(words[i:i + CHUNK_WORDS]) for i in range(0, len(words), step)
            if words[i:i + CHUNK_WORDS]]


def _caption_lines(page) -> list[dict]:
    """Caption lines on a page with their label, kind and vertical position."""
    caps = []
    for ln in page.extract_text_lines():
        raw = _clean(ln["text"])
        m = CAPTION_RE.match(raw)
        if m:
            # re-run the label regex on the cleaned line to capture the exact label text
            lm = re.search(LABEL_PATTERN, raw)
            caps.append({"label": lm.group(0), "kind": lm.group(1), "top": ln["top"],
                         "bottom": ln["bottom"], "text": raw})
    return caps


def _union_bbox(objs, page, pad=6):
    xs0 = [o["x0"] for o in objs]; xs1 = [o["x1"] for o in objs]
    ts = [o["top"] for o in objs]; bs = [o["bottom"] for o in objs]
    return (max(0, min(xs0) - pad), max(0, min(ts) - pad),
            min(page.width, max(xs1) + pad), min(page.height, max(bs) + pad))


def _figure_bbox(page, cap: dict, used: set) -> tuple:
    """Region of the page that holds the figure for this caption.

    DOC2/DOC3 figures are raster images: pick the nearest unused image at/below the caption.
    DOC1 figures are vector drawings: use the bounding box of the drawing primitives, falling back
    to the whole page when there are too few to trust.
    """
    imgs = [i for i, im in enumerate(page.images) if i not in used]
    if imgs:
        below = [i for i in imgs if page.images[i]["top"] >= cap["top"] - 8]
        pick = min(below or imgs, key=lambda i: abs(page.images[i]["top"] - cap["top"]))
        used.add(pick)
        return _union_bbox([page.images[pick]], page, pad=2)
    prims = list(page.curves) + list(page.rects)
    big = [p for p in prims if (p["x1"] - p["x0"]) * (p["bottom"] - p["top"]) < page.width * page.height * 0.6]
    if len(big) >= 20:
        return _union_bbox(big, page, pad=10)
    return (0, 0, page.width, page.height)


def _table_text(table) -> str:
    rows = []
    for r in table.extract():
        cells = [_clean(c or "") for c in r]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def ingest_pdf(pdf_path: Path, doc_meta: dict, fig_dir: Path) -> list[Item]:
    doc_id = doc_meta["doc_id"]
    items: list[Item] = []
    seen_labels: set[str] = set()
    fig_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for pno, page in enumerate(pdf.pages, start=1):
            trust = _trust_for_page(doc_meta, pno)
            page_text = _clean(page.extract_text() or "")
            lines = [_clean(l["text"]) for l in page.extract_text_lines()]
            section = next((l for l in lines if SECTION_RE.match(l)), "")
            caps = _caption_lines(page)

            # ---- tables: attach each detected table to the nearest "Table x-y" caption above it
            tables = page.find_tables()
            table_caps = [c for c in caps if c["kind"] == "Table"]
            for t in tables:
                above = [c for c in table_caps if c["top"] <= t.bbox[1] + 6]
                if not above:
                    continue
                cap = max(above, key=lambda c: c["top"])
                if cap["label"] in seen_labels:
                    continue
                body = _table_text(t)
                if len(body) < 20:
                    continue
                seen_labels.add(cap["label"])
                items.append(Item(f"{doc_id}-p{pno}-{cap['label']}", "table", doc_id, pno, trust,
                                  f"{cap['text']}\n{body}", cap["label"], section))

            # ---- figures: one crop per caption, labelled exactly as printed
            used_imgs: set = set()
            for cap in [c for c in caps if c["kind"] == "Figure"]:
                if cap["label"] in seen_labels:
                    continue
                seen_labels.add(cap["label"])
                bbox = _figure_bbox(page, cap, used_imgs)
                safe = re.sub(r"[^A-Za-z0-9]+", "_", f"{doc_id}_{cap['label']}")
                img_path = fig_dir / f"{safe}.png"
                page.crop(bbox).to_image(resolution=170).save(str(img_path))
                # body sentences that mention the figure help text retrieval of figure items
                mentions = [l for l in lines if cap["label"] in l and not CAPTION_RE.match(l)]
                text = f"{cap['text']}\n" + " ".join(mentions[:3])
                items.append(Item(f"{doc_id}-p{pno}-{cap['label']}", "figure", doc_id, pno, trust,
                                  text.strip(), cap["label"], section, str(img_path)))

            # ---- running text chunks (kept for every page, trust attached)
            for ci, ch in enumerate(_chunk_words(page_text)):
                items.append(Item(f"{doc_id}-p{pno}-c{ci}", "text", doc_id, pno, trust, ch, "", section,
                                  is_revision=bool(REVISION_RE.search(ch))))
    return items


def ingest_all(data_dir: Path, fig_dir: Path) -> list[Item]:
    """Ingest every document listed in kb_manifest.json."""
    manifest = load_manifest(data_dir)
    items: list[Item] = []
    for doc in manifest["documents"]:
        items += ingest_pdf(Path(data_dir) / doc["file"], doc, fig_dir)
    return items
