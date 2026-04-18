"""
Custom markdown assembly for the Docling extraction pipeline.

Uses Docling for layout detection (bounding boxes, labels, reading
order) and pypdfium2 for text extraction.  For each detected region,
text is pulled directly from the PDF text layer using the region's
bounding box, with an automatic margin filter that strips line-number
columns.

All noise detection is spatial/structural -- no hardcoded document
patterns.  Margin line numbers are excluded via glyph-level gap
detection, and page headers/footers are identified by finding short
text that repeats across pages in the top/bottom zones.

Tables still use Docling's TableFormer for cell-level structure.
"""

import re
from typing import Dict, FrozenSet, List, Tuple

import pypdfium2 as pdfium

from benchmark.tier2_hybrid.assembly_core import (
    detect_margin_boundary,
    detect_page_chrome,
    is_chrome,
)


# =========================================================================
# PDF text extraction helpers
# =========================================================================

class _PageCache:
    """
    Lazy cache of pypdfium2 page and textpage objects, margin
    boundaries, and cross-page chrome detection.
    """

    def __init__(self, pdf_path: str):
        self._doc = pdfium.PdfDocument(pdf_path)
        self._pages: Dict[int, Tuple] = {}

        # Pre-compute margin boundaries so chrome detection can use them.
        self._margins: Dict[int, float] = {}
        for pi in range(len(self._doc)):
            page = self._doc[pi]
            tp = page.get_textpage()
            w, _ = page.get_size()
            self._margins[pi] = detect_margin_boundary(tp, w)
            tp.close()
            page.close()

        self.chrome: FrozenSet[str] = detect_page_chrome(
            self._doc, self._margins)

    def get(self, page_no: int):
        """
        Return ``(text_page, margin_x, page_width, page_height)``
        for 1-based *page_no*.
        """
        if page_no not in self._pages:
            page = self._doc[page_no - 1]
            tp = page.get_textpage()
            w, h = page.get_size()
            mx = self._margins.get(page_no - 1, 0.0)
            self._pages[page_no] = (tp, mx, w, h, page)
        tp, mx, w, h, _ = self._pages[page_no]
        return tp, mx, w, h

    def close(self):
        for tp, _, _, _, page in self._pages.values():
            tp.close()
            page.close()
        self._doc.close()


def _normalize_spaces(text: str) -> str:
    """Collapse runs of 2+ spaces (from justified PDF text) to one."""
    return re.sub(r"  +", " ", text).strip()


def _extract_text_from_bbox(cache: _PageCache, page_no: int,
                            bbox) -> str:
    """
    Extract text from the PDF text layer within *bbox*, excluding
    the margin zone and any leading/trailing page chrome lines.

    The bbox is expected to use BOTTOMLEFT coordinate origin (the
    PDF default, which Docling provenance items use).
    """
    tp, margin_x, w, h = cache.get(page_no)

    left = max(bbox.l, margin_x)
    raw = tp.get_text_bounded(left=left, bottom=bbox.b,
                              right=bbox.r, top=bbox.t)

    # Normalise \r\n to \n so line operations are consistent.
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")

    # Strip chrome lines BEFORE collapsing to a single paragraph.
    # Bboxes that span into header/footer zones would otherwise include
    # repeated chrome text at the start or end of the extracted string.
    if cache.chrome:
        lines = raw.split("\n")
        while lines and is_chrome(lines[0].strip(), cache.chrome):
            lines.pop(0)
        while lines and is_chrome(lines[-1].strip(), cache.chrome):
            lines.pop()
        raw = "\n".join(lines)

    # Collapse visual PDF line-wraps (single \n) to spaces so the text
    # flows as one paragraph, then normalise runs of spaces.
    text = re.sub(r"  +", " ", raw.replace("\n", " ")).strip()
    return text


# =========================================================================
# Cross-block continuation detection
# =========================================================================

# Labels that can absorb a following continuation fragment.
_JOINABLE_LABELS = {"text", "paragraph", "body_text", "footnote",
                    "document_index"}

# Labels that always start a new paragraph -- never joined to anything.
_HARD_BREAK_LABELS = {"title", "section_header", "table", "caption",
                      "formula", "code", "page_header", "page_footer"}


def _is_continuation(prev: dict, curr: dict) -> bool:
    """
    Return True when *curr* is a wrapped continuation of *prev* rather
    than a new paragraph.

    Two signals, either one is sufficient:
      1. The current block starts with a lowercase letter (cannot be the
         start of a new sentence or structural element).
      2. The previous block ends with an alphabetic character -- the
         clearest sign of a mid-sentence word wrap (e.g. "...the",
         "...designated").  Digits, commas, semicolons, and closing
         punctuation indicate the previous item is structurally complete.
    """
    if curr["label"] in _HARD_BREAK_LABELS:
        return False
    if prev["label"] in _HARD_BREAK_LABELS:
        return False
    # A non-list block can be absorbed into the previous list item
    # (wrapped continuation of the item's text), but a new list item
    # cannot start as a continuation off a non-list block.
    if curr["label"] == "list_item":
        return False

    text = curr["text"]
    first_alpha = next((c for c in text if c.isalpha()), None)

    # Signal 1: current block starts with a lowercase letter.
    if first_alpha is not None and first_alpha.islower():
        return True

    # Signal 2: previous block ends mid-sentence.
    # Only fire when the last character is an alphabetic letter -- that
    # is the clearest sign of a broken mid-sentence wrap (e.g. "...the"
    # or "...designated").  Digits (dates, counts), commas, semicolons,
    # and closing punctuation all indicate the previous item is complete.
    # The colon guard is already covered since ':'.isalpha() is False.
    prev_end = prev["text"].rstrip()
    if prev_end and prev_end[-1].isalpha():
        return True

    return False


def _merge_continuations(blocks: List[dict]) -> List[dict]:
    """
    Merge consecutive text blocks where the latter is a lowercase
    continuation of the former.

    Produces a flat list where each entry is either a standalone block
    or the result of joining several wrapped-line fragments into one.
    """
    merged: List[dict] = []
    for blk in blocks:
        if merged and _is_continuation(merged[-1], blk):
            prev_text = merged[-1]["text"]
            # If the previous block ended mid-word (soft hyphen), rejoin
            # without a space; otherwise join with a space.
            if prev_text.endswith("-"):
                merged[-1]["text"] = prev_text[:-1] + blk["text"]
            else:
                merged[-1]["text"] = prev_text + " " + blk["text"]
        else:
            merged.append(dict(blk))
    return merged


_CONTINUATION_RE = re.compile(
    r"^(of |to |and |or |the |for |in |that |than |such |with |"
    r"by |on |at |from |as |not |nor |but |an? |is |are |was |"
    r"one |[a-z])"
)


def _merge_list_blocks(blocks: List[dict]) -> List[dict]:
    """
    Merge consecutive list_item blocks that are clearly wrapped
    continuation lines rather than separate items.

    A list_item whose text starts with a lowercase word or common
    preposition is joined onto the previous list_item.
    """
    merged: List[dict] = []
    for blk in blocks:
        if (blk["label"] == "list_item"
                and merged
                and merged[-1]["label"] == "list_item"
                and _CONTINUATION_RE.match(blk["text"])):
            merged[-1]["text"] = merged[-1]["text"] + " " + blk["text"]
        else:
            merged.append(blk)
    return merged


# =========================================================================
# Main assembler
# =========================================================================

def assemble_markdown(result, pdf_path: str) -> str:
    """
    Build markdown from a Docling ``ConversionResult``.

    Iterates items in Docling's resolved reading order.  For each
    item with a bounding box, text is extracted directly from the
    PDF text layer using pypdfium2 -- bypassing Docling's lossy
    text assembly.  An automatic margin detector excludes line-
    number columns at the glyph level.

    Tables still use Docling's TableFormer export to preserve
    cell-level structure.

    Args:
        result:   Docling ``ConversionResult``.
        pdf_path: Path to the source PDF file.

    Returns:
        Markdown string.
    """
    doc = result.document
    cache = _PageCache(pdf_path)

    try:
        blocks: List[dict] = []

        for item, _level in doc.iterate_items():
            label = (item.label.value
                     if hasattr(item.label, "value") else str(item.label))

            # Skip page chrome.
            if label in ("page_header", "page_footer"):
                continue

            # --- Extract text ---
            if label == "table":
                try:
                    text = item.export_to_markdown(doc)
                except Exception:
                    text = getattr(item, "text", "") or ""
                text = _normalize_spaces(text)
            elif item.prov:
                prov = item.prov[0]
                text = _extract_text_from_bbox(
                    cache, prov.page_no, prov.bbox)
            else:
                text = _normalize_spaces(
                    getattr(item, "text", "") or "")

            if not text:
                continue

            if is_chrome(text, cache.chrome):
                continue

            # Universal typographic convention: centered page numbers
            # like "- 3 -" or "-- 7 --".
            if re.fullmatch(r"-+\s*\d+\s*-+", text):
                continue

            blocks.append({"label": label, "text": text})

        # Merge wrapped continuations then merge wrapped list items.
        blocks = _merge_continuations(blocks)
        blocks = _merge_list_blocks(blocks)

        # --- Format blocks into markdown ---
        # Blocks are separated by a blank line, EXCEPT consecutive list
        # items which are joined with a single newline so they render as
        # one list rather than individual paragraphs.
        parts: List[str] = []
        for blk in blocks:
            label = blk["label"]
            text = blk["text"]

            if label == "title":
                formatted = f"# {text}"
            elif label == "section_header":
                formatted = f"## {text}"
            elif label == "list_item":
                formatted = f"- {text}"
            elif label == "caption":
                formatted = f"*{text}*"
            elif label in ("formula", "code"):
                formatted = f"```\n{text}\n```"
            else:
                formatted = text

            if (parts
                    and label == "list_item"
                    and parts[-1].startswith("- ")):
                # Consecutive list items: single newline, no blank line.
                parts[-1] = parts[-1] + "\n" + formatted
            else:
                parts.append(formatted)

        md = "\n\n".join(parts)
        md = re.sub(r"\n{4,}", "\n\n\n", md)
        return md.strip() + "\n"

    finally:
        cache.close()
