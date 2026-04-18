"""
Shared PDF assembly primitives for tier 2 hybrid extraction methods.

Contains the chrome and margin detection routines that were
independently implemented (identically) in both the YOLO glyph
assembler and the Docling bbox assembler.  Kept here as the single
source of truth so that improvements to either heuristic propagate to
both pipelines.

All primitives are document-agnostic and stateless -- no hardcoded
patterns.  Margin line numbers are detected via glyph-level gap
analysis in the left quarter of the page, and page headers/footers
are identified by text that repeats in the top/bottom spatial zones
across multiple pages of the same document.
"""

import re
from collections import Counter
from typing import Dict, FrozenSet, List, Set, Tuple

import pypdfium2 as pdfium


# =========================================================================
# Chrome detection tuning constants
#
# Chrome = repeated header / footer text that appears on a significant
# fraction of a document's pages.  The constants below define the
# geometry of the "chrome zone" scanned on each page and the repetition
# thresholds needed to classify a line as chrome.
# =========================================================================

_CHROME_ZONE_PTS = 100.0
_CHROME_MIN_PAGES = 2
_CHROME_PAGE_RATIO = 0.4
_CHROME_SHORT_THRESHOLD = 30


# =========================================================================
# Margin detection (glyph-level line-number column suppression)
# =========================================================================

def detect_margin_boundary(text_page, page_width: float) -> float:
    """
    Detect the right edge of a line-number margin column.

    Scans character positions in the left quarter of the page and
    looks for a spatial gap between a digit-only column (margin line
    numbers) and the start of body text.  Returns the x-coordinate
    just before body text starts, or 0.0 if no margin column is found.

    The algorithm is purely geometric and requires no regex patterns
    for specific document types.  It is used by both the YOLO glyph
    assembler and the Docling bbox assembler to remove margin line
    numbers at the glyph level before any markdown is generated.

    Args:
        text_page: pypdfium2 ``PdfTextPage``.
        page_width: Page width in PDF points.

    Returns:
        x-coordinate boundary separating margin from body text.
    """
    n = text_page.count_chars()
    if n < 10:
        return 0.0

    # Collect x-positions of visible characters in the left 25% of the page.
    left_quarter = page_width * 0.25
    xs: List[Tuple[float, float, str]] = []
    for i in range(n):
        ch = text_page.get_text_range(i, 1)
        if not ch.strip():
            continue
        try:
            box = text_page.get_charbox(i)
        except Exception:
            continue
        if box[0] < left_quarter:
            xs.append((box[0], box[2], ch))

    if len(xs) < 5:
        return 0.0

    # Unique left-edge positions rounded to the nearest point.
    unique_x = sorted(set(round(c[0]) for c in xs))
    if len(unique_x) < 2:
        return 0.0

    # Find the first significant spatial gap (>= 8 pts) from the left.
    margin_zone_end = None
    body_zone_start = None
    for i in range(1, len(unique_x)):
        gap = unique_x[i] - unique_x[i - 1]
        if gap >= 8:
            margin_zone_end = unique_x[i - 1]
            body_zone_start = unique_x[i]
            break

    if margin_zone_end is None:
        return 0.0

    # Verify: characters to the left of the gap must be mostly digits,
    # otherwise this is just a regular paragraph indent, not a
    # line-number margin.
    digit_count = 0
    total_count = 0
    for left, right, ch in xs:
        if left <= margin_zone_end + 2:
            total_count += 1
            if ch.isdigit():
                digit_count += 1

    if total_count == 0 or digit_count / total_count < 0.7:
        return 0.0

    return body_zone_start - 2.0


# =========================================================================
# Page chrome detection (repeated headers / footers)
# =========================================================================

def detect_page_chrome(
    doc: pdfium.PdfDocument,
    margins: Dict[int, float],
) -> FrozenSet[str]:
    """
    Identify repeated header/footer text across pages.

    Extracts text from the top and bottom spatial zones of every page
    (excluding the margin column), splits into lines, and returns any
    line that appears on at least 40% of pages (minimum 2).

    This is fully document-agnostic -- no patterns are needed.  It lets
    both assemblers filter out running headers and footers at a content
    level rather than relying on the layout detector to always classify
    them as ``Page-header`` / ``Page-footer`` regions, which the
    detectors routinely fail to do consistently.

    Args:
        doc:     An open ``pypdfium2.PdfDocument``.
        margins: Mapping of 0-based page index to margin-x boundary
                 (as returned by ``detect_margin_boundary``).

    Returns:
        Frozen set of normalised chrome strings.
    """
    n_pages = len(doc)
    if n_pages < 2:
        return frozenset()

    top_lines: List[str] = []
    bot_lines: List[str] = []

    for pi in range(n_pages):
        page = doc[pi]
        tp = page.get_textpage()
        w, h = page.get_size()
        mx = margins.get(pi, 0.0)

        top_raw = tp.get_text_bounded(
            left=mx, bottom=h - _CHROME_ZONE_PTS, right=w, top=h)
        bot_raw = tp.get_text_bounded(
            left=mx, bottom=0, right=w, top=_CHROME_ZONE_PTS)

        for raw in (top_raw, bot_raw):
            for line in raw.split("\n"):
                s = re.sub(r"  +", " ", line).strip()
                if s:
                    (top_lines if raw is top_raw else bot_lines).append(s)

        tp.close()
        page.close()

    chrome: Set[str] = set()
    threshold = max(_CHROME_MIN_PAGES, int(n_pages * _CHROME_PAGE_RATIO))

    for zone in (top_lines, bot_lines):
        for text, count in Counter(zone).items():
            if count >= threshold:
                chrome.add(text)

    return frozenset(chrome)


def is_chrome(text: str, chrome: FrozenSet[str]) -> bool:
    """
    Return True if *text* matches detected page chrome.

    Exact match is tried first.  For short text (below
    ``_CHROME_SHORT_THRESHOLD`` chars) a substring check handles cases
    where the layout model splits a single header line (e.g.
    "25 LC 57 0207S/AP" or "H. B. 89") into multiple regions that
    each hold only part of the repeated chrome.
    """
    if text in chrome:
        return True
    if len(text) <= _CHROME_SHORT_THRESHOLD:
        return any(text in c for c in chrome)
    return False
