"""
YOLO text-assembly pipeline.

Region merging heuristics, glyph-level text-layer extraction, and
markdown reconstruction for the YOLO-based hybrid extraction methods.

Key design principle: YOLO bounding boxes act as structural *groupers*,
not hard clip windows.  All text content comes from the PDF text layer.
Each glyph is assigned to the region whose bounds contain its centre
point, so no characters are clipped at region edges.

This module is where active tuning of merge strategies and text-
grouping logic happens.  It is intentionally separated from the thin
shared module (Region + geometric sort) and the stable reading-order
module so changes here do not risk breaking unrelated code paths.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pypdfium2 as pdfium

from benchmark.config import (
    LABEL_TO_MD,
    NON_TEXT_LABELS,
)
from benchmark.tier2_hybrid.assembly_core import (
    detect_margin_boundary,
    detect_page_chrome,
    is_chrome,
)
from benchmark.tier2_hybrid.shared import Region


# =========================================================================
# Overlapping region merging
# =========================================================================

def _iou_contained(small: Region, big: Region) -> float:
    """
    Fraction of *small*'s area that overlaps with *big*.
    Returns 0-1; 1 means small is entirely inside big.
    """
    ix1 = max(small.x1, big.x1)
    iy1 = max(small.y1, big.y1)
    ix2 = min(small.x2, big.x2)
    iy2 = min(small.y2, big.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    small_area = max((small.x2 - small.x1) * (small.y2 - small.y1), 1e-6)
    return inter / small_area


# Labels whose structural role should override smaller overlapping
# detections during the merge phase.  YOLO sometimes fires duplicate
# boxes on the same region (e.g. a low-confidence Text box inside a
# higher-confidence Page-footer).  Letting the duplicate survive causes
# content to leak into the wrong output section.
#
# Page-footer / Page-header: prevents footer/header text from appearing
#     as body content.
# Table: prevents List-item, Text, or other small detections inside a
#     table bounding box from being extracted separately as body content,
#     which produces fragmented, out-of-context output.
_STRUCTURAL_OVERRIDE_LABELS = {"Page-footer", "Page-header", "Table"}

# Labels that should never be absorbed by a larger region of a different
# label, even when both are text-like.  Captions carry semantic formatting
# (italic in markdown) that is lost if the region is swallowed by a
# neighbouring Text box during containment-based merging.
_PROTECTED_LABELS = {"Caption", "Section-header", "Title", "Footnote", "Formula"}


def merge_overlapping_regions(
    regions: List[Region], threshold: float = 0.6,
) -> List[Region]:
    """
    Remove smaller regions that are mostly contained inside a larger one.

    YOLO often detects both a big bounding box around a group (e.g. the
    full author+affiliations block) AND smaller boxes for each sub-line.
    Keeping both causes the glyph assignment to split the text into
    separate paragraphs.  By dropping the sub-boxes we let the parent
    region capture all glyphs as one flowing paragraph.

    Cross-type merging (text vs non-text) is normally skipped so that,
    for example, a caption inside a picture region is preserved.  The one
    exception is Page-footer / Page-header: if a smaller Text region is
    mostly inside a larger footer or header box, the Text duplicate is
    removed so the structural classification wins and the content is
    correctly excluded from body text.

    Protected labels (Caption, Section-header, etc.) are never absorbed
    by a region of a *different* label so their semantic formatting is
    preserved.
    """
    if len(regions) <= 1:
        return regions

    areas = [
        (r.x2 - r.x1) * (r.y2 - r.y1) for r in regions
    ]
    by_area = sorted(range(len(regions)), key=lambda i: areas[i], reverse=True)

    keep = set(range(len(regions)))

    for i, big_idx in enumerate(by_area):
        if big_idx not in keep:
            continue
        big_is_text = regions[big_idx].label not in NON_TEXT_LABELS
        for small_idx in by_area[i + 1:]:
            if small_idx not in keep:
                continue
            small_is_text = regions[small_idx].label not in NON_TEXT_LABELS

            same_type = big_is_text == small_is_text

            structural_override = (
                not big_is_text
                and small_is_text
                and regions[big_idx].label in _STRUCTURAL_OVERRIDE_LABELS
            )

            if not (same_type or structural_override):
                continue

            # Never absorb a protected label into a region with a
            # different label -- e.g. a Caption inside a Text box must
            # survive so its italic markdown formatting is preserved.
            if (regions[small_idx].label in _PROTECTED_LABELS
                    and regions[small_idx].label != regions[big_idx].label):
                continue

            if _iou_contained(regions[small_idx], regions[big_idx]) >= threshold:
                keep.discard(small_idx)

    result = [regions[i] for i in sorted(keep)]
    return _merge_adjacent_same_label(result)


# Labels that commonly appear as vertically stacked boxes that should be
# merged into a single region.  List-item boxes often fire once per
# bullet, so merging lets the glyph assignment capture them in one pass.
#
# "Text" was previously included here, but blind region-level merging
# collapses separate paragraphs on single-column pages (the gap between
# paragraphs is within y_gap and the boxes span the same width).  Text
# fragment reconnection is now handled more precisely at the text level
# by the cross-region lowercase continuation merge in regions_to_markdown.
_STACKABLE_LABELS = {"List-item"}


def _merge_adjacent_same_label(
    regions: List[Region],
    y_gap: float = 15.0,
    x_overlap_ratio: float = 0.5,
) -> List[Region]:
    """
    Merge vertically adjacent regions that share the same stackable label.

    YOLO often fires a separate bounding box for every bullet in a list.
    Because the boxes are similar in size, the containment-based merge
    does not combine them.  This pass groups same-label regions that are
    close vertically and overlap horizontally, replacing each group with
    a single union bounding box.
    """
    if len(regions) <= 1:
        return regions

    stackable = [r for r in regions if r.label in _STACKABLE_LABELS]
    if len(stackable) <= 1:
        return regions

    others = [r for r in regions if r.label not in _STACKABLE_LABELS]

    by_label: Dict[str, List[Region]] = defaultdict(list)
    for r in stackable:
        by_label[r.label].append(r)

    merged: List[Region] = list(others)

    for label, group in by_label.items():
        group.sort(key=lambda r: r.y1)
        clusters: List[List[Region]] = [[group[0]]]

        for r in group[1:]:
            cl = clusters[-1]
            cl_x1 = min(c.x1 for c in cl)
            cl_x2 = max(c.x2 for c in cl)
            cl_y2 = max(c.y2 for c in cl)

            gap = r.y1 - cl_y2
            overlap_x = min(r.x2, cl_x2) - max(r.x1, cl_x1)
            min_w = min(r.x2 - r.x1, cl_x2 - cl_x1)

            if gap <= y_gap and overlap_x / max(min_w, 1e-6) >= x_overlap_ratio:
                clusters[-1].append(r)
            else:
                clusters.append([r])

        for cluster in clusters:
            if len(cluster) == 1:
                merged.append(cluster[0])
            else:
                merged.append(Region(
                    label,
                    min(c.x1 for c in cluster),
                    min(c.y1 for c in cluster),
                    max(c.x2 for c in cluster),
                    max(c.y2 for c in cluster),
                    max(c.confidence for c in cluster),
                ))

    return merged


# =========================================================================
# Glyph-level text-layer extraction
# =========================================================================

def _collect_glyphs(
    text_page, page_height: float, margin_x: float = 0.0,
) -> List[Tuple[int, float, float]]:
    """
    Collect every character on the page with its centre position
    converted to top-down coordinates (matching Region coordinate space).

    Glyphs whose horizontal centre falls within the left-margin zone
    (x-centre < *margin_x*) are excluded.  This removes line-number
    columns that appear in legislative bills and similar documents, where
    each body line is numbered in a narrow left margin.

    Returns list of (char_index, cx, cy) tuples.
    """
    n = text_page.count_chars()
    glyphs: List[Tuple[int, float, float]] = []

    for i in range(n):
        try:
            left, bottom, right, top = text_page.get_charbox(i)
        except Exception:
            continue
        cx = (left + right) / 2.0
        cy = page_height - (bottom + top) / 2.0
        if cx < margin_x:
            continue
        glyphs.append((i, cx, cy))

    return glyphs


def _assign_glyphs(
    glyphs: List[Tuple[int, float, float]],
    regions: List[Region],
    page_height: float = 0.0,
) -> Dict[int, List[int]]:
    """
    Assign each glyph to a region.

    Containment is checked with a small margin (GLYPH_MARGIN) so that
    characters just outside a YOLO bounding box are still captured --
    the boxes are structural guides, not pixel-perfect clip windows.

    If a glyph falls inside (with margin) one or more regions it is
    assigned to the smallest one.  Otherwise it goes to the nearest
    region by edge distance, with non-text regions (headers, footers)
    deprioritised so readable content is not silently dropped.

    Glyphs near the top or bottom page edge bypass the non-text penalty
    for Page-header / Page-footer regions so that stray page-number
    characters are routed to the correct structural region instead of
    leaking into adjacent text columns.

    Returns {region_index: [char_indices in text-stream order]}.
    """
    GLYPH_MARGIN = 5.0
    PAGE_EDGE_MARGIN = 50.0

    assignments: Dict[int, List[int]] = {i: [] for i in range(len(regions))}
    if not regions:
        return assignments

    for char_idx, cx, cy in glyphs:
        best_ri = -1
        best_area = float("inf")

        for ri, r in enumerate(regions):
            if (r.x1 - GLYPH_MARGIN <= cx <= r.x2 + GLYPH_MARGIN
                    and r.y1 - GLYPH_MARGIN <= cy <= r.y2 + GLYPH_MARGIN):
                area = (r.x2 - r.x1) * (r.y2 - r.y1)
                if area < best_area:
                    best_area = area
                    best_ri = ri

        if best_ri == -1:
            near_top = page_height > 0 and cy < PAGE_EDGE_MARGIN
            near_bottom = page_height > 0 and cy > page_height - PAGE_EDGE_MARGIN

            best_dist = float("inf")
            for ri, r in enumerate(regions):
                dx = max(r.x1 - cx, 0.0, cx - r.x2)
                dy = max(r.y1 - cy, 0.0, cy - r.y2)
                dist = dx * dx + dy * dy
                if r.label in NON_TEXT_LABELS:
                    if (r.label == "Page-footer" and near_bottom) or \
                       (r.label == "Page-header" and near_top):
                        pass
                    else:
                        dist += 1e6
                if dist < best_dist:
                    best_dist = dist
                    best_ri = ri

        assignments[best_ri].append(char_idx)

    return assignments


# =========================================================================
# Markdown reconstruction
# =========================================================================

_INTERNAL_WHITESPACE = re.compile(r"[\r\n]+")
_COLLAPSE_SPACES = re.compile(r"  +")

_LEADING_BULLET = re.compile(r"^[•·∙▪▸►‣⁃]\s*", re.MULTILINE)

_PRINT_JOB_META = re.compile(
    r"^.*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M\s+Page.*$",
    re.MULTILINE | re.IGNORECASE,
)

# Characters that mark the end of a sentence for paragraph-break
# detection.  We intentionally list only the true terminal punctuation
# here: closing brackets or quotes are handled separately by stripping
# one trailing enclosure character before checking.  That way a line
# like "Hit Dice: 6d10 (33 hp)" -- where the trailing ")" is just a
# closing paren, not the end of a sentence -- is not mistaken for a
# paragraph boundary.
_SENTENCE_TERMINATORS = frozenset(".!?\u2026")
_TRAILING_ENCLOSURE = frozenset(")]\u201D\u2019\"'")


def _ends_sentence(text: str) -> bool:
    """Return True when *text* ends with a sentence-terminal mark.

    A single closing bracket or quote at the very end is peeled off
    first so that ``saw it.")``, ``(Campaigne, 1990).``, and ``he said
    'yes.'`` all count, while ``(33 hp)`` does not.
    """
    s = text.rstrip()
    if not s:
        return False
    if s[-1] in _TRAILING_ENCLOSURE:
        s = s[:-1]
    return s[-1:] in _SENTENCE_TERMINATORS

# Soft hyphens (U+00AD) and the replacement character (U+FFFE) that
# some PDF producers emit for mid-word line breaks *within* a region.
_SOFT_HYPHEN_TRAIL = re.compile(r"[\u00AD\uFFFE]\s*$")

# Some PDF text streams embed the soft-hyphen marker inline in the
# middle of a visual line (e.g. "bal" + U+FFFE + " " + "ance") rather
# than only at the end of a wrapped line.  This global pattern strips
# any soft-hyphen / U+FFFE (plus any whitespace that immediately
# follows it) when the next character is a word character, rejoining
# the split word without inserting a space.  The lookahead guard keeps
# us from accidentally removing marks that are followed by punctuation
# or end-of-string.
_SOFT_HYPHEN_GLOBAL = re.compile(r"[\u00AD\uFFFE]\s*(?=\w)")


def _build_line_text(
    line_chars: List[Tuple[str, float, float, float, float, float, int]],
    text_page=None,
) -> str:
    """
    Build text for a visual line, recovering word spaces that may have
    been lost during glyph-to-region assignment.

    Strategy, in order of preference:
      1. If the line's assigned chars already contain stream whitespace,
         just concatenate (the stream has the spacing we want).
      2. Otherwise, consult the text stream to see whether whitespace
         characters *exist* between our visible glyphs but got routed to
         a neighbouring region because their y-centre happened to land on
         a region boundary.  Whenever the stream has a space between two
         of our consecutive visible glyphs, inject one here.
      3. Fallback: use glyph-gap heuristics.  Only reached when the PDF
         genuinely omits space glyphs and relies on positioning (produces
         crushed text like "dininghalls").
    """
    has_stream_spaces = any(
        ch in (" ", "\t") for ch, _, _, _, _, _, _ in line_chars
    )
    if has_stream_spaces:
        return "".join(ch for ch, _, _, _, _, _, _ in line_chars)

    if text_page is not None:
        stream_parts = _join_via_stream_gaps(line_chars, text_page)
        if stream_parts is not None:
            return stream_parts

    visible = [(left, right) for ch, left, right, _, _, _, _ in line_chars
               if ch.strip() and left > 0 and right > left]

    if len(visible) < 2:
        return "".join(ch for ch, _, _, _, _, _, _ in line_chars)

    avg_width = sum(r - l for l, r in visible) / len(visible)
    space_threshold = max(avg_width * 0.3, 1.0)

    parts: List[str] = []
    prev_right = 0.0

    for ch, left, right, _, _, _, _ in line_chars:
        if not ch.strip():
            parts.append(ch)
            continue

        if left > 0 and prev_right > 0:
            gap = left - prev_right
            if gap > space_threshold:
                parts.append(" ")

        parts.append(ch)
        if left > 0 and right > left:
            prev_right = right

    return "".join(parts)


def _join_via_stream_gaps(
    line_chars: List[Tuple[str, float, float, float, float, float, int]],
    text_page,
) -> str:
    """
    Reconstruct word spaces by consulting the text stream.

    When adjacent YOLO regions share a boundary, space and punctuation
    glyphs whose baseline boxes sit just off the letter baseline can be
    routed to the neighbouring region, leaving our line with only
    letters.  We cannot see those spaces in ``line_chars`` -- but the
    original character-index sequence tells us they *did* exist in the
    stream.  For each pair of consecutive visible glyphs in our line we
    scan the stream indices that lie between them; if any is a space
    or tab, we emit a space in the rebuilt text.

    Returns ``None`` when no cross-index whitespace was recovered, so
    the caller can fall back to the gap-based heuristic for PDFs whose
    streams genuinely omit space glyphs.
    """
    visible = [
        (j, line_chars[j][6]) for j in range(len(line_chars))
        if line_chars[j][0].strip()
    ]
    if len(visible) < 2:
        return None

    space_after_pos: set = set()
    for k in range(len(visible) - 1):
        j_cur, ci_cur = visible[k]
        j_next, ci_next = visible[k + 1]
        if ci_next - ci_cur <= 1:
            continue
        for mid in range(ci_cur + 1, ci_next):
            try:
                mid_ch = text_page.get_text_range(mid, 1)
            except Exception:
                continue
            if mid_ch and mid_ch[0] in (" ", "\t"):
                space_after_pos.add(j_cur)
                break

    if not space_after_pos:
        return None

    parts: List[str] = []
    for j, (ch, _, _, _, _, _, _) in enumerate(line_chars):
        parts.append(ch)
        if j in space_after_pos:
            parts.append(" ")
    return "".join(parts)


def _join_region_text(
    text_page, char_indices: List[int],
    page_height: float, region: Region,
) -> str:
    """
    Build text from the characters assigned to a region, using the
    left-margin position of each visual line to decide line breaks.

    Lines that start at the left margin (within INDENT_THRESHOLD of the
    leftmost line in the region) are treated as new entries and get a
    newline separator.  Lines that start indented -- wrapped
    continuations, sub-entries, or 'See also' references -- are joined
    to the preceding entry with a space.
    """
    if not char_indices:
        return ""

    prev_cy = 0.0
    char_data: List[Tuple[str, float, float, float, float, float, int]] = []
    for ci in char_indices:
        ch = text_page.get_text_range(ci, 1)
        left = right = top = bottom = 0.0
        cy = prev_cy
        td_top = prev_cy
        td_bottom = prev_cy
        if ch.strip():
            try:
                left, bottom, right, top = text_page.get_charbox(ci)
                cy = page_height - (bottom + top) / 2.0
                td_top = page_height - top
                td_bottom = page_height - bottom
                prev_cy = cy
            except Exception:
                pass
        char_data.append((ch, left, right, cy, td_top, td_bottom, ci))

    lines: List[List[Tuple[str, float, float, float, float, float, int]]] = [[char_data[0]]]

    for cd in char_data[1:]:
        if not cd[0].strip():
            lines[-1].append(cd)
            continue

        # A character belongs to the current line if its vertical center is close
        # to the line's center, OR if its bounding box significantly overlaps the
        # line's bounding box (helps with superscripts/subscripts).
        cur_line = lines[-1]
        valid_chars = [c for c in cur_line if c[0].strip()]

        if not valid_chars:
            lines[-1].append(cd)
            continue

        line_cy = sum(c[3] for c in valid_chars) / len(valid_chars)
        line_top = min(c[4] for c in valid_chars)
        line_bottom = max(c[5] for c in valid_chars)

        ch_cy, ch_top, ch_bottom = cd[3], cd[4], cd[5]

        overlap = min(line_bottom, ch_bottom) - max(line_top, ch_top)

        if abs(ch_cy - line_cy) <= 4.0 or overlap > 0:
            lines[-1].append(cd)
        else:
            lines.append([cd])

    line_info: List[Tuple[str, float, float, float]] = []
    for lc in lines:
        raw = _build_line_text(lc, text_page)
        text = _COLLAPSE_SPACES.sub(" ", _INTERNAL_WHITESPACE.sub(" ", raw)).strip()
        if not text:
            continue
        lefts = [l for ch, l, _, _, _, _, _ in lc if ch.strip() and l > 0]
        rights = [r for ch, _, r, _, _, _, _ in lc if ch.strip() and r > 0]
        x_start = min(lefts) if lefts else region.x1
        x_end = max(rights) if rights else region.x2
        valid_chars = [c for c in lc if c[0].strip()]
        cy = sum(c[3] for c in valid_chars) / len(valid_chars) if valid_chars else prev_cy
        line_info.append((text, x_start, x_end, cy))

    if not line_info:
        return ""

    while len(line_info) > 1 and re.fullmatch(r"\d{1,4}", line_info[-1][0]):
        line_info.pop()
    while len(line_info) > 1 and re.fullmatch(r"\d{1,4}", line_info[0][0]):
        line_info.pop(0)

    if not line_info:
        return ""

    # Merge punctuation-only lines (e.g. a trailing period that the PDF
    # placed as a separate glyph cluster) into the preceding line so the
    # punctuation is not orphaned as a floating character.
    _pi = 1
    while _pi < len(line_info):
        txt = line_info[_pi][0]
        if not any(c.isalnum() for c in txt) and len(txt.strip()) <= 3:
            ptxt, pxs, pxe, pcy = line_info[_pi - 1]
            line_info[_pi - 1] = (ptxt + txt, pxs, max(pxe, line_info[_pi][2]), pcy)
            line_info.pop(_pi)
        else:
            _pi += 1

    if not line_info:
        return ""

    start_counts: Dict[int, int] = {}
    for _, xs, _, _ in line_info:
        b = round(xs / 2.0)
        start_counts[b] = start_counts.get(b, 0) + 1
    dominant_bin = min(start_counts, key=lambda k: (-start_counts[k], k))
    true_left = dominant_bin * 2.0
    INDENT_THRESHOLD = 3.0

    max_right = max(xe for _, _, xe, _ in line_info)

    gaps = [line_info[i][3] - line_info[i-1][3] for i in range(1, len(line_info))]
    if gaps:
        median_gap = sorted(gaps)[len(gaps) // 2]
    else:
        median_gap = 12.0

    parts = [line_info[0][0]]
    for i in range(1, len(line_info)):
        cur_text, cur_x_start, _, cur_cy = line_info[i]
        prev_text, _, prev_x_end, prev_cy = line_info[i-1]

        if (_SOFT_HYPHEN_TRAIL.search(parts[-1])
                and cur_text and cur_text[0].islower()):
            parts[-1] = _SOFT_HYPHEN_TRAIL.sub("", parts[-1])
            parts.append(cur_text)
            continue

        is_indented = cur_x_start > true_left + INDENT_THRESHOLD

        line_gap = cur_cy - prev_cy
        prev_wrapped = prev_x_end >= max_right - 30.0

        # Paragraph-break detection combines two complementary signals
        # so short regions (2-3 lines, where median_gap alone is
        # meaningless) still break correctly:
        #
        #  1. Vertical: the line gap is clearly larger than the rest of
        #     the region's typical gaps.  Dominant signal when the
        #     region has enough lines to establish a baseline rhythm.
        #  2. Typographic: the previous line ends with sentence-terminal
        #     punctuation AND the current line begins with an uppercase
        #     letter AND the current line has a first-line indent.  This
        #     is the classic indented-paragraph signature in running
        #     prose and works with only two lines.  We intentionally do
        #     *not* treat a short previous line + uppercase start alone
        #     as a paragraph break, because that pattern also describes
        #     every line in a stat block or definition list.
        is_gap_break = line_gap > median_gap * 1.4
        cur_starts_upper = bool(cur_text) and cur_text[0].isupper()
        is_indent_break = (
            is_indented
            and cur_starts_upper
            and _ends_sentence(prev_text)
        )

        is_new_paragraph = is_gap_break or is_indent_break

        if is_new_paragraph:
            parts.append("\n\n" + cur_text)
        elif not prev_wrapped:
            parts.append("\n" + cur_text)
        elif is_indented and cur_text and cur_text[0].isupper():
            parts.append("\n" + cur_text)
        else:
            parts.append(" " + cur_text)

    # Strip any inline soft-hyphen markers that survived line-joining
    # (the line-boundary rule above only fires when the hyphen is the
    # last character of a line segment).
    return _SOFT_HYPHEN_GLOBAL.sub("", "".join(parts))


def _same_column(a: Region, b: Region, min_overlap: float = 0.5) -> bool:
    """
    Check whether two regions belong to the same horizontal column.

    Returns True when the horizontal overlap between the two regions
    exceeds *min_overlap* of the narrower region's width.
    """
    overlap = min(a.x2, b.x2) - max(a.x1, b.x1)
    if overlap <= 0:
        return False
    min_width = min(a.x2 - a.x1, b.x2 - b.x1)
    return overlap / max(min_width, 1e-6) >= min_overlap


# How far above *prev* the top of *curr* must sit for us to treat the
# jump as a column break rather than a minor layout artefact.
_COLUMN_JUMP_TOLERANCE = 20.0


def _is_column_transition(curr: Region, prev: Region) -> bool:
    """
    Return True when *curr* starts geometrically above *prev* ends.

    Normal same-column reading order moves monotonically downward, so
    an upward y-jump between two adjacent regions is the unambiguous
    signature of a column break.  Used to detect the column-wrap case
    where a sentence flows from the bottom of column 1 into the top of
    column 2; in that case ``_same_column`` returns False (the two
    regions don't overlap horizontally) but the merge is still
    legitimate.

    Coordinates here follow the ``Region`` convention used throughout
    this module: ``y1`` is the top edge and ``y2`` is the bottom edge,
    with larger values lower on the page.
    """
    return curr.y1 < prev.y2 - _COLUMN_JUMP_TOLERANCE


# Body-text labels used to decide where the "reading area" of a page
# lies.  Pictures and Tables are excluded on purpose: a logo in the
# top margin should not prevent us from recognising an adjacent
# running header as chrome.
_BODY_LABELS = ("Text", "Caption", "List-item")

# Fraction of page height considered the outer margin band.  A
# Section-header/Title sitting entirely inside this band on the top
# (or mirrored at the bottom) is a candidate for running-chrome
# removal.  Kept generous at 15 % so that even loosely-cropped folio
# lines still qualify, while staying well above the practical top of
# any normal column.
_OUTER_MARGIN_FRACTION = 0.15

# Matches a running-header folio pattern: a 1-4 digit page number at
# the very start or very end of the text, separated by whitespace or
# punctuation from a non-digit neighbour.  Deliberately narrow so that
# inline numeric headings such as "3.1 Methods" or "Chapter 3" do not
# qualify -- the page number must stand alone on one end of the line.
_FOLIO_LEAD = re.compile(r"^\d{1,4}[\s.\u2013\u2014\-]+\D")
_FOLIO_TRAIL = re.compile(r"\D[\s.\u2013\u2014\-]+\d{1,4}$")


def _looks_like_folio(text: str) -> bool:
    """
    Return True when *text* carries the page-folio signature used by
    running headers and footers (a standalone page number on one end
    of the line, with a short title/author reference on the other).
    """
    s = text.strip()
    if not s:
        return False
    return bool(_FOLIO_LEAD.match(s) or _FOLIO_TRAIL.search(s))


def _is_multicolumn_body(regions: List[Region]) -> bool:
    """
    Return True when the page's body-text regions form two or more
    horizontally disjoint clusters -- the geometric fingerprint of a
    multi-column article page.

    A title page is typically single-column (all body regions share a
    common horizontal span), so a positive result here is strong
    evidence that a top/bottom-margin heading is a running page header
    rather than the article's own title.
    """
    body = [r for r in regions if r.label in _BODY_LABELS]
    if len(body) < 2:
        return False
    centers = sorted((r.x1 + r.x2) / 2.0 for r in body)
    widths = sorted(r.x2 - r.x1 for r in body)
    median_width = widths[len(widths) // 2]
    # Two centers belong to different columns when their separation is
    # at least half the median region width -- well beyond any
    # plausible horizontal jitter inside a single column.
    for i in range(1, len(centers)):
        if centers[i] - centers[i - 1] > median_width * 0.5:
            return True
    return False


def _is_running_header_chrome(
    region: Region,
    text: str,
    all_regions: List[Region],
    page_height: float,
) -> bool:
    """
    Decide whether *region* is a running page header/footer masquerading
    as a ``Title`` or ``Section-header``.

    Safety strategy: require three independent signals to agree before
    stripping.  The combination is conservative enough that a genuine
    article title or body section heading cannot satisfy all three
    simultaneously:

      1. The region is entirely inside the page's outer 15 % margin
         band (top or bottom).  Real body headings are embedded in
         column reading flow and never live in the margin.
      2. The region lies on the margin side of every body Text /
         Caption / List-item region on the page -- i.e. it is
         *isolated* in the outer strip, with all readable content on
         the interior side.  A legitimate top-of-page article title is
         typically followed directly by metadata (authors,
         affiliation) before any body text; stripping requires the
         header to be cleanly separated from all such content.
      3. Either (a) the text carries a page-folio signature (a bare
         1-4 digit page number leading or trailing the line), or
         (b) the page's body layout is unambiguously multi-column.
         These are independent fingerprints of running chrome: (a)
         catches folio lines directly, (b) catches article pages
         whose chrome was styled without a visible page number.
    """
    if region.label not in ("Title", "Section-header"):
        return False

    top_band = page_height * _OUTER_MARGIN_FRACTION
    bottom_band = page_height * (1.0 - _OUTER_MARGIN_FRACTION)
    in_top = region.y2 <= top_band
    in_bottom = region.y1 >= bottom_band
    if not (in_top or in_bottom):
        return False

    body = [
        r for r in all_regions
        if r.label in _BODY_LABELS and r is not region
    ]
    if not body:
        return False

    if in_top:
        if not all(r.y1 >= region.y2 for r in body):
            return False
    else:
        if not all(r.y2 <= region.y1 for r in body):
            return False

    if _looks_like_folio(text):
        return True
    if _is_multicolumn_body(all_regions):
        return True
    return False


def _first_alpha(text: str):
    """Return the first alphabetic character in *text*, or None."""
    for ch in text:
        if ch.isalpha():
            return ch
    return None


# Trailing soft-hyphen or regular hyphen at a region boundary indicates
# a word was split across two YOLO detections.  When merging, the hyphen
# should be removed and the two fragments joined without a space so the
# word is reassembled (e.g. "bal\uFFFE" + "ance" -> "balance").
_TRAILING_HYPHEN = re.compile(r"[-\u00AD\uFFFE]\s*$")

# Sentence-ending punctuation stranded at the start of a region because
# the final glyph of the preceding sentence fell into the next YOLO box.
_LEADING_SENTENCE_PUNCT = re.compile(r"^([.!?;:,)]+)\s*", re.DOTALL)


def regions_to_markdown(
    regions: List[Region], pdf_path: str, page_index: int = 0,
    page_image=None, enable_tables: bool = False,
    table_mode: str = "accurate",
) -> str:
    """
    Build a markdown string from ordered, labelled regions.

    Each character in the PDF text layer is assigned to the YOLO region
    that contains its centre point.  YOLO provides structure (labels and
    reading order); the text layer provides the actual character content.

    Meaningful line breaks within a region are preserved; only
    paragraph wrapping and indented continuations are collapsed to
    spaces.  Regions are separated by blank lines.

    Cross-region continuation: when a Text region's content begins with
    a lowercase letter it is almost certainly a sentence fragment that
    continues from a preceding region in the same column (YOLO sometimes
    splits one paragraph across multiple bounding boxes).  These
    fragments are merged back into the preceding same-column Text block
    so the sentence is not severed mid-word.

    Table extraction: when *enable_tables* is True and *page_image* is
    supplied, each YOLO ``Table`` region is handed to Docling's
    TableFormer for cell-level structure recovery and the result is
    inlined as HTML.  When TableFormer is unavailable or fails for a
    specific region, the output falls back to the ``[Table]``
    placeholder that non-table mode always produces.

    Parameters
    ----------
    regions
        Ordered list of YOLO-detected regions for this page.
    pdf_path
        Path to the source PDF.
    page_index
        Zero-based page to process.
    page_image
        Optional PIL image of the rendered page.  Required for
        TableFormer extraction; unused otherwise.
    enable_tables
        When True, run TableFormer on ``Table`` regions.
    table_mode
        TableFormer quality mode: ``"accurate"`` (default) or ``"fast"``.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    text_page = page.get_textpage()
    page_width, page_height = page.get_size()

    # Detect and exclude the left-margin line-number column so that
    # numbered legislative lines (e.g. "1 To amend...") do not have the
    # line number prepended to the body text.
    margin_x = detect_margin_boundary(text_page, page_width)

    # Detect repeated header/footer text across all pages so it can be
    # filtered from this page's output, regardless of how YOLO labels it.
    margins_all: Dict[int, float] = {}
    for pi in range(len(doc)):
        if pi == page_index:
            margins_all[pi] = margin_x
        else:
            p = doc[pi]
            tp = p.get_textpage()
            w, _ = p.get_size()
            margins_all[pi] = detect_margin_boundary(tp, w)
            tp.close()
            p.close()
    chrome = detect_page_chrome(doc, margins_all)

    regions = merge_overlapping_regions(regions)
    glyphs = _collect_glyphs(text_page, page_height, margin_x)
    assignments = _assign_glyphs(glyphs, regions, page_height)

    parts: List[str] = []
    text_part_log: List[Tuple[int, Region]] = []

    for ri, region in enumerate(regions):
        char_indices = assignments.get(ri, [])

        if region.label in NON_TEXT_LABELS:
            if region.label == "Table":
                table_html = ""
                if enable_tables and page_image is not None:
                    from benchmark.tier2_hybrid.yolo.table_extraction import (
                        collect_pdf_cells_for_region,
                        extract_table,
                    )
                    region_bbox = (region.x1, region.y1, region.x2, region.y2)
                    pdf_cells = collect_pdf_cells_for_region(
                        text_page, region_bbox, page_height,
                    )
                    table_html = extract_table(
                        page_image=page_image,
                        region_bbox=region_bbox,
                        pdf_cells=pdf_cells,
                        page_pdf_size=(page_width, page_height),
                        mode=table_mode,
                    )
                if table_html:
                    parts.append("\n" + table_html + "\n")
                else:
                    parts.append("\n[Table]\n")
            elif region.label == "Picture":
                parts.append("\n[Image]\n")
            continue

        if not char_indices:
            continue

        text = _join_region_text(text_page, char_indices, page_height, region)
        if not text:
            continue

        text = _PRINT_JOB_META.sub("", text).strip()
        if not text:
            continue

        if re.fullmatch(r"\d{1,4}", text):
            continue

        # Centered page numbers: "- 3 -", "-- 7 --", or with en/em dashes.
        if re.fullmatch(r"[-\u2013\u2014]+\s*\d+\s*[-\u2013\u2014]+", text):
            continue

        # Repeated header/footer chrome detected across pages.
        # 1. Full-block exact/substring check (fast path).
        # 2. Strip any leading/trailing lines that match chrome so that a
        #    YOLO box spanning both a header line and body text is trimmed
        #    rather than kept whole.
        if chrome:
            if is_chrome(text, chrome):
                continue
            if "\n" in text:
                lines = text.split("\n")
                while lines and is_chrome(lines[0].strip(), chrome):
                    lines.pop(0)
                while lines and is_chrome(lines[-1].strip(), chrome):
                    lines.pop()
                text = "\n".join(lines).strip()
                if not text:
                    continue

        if region.label in ("Title", "Section-header"):
            text = text.replace("\n", " ")
            if _is_running_header_chrome(region, text, regions, page_height):
                continue

        if region.label == "List-item":
            text = _LEADING_BULLET.sub("", text).strip()
            if not text:
                continue
            lines = [ln for ln in text.split("\n") if ln.strip()]
            parts.append("\n".join(f"- {ln}" for ln in lines))
            continue

        # Text regions can legitimately contain multiple paragraphs when
        # YOLO's bounding box covers the tail of one paragraph plus the
        # head of another (common when a box is redrawn around a page
        # header or image that interrupts the text flow).  Split the
        # region's text on paragraph boundaries so only the leading
        # paragraph participates in cross-region continuation merging;
        # trailing paragraphs are emitted as their own entries.  For
        # non-Text regions there is a single logical paragraph.
        if region.label == "Text":
            paragraphs = [p for p in text.split("\n\n") if p.strip()]
        else:
            paragraphs = [text]

        if not paragraphs:
            continue

        head = paragraphs[0]
        tail_paragraphs = paragraphs[1:]

        # --- Leading punctuation reattachment (head only) ---
        # When a sentence-ending glyph (e.g. ".") falls just outside the
        # preceding YOLO box it becomes the first character of the next
        # region.  Split it off and attach it to the previous Text region
        # that logically precedes this one so the punctuation lands at
        # the end of its sentence.
        #
        # Preferred target: a same-column predecessor in text_part_log.
        # Fallback: the immediately-prior text region when it sits
        # geometrically below us (column-transition signature).  Without
        # the fallback, punctuation stranded at the top of a new column
        # whose paragraph actually wraps from the previous column would
        # never find a same-column match and would be emitted as a
        # floating glyph on its own line.
        lead_m = _LEADING_SENTENCE_PUNCT.match(head)
        if lead_m and region.label == "Text" and text_part_log:
            punct = lead_m.group(1)
            merge_idx = None
            for prev_idx, prev_region in reversed(text_part_log):
                if _same_column(region, prev_region):
                    merge_idx = prev_idx
                    break
            if merge_idx is None:
                last_idx, last_region = text_part_log[-1]
                if _is_column_transition(region, last_region):
                    merge_idx = last_idx
            if merge_idx is not None:
                parts[merge_idx] = parts[merge_idx] + punct
                head = head[lead_m.end():].strip()

        # --- Cross-region continuation merge (head only) ---
        # Catches both lowercase sentence fragments that continue from a
        # preceding region AND tiny punctuation-only fragments (e.g. a
        # lone period) that YOLO split into their own bounding box.
        #
        # Merge strategy:
        #   1. Preferred: same-column history match (walks text_part_log
        #      in reverse until we find an overlapping column).
        #   2. Fallback: column-transition against the immediately-prior
        #      text region.  This handles the case where a sentence
        #      wraps from the bottom of column 1 into the top of
        #      column 2 -- the two regions don't share a column so
        #      _same_column returns False, but the upward y-jump in
        #      reading order gives us an unambiguous signal that a
        #      column break just happened and the fragment legitimately
        #      continues the previous region.  Scope is intentionally
        #      narrow (only text_part_log[-1]) to avoid the full
        #      permissive behaviour of the Docling assembler.
        head_consumed = False
        if head and region.label == "Text":
            first_ch = _first_alpha(head)
            is_lowercase_cont = first_ch is not None and first_ch.islower()
            is_punct_fragment = first_ch is None and len(head.strip()) <= 3
            if (is_lowercase_cont or is_punct_fragment) and text_part_log:
                merge_target = None
                for prev_idx, prev_region in reversed(text_part_log):
                    if _same_column(region, prev_region):
                        merge_target = (prev_idx, prev_region)
                        break
                column_transition = False
                if merge_target is None:
                    last_idx, last_region = text_part_log[-1]
                    if _is_column_transition(region, last_region):
                        merge_target = (last_idx, last_region)
                        column_transition = True

                if merge_target is not None:
                    prev_idx, _ = merge_target
                    prev_text = parts[prev_idx]
                    if _TRAILING_HYPHEN.search(prev_text):
                        joined = _TRAILING_HYPHEN.sub("", prev_text) + head
                    elif is_punct_fragment:
                        joined = prev_text + head
                    else:
                        joined = prev_text + " " + head
                    parts[prev_idx] = joined
                    head_consumed = True
                    # When a column-transition merge extends the target
                    # paragraph into a new column, record that column in
                    # the log so later same-column lookups (for leading
                    # punctuation, further continuations, or subsequent
                    # paragraphs of this same region) can find it.
                    if column_transition:
                        text_part_log.append((prev_idx, region))

        prefix = LABEL_TO_MD.get(region.label, "")

        if not head_consumed and head:
            if region.label == "Caption":
                parts.append(f"{prefix}{head}{prefix}")
            else:
                parts.append(f"{prefix}{head}")
            if region.label == "Text":
                text_part_log.append((len(parts) - 1, region))

        for para in tail_paragraphs:
            parts.append(f"{prefix}{para}")
            if region.label == "Text":
                text_part_log.append((len(parts) - 1, region))

    text_page.close()
    page.close()
    doc.close()

    return "\n\n".join(parts)
