"""
Shared components for Tier 2 hybrid extraction methods.

Contains the Region data class, geometric reading-order sort, glyph-level
text-layer extraction, markdown reconstruction, and LayoutReader utilities
used across YOLO-based and SAM-based hybrid pipelines.

Key design principle: YOLO (or SAM) bounding boxes act as structural
*groupers*, not hard clip windows.  All text content comes from the PDF
text layer.  Each glyph in the text layer is assigned to the region whose
bounds contain its centre point, so no characters are clipped at region
edges.
"""

import re
from collections import defaultdict
from typing import Dict, List, Tuple

import pypdfium2 as pdfium
import torch
from transformers import LayoutLMv3ForTokenClassification

from benchmark.config import (
    DOCLAYNET_LABELS,
    LABEL_TO_MD,
    NON_TEXT_LABELS,
)
from benchmark.pdf_render import pixel_to_pdf_coords


# =========================================================================
# Region data class
# =========================================================================

class Region:
    """
    A detected document region with a label and bounding box.

    Coordinates are in pixel-divided-by-scale space (top-down, matching
    the rendered image orientation).
    """

    def __init__(
        self,
        label: str,
        x1: float, y1: float, x2: float, y2: float,
        confidence: float,
    ):
        self.label = label
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.confidence = confidence

    @property
    def center_y(self):
        return (self.y1 + self.y2) / 2.0

    @property
    def center_x(self):
        return (self.x1 + self.x2) / 2.0


# =========================================================================
# Geometric reading-order sort
# =========================================================================

def sort_regions_geometric(regions: List[Region]) -> List[Region]:
    """
    Sort regions in a simple top-to-bottom, left-to-right order.

    Regions whose vertical centres are within 10 PDF points of each
    other are treated as being on the same line and sorted left-to-right.
    """
    if not regions:
        return regions

    regions = sorted(regions, key=lambda r: (r.center_y, r.center_x))

    LINE_TOLERANCE = 10.0
    lines: List[List[Region]] = []
    current_line: List[Region] = [regions[0]]

    for r in regions[1:]:
        if abs(r.center_y - current_line[0].center_y) <= LINE_TOLERANCE:
            current_line.append(r)
        else:
            lines.append(current_line)
            current_line = [r]
    lines.append(current_line)

    ordered: List[Region] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda r: r.center_x))
    return ordered


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

            # Same-type overlap: always eligible for merging (the
            # original behaviour -- e.g. nested Text boxes).
            same_type = big_is_text == small_is_text

            # Cross-type exception: a Text box inside a Page-footer or
            # Page-header should be absorbed so the footer/header label
            # takes precedence and the text is excluded from body output.
            structural_override = (
                not big_is_text
                and small_is_text
                and regions[big_idx].label in _STRUCTURAL_OVERRIDE_LABELS
            )

            if not (same_type or structural_override):
                continue

            if _iou_contained(regions[small_idx], regions[big_idx]) >= threshold:
                keep.discard(small_idx)

    result = [regions[i] for i in sorted(keep)]
    return _merge_adjacent_same_label(result)


# Labels that commonly appear as vertically stacked boxes that should be
# merged into a single region.  List-item boxes often fire once per
# bullet; Text boxes can fragment when indented paragraphs cause YOLO
# to split a column into separate detections.  Merging lets the glyph
# assignment capture all the text in one pass.
_STACKABLE_LABELS = {"List-item", "Text"}


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
            # Compare against the cluster's current bounding box so a
            # tall earlier region doesn't leave a false gap.
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
    text_page, page_height: float,
) -> List[Tuple[int, float, float]]:
    """
    Collect every character on the page with its centre position
    converted to top-down coordinates (matching Region coordinate space).

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
        # PDF canvas is bottom-up; flip to top-down to match regions
        cy = page_height - (bottom + top) / 2.0
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

# Strips leading bullet characters from list-item text.  We already add
# a "- " markdown prefix for List-item regions, so the raw bullet symbol
# from the PDF text layer (e.g. "•") would create a redundant "- •".
_LEADING_BULLET = re.compile(r"^[•·∙▪▸►‣⁃]\s*", re.MULTILINE)

# Matches invisible print-job header/footer metadata embedded in the
# PDF text layer (e.g. "Grzimek Index - A to O 11/20/03 12:55 PM Page").
_PRINT_JOB_META = re.compile(
    r"^.*\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*[AP]M\s+Page.*$",
    re.MULTILINE | re.IGNORECASE,
)


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
    char_data: List[Tuple[str, float, float, float]] = []
    for ci in char_indices:
        ch = text_page.get_text_range(ci, 1)
        left = right = 0.0
        cy = prev_cy
        if ch.strip():
            try:
                left, bottom, right, top = text_page.get_charbox(ci)
                cy = page_height - (bottom + top) / 2.0
                prev_cy = cy
            except Exception:
                pass
        char_data.append((ch, left, right, cy))

    LINE_Y_TOL = 3.0
    lines: List[List[Tuple[str, float, float, float]]] = [[char_data[0]]]
    ref_y = char_data[0][3]

    for cd in char_data[1:]:
        if cd[0].strip() and abs(cd[3] - ref_y) > LINE_Y_TOL:
            lines.append([cd])
            ref_y = cd[3]
        else:
            lines[-1].append(cd)
            if cd[0].strip():
                ref_y = cd[3]

    line_info: List[Tuple[str, float, float]] = []
    for lc in lines:
        raw = "".join(ch for ch, _, _, _ in lc)
        text = _COLLAPSE_SPACES.sub(" ", _INTERNAL_WHITESPACE.sub(" ", raw)).strip()
        if not text:
            continue
        lefts = [l for ch, l, _, _ in lc if ch.strip() and l > 0]
        rights = [r for ch, _, r, _ in lc if ch.strip() and r > 0]
        x_start = min(lefts) if lefts else region.x1
        x_end = max(rights) if rights else region.x2
        line_info.append((text, x_start, x_end))

    if not line_info:
        return ""

    # Drop trailing/leading lines that are just 1-4 digits -- stray
    # page-number fragments that slipped inside this region's bbox.
    while len(line_info) > 1 and re.fullmatch(r"\d{1,4}", line_info[-1][0]):
        line_info.pop()
    while len(line_info) > 1 and re.fullmatch(r"\d{1,4}", line_info[0][0]):
        line_info.pop(0)

    if not line_info:
        return ""

    start_counts: Dict[int, int] = {}
    for _, xs, _ in line_info:
        b = round(xs / 2.0)
        start_counts[b] = start_counts.get(b, 0) + 1
    dominant_bin = min(start_counts, key=lambda k: (-start_counts[k], k))
    true_left = dominant_bin * 2.0
    INDENT_THRESHOLD = 3.0

    parts = [line_info[0][0]]
    for i in range(1, len(line_info)):
        cur_text, cur_x_start, _ = line_info[i]

        is_indented = cur_x_start > true_left + INDENT_THRESHOLD

        if is_indented and cur_text and cur_text[0].isupper():
            parts.append("\n" + cur_text)
        elif not is_indented:
            parts.append("\n" + cur_text)
        else:
            parts.append(" " + cur_text)

    return "".join(parts)


def regions_to_markdown(
    regions: List[Region], pdf_path: str, page_index: int = 0,
) -> str:
    """
    Build a markdown string from ordered, labelled regions.

    Each character in the PDF text layer is assigned to the YOLO region
    that contains its centre point.  YOLO provides structure (labels and
    reading order); the text layer provides the actual character content.

    Meaningful line breaks within a region are preserved; only
    paragraph wrapping and indented continuations are collapsed to
    spaces.  Regions are separated by blank lines.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    text_page = page.get_textpage()
    _width, page_height = page.get_size()

    regions = merge_overlapping_regions(regions)
    glyphs = _collect_glyphs(text_page, page_height)
    assignments = _assign_glyphs(glyphs, regions, page_height)

    parts: List[str] = []

    for ri, region in enumerate(regions):
        char_indices = assignments.get(ri, [])

        if region.label in NON_TEXT_LABELS:
            # Tables and pictures are structural placeholders.  Table
            # content has 2D column/row structure that flat text-layer
            # extraction cannot faithfully reproduce, so we emit a
            # placeholder rather than mangled text.
            if region.label == "Table":
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

        # Titles and section headers are single logical lines even when
        # the text wraps visually across multiple lines in the PDF.
        # Collapse internal newlines so the markdown heading stays on
        # one line (e.g. "# Full Title Here" instead of "# Full\nTitle").
        if region.label in ("Title", "Section-header"):
            text = text.replace("\n", " ")

        # List-item regions may contain multiple bullets (especially after
        # adjacent boxes are merged into one region).  Strip the raw bullet
        # symbols from the text layer, then add "- " to every line so each
        # item is a proper markdown list entry.
        if region.label == "List-item":
            text = _LEADING_BULLET.sub("", text).strip()
            if not text:
                continue
            lines = [ln for ln in text.split("\n") if ln.strip()]
            parts.append("\n".join(f"- {ln}" for ln in lines))
            continue

        prefix = LABEL_TO_MD.get(region.label, "")
        if region.label == "Caption":
            parts.append(f"{prefix}{text}{prefix}")
        else:
            parts.append(f"{prefix}{text}")

    text_page.close()
    page.close()
    doc.close()

    return "\n\n".join(parts)


# =========================================================================
# LayoutReader helpers
# =========================================================================

_MAX_LEN = 510
_CLS_TOKEN_ID = 0
_UNK_TOKEN_ID = 3
_EOS_TOKEN_ID = 2


def _boxes2inputs(boxes: List[List[int]]) -> Dict[str, torch.Tensor]:
    """Convert a list of [l, t, r, b] boxes (0-1000) into model inputs."""
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [_CLS_TOKEN_ID] + [_UNK_TOKEN_ID] * len(boxes) + [_EOS_TOKEN_ID]
    attention_mask = [1] * len(bbox)
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }


def _prepare_inputs(
    inputs: Dict[str, torch.Tensor], model: LayoutLMv3ForTokenClassification
) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        out[k] = v
    return out


def _parse_logits(logits: torch.Tensor, length: int) -> List[int]:
    """Greedy order decoding with duplicate resolution."""
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]

    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        order_to_idxes = {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        if not order_to_idxes:
            break
        for order, idxes in order_to_idxes.items():
            scored = sorted(
                ((idx, logits[idx, order]) for idx in idxes),
                key=lambda x: x[1],
                reverse=True,
            )
            for idx, _ in scored[1:]:
                ret[idx] = orders[idx].pop()
    return ret


def predict_reading_order(
    lr_model: LayoutLMv3ForTokenClassification,
    regions: List[Region],
    page_width: float,
    page_height: float,
) -> List[Region]:
    """
    Use LayoutReader to reorder *regions* into predicted reading order.

    Bounding boxes are scaled from PDF-point space to the 0-1000 integer
    range that LayoutReader expects.
    """
    if len(regions) <= 1:
        return regions

    x_scale = 1000.0 / page_width
    y_scale = 1000.0 / page_height

    boxes_1000: List[List[int]] = []
    for r in regions:
        boxes_1000.append([
            max(0, min(1000, round(r.x1 * x_scale))),
            max(0, min(1000, round(r.y1 * y_scale))),
            max(0, min(1000, round(r.x2 * x_scale))),
            max(0, min(1000, round(r.y2 * y_scale))),
        ])

    truncated = boxes_1000[: _MAX_LEN]

    inputs = _boxes2inputs(truncated)
    inputs = _prepare_inputs(inputs, lr_model)

    with torch.no_grad():
        logits = lr_model(**inputs).logits.cpu().squeeze(0)

    order = _parse_logits(logits, len(truncated))

    ordered = [regions[i] for i in order if i < len(regions)]

    if len(regions) > _MAX_LEN:
        ordered.extend(regions[_MAX_LEN:])

    return ordered
