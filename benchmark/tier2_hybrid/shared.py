"""
Shared components for Tier 2 hybrid extraction methods.

This module contains the truly shared, stable primitives:
  - Region: the data class used by all detection backends
  - detect_page_columns: vertical gutter detection for multi-column pages
  - sort_columns_reading_order: interleave column groups with spanning regions
  - sort_regions_geometric: column-aware top-to-bottom, left-to-right sort

YOLO-specific merge heuristics, glyph extraction, and markdown
reconstruction live in ``benchmark.tier2_hybrid.yolo.text_assembly``.
LayoutReader reading-order prediction lives in
``benchmark.tier2_hybrid.reading_order``.
"""

from typing import List, Tuple


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
# Column detection
# =========================================================================

_FULL_WIDTH_RATIO = 0.70
_GAP_MIN_RATIO = 0.15
_CENTER_TOLERANCE_RATIO = 0.20
_MIN_NARROW_PER_SIDE = 2


def detect_page_columns(
    regions: List[Region],
) -> Tuple[List[List[Region]], List[Region]]:
    """
    Detect whether a page has a multi-column layout by looking for a
    vertical gutter -- a horizontal gap in region coverage near the
    center of the content area.

    Returns ``(columns, spanning)`` where *columns* is a list of region
    lists (one per detected column, ordered left-to-right) and *spanning*
    contains regions that stretch across the full page width (e.g.
    titles).  For single-column pages, ``columns`` has one entry holding
    all regions and ``spanning`` is empty.
    """
    if len(regions) < 4:
        return [list(regions)], []

    min_x = min(r.x1 for r in regions)
    max_x = max(r.x2 for r in regions)
    content_width = max_x - min_x
    if content_width < 1:
        return [list(regions)], []

    spanning: List[Region] = []
    narrow: List[Region] = []
    for r in regions:
        if (r.x2 - r.x1) > content_width * _FULL_WIDTH_RATIO:
            spanning.append(r)
        else:
            narrow.append(r)

    if len(narrow) < _MIN_NARROW_PER_SIDE * 2:
        return [list(regions)], []

    centers = sorted(r.center_x for r in narrow)

    best_gap = 0.0
    best_gap_mid = 0.0
    for i in range(1, len(centers)):
        gap = centers[i] - centers[i - 1]
        if gap > best_gap:
            best_gap = gap
            best_gap_mid = (centers[i] + centers[i - 1]) / 2.0

    content_center = (min_x + max_x) / 2.0
    near_center = abs(best_gap_mid - content_center) < content_width * _CENTER_TOLERANCE_RATIO

    if best_gap < content_width * _GAP_MIN_RATIO or not near_center:
        return [list(regions)], []

    left = [r for r in narrow if r.center_x < best_gap_mid]
    right = [r for r in narrow if r.center_x >= best_gap_mid]

    if len(left) < _MIN_NARROW_PER_SIDE or len(right) < _MIN_NARROW_PER_SIDE:
        return [list(regions)], []

    return [left, right], spanning


def sort_columns_reading_order(
    columns: List[List[Region]],
    spanning: List[Region],
) -> List[Region]:
    """
    Merge per-column region lists and full-width spanning regions into a
    single reading-order list.

    Spanning regions act as "fences" -- all column content above a
    spanning region is flushed (left column first, then right) before
    the spanning region is emitted.  After the last spanning region,
    remaining column content is flushed in the same left-then-right
    order.  Assumes English (LTR) reading direction.
    """
    for col in columns:
        col.sort(key=lambda r: r.center_y)
    spanning = sorted(spanning, key=lambda r: r.center_y)

    col_cursors = [0] * len(columns)
    result: List[Region] = []

    def _flush_above(y_limit: float) -> None:
        """Emit column regions whose center_y is above *y_limit*."""
        for ci, col in enumerate(columns):
            while col_cursors[ci] < len(col) and col[col_cursors[ci]].center_y < y_limit:
                result.append(col[col_cursors[ci]])
                col_cursors[ci] += 1

    for sr in spanning:
        _flush_above(sr.center_y)
        result.append(sr)

    _flush_above(float("inf"))

    return result


# =========================================================================
# Geometric reading-order sort
# =========================================================================

def sort_single_column(regions: List[Region]) -> List[Region]:
    """
    Flat top-to-bottom, left-to-right sort for a single column of regions.

    Regions whose vertical centres are within LINE_TOLERANCE of each
    other are treated as one visual line and sorted left-to-right within
    that line.  Safe to call on a column list from
    ``detect_page_columns`` as well as on a full page's regions.
    """
    if not regions:
        return []

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


# Back-compat alias; prefer the public name in new code.
_sort_single_column = sort_single_column


def sort_regions_geometric(regions: List[Region]) -> List[Region]:
    """
    Sort regions in reading order using column-aware logic.

    First detects whether the page is multi-column.  If a vertical
    gutter is found, regions are partitioned into columns and read
    left-to-right (English order), with full-width spanning elements
    interleaved at their vertical position.  Single-column pages fall
    back to the simple line-tolerance sort.
    """
    if not regions:
        return regions

    columns, spanning = detect_page_columns(regions)
    if len(columns) == 1 and not spanning:
        return _sort_single_column(regions)
    return sort_columns_reading_order(columns, spanning)
