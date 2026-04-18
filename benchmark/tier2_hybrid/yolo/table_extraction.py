"""
Standalone TableFormer wrapper for YOLO-detected Table regions.

We keep YOLO responsible for finding table bounding boxes on the page
but hand each table off to Docling's TableFormer predictor for
cell-level structure recovery.  The PDF text layer (already extracted
in ``text_assembly``) supplies the cell text so the output HTML has
exact characters rather than OCR guesses.

This module isolates the Docling / ibm_models dependency chain so that
the rest of the tier-2 YOLO pipeline stays lightweight and continues
to work when ``docling_ibm_models`` is not installed -- in that case
``extract_table`` simply returns an empty string and the caller
emits the ``[Table]`` placeholder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# =========================================================================
# Public constants and module-level state
# =========================================================================

# Extra padding (in image pixels) applied to the YOLO bbox before
# cropping the table out of the page.  Small cushion avoids clipping
# rules / borders that sit just outside the detected box.
_CROP_PADDING_PX = 5

# TableFormer was trained on 144 dpi crops; the inference helper
# rescales as needed, but we pass the image at whatever DPI the caller
# rendered the page at and let the predictor handle its own resizing.

_TF_PREDICTOR: Optional[object] = None
_TF_MODE: Optional[str] = None
_TF_UNAVAILABLE_REASON: Optional[str] = None
_WARNED_ONCE: bool = False


# =========================================================================
# Predictor loading
# =========================================================================

def _download_tableformer_artifacts() -> Path:
    """
    Return the path to the cached TableFormer model_artifacts directory.

    Uses Docling's own download helper so we pull the exact revision
    Docling ships with, avoiding version mismatches between the
    weights and the inference code.
    """
    from docling.models.utils.hf_model_download import download_hf_model

    cache_root = download_hf_model(
        repo_id="docling-project/docling-models",
        revision="v2.3.0",
    )
    return cache_root / "model_artifacts" / "tableformer"


def _load_tf_config(mode: str, artifacts_root: Path) -> dict:
    """
    Load a TableFormer config JSON for the requested quality *mode*
    and patch it so the predictor knows where to find the weights.

    Mirrors Docling's own ``TableStructureModel.__init__`` logic to
    avoid any subtle divergence between how Docling runs TableFormer
    and how we do.
    """
    import docling_ibm_models.tableformer.common as c

    sub = "accurate" if mode == "accurate" else "fast"
    weights_dir = artifacts_root / sub
    tm_config = c.read_config(str(weights_dir / "tm_config.json"))
    tm_config["model"]["save_dir"] = str(weights_dir)
    return tm_config


def _pick_device() -> str:
    """
    Return a device string suitable for TFPredictor.

    TableFormer runs fine on CPU and is only a few hundred MB, so we
    fall back to CPU whenever CUDA isn't available instead of raising.
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _get_predictor(mode: str = "accurate"):
    """
    Return a cached ``TFPredictor`` instance for the given *mode*.

    The predictor loads ~200 MB of weights the first time it is
    called; subsequent calls reuse the cached instance.  If loading
    fails for any reason (``docling_ibm_models`` not installed,
    weights cannot be downloaded, config is malformed) the reason is
    stashed in ``_TF_UNAVAILABLE_REASON`` and we return ``None`` from
    every subsequent call without retrying.
    """
    global _TF_PREDICTOR, _TF_MODE, _TF_UNAVAILABLE_REASON

    if _TF_UNAVAILABLE_REASON is not None:
        return None

    if _TF_PREDICTOR is not None and _TF_MODE == mode:
        return _TF_PREDICTOR

    try:
        from docling_ibm_models.tableformer.data_management.tf_predictor \
            import TFPredictor

        artifacts_root = _download_tableformer_artifacts()
        config = _load_tf_config(mode, artifacts_root)
        device = _pick_device()
        _TF_PREDICTOR = TFPredictor(config, device=device, num_threads=4)
        _TF_MODE = mode
        return _TF_PREDICTOR
    except Exception as exc:
        _TF_UNAVAILABLE_REASON = f"{type(exc).__name__}: {exc}"
        _TF_PREDICTOR = None
        _TF_MODE = None
        return None


# =========================================================================
# Coordinate helpers
# =========================================================================

def _pt_to_px(value: float, scale: float) -> float:
    """Convert a single PDF-point measurement to image pixels."""
    return value * scale


def _filter_cells_in_bbox(
    pdf_cells: List[Dict],
    region_bbox: Tuple[float, float, float, float],
) -> List[Dict]:
    """
    Return the subset of *pdf_cells* whose bbox centre lies inside the
    given table region bbox.  Cells that straddle the boundary are
    dropped rather than clipped -- TableFormer handles that well and
    partial cells can confuse the cell-matching step.
    """
    rx1, ry1, rx2, ry2 = region_bbox
    inside: List[Dict] = []
    for cell in pdf_cells:
        l, t, r, b = cell["bbox"]
        cx = (l + r) / 2.0
        cy = (t + b) / 2.0
        if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
            inside.append(cell)
    return inside


def collect_pdf_cells_for_region(
    text_page,
    region_bbox: Tuple[float, float, float, float],
    page_height: float,
) -> List[Dict]:
    """
    Walk the PDF text page's glyph stream and return word-level cell
    dicts for all glyphs whose centre falls inside *region_bbox*.

    TableFormer's ``do_matching`` step needs word-granularity tokens
    with accurate bboxes to assign them to cells.  We mirror Docling's
    approach (word-level text cells via segmented-page queries) by
    grouping consecutive glyphs on the same visual line, splitting at
    whitespace, and emitting one dict per word.

    Returned cells use the same y-down "PDF-point / scale" coordinate
    system as ``region_bbox`` -- the glyph bboxes pypdfium2 reports
    are in PDF units with y-up, so we flip y against *page_height*
    before emitting.
    """
    rx1, ry1, rx2, ry2 = region_bbox

    # (char, flipped-centre-y, l, t_flipped, r, b_flipped)
    glyphs: List[Tuple[str, float, float, float, float, float]] = []
    n = text_page.count_chars()
    for i in range(n):
        ch = text_page.get_text_range(i, 1)
        if not ch:
            continue
        try:
            left, bottom, right, top = text_page.get_charbox(i)
        except Exception:
            continue
        flipped_top = page_height - top
        flipped_bottom = page_height - bottom
        cx = (left + right) / 2.0
        cy = (flipped_top + flipped_bottom) / 2.0
        if not (rx1 <= cx <= rx2 and ry1 <= cy <= ry2):
            continue
        glyphs.append((ch, cy, left, flipped_top, right, flipped_bottom))

    if not glyphs:
        return []

    # Bucket glyphs into visual lines by centre-y similarity.  Using a
    # small tolerance keeps tight tabular rows in the same bucket.
    LINE_TOLERANCE = 3.0
    glyphs.sort(key=lambda g: (g[1], g[2]))

    lines: List[List[Tuple[str, float, float, float, float, float]]] = []
    current: List[Tuple[str, float, float, float, float, float]] = [glyphs[0]]
    for g in glyphs[1:]:
        if abs(g[1] - current[0][1]) <= LINE_TOLERANCE:
            current.append(g)
        else:
            lines.append(current)
            current = [g]
    lines.append(current)

    # Within each line, sort left-to-right and split into words at
    # whitespace.
    cells: List[Dict] = []
    for line in lines:
        line.sort(key=lambda g: g[2])
        word_glyphs: List[Tuple[str, float, float, float, float, float]] = []
        for g in line:
            ch = g[0]
            if ch.isspace():
                if word_glyphs:
                    cells.append(_word_from_glyphs(word_glyphs))
                    word_glyphs = []
            else:
                word_glyphs.append(g)
        if word_glyphs:
            cells.append(_word_from_glyphs(word_glyphs))

    return cells


def _word_from_glyphs(
    glyphs: List[Tuple[str, float, float, float, float, float]],
) -> Dict:
    """
    Merge consecutive glyphs into a single word-level cell dict.

    The merged bbox is the union of the individual glyph bboxes in
    y-down coords; the text is the literal concatenation of the
    glyph characters (no space normalisation -- whitespace runs are
    already handled upstream).
    """
    text = "".join(g[0] for g in glyphs)
    l = min(g[2] for g in glyphs)
    t = min(g[3] for g in glyphs)
    r = max(g[4] for g in glyphs)
    b = max(g[5] for g in glyphs)
    return {"text": text, "bbox": (l, t, r, b)}


def _build_page_input(
    table_image: Image.Image,
    cell_tokens: List[Dict],
) -> dict:
    """
    Package the cropped table image and translated cell tokens into
    the dict format ``TFPredictor.multi_table_predict`` expects.
    """
    return {
        "width": float(table_image.width),
        "height": float(table_image.height),
        "image": np.asarray(table_image),
        "tokens": cell_tokens,
    }


# =========================================================================
# OTSL / cell grid -> HTML
# =========================================================================

def _escape_html(text: str) -> str:
    """Minimal HTML-safe escape (keeps output readable; no attributes)."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def _cells_to_html(
    tf_responses: List[dict],
    num_rows: int,
    num_cols: int,
) -> str:
    """
    Build an HTML ``<table>`` string from TableFormer's per-cell
    responses.

    Each response is a dict shaped like ``docling_core.types.doc.TableCell``
    with ``start_row_offset_idx``, ``start_col_offset_idx``,
    ``row_span``, ``col_span``, ``text``, and ``column_header`` fields.
    We lay out a ``num_rows`` x ``num_cols`` grid, placing each cell at
    its start row/col and marking the spanned region so later cells
    don't overwrite it.

    This is a minimal replacement for Docling's ``Table.export_to_html``
    that lets us generate HTML without instantiating a full Docling
    ``DoclingDocument``.
    """
    if num_rows <= 0 or num_cols <= 0 or not tf_responses:
        return ""

    occupied = [[False] * num_cols for _ in range(num_rows)]
    cell_map: Dict[Tuple[int, int], dict] = {}

    for resp in tf_responses:
        r = int(resp.get("start_row_offset_idx", 0))
        c = int(resp.get("start_col_offset_idx", 0))
        rs = max(1, int(resp.get("row_span", 1)))
        cs = max(1, int(resp.get("col_span", 1)))
        if not (0 <= r < num_rows and 0 <= c < num_cols):
            continue
        cell_map[(r, c)] = {
            "text": str(resp.get("text", "") or ""),
            "row_span": rs,
            "col_span": cs,
            "column_header": bool(resp.get("column_header", False)),
            "row_header": bool(resp.get("row_header", False)),
        }
        for rr in range(r, min(r + rs, num_rows)):
            for cc in range(c, min(c + cs, num_cols)):
                occupied[rr][cc] = True

    lines: List[str] = ["<table>"]
    for r in range(num_rows):
        lines.append("  <tr>")
        c = 0
        while c < num_cols:
            cell = cell_map.get((r, c))
            if cell is not None:
                tag = "th" if (cell["column_header"] or cell["row_header"]) else "td"
                attrs = []
                if cell["row_span"] > 1:
                    attrs.append(f'rowspan="{cell["row_span"]}"')
                if cell["col_span"] > 1:
                    attrs.append(f'colspan="{cell["col_span"]}"')
                attr_str = (" " + " ".join(attrs)) if attrs else ""
                text = _escape_html(cell["text"].strip())
                lines.append(f"    <{tag}{attr_str}>{text}</{tag}>")
                c += cell["col_span"]
            elif occupied[r][c]:
                c += 1
            else:
                lines.append("    <td></td>")
                c += 1
        lines.append("  </tr>")
    lines.append("</table>")
    return "\n".join(lines)


# =========================================================================
# Main entrypoint
# =========================================================================

def extract_table(
    page_image: Image.Image,
    region_bbox: Tuple[float, float, float, float],
    pdf_cells: List[Dict],
    page_pdf_size: Tuple[float, float],
    mode: str = "accurate",
) -> str:
    """
    Run TableFormer on a single YOLO-detected Table region and return
    an HTML representation of the recovered cell structure.

    Parameters
    ----------
    page_image
        The full rendered page as a PIL image.  The caller normally
        produces this with ``benchmark.pdf_render.render_page``.
    region_bbox
        ``(x1, y1, x2, y2)`` for the table region in the same
        "y-down PDF-points divided by scale" coordinate system used by
        the rest of the YOLO pipeline (i.e. the same units as
        ``Region.x1..y2``).
    pdf_cells
        List of glyph-run dicts shaped like
        ``{"text": str, "bbox": (l, t, r, b)}`` with bboxes in the
        same coord system as ``region_bbox``.  May already be filtered
        to this table; this function filters again defensively.
    page_pdf_size
        ``(page_width_pts, page_height_pts)`` for the PDF page.  Used
        to compute the pixels-per-point scale.
    mode
        TableFormer quality mode: ``"accurate"`` or ``"fast"``.

    Returns
    -------
    str
        HTML markup for the table, or an empty string when
        TableFormer is unavailable / the extraction fails / no PDF
        cells land inside the region.  Callers fall back to the
        ``[Table]`` placeholder on empty output.
    """
    global _WARNED_ONCE

    predictor = _get_predictor(mode)
    if predictor is None:
        if not _WARNED_ONCE:
            log.warning(
                "TableFormer unavailable (%s); falling back to [Table] placeholder.",
                _TF_UNAVAILABLE_REASON,
            )
            _WARNED_ONCE = True
        return ""

    cells_in_region = _filter_cells_in_bbox(pdf_cells, region_bbox)
    if not cells_in_region:
        # Image-only table or detector mis-classification; fall back.
        return ""

    page_pdf_w, page_pdf_h = page_pdf_size
    if page_pdf_w <= 0 or page_pdf_h <= 0:
        return ""

    scale_x = page_image.width / page_pdf_w
    scale_y = page_image.height / page_pdf_h

    rx1, ry1, rx2, ry2 = region_bbox
    px1 = max(0, int(_pt_to_px(rx1, scale_x)) - _CROP_PADDING_PX)
    py1 = max(0, int(_pt_to_px(ry1, scale_y)) - _CROP_PADDING_PX)
    px2 = min(page_image.width, int(_pt_to_px(rx2, scale_x)) + _CROP_PADDING_PX)
    py2 = min(page_image.height, int(_pt_to_px(ry2, scale_y)) + _CROP_PADDING_PX)
    if px2 <= px1 or py2 <= py1:
        return ""

    table_image = page_image.crop((px1, py1, px2, py2))

    # Build cell tokens in the cropped image's pixel space.  Each token
    # must carry a unique ``id`` (the predictor's cell-matching step
    # keys on it -- duplicates collapse cells).
    cell_tokens: List[Dict] = []
    for idx, cell in enumerate(cells_in_region):
        text = (cell.get("text") or "").strip()
        if not text:
            continue
        l, t, r, b = cell["bbox"]
        tx1 = _pt_to_px(l, scale_x) - px1
        ty1 = _pt_to_px(t, scale_y) - py1
        tx2 = _pt_to_px(r, scale_x) - px1
        ty2 = _pt_to_px(b, scale_y) - py1
        cell_tokens.append({
            "id": idx,
            "text": text,
            "bbox": {
                "l": float(tx1),
                "t": float(ty1),
                "r": float(tx2),
                "b": float(ty2),
                "coord_origin": "TOPLEFT",
            },
        })

    if not cell_tokens:
        return ""

    page_input = _build_page_input(table_image, cell_tokens)
    tbl_box = [0.0, 0.0, float(table_image.width), float(table_image.height)]

    try:
        tf_output = predictor.multi_table_predict(
            page_input, [tbl_box], do_matching=True,
        )
    except Exception as exc:
        log.warning("TableFormer prediction failed: %s", exc)
        return ""

    if not tf_output:
        return ""

    table_out = tf_output[0]
    tf_responses = table_out.get("tf_responses") or []
    details = table_out.get("predict_details") or {}
    num_rows = int(details.get("num_rows", 0))
    num_cols = int(details.get("num_cols", 0))

    return _cells_to_html(tf_responses, num_rows, num_cols)
