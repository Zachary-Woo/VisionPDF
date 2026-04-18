"""
Layout visualization for the Docling extraction pipeline.

Renders each PDF page and draws the layout elements that Docling
detected, coloured by type and labelled with reading-order numbers.
The output mirrors the annotated images produced by
``demo/visualize_regions.py`` for the YOLO pipeline so both can be
compared side-by-side.

Output (per page):
  {stem}_docling_layout.png          -- single-page documents
  {stem}_docling_layout_p{n}.png     -- multi-page documents (one file per page)
"""

from pathlib import Path
from typing import List

import pypdfium2 as pdfium
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Label -> colour mapping.  Values match Docling's DocItemLabel.value strings
# (snake_case).  Colours chosen to be consistent with demo/visualize_regions.py
# wherever the labels overlap.
# ---------------------------------------------------------------------------
_LABEL_COLORS = {
    "caption":            (255, 165,   0),
    "footnote":           (128, 128, 128),
    "formula":            (148,   0, 211),
    "list_item":          (  0, 128, 128),
    "page_footer":        (169, 169, 169),
    "page_header":        (169, 169, 169),
    "picture":            ( 65, 105, 225),
    "section_header":     (220,  20,  60),
    "table":              ( 34, 139,  34),
    "text":               ( 30, 144, 255),
    "title":              (255,  50,  50),
    "code":               (255, 140,   0),
    "key_value_region":   (100, 200, 100),
    "checkbox_selected":  (200, 100, 100),
    "checkbox_unselected":(150, 150, 150),
    "document_index":     (180, 180,  50),
    "form":               (200,  50, 200),
}

# Human-readable label names for the annotation tags.
_LABEL_DISPLAY = {
    "caption":            "Caption",
    "footnote":           "Footnote",
    "formula":            "Formula",
    "list_item":          "List-item",
    "page_footer":        "Page-footer",
    "page_header":        "Page-header",
    "picture":            "Picture",
    "section_header":     "Section-header",
    "table":              "Table",
    "text":               "Text",
    "title":              "Title",
    "code":               "Code",
    "key_value_region":   "Key-Value",
    "checkbox_selected":  "Checkbox (on)",
    "checkbox_unselected":"Checkbox (off)",
    "document_index":     "Doc-Index",
    "form":               "Form",
}

_PDF_BASE_DPI = 72


def _get_font(size: int = 14) -> ImageFont.ImageFont:
    """Load a TrueType font, falling back to the default bitmap font."""
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_layout(
    image: Image.Image,
    page_items: list,
    scale: float,
    page_height_pts: float,
    title: str,
) -> Image.Image:
    """
    Draw labelled bounding boxes on a copy of *image*.

    Each item receives a coloured rectangle, a small label tag above
    the box, and a reading-order number in its top-left corner.

    Args:
        image:           PIL image of the rendered PDF page.
        page_items:      List of ``(order_number, item, prov)`` tuples
                         for this page, in reading order.
        scale:           Pixels-per-PDF-point (``dpi / 72``).
        page_height_pts: Page height in PDF points (for y-axis flip).
        title:           Text drawn in the top-left corner of the image.

    Returns:
        Annotated copy of *image*.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    label_font = _get_font(11)
    order_font = _get_font(22)
    title_font = _get_font(16)

    for order_num, item, prov in page_items:
        bbox = prov.bbox

        # Docling stores bboxes with CoordOrigin.BOTTOMLEFT (PDF spec).
        # Convert to top-left origin so y values map directly to PIL pixels.
        try:
            bbox_tl = bbox.to_top_left_origin(page_height=page_height_pts)
        except (AttributeError, TypeError):
            # Already top-left or no conversion method available; use as-is.
            bbox_tl = bbox

        px1 = bbox_tl.l * scale
        py1 = bbox_tl.t * scale
        px2 = bbox_tl.r * scale
        py2 = bbox_tl.b * scale

        # Guard against degenerate boxes (can happen with empty regions).
        if px2 <= px1 or py2 <= py1:
            continue

        label_val = (
            item.label.value if hasattr(item.label, "value") else str(item.label)
        )
        color = _LABEL_COLORS.get(label_val, (200, 200, 200))
        display = _LABEL_DISPLAY.get(label_val,
                                     label_val.replace("_", " ").title())

        # Bounding box outline.
        draw.rectangle([px1, py1, px2, py2], outline=color, width=2)

        # Label tag above the box.
        tag_y = max(py1 - 14, 0)
        tag_w = len(display) * 6.5 + 4
        draw.rectangle([px1, tag_y, px1 + tag_w, tag_y + 13], fill=color)
        draw.text((px1 + 2, tag_y), display, fill="white", font=label_font)

        # Reading-order number badge in the top-left corner of the box.
        num = str(order_num)
        draw.rectangle([px1 + 1, py1 + 1, px1 + 26, py1 + 26], fill=color)
        draw.text((px1 + 4, py1 + 1), num, fill="white", font=order_font)

    # Title banner.
    draw.rectangle([4, 4, len(title) * 10 + 12, 24], fill="white")
    draw.text((8, 5), title, fill="black", font=title_font)

    return img


def save_layout_images(
    result,
    pdf_path: str,
    output_dir: str,
    dpi: int = 144,
) -> List[Path]:
    """
    Render annotated layout images for every page in the conversion result.

    For each page, the PDF is rendered to a PIL image using pypdfium2, then
    all items from Docling's DoclingDocument are drawn as coloured bounding
    boxes with reading-order numbers.

    Args:
        result:     Docling ``ConversionResult`` from ``converter.convert()``.
        pdf_path:   Path to the source PDF (used for rendering and naming).
        output_dir: Directory where PNG files are written.
        dpi:        Rendering resolution (pixels per inch).

    Returns:
        List of ``Path`` objects for the saved PNG files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale = dpi / _PDF_BASE_DPI
    stem = Path(pdf_path).stem

    # --- Collect items grouped by page in reading order ---
    # iterate_items() yields elements in Docling's resolved reading order.
    items_by_page: dict = {}
    order_num = 0
    for item, _level in result.document.iterate_items():
        if not item.prov:
            continue
        order_num += 1
        for prov in item.prov:
            page_no = prov.page_no  # 1-based
            if page_no not in items_by_page:
                items_by_page[page_no] = []
            items_by_page[page_no].append((order_num, item, prov))

    if not items_by_page:
        return []

    # --- Render pages and annotate ---
    doc_pdf = pdfium.PdfDocument(pdf_path)
    saved: List[Path] = []
    multi_page = len(items_by_page) > 1

    for page_no in sorted(items_by_page.keys()):
        pdf_page = doc_pdf[page_no - 1]  # pypdfium2 is 0-indexed
        page_width_pts, page_height_pts = pdf_page.get_size()

        bitmap = pdf_page.render(scale=scale)
        image = bitmap.to_pil().convert("RGB")
        pdf_page.close()

        items = items_by_page[page_no]
        title = f"Docling Layout - {len(items)} regions"
        annotated = draw_layout(image, items, scale, page_height_pts, title)

        fname = (f"{stem}_docling_layout_p{page_no}.png"
                 if multi_page else f"{stem}_docling_layout.png")
        out_path = output_dir / fname
        annotated.save(str(out_path))
        saved.append(out_path)

    doc_pdf.close()
    return saved
