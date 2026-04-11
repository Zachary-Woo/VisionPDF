"""
PDF-to-image rendering helper shared by all vision-based extraction scripts.

Uses pypdfium2 which is already a dependency for Tier 1 extraction.  The
helper also exposes the scale factor so callers can convert between pixel
coordinates (from a vision model) and PDF points (for text-layer queries).
"""

from typing import Tuple

import pypdfium2 as pdfium
from PIL import Image

from benchmark.config import RENDER_DPI

# PDF specification base resolution.
_PDF_BASE_DPI = 72


def render_page(
    pdf_path: str,
    page_index: int = 0,
    dpi: int = RENDER_DPI,
) -> Tuple[Image.Image, float]:
    """
    Render a single PDF page to a PIL Image.

    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.
    page_index : int
        Zero-based page number.
    dpi : int
        Target rendering resolution.

    Returns
    -------
    image : PIL.Image.Image
        Rendered page as an RGB image.
    scale : float
        Ratio ``dpi / 72``.  Multiply PDF-unit coordinates by this value
        to get pixel coordinates in the returned image (or divide pixel
        coords by it to get PDF units).
    """
    scale = dpi / _PDF_BASE_DPI
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    bitmap = page.render(scale=scale)
    image = bitmap.to_pil().convert("RGB")
    page.close()
    doc.close()
    return image, scale


def pixel_to_pdf_coords(
    x1: float, y1: float, x2: float, y2: float, scale: float
) -> Tuple[float, float, float, float]:
    """
    Convert pixel-space bounding box to PDF-point coordinates.
    """
    return x1 / scale, y1 / scale, x2 / scale, y2 / scale
