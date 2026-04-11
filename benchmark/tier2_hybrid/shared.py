"""
Shared components for Tier 2 hybrid extraction methods.

Contains the Region data class, geometric reading-order sort, text-layer
extraction helpers, markdown reconstruction, and LayoutReader utilities
used across YOLO-based and SAM-based hybrid pipelines.
"""

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
    """A detected document region with a label and bounding box."""

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

    LINE_TOLERANCE = 10.0  # PDF points
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
# Text-layer extraction per region
# =========================================================================

def extract_text_in_region(text_page, region: Region) -> str:
    """
    Extract text bounded by the region's box (in PDF-point coordinates)
    from an already-open pypdfium2 text page.
    """
    text = text_page.get_text_bounded(
        region.x1, region.y1, region.x2, region.y2
    )
    return text.strip()


# =========================================================================
# Markdown reconstruction
# =========================================================================

def regions_to_markdown(regions: List[Region], pdf_path: str, page_index: int = 0) -> str:
    """
    Build a markdown string from ordered, labelled regions by pulling
    text content from the native PDF text layer.

    Opens the PDF once and reuses the text page handle for all regions.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    text_page = page.get_textpage()

    parts: List[str] = []

    for region in regions:
        if region.label in NON_TEXT_LABELS:
            if region.label == "Table":
                text = extract_text_in_region(text_page, region)
                if text:
                    parts.append(f"\n{text}\n")
            elif region.label == "Picture":
                parts.append("\n[Image]\n")
            continue

        text = extract_text_in_region(text_page, region)
        if not text:
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
