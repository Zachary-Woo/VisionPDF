"""
LayoutReader reading-order prediction.

Uses a LayoutLMv3 model (LayoutReader) to reorder detected document
regions into their predicted reading sequence.  Bounding boxes are
scaled to the 0-1000 integer range the model expects, fed through the
network, and the output logits are decoded greedily with duplicate
resolution.

Shared by YOLO-based hybrid Tier 2 pipelines.
"""

from collections import defaultdict
from typing import Dict, List

import torch
from transformers import LayoutLMv3ForTokenClassification

from benchmark.tier2_hybrid.shared import Region

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
