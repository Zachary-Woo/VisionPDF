"""
Post-processing utilities for DocDet predictions.

Covers the three steps between raw head outputs and the final list of
bounding boxes consumed by downstream pipeline code:

    decode_predictions  : map per-level (cls, reg, ctr) tensors to
                          a flat batch of (boxes, scores, class_ids).
    class_aware_nms     : IoU suppression among detections of the
                          SAME class (different classes may overlap
                          so e.g. a caption sitting just below a
                          figure is allowed).
    dilate_table_bboxes : expand Table-class boxes by a few pixels so
                          the TableFormer cell-detection model downstream
                          sees the full table including border rows.
"""

from typing import List, Tuple

import torch
import torchvision.ops as tv_ops

from .loss import LEVEL_NAMES, LEVEL_STRIDES


# ---------------------------------------------------------------------------
# Per-level decoding
# ---------------------------------------------------------------------------

def decode_predictions(
    outputs,  # Dict[level_name, (cls_logits, reg_stride_units, cent_logits)]
    score_threshold: float = 0.3,
    top_k_per_level: int = 1000,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Convert per-level head outputs into a flat list of detections per image.

    Expects the regression tensor to already be in STRIDE-NORMALISED
    units (i.e. multiplied by the per-level Scale and passed through
    ``exp``).  This function multiplies by the level stride to get
    pixel-space distances.

    Parameters
    ----------
    outputs : dict level_name -> (cls_logits, reg_pred, cent_logits)
        Each tensor shape matches the corresponding feature map:
        (B, num_classes, H, W), (B, 4, H, W), (B, 1, H, W).
    score_threshold : minimum combined score (cls * centerness).
    top_k_per_level : maximum retained detections per level per image
        before NMS.  Helps cap memory when many anchors fire.

    Returns
    -------
    all_boxes  : list of length B, each a (K, 4) float tensor xyxy.
    all_scores : list of length B, each a (K,) float tensor.
    all_cls    : list of length B, each a (K,) long tensor.
    """
    device = next(iter(outputs.values()))[0].device

    # Shape of cls_logits is (B, C, H, W); take first entry to get B.
    first_level = next(iter(outputs.values()))
    B = first_level[0].shape[0]
    num_classes = first_level[0].shape[1]

    boxes_per_image: List[List[torch.Tensor]] = [[] for _ in range(B)]
    scores_per_image: List[List[torch.Tensor]] = [[] for _ in range(B)]
    labels_per_image: List[List[torch.Tensor]] = [[] for _ in range(B)]

    for level_name in LEVEL_NAMES:
        if level_name not in outputs:
            continue
        cls_logits, reg_pred, cent_logits = outputs[level_name]
        stride = LEVEL_STRIDES[level_name]
        _, _, H, W = cls_logits.shape

        shifts_y = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) * stride
        shifts_x = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) * stride
        sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        for b in range(B):
            cls_scores = cls_logits[b].sigmoid()  # (C, H, W)
            cent = cent_logits[b, 0].sigmoid()    # (H, W)
            ltrb = reg_pred[b] * stride            # (4, H, W) pixel-space

            scores = cls_scores * cent.unsqueeze(0)
            scores_flat = scores.permute(1, 2, 0).reshape(-1, num_classes)
            max_scores, cls_ids = scores_flat.max(dim=1)
            keep = max_scores > score_threshold
            if not keep.any():
                continue

            kept_scores = max_scores[keep]
            kept_cls = cls_ids[keep]
            kept_ltrb = ltrb.permute(1, 2, 0).reshape(-1, 4)[keep]
            kept_cx = sx.reshape(-1)[keep]
            kept_cy = sy.reshape(-1)[keep]

            if kept_scores.shape[0] > top_k_per_level:
                topk = torch.topk(kept_scores, top_k_per_level)
                sel = topk.indices
                kept_scores = topk.values
                kept_cls = kept_cls[sel]
                kept_ltrb = kept_ltrb[sel]
                kept_cx = kept_cx[sel]
                kept_cy = kept_cy[sel]

            x1 = kept_cx - kept_ltrb[:, 0]
            y1 = kept_cy - kept_ltrb[:, 1]
            x2 = kept_cx + kept_ltrb[:, 2]
            y2 = kept_cy + kept_ltrb[:, 3]
            boxes = torch.stack([x1, y1, x2, y2], dim=1)

            boxes_per_image[b].append(boxes)
            scores_per_image[b].append(kept_scores)
            labels_per_image[b].append(kept_cls)

    out_boxes: List[torch.Tensor] = []
    out_scores: List[torch.Tensor] = []
    out_labels: List[torch.Tensor] = []
    for b in range(B):
        if boxes_per_image[b]:
            out_boxes.append(torch.cat(boxes_per_image[b], dim=0))
            out_scores.append(torch.cat(scores_per_image[b], dim=0))
            out_labels.append(torch.cat(labels_per_image[b], dim=0))
        else:
            out_boxes.append(torch.zeros((0, 4), device=device))
            out_scores.append(torch.zeros((0,), device=device))
            out_labels.append(torch.zeros((0,), dtype=torch.long, device=device))

    return out_boxes, out_scores, out_labels


# ---------------------------------------------------------------------------
# Class-aware NMS
# ---------------------------------------------------------------------------

def class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float = 0.5,
    max_detections: int = 300,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Non-max suppression applied independently per class.

    Boxes of the same class that overlap more than ``iou_threshold``
    are suppressed, but boxes of different classes are always kept
    regardless of overlap.  This intentionally allows a caption box
    to sit directly below its figure box, a footnote under the main
    text, etc., all of which are valid document layouts.

    Parameters
    ----------
    boxes           : (N, 4) xyxy float tensor.
    scores          : (N,)   float tensor.
    labels          : (N,)   long tensor (class IDs).
    iou_threshold   : same-class overlap threshold.
    max_detections  : final cap on total boxes returned (by score).

    Returns
    -------
    kept_boxes, kept_scores, kept_labels : tensors filtered + ordered
        by descending score, with at most ``max_detections`` rows.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    kept = tv_ops.batched_nms(boxes, scores, labels, iou_threshold)
    if kept.numel() > max_detections:
        # batched_nms returns indices in score-descending order per class
        # but not globally; re-rank by score for the final cap.
        score_order = scores[kept].argsort(descending=True)
        kept = kept[score_order[:max_detections]]
    return boxes[kept], scores[kept], labels[kept]


# ---------------------------------------------------------------------------
# Table bbox dilation
# ---------------------------------------------------------------------------

def dilate_table_bboxes(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    table_class_id: int,
    pad_px: int = 2,
    image_size: Tuple[int, int] = None,
) -> torch.Tensor:
    """
    Expand Table-class boxes by ``pad_px`` pixels on each side.

    TableFormer's cell detector is sensitive to tight crops that
    clip off the outer border row / column.  A small uniform pad
    buys margin without risking overlap with neighbouring content.

    Parameters
    ----------
    boxes         : (N, 4) xyxy float tensor (modified out-of-place).
    labels        : (N,)   long tensor matching ``boxes``.
    table_class_id : class ID of the "Table" label (see
                    benchmark/config.py DOCLAYNET_LABELS, index 8).
    pad_px        : pixels to expand on each side.
    image_size    : optional (H, W) to clamp dilated boxes inside the
                    image bounds.  If None, no clamping is performed.

    Returns
    -------
    new_boxes : cloned tensor with table boxes dilated.
    """
    new_boxes = boxes.clone()
    if new_boxes.numel() == 0:
        return new_boxes

    mask = labels == table_class_id
    if not mask.any():
        return new_boxes

    new_boxes[mask, 0] -= pad_px
    new_boxes[mask, 1] -= pad_px
    new_boxes[mask, 2] += pad_px
    new_boxes[mask, 3] += pad_px

    if image_size is not None:
        H, W = image_size
        new_boxes[:, 0].clamp_(min=0, max=W)
        new_boxes[:, 1].clamp_(min=0, max=H)
        new_boxes[:, 2].clamp_(min=0, max=W)
        new_boxes[:, 3].clamp_(min=0, max=H)

    return new_boxes
