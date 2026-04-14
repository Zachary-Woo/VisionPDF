"""
SAM encoder with multi-scale FPN and FCOS detection head for DocLayNet.

Reuses the SAM ViT-B encoder from encoder.py but exposes intermediate
feature maps at three scales (P3=64x64, P4=32x32, P5=16x16).  A
lightweight Feature Pyramid Network fuses these into 256-channel maps at
each level, and an anchor-free FCOS head predicts bounding boxes with
DocLayNet class labels.

Trainable parameters are only in the FPN + FCOS head (~3-5M).  The SAM
encoder is kept frozen with pretrained DeepSeek OCR 2 weights.
"""

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from benchmark.tier2_hybrid.sam.encoder import SAMEncoder, _get_abs_pos
from benchmark.config import DOCLAYNET_LABELS

NUM_CLASSES = len(DOCLAYNET_LABELS)  # 11


class MultiScaleSAMEncoder(nn.Module):
    """
    Wraps SAMEncoder to return multi-scale feature maps instead of only
    the final 16x16x896 output.

    Returns a dict with keys "p3", "p4", "p5":
      p3: (B, 256, H/16, W/16)  -- after neck
      p4: (B, 512, H/32, W/32)  -- after net_2
      p5: (B, 896, H/64, W/64)  -- after net_3
    """

    def __init__(self, sam_encoder: SAMEncoder):
        super().__init__()
        self.sam = sam_encoder

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        s = self.sam
        x = s.patch_embed(x)
        x = x + _get_abs_pos(s.pos_embed, x.size(1))
        for blk in s.blocks:
            x = blk(x)
        p3 = s.neck(x.permute(0, 3, 1, 2))
        p4 = s.net_2(p3)
        p5 = s.net_3(p4)
        return {"p3": p3, "p4": p4, "p5": p5}


# ---------------------------------------------------------------------------
# Feature Pyramid Network
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    """
    Lightweight top-down FPN that projects P3/P4/P5 to a common channel
    width (256) and fuses them with bilinear upsampling + elementwise add.
    """

    def __init__(self, in_channels_p3: int = 256, in_channels_p4: int = 512,
                 in_channels_p5: int = 896, out_channels: int = 256):
        super().__init__()
        self.lateral_p5 = nn.Conv2d(in_channels_p5, out_channels, 1)
        self.lateral_p4 = nn.Conv2d(in_channels_p4, out_channels, 1)
        self.lateral_p3 = nn.Conv2d(in_channels_p3, out_channels, 1)

        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        p3_in, p4_in, p5_in = features["p3"], features["p4"], features["p5"]

        p5 = self.lateral_p5(p5_in)

        p4 = self.lateral_p4(p4_in) + F.interpolate(
            p5, size=p4_in.shape[2:], mode="bilinear", align_corners=False
        )
        p4 = self.smooth_p4(p4)

        p3 = self.lateral_p3(p3_in) + F.interpolate(
            p4, size=p3_in.shape[2:], mode="bilinear", align_corners=False
        )
        p3 = self.smooth_p3(p3)

        return {"p3": p3, "p4": p4, "p5": p5}


# ---------------------------------------------------------------------------
# FCOS detection head
# ---------------------------------------------------------------------------

class FCOSHead(nn.Module):
    """
    Anchor-free FCOS-style detection head applied at each FPN level.

    Predicts per-location:
      - class logits  (NUM_CLASSES)
      - bbox offsets   (4: left, top, right, bottom distances)
      - centerness     (1: scalar indicating proximity to box centre)
    """

    def __init__(self, in_channels: int = 256, num_convs: int = 4):
        super().__init__()
        cls_tower = []
        reg_tower = []
        for _ in range(num_convs):
            cls_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU(inplace=True))
            reg_tower.append(nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False))
            reg_tower.append(nn.GroupNorm(32, in_channels))
            reg_tower.append(nn.ReLU(inplace=True))

        self.cls_tower = nn.Sequential(*cls_tower)
        self.reg_tower = nn.Sequential(*reg_tower)

        self.cls_logits = nn.Conv2d(in_channels, NUM_CLASSES, 3, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, 3, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for modules in [self.cls_tower, self.reg_tower]:
            for m in modules.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_logits.weight, std=0.01)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

        nn.init.normal_(self.bbox_pred.weight, std=0.01)
        nn.init.zeros_(self.bbox_pred.bias)
        nn.init.normal_(self.centerness.weight, std=0.01)
        nn.init.zeros_(self.centerness.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, C, H, W) single FPN level feature map

        Returns
        -------
        cls_logits  : (B, NUM_CLASSES, H, W)
        bbox_pred   : (B, 4, H, W) -- exp-transformed LTRB distances
        centerness  : (B, 1, H, W)
        """
        cls_feat = self.cls_tower(x)
        reg_feat = self.reg_tower(x)
        return (
            self.cls_logits(cls_feat),
            torch.exp(self.bbox_pred(reg_feat).clamp(max=16.0)),
            self.centerness(cls_feat),
        )


# ---------------------------------------------------------------------------
# Combined model
# ---------------------------------------------------------------------------

class SAMDetector(nn.Module):
    """
    Frozen SAM encoder + trainable FPN + trainable FCOS head.

    At inference time, call ``detect()`` to get a list of
    (label, x1, y1, x2, y2, confidence) tuples in the input image's
    pixel coordinate space.
    """

    LEVEL_RANGES = {
        "p3": (0, 64),
        "p4": (64, 256),
        "p5": (256, 1e8),
    }
    STRIDES = {"p3": 16, "p4": 32, "p5": 64}

    def __init__(self, sam_encoder: SAMEncoder):
        super().__init__()
        self.backbone = MultiScaleSAMEncoder(sam_encoder)
        self.fpn = FPN()
        self.head = FCOSHead()

    def forward(self, images: torch.Tensor) -> Dict[str, List[Tuple[torch.Tensor, ...]]]:
        """
        Training forward pass.  Returns per-level predictions for loss
        computation.
        """
        features = self.backbone(images)
        fpn_features = self.fpn(features)
        outputs = {}
        for level_name, feat in fpn_features.items():
            outputs[level_name] = self.head(feat)
        return outputs

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
    ) -> List[List[Tuple[str, float, float, float, float, float]]]:
        """
        Inference: returns a list (per image in batch) of detected regions.
        Each region is (label, x1, y1, x2, y2, score).
        """
        self.eval()
        outputs = self.forward(images)
        batch_size = images.shape[0]

        all_detections = [[] for _ in range(batch_size)]

        for level_name, (cls_logits, bbox_pred, centerness) in outputs.items():
            stride = self.STRIDES[level_name]
            B, C, H, W = cls_logits.shape

            shifts_y = (torch.arange(H, device=cls_logits.device) + 0.5) * stride
            shifts_x = (torch.arange(W, device=cls_logits.device) + 0.5) * stride
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

            for b in range(B):
                cls_scores = cls_logits[b].sigmoid()
                cent = centerness[b, 0].sigmoid()
                ltrb = bbox_pred[b]

                scores = cls_scores * cent.unsqueeze(0)

                scores_flat = scores.permute(1, 2, 0).reshape(-1, C)
                max_scores, cls_ids = scores_flat.max(dim=1)
                keep_mask = max_scores > score_threshold
                if not keep_mask.any():
                    continue

                kept_scores = max_scores[keep_mask]
                kept_cls = cls_ids[keep_mask]
                kept_ltrb = ltrb.permute(1, 2, 0).reshape(-1, 4)[keep_mask]
                kept_cx = shift_x.reshape(-1)[keep_mask]
                kept_cy = shift_y.reshape(-1)[keep_mask]

                x1 = kept_cx - kept_ltrb[:, 0]
                y1 = kept_cy - kept_ltrb[:, 1]
                x2 = kept_cx + kept_ltrb[:, 2]
                y2 = kept_cy + kept_ltrb[:, 3]
                boxes = torch.stack([x1, y1, x2, y2], dim=1)

                nms_keep = torchvision.ops.batched_nms(
                    boxes, kept_scores, kept_cls, nms_threshold
                )
                for idx in nms_keep:
                    ci = int(kept_cls[idx].item())
                    label = DOCLAYNET_LABELS[ci] if ci < NUM_CLASSES else "Text"
                    bx = boxes[idx]
                    all_detections[b].append((
                        label,
                        float(bx[0]), float(bx[1]),
                        float(bx[2]), float(bx[3]),
                        float(kept_scores[idx]),
                    ))

        return all_detections


# ---------------------------------------------------------------------------
# FCOS target assignment for training
# ---------------------------------------------------------------------------

def compute_fcos_targets(
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
    fpn_levels: Dict[str, Tuple[int, int, int]],
    strides: Dict[str, int],
    level_ranges: Dict[str, Tuple[float, float]],
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Assign FCOS regression/classification targets for one image.

    Parameters
    ----------
    gt_boxes : (N, 4) in xyxy pixel coords
    gt_labels : (N,) integer class indices (0-based)
    fpn_levels : dict mapping level name -> (C, H, W) spatial size
    strides : dict mapping level name -> pixel stride
    level_ranges : dict mapping level name -> (lo, hi) size range

    Returns
    -------
    dict of level_name -> (cls_targets, reg_targets, centerness_targets)
    """
    device = gt_boxes.device
    targets = {}

    for level_name, (_, H, W) in fpn_levels.items():
        stride = strides[level_name]
        lo, hi = level_ranges[level_name]

        cls_target = torch.zeros(H, W, dtype=torch.long, device=device)
        reg_target = torch.zeros(H, W, 4, dtype=torch.float32, device=device)
        cent_target = torch.zeros(H, W, dtype=torch.float32, device=device)

        if gt_boxes.numel() == 0:
            targets[level_name] = (cls_target, reg_target, cent_target)
            continue

        shifts_y = (torch.arange(H, device=device).float() + 0.5) * stride
        shifts_x = (torch.arange(W, device=device).float() + 0.5) * stride
        sy, sx = torch.meshgrid(shifts_y, shifts_x, indexing="ij")

        N = gt_boxes.shape[0]
        sx_exp = sx.unsqueeze(-1).expand(H, W, N)
        sy_exp = sy.unsqueeze(-1).expand(H, W, N)

        l = sx_exp - gt_boxes[:, 0].view(1, 1, N)
        t = sy_exp - gt_boxes[:, 1].view(1, 1, N)
        r = gt_boxes[:, 2].view(1, 1, N) - sx_exp
        b = gt_boxes[:, 3].view(1, 1, N) - sy_exp
        ltrb = torch.stack([l, t, r, b], dim=-1)

        inside = ltrb.min(dim=-1).values > 0
        max_ltrb = ltrb.max(dim=-1).values
        in_range = (max_ltrb >= lo) & (max_ltrb < hi)
        valid = inside & in_range

        areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        areas_exp = areas.view(1, 1, N).expand(H, W, N)
        areas_exp = torch.where(valid, areas_exp, torch.tensor(1e8, device=device))
        min_area_idx = areas_exp.argmin(dim=-1)

        has_gt = valid.any(dim=-1)

        for hy in range(H):
            for wx in range(W):
                if not has_gt[hy, wx]:
                    continue
                gi = min_area_idx[hy, wx]
                cls_target[hy, wx] = gt_labels[gi] + 1
                l_val = ltrb[hy, wx, gi, 0]
                t_val = ltrb[hy, wx, gi, 1]
                r_val = ltrb[hy, wx, gi, 2]
                b_val = ltrb[hy, wx, gi, 3]
                reg_target[hy, wx] = torch.tensor([l_val, t_val, r_val, b_val])
                lr_min = min(l_val, r_val)
                lr_max = max(l_val, r_val)
                tb_min = min(t_val, b_val)
                tb_max = max(t_val, b_val)
                if lr_max > 0 and tb_max > 0:
                    cent_target[hy, wx] = math.sqrt(
                        (lr_min / lr_max) * (tb_min / tb_max)
                    )

        targets[level_name] = (cls_target, reg_target, cent_target)

    return targets


# ---------------------------------------------------------------------------
# FCOS losses
# ---------------------------------------------------------------------------

def focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Multi-class focal loss.
    pred: (N, C) logits
    target: (N,) long -- 0 = background, 1..C = class
    """
    num_classes = pred.shape[1]
    t = torch.zeros_like(pred)
    fg_mask = target > 0
    if fg_mask.any():
        t[fg_mask, target[fg_mask] - 1] = 1.0

    p = pred.sigmoid()
    ce = F.binary_cross_entropy_with_logits(pred, t, reduction="none")
    pt = torch.where(t == 1, p, 1 - p)
    loss = alpha * (1 - pt) ** gamma * ce
    return loss.sum() / max(fg_mask.sum().item(), 1)


def giou_loss(pred_ltrb: torch.Tensor, target_ltrb: torch.Tensor) -> torch.Tensor:
    """
    GIoU loss on LTRB distances.
    """
    if pred_ltrb.numel() == 0:
        return pred_ltrb.sum() * 0.0

    pred_area = (pred_ltrb[:, 0] + pred_ltrb[:, 2]) * (pred_ltrb[:, 1] + pred_ltrb[:, 3])
    target_area = (target_ltrb[:, 0] + target_ltrb[:, 2]) * (target_ltrb[:, 1] + target_ltrb[:, 3])

    w_inter = torch.min(pred_ltrb[:, 0], target_ltrb[:, 0]) + torch.min(pred_ltrb[:, 2], target_ltrb[:, 2])
    h_inter = torch.min(pred_ltrb[:, 1], target_ltrb[:, 1]) + torch.min(pred_ltrb[:, 3], target_ltrb[:, 3])
    inter = w_inter.clamp(min=0) * h_inter.clamp(min=0)

    union = pred_area + target_area - inter + 1e-7
    iou = inter / union

    w_enclose = torch.max(pred_ltrb[:, 0], target_ltrb[:, 0]) + torch.max(pred_ltrb[:, 2], target_ltrb[:, 2])
    h_enclose = torch.max(pred_ltrb[:, 1], target_ltrb[:, 1]) + torch.max(pred_ltrb[:, 3], target_ltrb[:, 3])
    enclose_area = w_enclose * h_enclose + 1e-7

    giou = iou - (enclose_area - union) / enclose_area
    return (1 - giou).mean()
