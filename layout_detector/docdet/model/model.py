"""
Top-level DocDet detector assembly.

Wires the backbone, PANet neck, shared FCOS head, per-level Scale
layers, and postprocess utilities into a single ``nn.Module``.

Training flow
-------------
Call ``model(images)`` to get a dict::

    {
        "p2": (cls_logits, reg_stride_units, centerness_logits),
        "p3": ...,
        "p4": ...,
    }

where ``reg_stride_units`` is already after ``exp(raw * scale)`` so
both the loss (stride-normalised target) and the postprocess (stride-
multiplied pixel-space distances) can read from the same tensor.

Inference flow
--------------
Call ``model.detect(images, ...)`` to get, per-image, a list of
``(label, x1, y1, x2, y2, score)`` tuples using the same label set
as ``benchmark.config.DOCLAYNET_LABELS``.

Number of classes
-----------------
Defaults to 11 (the full DocLayNet taxonomy).  Override via the
``num_classes`` kwarg if you need to warm-up the head on COCO
(num_classes=80) or another dataset.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .backbone import DocDetBackbone, build_backbone
from .blocks import Scale
from .head import FCOSHead
from .loss import LEVEL_NAMES, LEVEL_STRIDES
from .postprocess import class_aware_nms, decode_predictions

# Import the DocLayNet label list lazily in detect() so that the
# training code does not force a benchmark.config import.  Users
# should pass label_names=None to get raw integer class IDs or
# provide their own label_names list.


_DEFAULT_NUM_CLASSES = 11
_DEFAULT_REG_MAX = 16.0


class DocDet(nn.Module):
    """
    Full detector: backbone + neck + FCOS head + per-level Scale.

    Parameters
    ----------
    num_classes    : number of foreground classes (11 for DocLayNet).
    backbone_name  : "mobilenetv3_small" (default) or "efficientnet_b0".
    pretrained     : load ImageNet pretrained backbone weights.
    proj_channels  : shared channel width across backbone projections,
                     neck, and head (default 128).
    csp_depth      : depth of Bottleneck stacks inside each CSPBlock
                     in the neck (default 1).
    head_convs     : depth of each head tower (default 4).
    """

    def __init__(
        self,
        num_classes: int = _DEFAULT_NUM_CLASSES,
        backbone_name: str = "mobilenetv3_small",
        pretrained: bool = True,
        proj_channels: int = 128,
        csp_depth: int = 1,
        head_convs: int = 4,
    ):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = build_backbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            proj_channels=proj_channels,
        )

        # Neck import deferred so this module file stays above the
        # neck module in the import graph (neck depends on blocks,
        # which are now independent).
        from .neck import PANetNeck

        self.neck = PANetNeck(channels=proj_channels, csp_depth=csp_depth)

        self.head = FCOSHead(
            num_classes=num_classes,
            in_channels=proj_channels,
            num_convs=head_convs,
        )

        # Per-level learnable regression scale.  Registered as a
        # ModuleDict so state_dict serialisation is clean.
        self.scales = nn.ModuleDict(
            {name: Scale(init_value=1.0) for name in LEVEL_NAMES}
        )

    def _apply_reg_scale_and_exp(
        self, reg_raw: torch.Tensor, level_name: str
    ) -> torch.Tensor:
        """
        Multiply the head's raw regression by the per-level learnable
        scale, then apply ``exp`` (clamped) to get strictly positive
        stride-normalised LTRB distances.  The clamp prevents
        overflow early in training when the raw value may be large.
        """
        scaled = self.scales[level_name](reg_raw)
        return torch.exp(scaled.clamp(max=_DEFAULT_REG_MAX))

    def forward(
        self, images: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Training forward pass.

        Parameters
        ----------
        images : (B, 3, H, W) float tensor, H and W divisible by 16.

        Returns
        -------
        dict of level_name -> (cls_logits, reg_stride_units, centerness_logits)
        suitable for direct consumption by ``DocDetLoss``.
        """
        p2, p3, p4 = self.backbone(images)
        p2, p3, p4 = self.neck(p2, p3, p4)
        level_feats = {"p2": p2, "p3": p3, "p4": p4}

        outputs: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for level_name in LEVEL_NAMES:
            feat = level_feats[level_name]
            cls_logits, reg_raw, centerness = self.head(feat)
            reg = self._apply_reg_scale_and_exp(reg_raw, level_name)
            outputs[level_name] = (cls_logits, reg, centerness)
        return outputs

    @torch.no_grad()
    def detect(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        max_detections: int = 300,
        label_names: Optional[List[str]] = None,
    ) -> List[List[Tuple[str, float, float, float, float, float]]]:
        """
        Inference helper returning per-image detections in pixel space.

        Parameters
        ----------
        images          : (B, 3, H, W) float tensor.
        score_threshold : minimum cls * centerness score to keep.
        nms_threshold   : IoU threshold for same-class NMS.
        max_detections  : per-image cap after NMS.
        label_names     : list of length ``num_classes`` mapping class
                          IDs to strings (e.g. DOCLAYNET_LABELS).  If
                          ``None``, class IDs are returned as str(int).

        Returns
        -------
        List of length B, each a list of
        ``(label, x1, y1, x2, y2, score)`` tuples.
        """
        self.eval()
        outputs = self.forward(images)
        boxes_per_image, scores_per_image, labels_per_image = decode_predictions(
            outputs, score_threshold=score_threshold,
        )

        results: List[List[Tuple[str, float, float, float, float, float]]] = []
        for b in range(len(boxes_per_image)):
            boxes, scores, labels = class_aware_nms(
                boxes_per_image[b],
                scores_per_image[b],
                labels_per_image[b],
                iou_threshold=nms_threshold,
                max_detections=max_detections,
            )

            detections: List[Tuple[str, float, float, float, float, float]] = []
            for i in range(boxes.shape[0]):
                cls_id = int(labels[i].item())
                if label_names is not None and 0 <= cls_id < len(label_names):
                    label_name = label_names[cls_id]
                else:
                    label_name = str(cls_id)
                bx = boxes[i].tolist()
                detections.append((
                    label_name,
                    float(bx[0]), float(bx[1]),
                    float(bx[2]), float(bx[3]),
                    float(scores[i].item()),
                ))
            results.append(detections)

        return results

    def freeze_backbone_stages(self, num_stages: int = 2) -> None:
        """
        Freeze the first ``num_stages`` backbone stages.

        Used for Phase 1 synthetic pretraining: keeping the earliest
        ImageNet-derived visual primitives fixed prevents
        overfitting to synthetic texture artefacts while the neck
        and head learn document layout structure.

        Parameters
        ----------
        num_stages : 0 = nothing frozen, 1 = P2 stage frozen,
                     2 = P2 + P3 stages frozen, 3 = everything frozen.
        """
        stages = [self.backbone.stage_p2, self.backbone.stage_p3, self.backbone.stage_p4]
        for i, stage in enumerate(stages):
            if i < num_stages:
                for p in stage.parameters():
                    p.requires_grad = False
            else:
                for p in stage.parameters():
                    p.requires_grad = True

    def unfreeze_backbone(self) -> None:
        """Re-enable gradients on all backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Return total (or trainable) parameter count."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())
