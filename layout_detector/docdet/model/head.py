"""
DocDet FCOS detection head.

Applies the same two prediction towers across every FPN level:

    feature map (B, C, H, W)
        |
        +--- cls_tower (4x Conv+GN+ReLU) ---> cls_logits (B, num_classes, H, W)
        |
        +--- reg_tower (4x Conv+GN+ReLU) ---> bbox_pred   (B, 4, H, W)  LTRB distances
        |                                 \\-> centerness  (B, 1, H, W)

Regression output is raw linear; the enclosing DocDet module
multiplies by a per-level ``Scale`` scalar, applies ``exp`` (so
distances stay positive), and finally multiplies by the level stride
to get pixel-space distances.  We deliberately keep the exp/scale
application outside this module so the head is compact, stateless
across pyramid levels, and easy to export to ONNX.

Classification output is raw logits (apply ``sigmoid`` downstream for
multi-label style focal loss).  Cls bias is initialised using a focal
loss prior so that, at the start of training, every location has an
object probability of about 1%, which stabilises training.
"""

import math
from typing import List, Tuple

import torch
import torch.nn as nn


# Focal-loss prior: sigmoid(bias) = 0.01 at initialisation.
_FOCAL_INIT_BIAS = -math.log((1 - 0.01) / 0.01)


class FCOSHead(nn.Module):
    """
    Shared-weights FCOS head applied to every pyramid level.

    Parameters
    ----------
    num_classes : int
        Number of foreground classes.
    in_channels : int
        Feature map channel width (must match neck output, default 128).
    num_convs : int
        Depth of each tower (cls and reg).  Spec says 4.
    num_groups : int
        GroupNorm group count.  Uses the largest power-of-two that
        divides ``in_channels`` up to 32.

    Notes
    -----
    * GroupNorm (not BatchNorm) is used because FCOS heads are shared
      across levels; using BN would require per-level statistics and
      make export brittle.
    * The ``centerness`` predictor sits on top of the reg tower (per
      the original FCOS paper) because centerness correlates with
      regression quality, not classification.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 128,
        num_convs: int = 4,
        num_groups: int = 32,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # GroupNorm requires num_groups to divide num_channels.
        effective_groups = max(
            g for g in (num_groups, 16, 8, 4, 2, 1) if in_channels % g == 0
        )

        self.cls_tower = self._make_tower(
            in_channels, num_convs, effective_groups,
        )
        self.reg_tower = self._make_tower(
            in_channels, num_convs, effective_groups,
        )

        self.cls_pred = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1)
        self.centerness_pred = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self._init_weights()

    @staticmethod
    def _make_tower(channels: int, depth: int, groups: int) -> nn.Sequential:
        """Stack ``depth`` Conv+GroupNorm+ReLU triplets."""
        layers: List[nn.Module] = []
        for _ in range(depth):
            layers.append(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
            )
            layers.append(nn.GroupNorm(groups, channels))
            layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """Init all trainable weights (focal-loss prior on cls bias)."""
        for module in (self.cls_tower, self.reg_tower):
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        nn.init.normal_(self.cls_pred.weight, std=0.01)
        nn.init.constant_(self.cls_pred.bias, _FOCAL_INIT_BIAS)

        nn.init.normal_(self.reg_pred.weight, std=0.01)
        nn.init.zeros_(self.reg_pred.bias)

        nn.init.normal_(self.centerness_pred.weight, std=0.01)
        nn.init.zeros_(self.centerness_pred.bias)

    def forward(
        self, feature: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        feature : (B, C, H, W) single FPN level feature map.

        Returns
        -------
        cls_logits : (B, num_classes, H, W) raw logits (no sigmoid).
        reg_raw    : (B, 4, H, W) raw linear LTRB regression values.
                     The caller is expected to multiply by a
                     per-level Scale and apply ``exp`` before
                     multiplying by the level stride.
        centerness : (B, 1, H, W) raw centerness logits.
        """
        cls_feat = self.cls_tower(feature)
        reg_feat = self.reg_tower(feature)

        cls_logits = self.cls_pred(cls_feat)
        reg_raw = self.reg_pred(reg_feat)
        centerness = self.centerness_pred(reg_feat)

        return cls_logits, reg_raw, centerness
