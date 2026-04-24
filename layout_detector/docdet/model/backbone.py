"""
DocDet backbone: multi-scale feature extraction with 1x1 projections.

Wraps a standard ImageNet-pretrained CNN (MobileNetV3-Small by
default, EfficientNet-B0 optional) and taps three intermediate
feature maps corresponding to strides 4, 8, and 16 relative to the
input image.  Each tapped feature map is projected to a uniform
channel width (128 by default) via a 1x1 convolution to simplify
the downstream neck and head.

Design notes
------------
* Strides 4/8/16 are chosen to match spec 4.2: at 800x1120 input
  this gives 200x280 / 100x140 / 50x70 feature maps, sufficient to
  localise footnotes and narrow columns while keeping compute low.
* We deliberately do NOT tap stride 32.  Document regions are not
  "deep" semantic objects requiring very coarse features, and the
  extra level would add compute for little benefit.
* Slicing indices and per-stage channel counts are verified by
  running a dummy forward pass in ``_STAGE_SPECS`` sanity tests
  (see ``layout_detector/docdet/tests/test_backbone_shapes.py``).
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

_DEFAULT_PROJ_CHANNELS = 128


@dataclass(frozen=True)
class BackboneSpec:
    """
    Static description of a backbone family.

    Attributes
    ----------
    name          : short identifier used in configs and checkpoints
    p2_slice      : (start, end) index slice into model.features producing P2
    p3_slice      : slice producing P3
    p4_slice      : slice producing P4
    p2_channels   : raw channel count at P2 output (before projection)
    p3_channels   : raw channel count at P3
    p4_channels   : raw channel count at P4
    """

    name: str
    p2_slice: Tuple[int, int]
    p3_slice: Tuple[int, int]
    p4_slice: Tuple[int, int]
    p2_channels: int
    p3_channels: int
    p4_channels: int


# Verified by running a 1x3x800x1120 dummy input through torchvision
# models:mobilenet_v3_small.features and inspecting per-layer shapes.
_MOBILENETV3_SMALL_SPEC = BackboneSpec(
    name="mobilenetv3_small",
    p2_slice=(0, 2),    # 16ch, stride 4
    p3_slice=(2, 4),    # 24ch, stride 8
    p4_slice=(4, 9),    # 48ch, stride 16
    p2_channels=16,
    p3_channels=24,
    p4_channels=48,
)

# Verified by running a 1x3x800x1120 dummy input through torchvision
# models:efficientnet_b0.features.
_EFFICIENTNET_B0_SPEC = BackboneSpec(
    name="efficientnet_b0",
    p2_slice=(0, 3),    # 24ch, stride 4
    p3_slice=(3, 4),    # 40ch, stride 8
    p4_slice=(4, 6),    # 112ch, stride 16
    p2_channels=24,
    p3_channels=40,
    p4_channels=112,
)


class DocDetBackbone(nn.Module):
    """
    Multi-scale feature extractor for DocDet.

    Parameters
    ----------
    backbone_name : str
        One of ``"mobilenetv3_small"`` (default) or
        ``"efficientnet_b0"``.
    pretrained : bool
        If True, load ImageNet-1K pretrained weights from torchvision
        (MobileNetV3) or timm (EfficientNet).  Must be True for
        meaningful training; False is only used by unit tests.
    proj_channels : int
        Output channel count for each projection.  Downstream neck
        and head assume all three feature maps share this width.

    Attributes
    ----------
    spec : BackboneSpec
        Backbone description (used by the neck to know tap points).
    out_channels : int
        Equals ``proj_channels``; exposed for convenience.
    """

    def __init__(
        self,
        backbone_name: str = "mobilenetv3_small",
        pretrained: bool = True,
        proj_channels: int = _DEFAULT_PROJ_CHANNELS,
    ):
        super().__init__()

        self.spec = _resolve_spec(backbone_name)
        self.out_channels = proj_channels

        base = _load_base_model(backbone_name, pretrained)

        features = base.features
        p2_start, p2_end = self.spec.p2_slice
        p3_start, p3_end = self.spec.p3_slice
        p4_start, p4_end = self.spec.p4_slice

        # Each stage is a contiguous slice of the backbone's feature
        # list.  We cannot just re-run from scratch each time; the
        # stages share no state beyond the input, so we execute them
        # sequentially in forward().
        self.stage_p2 = nn.Sequential(*features[p2_start:p2_end])
        self.stage_p3 = nn.Sequential(*features[p3_start:p3_end])
        self.stage_p4 = nn.Sequential(*features[p4_start:p4_end])

        self.proj_p2 = nn.Conv2d(self.spec.p2_channels, proj_channels, kernel_size=1)
        self.proj_p3 = nn.Conv2d(self.spec.p3_channels, proj_channels, kernel_size=1)
        self.proj_p4 = nn.Conv2d(self.spec.p4_channels, proj_channels, kernel_size=1)

        self._init_projections()

    def _init_projections(self) -> None:
        """Kaiming-init the 1x1 projection convs (pretrained base is left alone)."""
        for m in (self.proj_p2, self.proj_p3, self.proj_p4):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, H, W) float tensor
            Input image batch, normalised with ImageNet stats.
            H and W should be divisible by 16 so all three feature
            map resolutions are integers.

        Returns
        -------
        p2 : (B, proj_channels, H/4,  W/4)
        p3 : (B, proj_channels, H/8,  W/8)
        p4 : (B, proj_channels, H/16, W/16)
        """
        h = self.stage_p2(x)
        p2 = self.proj_p2(h)

        h = self.stage_p3(h)
        p3 = self.proj_p3(h)

        h = self.stage_p4(h)
        p4 = self.proj_p4(h)

        return p2, p3, p4


def _resolve_spec(name: str) -> BackboneSpec:
    """Look up the static BackboneSpec for the requested backbone name."""
    name = name.lower()
    if name == "mobilenetv3_small":
        return _MOBILENETV3_SMALL_SPEC
    if name == "efficientnet_b0":
        return _EFFICIENTNET_B0_SPEC
    raise ValueError(
        f"Unknown backbone name '{name}'. "
        f"Supported: 'mobilenetv3_small', 'efficientnet_b0'."
    )


def _load_base_model(name: str, pretrained: bool) -> nn.Module:
    """
    Instantiate the torchvision/timm base classifier whose ``.features``
    attribute we will slice into three stages.  Classification heads
    are discarded; only ``model.features`` is used downstream.
    """
    name = name.lower()
    if name == "mobilenetv3_small":
        import torchvision.models as tv_models

        weights = (
            tv_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        return tv_models.mobilenet_v3_small(weights=weights)

    if name == "efficientnet_b0":
        # Torchvision also ships EfficientNet-B0 under an Apache-2.0
        # compatible license; we prefer it over timm to avoid an
        # extra dependency by default.  Users who want timm's
        # pretrained variants can swap this call.
        import torchvision.models as tv_models

        weights = (
            tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )
        return tv_models.efficientnet_b0(weights=weights)

    raise ValueError(f"Unknown backbone name '{name}'")


def build_backbone(
    backbone_name: str = "mobilenetv3_small",
    pretrained: bool = True,
    proj_channels: int = _DEFAULT_PROJ_CHANNELS,
) -> DocDetBackbone:
    """
    Factory helper preferred over directly constructing DocDetBackbone.

    Makes training configs easier to serialise (they carry a string
    name rather than a Python class), and keeps the backbone
    registry in one place.
    """
    return DocDetBackbone(
        backbone_name=backbone_name,
        pretrained=pretrained,
        proj_channels=proj_channels,
    )
