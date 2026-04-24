"""
DocDet neck: PANet fusion across P2, P3, P4 feature maps.

Performs a top-down pass followed by a bottom-up pass:

    P4 ---------------------------\\
                                   + --> P4_out
    P3 ------------+ --> P3_td -- + --> P3_out
                    \\                \\
    P2 --- + -- P2_td/P2_out          |
           |                          |
    (flows: up-sample P4 into P3, then P3_td into P2_td;
            then stride-2 conv P2 into P3, P3 into P4)

Top-down pass propagates high-level semantic information to finer
resolutions.  Bottom-up pass injects precise localisation cues from
the highest-resolution map back into the deeper levels.  Following
CSPBlock smoothing removes upsampling artefacts.

All channel widths are kept uniform (``channels`` parameter, default
128) so the detection head can share weights across levels.
"""

from typing import Tuple

import torch
import torch.nn as nn

from .blocks import CSPBlock, ConvBNAct


class PANetNeck(nn.Module):
    """
    Three-level PANet neck feeding the FCOS head.

    Parameters
    ----------
    channels : int
        Channel width used throughout the neck.  Must match the
        backbone's ``proj_channels``.
    csp_depth : int
        Number of Bottleneck blocks inside each CSPBlock stage.
        Default 1 is intentionally conservative for edge deployment;
        increase to 2 for slightly more capacity on the larger
        backbones (EfficientNet-B0).

    Input / output
    --------------
    Inputs are the three projected backbone feature maps p2/p3/p4 at
    strides 4/8/16 with the same channel count.  Outputs three fused
    maps with unchanged resolutions and channel widths.
    """

    def __init__(self, channels: int = 128, csp_depth: int = 1):
        super().__init__()

        self.td_p3 = CSPBlock(channels, channels, n=csp_depth, shortcut=True)
        self.td_p2 = CSPBlock(channels, channels, n=csp_depth, shortcut=True)

        self.bu_p3 = CSPBlock(channels, channels, n=csp_depth, shortcut=True)
        self.bu_p4 = CSPBlock(channels, channels, n=csp_depth, shortcut=True)

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        self.down_p2_to_p3 = ConvBNAct(channels, channels, kernel_size=3, stride=2)
        self.down_p3_to_p4 = ConvBNAct(channels, channels, kernel_size=3, stride=2)

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        p2, p3, p4 : projected backbone feature maps at strides 4/8/16
            all with the same ``channels`` width.

        Returns
        -------
        p2_out, p3_out, p4_out : fused feature maps for the detection head
            (same resolutions and channel widths as inputs).
        """
        p3_td = self.td_p3(p3 + self.upsample(p4))
        p2_td = self.td_p2(p2 + self.upsample(p3_td))

        p3_bu = self.bu_p3(p3_td + self.down_p2_to_p3(p2_td))
        p4_bu = self.bu_p4(p4 + self.down_p3_to_p4(p3_bu))

        return p2_td, p3_bu, p4_bu
