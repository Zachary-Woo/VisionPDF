"""
Visual reading order head using SAM encoder features.

Unlike LayoutReader which predicts reading order from bounding box
positions alone, this head uses the actual visual features pooled from
the SAM encoder at each detected region.  This is closer to how
DeepSeek OCR 2's full model works: the VLM "reads" the page using
visual context, not just spatial coordinates.

Architecture:
  1. RoI Align on P3 features (64x64x256) to extract a 7x7x256 patch
     per detected region.
  2. Flatten + linear projection to 256-dim per region.
  3. Add 2D positional encoding from normalised box coordinates.
  4. 2-layer transformer encoder (256-dim, 4 heads) for cross-region
     attention -- each region attends to all others.
  5. Linear head produces a scalar ordering score per region.
  6. argsort of scores gives the predicted reading order.
"""

from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torchvision.ops import roi_align

from benchmark.tier2_hybrid.shared import Region


class ReadingOrderHead(nn.Module):
    """
    Predicts reading order from pooled visual features + box positions.

    Parameters
    ----------
    feat_channels : int
        Channel dimension of the P3 feature map (default 256).
    hidden_dim : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    roi_size : int
        RoI Align output spatial size (roi_size x roi_size).
    """

    def __init__(
        self,
        feat_channels: int = 256,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        roi_size: int = 7,
    ):
        super().__init__()
        self.roi_size = roi_size

        roi_flat_dim = feat_channels * roi_size * roi_size
        self.roi_proj = nn.Sequential(
            nn.Linear(roi_flat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.pos_enc = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
        )

        self.order_head = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        p3_features: torch.Tensor,
        boxes_pixel: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        p3_features : (1, C, H, W) P3 feature map from SAM encoder
        boxes_pixel : (N, 4) boxes in the 1024x1024 preprocessed image space (xyxy)
        img_h, img_w : int, preprocessed image dimensions (1024)

        Returns
        -------
        scores : (N,) ordering scores (lower = earlier in reading order)
        """
        N = boxes_pixel.shape[0]
        if N == 0:
            return torch.zeros(0, device=p3_features.device)

        batch_idx = torch.zeros(N, 1, device=boxes_pixel.device)
        rois = torch.cat([batch_idx, boxes_pixel], dim=1)

        pooled = roi_align(
            p3_features, rois,
            output_size=self.roi_size,
            spatial_scale=1.0 / 16.0,
            aligned=True,
        )

        roi_flat = pooled.flatten(1)
        roi_emb = self.roi_proj(roi_flat)

        norm_boxes = boxes_pixel.clone()
        norm_boxes[:, [0, 2]] /= max(img_w, 1)
        norm_boxes[:, [1, 3]] /= max(img_h, 1)
        pos_emb = self.pos_enc(norm_boxes)

        tokens = roi_emb + pos_emb
        tokens = tokens.unsqueeze(0)

        out = self.transformer(tokens)

        scores = self.order_head(out.squeeze(0)).squeeze(-1)
        return scores


def predict_visual_order(
    model: ReadingOrderHead,
    regions: List[Region],
    p3_features: torch.Tensor,
    img_scale: float,
    render_scale: float,
) -> List[Region]:
    """
    Reorder regions using the visual reading order head.

    Parameters
    ----------
    model : ReadingOrderHead
    regions : list of Region (in PDF-point coords)
    p3_features : (1, 256, H, W) from SAM encoder
    img_scale : float
        Scale factor from rendered image to 1024 preprocessed image
    render_scale : float
        Scale factor from PDF points to rendered image pixels
    """
    if len(regions) <= 1:
        return regions

    boxes = []
    for r in regions:
        rx1 = r.x1 * render_scale
        ry1 = r.y1 * render_scale
        rx2 = r.x2 * render_scale
        ry2 = r.y2 * render_scale
        boxes.append([
            rx1 * img_scale,
            ry1 * img_scale,
            rx2 * img_scale,
            ry2 * img_scale,
        ])

    boxes_tensor = torch.tensor(boxes, device=p3_features.device, dtype=torch.float32)

    with torch.no_grad():
        scores = model(p3_features, boxes_tensor, 1024, 1024)

    order = scores.argsort().tolist()
    return [regions[i] for i in order]


def load_order_head(ckpt_path: Path, device: torch.device) -> ReadingOrderHead:
    """Load a trained ReadingOrderHead from a checkpoint file."""
    model = ReadingOrderHead()
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    return model
