"""
Minimal SAM ViT-B encoder reimplementation for DeepSeek OCR 2.

Mirrors the architecture in DeepSeek OCR 2's sam_vary_sdpa.py but avoids
the flash_attn import which fails on many setups.  Includes weight loading
from HuggingFace checkpoints and image preprocessing.

Architecture:
  SAM ViT-B  -->  neck 768->256  -->  stride-2 conv 256->512  -->  stride-2 conv 512->896
  Input 1024x1024  =>  feature map 16x16x896  (each cell ~ 64x64 px)
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class _MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x):
        return self.lin2(self.act(self.lin1(x)))


def _get_abs_pos(abs_pos, tgt_size):
    src_size = abs_pos.size(1)
    if src_size != tgt_size:
        old = abs_pos.permute(0, 3, 1, 2).to(torch.float32)
        new = F.interpolate(old, size=(tgt_size, tgt_size), mode="bicubic",
                            antialias=True, align_corners=False).to(abs_pos.dtype)
        return new.permute(0, 2, 3, 1)
    return abs_pos


def _get_rel_pos(q_size, k_size, rel_pos):
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rp = rel_pos.to(torch.float32)
        rp = F.interpolate(
            rp.reshape(1, rp.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        ).to(rel_pos.dtype)
        rel_pos = rp.reshape(-1, max_rel_dist).permute(1, 0)
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos[relative_coords.long()]


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=12, qkv_bias=True,
                 use_rel_pos=False, rel_pos_zero_init=True, input_size=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            head_dim = dim // num_heads
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, head_dim))

    def forward(self, x):
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)
        attn_bias = None
        if self.use_rel_pos:
            Rh = _get_rel_pos(H, H, self.rel_pos_h)
            Rw = _get_rel_pos(W, W, self.rel_pos_w)
            r_q = q.view(B * self.num_heads, H, W, -1)
            rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh).unsqueeze(-1)
            rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw).unsqueeze(-2)
            attn_bias = (rel_h + rel_w).reshape(B * self.num_heads, H * W, H * W)
            q = q.view(B, self.num_heads, H * W, -1)
            k = k.view(B, self.num_heads, H * W, -1)
            v = v.view(B, self.num_heads, H * W, -1)
            attn_bias = attn_bias.view(B, self.num_heads, H * W, H * W)
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q.view(B, self.num_heads, H * W, -1)
            k = k.view(B, self.num_heads, H * W, -1)
            v = v.view(B, self.num_heads, H * W, -1)
            x = F.scaled_dot_product_attention(q, k, v)
        x = x.view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


def _window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C), (Hp, Wp)


def _window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 use_rel_pos=False, rel_pos_zero_init=True,
                 window_size=0, input_size=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos, rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLPBlock(dim, int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = _window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = _window_unpartition(x, self.window_size, pad_hw, (H, W))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class _PatchEmbed(nn.Module):
    def __init__(self, kernel_size=(16, 16), stride=(16, 16),
                 in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.proj(x).permute(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# SAM ViT-B encoder
# ---------------------------------------------------------------------------

class SAMEncoder(nn.Module):
    """
    Minimal SAM ViT-B encoder that mirrors the architecture in DeepSeek
    OCR 2's sam_vary_sdpa.py but avoids the flash_attn import.
    """

    def __init__(self):
        super().__init__()
        embed_dim = 768
        depth = 12
        num_heads = 12
        img_size = 1024
        patch_size = 16
        out_chans = 256
        global_attn_indexes = (2, 5, 8, 11)

        self.patch_embed = _PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=3, embed_dim=embed_dim,
        )
        feat_size = img_size // patch_size
        self.pos_embed = nn.Parameter(
            torch.zeros(1, feat_size, feat_size, embed_dim)
        )
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=4.0,
                qkv_bias=True, use_rel_pos=True, rel_pos_zero_init=True,
                window_size=14 if i not in global_attn_indexes else 0,
                input_size=(feat_size, feat_size),
            ))
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            _LayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            _LayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 896, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, 1024, 1024) or (B, 3, 768, 768) tensor

        Returns
        -------
        features : (B, 896, H, W) spatial feature map
        """
        x = self.patch_embed(x)
        x = x + _get_abs_pos(self.pos_embed, x.size(1))
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        x = self.net_2(x)
        x = self.net_3(x)
        return x


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_sam_encoder(model_path: str, device: torch.device) -> SAMEncoder:
    """
    Load SAM encoder weights from the full DeepSeek OCR 2 checkpoint.

    HuggingFace checkpoints store vision weights under keys like
    ``model.sam_model.<key>``.  We strip that prefix and load into
    our minimal reimplementation.
    """
    from huggingface_hub import snapshot_download

    if Path(model_path).is_dir():
        model_dir = Path(model_path)
    else:
        print(f"Downloading model files from {model_path} (vision weights only)...")
        model_dir = Path(snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json", "*.bin"],
        ))

    state_dict: Dict[str, torch.Tensor] = {}
    safetensor_files = list(model_dir.glob("*.safetensors"))

    if safetensor_files:
        from safetensors.torch import load_file
        for sf in safetensor_files:
            sd = load_file(str(sf), device="cpu")
            for k, v in sd.items():
                if "sam_model" in k:
                    state_dict[k] = v
    else:
        bin_files = list(model_dir.glob("*.bin"))
        for bf in bin_files:
            sd = torch.load(str(bf), map_location="cpu")
            for k, v in sd.items():
                if "sam_model" in k:
                    state_dict[k] = v

    cleaned: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in ("model.sam_model.", "sam_model."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
                break
        cleaned[new_key] = v

    encoder = SAMEncoder()
    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Warning: missing keys in SAM encoder: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Warning: unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    encoder = encoder.to(device=device, dtype=torch.float16).eval()
    return encoder


# ---------------------------------------------------------------------------
# Image preprocessing (matches DeepSeek's approach)
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image, target_size: int = 1024) -> Tuple[torch.Tensor, float, int, int]:
    """
    Resize and pad the image to target_size x target_size, then normalise
    to the range expected by the SAM encoder (ImageNet mean/std).
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    canvas.paste(image, (0, 0))

    arr = np.array(canvas, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, scale, new_w, new_h
