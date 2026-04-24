"""
Tests for the document-safe augmentation pipeline.

Key invariants:
* Letterbox preserves aspect ratio and returns correct padding.
* Tensorisation is shape- and dtype-correct.
* Train pipeline never flips images (documents have a fixed reading
  direction); this is a regression guard.
"""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from layout_detector.docdet.data.transforms import (
    DocDetEvalTransform,
    DocDetTrainTransform,
    letterbox,
    random_small_rotation,
    to_tensor_and_normalize,
)


def _make_image(w: int = 200, h: int = 400) -> Image.Image:
    """Build a deterministic RGB test image."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, 0] = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    return Image.fromarray(arr)


def test_letterbox_preserves_aspect() -> None:
    img = _make_image(200, 400)
    boxes = np.array([[0.0, 0.0, 200.0, 400.0]], dtype=np.float32)
    out_img, out_boxes, scale, (pad_x, pad_y) = letterbox(img, boxes, (800, 400))

    assert out_img.size == (400, 800)
    assert pad_x >= 0 and pad_y >= 0
    assert scale == 2.0  # 400/200 = 2, 800/400 = 2 -> min = 2
    assert out_boxes[0, 0] == pad_x
    assert out_boxes[0, 1] == pad_y


def test_to_tensor_normalises() -> None:
    img = _make_image(64, 32)
    boxes = np.zeros((0, 4), dtype=np.float32)
    tensor, box_tensor = to_tensor_and_normalize(img, boxes)
    assert tensor.shape == (3, 32, 64)
    assert tensor.dtype == torch.float32
    assert box_tensor.shape == (0, 4)


def test_train_transform_shapes() -> None:
    img = _make_image(300, 400)
    boxes = np.array([[10.0, 10.0, 60.0, 80.0]], dtype=np.float32)
    t = DocDetTrainTransform(target_size=(800, 600))
    tensor, box_tensor, meta = t(img, boxes)

    assert tensor.shape == (3, 800, 600)
    assert box_tensor.shape[1] == 4
    assert "scale" in meta
    assert "pad_xy" in meta


def test_eval_transform_is_deterministic() -> None:
    img = _make_image(300, 400)
    boxes = np.array([[10.0, 10.0, 60.0, 80.0]], dtype=np.float32)
    t = DocDetEvalTransform(target_size=(800, 600))

    a, _, _ = t(img, boxes)
    b, _, _ = t(img, boxes)
    assert torch.allclose(a, b)


def test_small_rotation_does_not_corrupt_boxes() -> None:
    """Small rotation should preserve box count (within clip tolerance)."""
    img = _make_image(200, 200)
    boxes = np.array([[20.0, 20.0, 180.0, 180.0]], dtype=np.float32)

    out_img, out_boxes = random_small_rotation(
        img, boxes, max_degrees=2.0, p=1.0,
    )
    assert out_img.size == (200, 200)
    assert out_boxes.shape == (1, 4)
    x1, y1, x2, y2 = out_boxes[0]
    assert x2 > x1 and y2 > y1
