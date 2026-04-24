"""
Document-safe image + bbox augmentation pipeline.

Unlike typical detection stacks (COCO, ImageNet, YOLO) documents
have three constraints that force a different augmentation set:

1. Reading direction is fixed (left-to-right, top-to-bottom) so
   horizontal and vertical flips INVERT the semantics of the page.
2. Regions are axis-aligned and tightly packed.  Rotation beyond a
   couple of degrees either clips content or invalidates bboxes.
3. Pages are high-contrast black-ink-on-paper so heavy colour or
   contrast jitter destroys signal rather than augmenting it.

Every transform in this module respects those rules.  Each
transform is a callable ``fn(image, boxes) -> (image, boxes)`` with
boxes in pixel xyxy format relative to the returned image.
"""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageFilter

# ImageNet normalisation (used by the backbone pretrained weights).
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

BBoxArray = np.ndarray  # shape (N, 4), xyxy float


# ---------------------------------------------------------------------------
# Letterbox resize preserving aspect ratio
# ---------------------------------------------------------------------------

def letterbox(
    image: Image.Image,
    boxes: BBoxArray,
    target_size: Tuple[int, int],
    pad_color: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[Image.Image, BBoxArray, float, Tuple[int, int]]:
    """
    Resize ``image`` to fit inside ``target_size`` preserving aspect
    ratio, then pad with ``pad_color`` to reach exactly that size.

    Parameters
    ----------
    image       : PIL image (RGB).
    boxes       : (N, 4) xyxy float array in the image's original pixels.
    target_size : (H, W) output size, e.g. (1120, 800) for DocDet.
    pad_color   : RGB triple for padding (white for documents).

    Returns
    -------
    out_image : PIL image of size target_size.
    out_boxes : (N, 4) xyxy float array in the padded image's pixels.
    scale     : float applied to the original image.
    pad_xy    : (pad_left, pad_top) in pixels.  Useful for un-letterbox
                at inference time.
    """
    target_h, target_w = target_size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_w, target_h), pad_color)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas.paste(resized, (pad_x, pad_y))

    if boxes.shape[0] > 0:
        new_boxes = boxes.copy().astype(np.float32)
        new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]] * scale + pad_x
        new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]] * scale + pad_y
    else:
        new_boxes = boxes.astype(np.float32)

    return canvas, new_boxes, scale, (pad_x, pad_y)


# ---------------------------------------------------------------------------
# Individual augmentations
# ---------------------------------------------------------------------------

def random_crop_preserve_aspect(
    image: Image.Image,
    boxes: BBoxArray,
    min_area: float = 0.70,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """
    Crop a random window covering at least ``min_area`` of the input.

    Preserves aspect ratio: if the image is 800x1120, the crop will
    also be sized (800*k x 1120*k) for some k in [sqrt(min_area), 1].

    Boxes are clipped to the crop bounds and any box whose area
    drops below 1 square pixel is discarded.
    """
    rng = rng or random
    w, h = image.size
    k = rng.uniform(np.sqrt(min_area), 1.0)
    crop_w = max(int(w * k), 1)
    crop_h = max(int(h * k), 1)
    x0 = rng.randint(0, w - crop_w)
    y0 = rng.randint(0, h - crop_h)

    cropped = image.crop((x0, y0, x0 + crop_w, y0 + crop_h))

    if boxes.shape[0] == 0:
        return cropped, boxes

    new_boxes = boxes.copy().astype(np.float32)
    new_boxes[:, [0, 2]] -= x0
    new_boxes[:, [1, 3]] -= y0
    new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, crop_w)
    new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, crop_h)

    keep = (
        (new_boxes[:, 2] - new_boxes[:, 0] >= 1)
        & (new_boxes[:, 3] - new_boxes[:, 1] >= 1)
    )
    return cropped, new_boxes[keep]


def random_scale(
    image: Image.Image,
    boxes: BBoxArray,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """Resize by a random isotropic factor in ``scale_range``."""
    rng = rng or random
    s = rng.uniform(*scale_range)
    w, h = image.size
    nw, nh = max(int(w * s), 1), max(int(h * s), 1)
    resized = image.resize((nw, nh), Image.BILINEAR)
    new_boxes = boxes.copy().astype(np.float32) * s if boxes.shape[0] else boxes
    return resized, new_boxes


def random_jpeg_compression(
    image: Image.Image,
    boxes: BBoxArray,
    quality_range: Tuple[int, int] = (70, 100),
    p: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """Simulate JPEG artefacts by round-tripping through JPEG encoder."""
    rng = rng or random
    if rng.random() >= p:
        return image, boxes
    q = rng.randint(*quality_range)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    compressed = Image.open(buf).convert("RGB")
    return compressed, boxes


def random_gaussian_blur(
    image: Image.Image,
    boxes: BBoxArray,
    sigma_range: Tuple[float, float] = (0.0, 0.5),
    p: float = 0.3,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """Apply a Gaussian blur with sigma sampled uniformly."""
    rng = rng or random
    if rng.random() >= p:
        return image, boxes
    sigma = rng.uniform(*sigma_range)
    if sigma <= 0.01:
        return image, boxes
    return image.filter(ImageFilter.GaussianBlur(radius=sigma)), boxes


def random_brightness_contrast(
    image: Image.Image,
    boxes: BBoxArray,
    brightness_range: Tuple[float, float] = (0.9, 1.1),
    contrast_range: Tuple[float, float] = (0.9, 1.1),
    p: float = 0.5,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """
    Jitter brightness and contrast within conservative bounds.

    Uses PIL.ImageEnhance to keep the code free of torchvision
    dependencies for this module (useful for smoke tests in
    environments without the full torchvision stack).
    """
    rng = rng or random
    if rng.random() >= p:
        return image, boxes
    from PIL import ImageEnhance

    b = rng.uniform(*brightness_range)
    c = rng.uniform(*contrast_range)
    image = ImageEnhance.Brightness(image).enhance(b)
    image = ImageEnhance.Contrast(image).enhance(c)
    return image, boxes


def random_grayscale(
    image: Image.Image,
    boxes: BBoxArray,
    p: float = 0.1,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """Convert to grayscale (stored as 3-channel RGB) with prob ``p``."""
    rng = rng or random
    if rng.random() >= p:
        return image, boxes
    gs = image.convert("L").convert("RGB")
    return gs, boxes


def random_small_rotation(
    image: Image.Image,
    boxes: BBoxArray,
    max_degrees: float = 2.0,
    p: float = 0.2,
    rng: Optional[random.Random] = None,
) -> Tuple[Image.Image, BBoxArray]:
    """
    Rotate by a tiny angle (+-max_degrees degrees).

    Bboxes are re-computed as the axis-aligned bounding box of the
    rotated corners.  This introduces a small amount of bbox slack
    around rotated regions, which is acceptable for documents.
    """
    rng = rng or random
    if rng.random() >= p or max_degrees <= 0:
        return image, boxes
    if max_degrees > 2.0:
        # Hard cap per spec 7 - documents are axis-aligned; rotation
        # beyond 2 degrees destroys layout signal.
        max_degrees = 2.0

    angle = rng.uniform(-max_degrees, max_degrees)
    w, h = image.size
    rotated = image.rotate(angle, resample=Image.BILINEAR, fillcolor=(255, 255, 255))

    if boxes.shape[0] == 0:
        return rotated, boxes

    cx, cy = w / 2.0, h / 2.0
    rad = np.deg2rad(-angle)  # PIL rotates CCW for +angle, bbox math wants CW
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack(
        [
            np.stack([x1, y1], axis=1),
            np.stack([x2, y1], axis=1),
            np.stack([x2, y2], axis=1),
            np.stack([x1, y2], axis=1),
        ],
        axis=1,
    )  # (N, 4, 2)

    shifted = corners - np.array([cx, cy])
    rotated_corners = np.zeros_like(shifted)
    rotated_corners[..., 0] = shifted[..., 0] * cos_a - shifted[..., 1] * sin_a
    rotated_corners[..., 1] = shifted[..., 0] * sin_a + shifted[..., 1] * cos_a
    rotated_corners += np.array([cx, cy])

    new_boxes = np.stack(
        [
            rotated_corners[..., 0].min(axis=1),
            rotated_corners[..., 1].min(axis=1),
            rotated_corners[..., 0].max(axis=1),
            rotated_corners[..., 1].max(axis=1),
        ],
        axis=1,
    )
    new_boxes[:, [0, 2]] = np.clip(new_boxes[:, [0, 2]], 0, w)
    new_boxes[:, [1, 3]] = np.clip(new_boxes[:, [1, 3]], 0, h)

    keep = (new_boxes[:, 2] - new_boxes[:, 0] >= 1) & (
        new_boxes[:, 3] - new_boxes[:, 1] >= 1
    )
    return rotated, new_boxes[keep].astype(np.float32)


# ---------------------------------------------------------------------------
# Final to-tensor conversion
# ---------------------------------------------------------------------------

def to_tensor_and_normalize(
    image: Image.Image,
    boxes: BBoxArray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert PIL RGB to a normalised float tensor (C, H, W) in [0, 1]
    minus ImageNet mean and divided by ImageNet std.
    """
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - np.array(_IMAGENET_MEAN)) / np.array(_IMAGENET_STD)
    tensor = torch.from_numpy(array.transpose(2, 0, 1)).float()
    box_tensor = torch.from_numpy(boxes.astype(np.float32)).float()
    return tensor, box_tensor


# ---------------------------------------------------------------------------
# Explicitly-disabled operations (documented so they are not
# accidentally re-added; see spec 7)
# ---------------------------------------------------------------------------

_DISABLED_AUGMENTATIONS = (
    "horizontal flip",
    "vertical flip",
    "mosaic (4-image or 9-image grids)",
    "rotation > +-2 degrees",
    "heavy colour jitter (saturation, hue)",
    "random erasing / cutout",
)


# ---------------------------------------------------------------------------
# Composable pipeline
# ---------------------------------------------------------------------------

@dataclass
class TrainAugmentConfig:
    """
    Dataclass aggregating all training augmentation hyperparameters.

    Override individual fields to tune phase-specific augmentation
    strength (Phase 1 uses conservative defaults below; Phase 2 can
    enable ``mixup_alpha`` via the trainer).
    """

    random_crop_min_area: float = 0.70
    random_scale_range: Tuple[float, float] = (0.8, 1.2)
    jpeg_quality_range: Tuple[int, int] = (70, 100)
    jpeg_prob: float = 0.5
    blur_sigma_range: Tuple[float, float] = (0.0, 0.5)
    blur_prob: float = 0.3
    brightness_range: Tuple[float, float] = (0.9, 1.1)
    contrast_range: Tuple[float, float] = (0.9, 1.1)
    color_prob: float = 0.5
    grayscale_prob: float = 0.1
    rotation_max_degrees: float = 2.0
    rotation_prob: float = 0.2


class DocDetTrainTransform:
    """
    Full train-time pipeline: augment -> letterbox -> tensor.

    Call with ``image`` (PIL), ``boxes`` (N, 4 numpy xyxy) and get
    back ``(tensor, box_tensor, meta)`` where ``meta`` carries
    letterbox parameters for un-padding at eval/inference.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (1120, 800),
        config: Optional[TrainAugmentConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        self.target_size = target_size
        self.config = config or TrainAugmentConfig()
        self.rng = rng or random

        self._ops: List[Callable[[Image.Image, BBoxArray], Tuple[Image.Image, BBoxArray]]] = [
            lambda img, bx: random_crop_preserve_aspect(
                img, bx, self.config.random_crop_min_area, self.rng
            ),
            lambda img, bx: random_scale(
                img, bx, self.config.random_scale_range, self.rng
            ),
            lambda img, bx: random_small_rotation(
                img, bx,
                self.config.rotation_max_degrees,
                self.config.rotation_prob,
                self.rng,
            ),
            lambda img, bx: random_jpeg_compression(
                img, bx,
                self.config.jpeg_quality_range,
                self.config.jpeg_prob,
                self.rng,
            ),
            lambda img, bx: random_gaussian_blur(
                img, bx,
                self.config.blur_sigma_range,
                self.config.blur_prob,
                self.rng,
            ),
            lambda img, bx: random_brightness_contrast(
                img, bx,
                self.config.brightness_range,
                self.config.contrast_range,
                self.config.color_prob,
                self.rng,
            ),
            lambda img, bx: random_grayscale(
                img, bx, self.config.grayscale_prob, self.rng
            ),
        ]

    def __call__(
        self,
        image: Image.Image,
        boxes: BBoxArray,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Parameters
        ----------
        image : PIL image in RGB.
        boxes : (N, 4) xyxy numpy array in pixels.

        Returns
        -------
        tensor   : (3, H, W) normalised float tensor.
        box_out  : (M, 4) float tensor in letterboxed pixel space.
        meta     : dict with keys 'scale', 'pad_xy' for un-letterbox.
        """
        img = image
        bx = boxes.astype(np.float32)

        for op in self._ops:
            img, bx = op(img, bx)

        img, bx, scale, pad_xy = letterbox(img, bx, self.target_size)
        tensor, box_tensor = to_tensor_and_normalize(img, bx)
        meta = {"scale": scale, "pad_xy": pad_xy, "letterbox_size": self.target_size}
        return tensor, box_tensor, meta


class DocDetEvalTransform:
    """Eval-time pipeline: only letterbox + to_tensor (no stochastic ops)."""

    def __init__(self, target_size: Tuple[int, int] = (1120, 800)):
        self.target_size = target_size

    def __call__(
        self,
        image: Image.Image,
        boxes: BBoxArray,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        img, bx, scale, pad_xy = letterbox(image, boxes.astype(np.float32), self.target_size)
        tensor, box_tensor = to_tensor_and_normalize(img, bx)
        meta = {"scale": scale, "pad_xy": pad_xy, "letterbox_size": self.target_size}
        return tensor, box_tensor, meta
