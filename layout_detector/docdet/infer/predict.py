"""
High-level inference wrapper for DocDet.

Provides a single ``DocDetPredictor`` that accepts a PIL image, a
filesystem path, or a numpy array, and returns a list of
``LayoutDetection`` dataclass instances in the source image's
original pixel coordinates.

The class hides three concerns:

1. Whether the weights came from a PyTorch checkpoint or an ONNX
   file (runtime is selected automatically from the extension).
2. Preprocessing (letterbox + ImageNet normalisation) so callers
   never manually build tensors.
3. Post-processing: class-aware NMS, score filtering, and optional
   dilate-tables +2px for downstream TableFormer crops.

The predictor is stateless beyond the loaded weights, so it is safe
to share across threads as long as a single ``__call__`` is not
interleaved.  (We do not hold any streaming state.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image

from ..data.label_map import DOCDET_CLASS_NAMES, NUM_DOCDET_CLASSES, canonical_id
from ..data.transforms import DocDetEvalTransform

logger = logging.getLogger(__name__)

ImageInput = Union[str, Path, Image.Image, np.ndarray]


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class LayoutDetection:
    """
    One detected region.

    Attributes
    ----------
    label    : canonical DocDet class name (matches DOCLAYNET_LABELS).
    class_id : integer class index (0..10).
    score    : combined cls * centerness score in [0, 1].
    bbox     : (x1, y1, x2, y2) in pixels of the ORIGINAL input image
               (NOT the letterbox resized image).
    """
    label: str
    class_id: int
    score: float
    bbox: Tuple[float, float, float, float]


# ---------------------------------------------------------------------------
# Runtime detection (PyTorch vs ONNX Runtime)
# ---------------------------------------------------------------------------

def _is_onnx(path: Path) -> bool:
    """Return True if ``path`` looks like an ONNX model."""
    return path.suffix.lower() == ".onnx"


# ---------------------------------------------------------------------------
# Main predictor
# ---------------------------------------------------------------------------

class DocDetPredictor:
    """
    Loadable, callable predictor for DocDet layout detection.

    Parameters
    ----------
    model_path        : .pt / .pth / .onnx file with trained weights.
    device            : "cpu", "cuda", or "auto" (PyTorch path only;
                        ONNX uses its own provider selection).
    input_size        : (H, W) to letterbox every image to.  Must
                        match the size the model was trained at
                        (default 1120x800 per spec 4).
    score_threshold   : post-NMS confidence threshold.
    nms_threshold     : class-aware NMS IoU threshold.
    dilate_tables_px  : pixels to grow Table bboxes on every side
                        (spec 8).  Set to 0 to disable.
    onnx_providers    : list of ONNX Runtime provider names.  None
                        lets ORT pick the default order.
    backbone_name     : hint passed to the PyTorch model constructor
                        when the checkpoint's metadata does not
                        record a backbone.  Ignored for ONNX.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        input_size: Tuple[int, int] = (1120, 800),
        score_threshold: float = 0.3,
        nms_threshold: float = 0.5,
        dilate_tables_px: int = 2,
        max_detections: int = 300,
        onnx_providers: Optional[Sequence[str]] = None,
        backbone_name: str = "mobilenetv3_small",
    ):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.dilate_tables_px = dilate_tables_px
        self.max_detections = max_detections
        self.transform = DocDetEvalTransform(target_size=input_size)

        self._is_onnx = _is_onnx(self.model_path)
        if self._is_onnx:
            self._load_onnx(onnx_providers)
        else:
            self._load_torch(device=device, backbone_name=backbone_name)

    # ------------------------------------------------------------------
    # Torch backend
    # ------------------------------------------------------------------

    def _load_torch(self, device: str, backbone_name: str) -> None:
        import torch

        from ..model.model import DocDet

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        payload = torch.load(self.model_path, map_location="cpu")
        backbone = payload.get("backbone_name", backbone_name)

        self._torch = torch
        self._torch_model = DocDet(
            num_classes=NUM_DOCDET_CLASSES,
            backbone_name=backbone,
            pretrained=False,
        )
        state = payload.get("model", payload)
        missing, unexpected = self._torch_model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("DocDetPredictor: missing keys in state_dict: %d", len(missing))
        if unexpected:
            logger.warning("DocDetPredictor: unexpected keys in state_dict: %d", len(unexpected))
        self._torch_model.eval().to(self.device)

    # ------------------------------------------------------------------
    # ONNX backend
    # ------------------------------------------------------------------

    def _load_onnx(self, providers: Optional[Sequence[str]]) -> None:
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required to load .onnx models. "
                "Install via `pip install onnxruntime`."
            ) from e

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        provider_list = (
            list(providers) if providers is not None else ort.get_available_providers()
        )
        self._ort_session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=provider_list,
        )
        self._ort_input_name = self._ort_session.get_inputs()[0].name
        self._ort_output_names = [o.name for o in self._ort_session.get_outputs()]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, image: ImageInput) -> List[LayoutDetection]:
        """Run detection on a single image and return detections."""
        return self.predict(image)

    def predict(self, image: ImageInput) -> List[LayoutDetection]:
        """
        Run detection on a single image.

        Parameters
        ----------
        image : PIL.Image, numpy HWC uint8 RGB array, or filesystem path.

        Returns
        -------
        List of LayoutDetection objects in descending score order.
        """
        pil = self._to_pil(image)
        orig_w, orig_h = pil.size

        tensor, _boxes_unused, meta = self.transform(
            pil, np.zeros((0, 4), dtype=np.float32),
        )
        batch = tensor.unsqueeze(0)

        if self._is_onnx:
            boxes, scores, classes = self._infer_onnx(batch)
        else:
            boxes, scores, classes = self._infer_torch(batch)

        boxes = self._unletterbox(boxes, meta, orig_h, orig_w)
        if self.dilate_tables_px > 0:
            boxes = self._dilate_tables(boxes, classes, orig_h, orig_w)

        detections: List[LayoutDetection] = []
        for box, score, cls_id in zip(boxes, scores, classes):
            cls_id = int(cls_id)
            if not (0 <= cls_id < NUM_DOCDET_CLASSES):
                continue
            detections.append(LayoutDetection(
                label=DOCDET_CLASS_NAMES[cls_id],
                class_id=cls_id,
                score=float(score),
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
            ))
        return detections

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil(image: ImageInput) -> Image.Image:
        """Accept a path, numpy array, or PIL image and return PIL RGB."""
        if isinstance(image, (str, Path)):
            return Image.open(image).convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.dtype != np.uint8:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            return Image.fromarray(image).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError(f"Unsupported image type: {type(image).__name__}")

    @staticmethod
    def _unletterbox(
        boxes: np.ndarray,
        meta: dict,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """Reverse letterbox scale+pad to map boxes back to original pixels."""
        if boxes.size == 0:
            return boxes
        scale = float(meta["scale"])
        pad_x, pad_y = meta["pad_xy"]
        out = boxes.copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / scale
        out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / scale
        out[:, 0] = np.clip(out[:, 0], 0, orig_w)
        out[:, 1] = np.clip(out[:, 1], 0, orig_h)
        out[:, 2] = np.clip(out[:, 2], 0, orig_w)
        out[:, 3] = np.clip(out[:, 3], 0, orig_h)
        return out

    def _dilate_tables(
        self,
        boxes: np.ndarray,
        classes: np.ndarray,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """Apply the table-bbox dilation in numpy space."""
        if boxes.size == 0:
            return boxes
        table_id = canonical_id("Table")
        mask = classes == table_id
        if not mask.any():
            return boxes
        out = boxes.copy()
        out[mask, 0] = np.clip(out[mask, 0] - self.dilate_tables_px, 0, orig_w)
        out[mask, 1] = np.clip(out[mask, 1] - self.dilate_tables_px, 0, orig_h)
        out[mask, 2] = np.clip(out[mask, 2] + self.dilate_tables_px, 0, orig_w)
        out[mask, 3] = np.clip(out[mask, 3] + self.dilate_tables_px, 0, orig_h)
        return out

    def _infer_torch(
        self, batch_tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the PyTorch model and return numpy arrays."""
        self._torch_model.eval()
        with self._torch.no_grad():
            result = self._torch_model.detect(
                batch_tensor.to(self.device),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                max_detections=self.max_detections,
                label_names=None,
            )
        detections = result[0]
        if not detections:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        boxes = np.array([[d[1], d[2], d[3], d[4]] for d in detections], dtype=np.float32)
        scores = np.array([d[5] for d in detections], dtype=np.float32)
        classes = np.array([int(d[0]) for d in detections], dtype=np.int64)
        return boxes, scores, classes

    def _infer_onnx(
        self, batch_tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the ONNX model and apply decode + NMS in numpy/torch."""
        import torch

        from ..model.postprocess import class_aware_nms, decode_predictions
        from ..model.loss import LEVEL_NAMES

        array_input = batch_tensor.numpy().astype(np.float32)
        ort_outputs = self._ort_session.run(
            self._ort_output_names,
            {self._ort_input_name: array_input},
        )

        outputs_by_level = {}
        out_idx = 0
        for level_name in LEVEL_NAMES:
            cls = torch.from_numpy(ort_outputs[out_idx]); out_idx += 1
            reg = torch.from_numpy(ort_outputs[out_idx]); out_idx += 1
            cent = torch.from_numpy(ort_outputs[out_idx]); out_idx += 1
            outputs_by_level[level_name] = (cls, reg, cent)

        boxes_list, scores_list, labels_list = decode_predictions(
            outputs_by_level, score_threshold=self.score_threshold,
        )
        boxes_t, scores_t, labels_t = class_aware_nms(
            boxes_list[0],
            scores_list[0],
            labels_list[0],
            iou_threshold=self.nms_threshold,
            max_detections=self.max_detections,
        )
        return (
            boxes_t.numpy().astype(np.float32),
            scores_t.numpy().astype(np.float32),
            labels_t.numpy().astype(np.int64),
        )
