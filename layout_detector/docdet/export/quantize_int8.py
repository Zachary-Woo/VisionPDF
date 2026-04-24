"""
Post-training INT8 static quantization for the exported DocDet ONNX.

Spec 9.2 calls out static quantization using ONNX Runtime's
``quantize_static`` with calibration on 500 DocLayNet val pages.
Static quantization is preferred over dynamic because convolution
activations dominate DocDet's compute and dynamic only helps
MatMul-heavy architectures.

The calibration reader is a thin adapter over
``DocDetEvalTransform`` so we use the exact preprocessing pipeline
the runtime will see at inference (ImageNet normalisation,
letterbox, same target size).

Reduce range (int7) is disabled on x86 because ONNX Runtime already
handles signed int8 saturation via clip; keeping full int8 range
preserves more accuracy.

Usage
-----
python -m layout_detector.docdet.export.quantize_int8 \\
    --fp32-onnx layout_detector/weights/docdet.onnx \\
    --int8-onnx layout_detector/weights/docdet_int8.onnx \\
    --calibration-images path/to/500_val_images \\
    --num-calibration-images 500
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

import numpy as np

from ..data.transforms import DocDetEvalTransform

logger = logging.getLogger(__name__)


def _import_ort_quant():
    try:
        from onnxruntime.quantization import (
            CalibrationDataReader,
            QuantFormat,
            QuantType,
            quantize_static,
        )
    except ImportError as e:
        raise ImportError(
            "onnxruntime is required for INT8 quantization. "
            "Install via `pip install onnxruntime`."
        ) from e
    return CalibrationDataReader, QuantFormat, QuantType, quantize_static


class _DocDetCalibrationReader:
    """
    Iterate a directory of calibration images, yielding preprocessed
    (1, 3, H, W) numpy tensors through ORT's ``CalibrationDataReader``
    API.
    """

    def __init__(
        self,
        image_paths: List[Path],
        input_name: str,
        input_size=(1120, 800),
    ):
        self.image_paths = image_paths
        self.input_name = input_name
        self.transform = DocDetEvalTransform(target_size=input_size)
        self._iter: Optional[Iterator[Path]] = None

    def _ensure_iter(self) -> None:
        if self._iter is None:
            self._iter = iter(self.image_paths)

    def get_next(self):
        """ORT calibration protocol: return dict or None."""
        from PIL import Image

        self._ensure_iter()
        try:
            path = next(self._iter)
        except StopIteration:
            return None

        pil = Image.open(path).convert("RGB")
        tensor, _boxes, _meta = self.transform(
            pil, np.zeros((0, 4), dtype=np.float32),
        )
        arr = tensor.unsqueeze(0).numpy().astype(np.float32)
        return {self.input_name: arr}

    def rewind(self) -> None:
        """Allow re-iteration (ORT can call this between percentile runs)."""
        self._iter = None


def _discover_images(
    root: Path,
    max_images: Optional[int],
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff"),
) -> List[Path]:
    """
    Walk ``root`` and return up to ``max_images`` image paths.

    We deliberately sort so calibration is reproducible between runs.
    """
    files: List[Path] = []
    for ext in extensions:
        files.extend(sorted(root.rglob(f"*{ext}")))
    if max_images is not None:
        files = files[:max_images]
    return files


def quantize_static_int8(
    fp32_onnx: Path,
    int8_onnx: Path,
    calibration_dir: Path,
    num_calibration_images: int = 500,
    input_height: int = 1120,
    input_width: int = 800,
) -> Path:
    """
    Quantize ``fp32_onnx`` to INT8 and write to ``int8_onnx``.

    Returns the resolved ``int8_onnx`` path.
    """
    CalibrationDataReader, QuantFormat, QuantType, quantize_static = _import_ort_quant()

    image_paths = _discover_images(calibration_dir, num_calibration_images)
    if not image_paths:
        raise RuntimeError(
            f"No calibration images found under {calibration_dir}. "
            "Point --calibration-images at a folder that contains "
            "PNG/JPG/TIFF files."
        )
    logger.info("Using %d calibration images", len(image_paths))

    reader = _DocDetCalibrationReader(
        image_paths=image_paths,
        input_name="images",
        input_size=(input_height, input_width),
    )

    int8_onnx.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Running ONNX Runtime static quantization...")

    quantize_static(
        model_input=str(fp32_onnx),
        model_output=str(int8_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QOperator,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
        op_types_to_quantize=["Conv", "MatMul"],
    )
    logger.info("Wrote INT8 model to %s", int8_onnx)
    return int8_onnx


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Statically quantize DocDet ONNX to INT8."
    )
    parser.add_argument("--fp32-onnx", type=Path, required=True)
    parser.add_argument("--int8-onnx", type=Path, required=True)
    parser.add_argument("--calibration-images", type=Path, required=True,
                        help="Directory of calibration images (e.g. DocLayNet val).")
    parser.add_argument("--num-calibration-images", type=int, default=500)
    parser.add_argument("--input-height", type=int, default=1120)
    parser.add_argument("--input-width", type=int, default=800)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    quantize_static_int8(
        fp32_onnx=args.fp32_onnx,
        int8_onnx=args.int8_onnx,
        calibration_dir=args.calibration_images,
        num_calibration_images=args.num_calibration_images,
        input_height=args.input_height,
        input_width=args.input_width,
    )


if __name__ == "__main__":
    main()
