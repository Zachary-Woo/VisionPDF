"""
End-to-end OmniDocBench evaluation driver.

This script:

1. Parses the OmniDocBench annotation file (a list of per-page
   dicts with ``page_info`` / ``layout_dets`` / ``extra`` keys).
2. Converts the layout annotations into COCO-compatible ground
   truth (dropping mask/abandon categories).
3. Runs a DocDet checkpoint over every image at inference size
   800x1120 with the standard eval transform.
4. Accumulates detections and hands both to the pycocotools-based
   ``evaluate_map`` helper.
5. Writes a JSON report and a one-line summary usable in CI.

The driver intentionally avoids any dependency on the OmniDocBench
repo's own evaluation code: we only consume the raw JSON + images.

Usage
-----
python -m layout_detector.docdet.eval.eval_omnidocbench \\
    --checkpoint layout_detector/weights/phase2/last.pt \\
    --annotations OmniDocBench/OmniDocBench.json \\
    --image-root OmniDocBench/images \\
    --output-dir layout_detector/weights/eval_omnidocbench
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

from ..data.label_map import DOCDET_CLASS_NAMES, NUM_DOCDET_CLASSES, map_class_with_logging
from ..data.transforms import DocDetEvalTransform
from ..model.model import DocDet
from .metrics import build_coco_detections, build_coco_gt, evaluate_map, save_report

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _poly_to_xyxy(poly: List[float]) -> Tuple[float, float, float, float]:
    """Convert an OmniDocBench polygon (8 numbers) to (x1,y1,x2,y2)."""
    xs = poly[0::2]
    ys = poly[1::2]
    return float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))


def _is_ignored(layout_det: Dict[str, Any]) -> bool:
    """True if the annotation should be ignored per OmniDocBench rules."""
    if layout_det.get("ignore") is True:
        return True
    if layout_det.get("ignore") == "True":
        return True
    return False


def _parse_omnidocbench(
    annotation_file: Path,
    image_root: Path,
) -> Tuple[List[Dict], List[Dict], List[Path]]:
    """
    Parse the OmniDocBench JSON into COCO format.

    Returns
    -------
    images      : list of COCO-style image dicts.
    annotations : list of COCO-style annotation dicts (category IDs
                  are 1-based DocDet class IDs).
    image_paths : absolute paths to every image (same order as
                  ``images``).
    """
    with open(annotation_file, "r", encoding="utf-8") as f:
        samples = json.load(f)

    images: List[Dict] = []
    annotations: List[Dict] = []
    image_paths: List[Path] = []

    ann_id = 1
    for image_id, sample in enumerate(samples, start=1):
        page_info = sample["page_info"]
        img_path = image_root / page_info["image_path"]

        images.append({
            "id": image_id,
            "file_name": page_info["image_path"],
            "width": int(page_info["width"]),
            "height": int(page_info["height"]),
        })
        image_paths.append(img_path)

        for ld in sample["layout_dets"]:
            if _is_ignored(ld):
                continue
            category_type = ld.get("category_type")
            docdet_id = map_class_with_logging("omnidocbench", category_type)
            if docdet_id is None:
                continue
            if "poly" not in ld:
                continue

            x1, y1, x2, y2 = _poly_to_xyxy(ld["poly"])
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": docdet_id + 1,  # COCO categories are 1-based
                "bbox": [x1, y1, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            ann_id += 1

    return images, annotations, image_paths


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_model(checkpoint: Path, device: torch.device) -> DocDet:
    """Load a DocDet checkpoint for eval."""
    payload = torch.load(checkpoint, map_location="cpu")
    backbone_name = payload.get("backbone_name", "mobilenetv3_small")
    model = DocDet(
        num_classes=NUM_DOCDET_CLASSES,
        backbone_name=backbone_name,
        pretrained=False,
    )
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys loading checkpoint: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys loading checkpoint: %d", len(unexpected))
    model.eval().to(device)
    return model


@torch.no_grad()
def _run_inference(
    model: DocDet,
    image_paths: List[Path],
    images_meta: List[Dict],
    device: torch.device,
    input_size: Tuple[int, int] = (1120, 800),
    score_threshold: float = 0.05,
) -> Tuple[Dict[int, List[Dict]], float]:
    """
    Run DocDet on every image and collect detections in COCO format.

    Parameters
    ----------
    input_size : (H, W) tuple matching the training target resolution.

    Returns
    -------
    detections : dict image_id -> list of COCO result dicts.
    total_time : wall-clock seconds over the full loop (useful for
                 the per-page latency report).
    """
    transform = DocDetEvalTransform(target_size=input_size)

    detections: Dict[int, List[Dict]] = {}
    total_time = 0.0

    for img_path, meta in zip(image_paths, images_meta):
        image_id = meta["id"]
        try:
            pil = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            logger.warning("Missing image %s; skipping", img_path)
            detections[image_id] = []
            continue

        img_t, _box_tensor, transform_meta = transform(
            pil, np.zeros((0, 4), dtype=np.float32),
        )
        img_t = img_t.unsqueeze(0).to(device)

        t0 = time.time()
        batch_dets = model.detect(
            img_t,
            score_threshold=score_threshold,
            label_names=None,  # we want raw integer class IDs
        )
        total_time += time.time() - t0

        scale = float(transform_meta["scale"])
        pad_x, pad_y = transform_meta["pad_xy"]

        dets: List[Dict] = []
        for label_str, x1, y1, x2, y2, score in batch_dets[0]:
            cls_id = int(label_str)
            # Undo letterbox to recover original-image coordinates.
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale
            x1 = float(np.clip(x1, 0.0, meta["width"]))
            y1 = float(np.clip(y1, 0.0, meta["height"]))
            x2 = float(np.clip(x2, 0.0, meta["width"]))
            y2 = float(np.clip(y2, 0.0, meta["height"]))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w <= 0 or h <= 0:
                continue
            dets.append({
                "bbox": [x1, y1, w, h],
                "category_id": cls_id + 1,
                "score": float(score),
            })
        detections[image_id] = dets

    return detections, total_time


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DocDet on OmniDocBench."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True,
                        help="Path to OmniDocBench.json")
    parser.add_argument("--image-root", type=Path, required=True,
                        help="Directory containing OmniDocBench images.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("layout_detector/weights/eval_omnidocbench"))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--input-height", type=int, default=1120)
    parser.add_argument("--input-width", type=int, default=800)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info("Using device %s", device)

    logger.info("Parsing annotations %s", args.annotations)
    images, annotations, image_paths = _parse_omnidocbench(
        args.annotations, args.image_root,
    )
    logger.info("Loaded %d images, %d annotations", len(images), len(annotations))

    coco_gt = build_coco_gt(images, annotations)

    logger.info("Loading model %s", args.checkpoint)
    model = _load_model(args.checkpoint, device)

    logger.info("Running inference on %d images", len(images))
    detections_per_image, total_time = _run_inference(
        model,
        image_paths,
        images,
        device,
        input_size=(args.input_height, args.input_width),
        score_threshold=args.score_threshold,
    )

    avg_latency_ms = (total_time / max(1, len(image_paths))) * 1000.0
    logger.info("Average per-page inference latency: %.1f ms", avg_latency_ms)

    logger.info("Scoring...")
    detections = build_coco_detections(detections_per_image)
    metrics = evaluate_map(
        coco_gt, detections,
        class_names=DOCDET_CLASS_NAMES,
        table_class_name="Table",
    )
    metrics["avg_latency_ms_per_page"] = avg_latency_ms

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "metrics.json"
    save_report(metrics, report_path, extra={
        "checkpoint": str(args.checkpoint),
        "annotations": str(args.annotations),
        "image_root": str(args.image_root),
        "input_size": [args.input_height, args.input_width],
        "score_threshold": args.score_threshold,
        "num_images": len(images),
    })

    logger.info(
        "mAP=%.3f  mAP50=%.3f  table_AP=%.3f  table_AP50=%.3f  report=%s",
        metrics["mAP"], metrics["mAP_50"],
        metrics["table_AP"], metrics["table_AP_50"],
        report_path,
    )


if __name__ == "__main__":
    main()
