"""
pycocotools-based evaluation metrics for DocDet.

Exports
-------
build_coco_gt           : assemble a ``pycocotools.coco.COCO`` object
                          from an OmniDocBench-style JSON layout.
build_coco_detections   : convert DocDet predictions to COCO result
                          format (one entry per (image, bbox)).
evaluate_map            : run the standard COCO evaluation and
                          return a dict of scalar metrics suitable
                          for TensorBoard logging.

Rationale
---------
We deliberately reuse pycocotools rather than reimplement mAP, for
three reasons:

1. Reproducibility: published papers use pycocotools so our numbers
   are directly comparable without hidden constant differences.
2. Auditability: the code is small and battle-tested.
3. Speed: pycocotools' internals are C-accelerated, faster than a
   pure-python reimplementation we could write in a weekend.

We additionally compute table-class AP separately because that is
the pipeline-critical metric (TableFormer input quality depends on
it; see spec 3).
"""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

from ..data.label_map import DOCDET_CLASS_NAMES


# ---------------------------------------------------------------------------
# pycocotools imports - deferred to keep the eval module optional
# ---------------------------------------------------------------------------

def _import_coco():
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as e:
        raise ImportError(
            "pycocotools is required for DocDet evaluation. "
            "Install via `pip install pycocotools`."
        ) from e
    return COCO, COCOeval


# ---------------------------------------------------------------------------
# Ground-truth and detection construction
# ---------------------------------------------------------------------------

def build_coco_gt(
    images: List[Dict],
    annotations: List[Dict],
    class_names: Iterable[str] = DOCDET_CLASS_NAMES,
):
    """
    Build a pycocotools ``COCO`` object in memory.

    Parameters
    ----------
    images      : list of {id, file_name, width, height} dicts.
    annotations : list of {id, image_id, category_id, bbox[xywh], area, iscrowd} dicts.
    class_names : iterable of class names; class IDs are positional.

    Returns
    -------
    A COCO() instance with the dataset loaded.
    """
    COCO, _ = _import_coco()

    gt = {
        "info": {"description": "DocDet eval"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i + 1, "name": name, "supercategory": "document"}
            for i, name in enumerate(class_names)
        ],
    }

    buf = io.StringIO()
    json.dump(gt, buf)
    buf.seek(0)

    coco = COCO()
    with contextlib.redirect_stdout(io.StringIO()):
        coco.dataset = gt
        coco.createIndex()
    return coco


def build_coco_detections(
    detections_per_image: Dict[int, List[Dict]],
) -> List[Dict]:
    """
    Convert per-image DocDet detections to COCO result format.

    Parameters
    ----------
    detections_per_image : dict image_id -> list of
        {"bbox": [x, y, w, h], "category_id": int (1-based),
         "score": float} dicts.

    Returns
    -------
    Flat list of dicts with an added ``image_id`` field, ready for
    ``coco.loadRes()``.
    """
    out: List[Dict] = []
    for image_id, dets in detections_per_image.items():
        for d in dets:
            out.append({
                "image_id": int(image_id),
                "category_id": int(d["category_id"]),
                "bbox": [float(x) for x in d["bbox"]],
                "score": float(d["score"]),
            })
    return out


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------

def evaluate_map(
    coco_gt,
    detections: List[Dict],
    iou_type: str = "bbox",
    class_names: Iterable[str] = DOCDET_CLASS_NAMES,
    table_class_name: str = "Table",
) -> Dict[str, float]:
    """
    Run COCOeval and return a flat metric dictionary.

    Parameters
    ----------
    coco_gt     : COCO() instance from build_coco_gt.
    detections  : list from build_coco_detections.
    iou_type    : "bbox" (default) - we do not train segmentation.
    class_names : order-matched iterable; used for per-class APs.
    table_class_name : which class to track separately for the
                       pipeline-critical Table AP@0.5.

    Returns
    -------
    metrics : dict with keys
        * "mAP"            = mAP @ IoU 0.5:0.95
        * "mAP_50"         = mAP @ IoU 0.5
        * "mAP_75"         = mAP @ IoU 0.75
        * "AP_<class>"     = per-class AP @ IoU 0.5:0.95
        * "AP50_<class>"   = per-class AP @ IoU 0.5
        * "table_AP"       = alias for AP_<table_class_name>
        * "table_AP_50"    = alias for AP50_<table_class_name>
    """
    _, COCOeval = _import_coco()

    if not detections:
        # Early exit to avoid pycocotools crashing on empty results.
        return {
            "mAP": 0.0, "mAP_50": 0.0, "mAP_75": 0.0,
            "table_AP": 0.0, "table_AP_50": 0.0,
        }

    coco_dt = coco_gt.loadRes(detections)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)

    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    stats = evaluator.stats
    metrics: Dict[str, float] = {
        "mAP": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "mAP_small": float(stats[3]),
        "mAP_medium": float(stats[4]),
        "mAP_large": float(stats[5]),
        "AR_100": float(stats[8]),
    }

    class_list = list(class_names)
    precisions = evaluator.eval["precision"]
    for idx, cname in enumerate(class_list):
        # precision shape: [T, R, K, A, M]
        #   T = IoU thresholds (0.5..0.95 by 0.05 = 10)
        #   K = categories
        if idx >= precisions.shape[2]:
            continue

        ap = float(np.nanmean(precisions[:, :, idx, 0, -1]))
        ap_50 = float(np.nanmean(precisions[0, :, idx, 0, -1]))
        metrics[f"AP_{cname}"] = ap
        metrics[f"AP50_{cname}"] = ap_50

    metrics["table_AP"] = metrics.get(f"AP_{table_class_name}", 0.0)
    metrics["table_AP_50"] = metrics.get(f"AP50_{table_class_name}", 0.0)
    return metrics


# ---------------------------------------------------------------------------
# Convenience: JSON dump for Phase 3 reports
# ---------------------------------------------------------------------------

def save_report(
    metrics: Dict[str, float],
    path: Path,
    extra: Optional[Dict] = None,
) -> None:
    """Persist ``metrics`` (+ optional ``extra`` context) as JSON."""
    out = dict(metrics)
    if extra is not None:
        out["_meta"] = extra
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
