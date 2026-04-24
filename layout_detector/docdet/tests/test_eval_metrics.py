"""
Tests for eval/metrics.py.

We skip the whole file if pycocotools is unavailable since the
rest of the DocDet stack does not require it.
"""

from __future__ import annotations

import pytest

pycocotools = pytest.importorskip("pycocotools")

from layout_detector.docdet.eval.metrics import (
    build_coco_detections,
    build_coco_gt,
    evaluate_map,
)


def _mock_gt():
    images = [
        {"id": 1, "file_name": "a.png", "width": 100, "height": 100},
        {"id": 2, "file_name": "b.png", "width": 100, "height": 100},
    ]
    annotations = [
        {
            "id": 1, "image_id": 1, "category_id": 9,  # 1-based Table (idx 8 + 1)
            "bbox": [10, 10, 30, 30], "area": 900, "iscrowd": 0,
        },
        {
            "id": 2, "image_id": 2, "category_id": 10,  # 1-based Text
            "bbox": [20, 20, 20, 20], "area": 400, "iscrowd": 0,
        },
    ]
    return images, annotations


def test_evaluate_map_perfect_predictions() -> None:
    """If we feed the GT back as predictions, mAP should be 1.0."""
    images, annotations = _mock_gt()
    gt = build_coco_gt(images, annotations)

    preds_per_image = {
        1: [{"bbox": [10, 10, 30, 30], "category_id": 9, "score": 0.99}],
        2: [{"bbox": [20, 20, 20, 20], "category_id": 10, "score": 0.99}],
    }
    dets = build_coco_detections(preds_per_image)
    metrics = evaluate_map(gt, dets)

    assert metrics["mAP"] > 0.9
    assert metrics["mAP_50"] > 0.99


def test_evaluate_map_no_predictions_returns_zero() -> None:
    images, annotations = _mock_gt()
    gt = build_coco_gt(images, annotations)
    metrics = evaluate_map(gt, [])
    assert metrics["mAP"] == 0.0
    assert metrics["mAP_50"] == 0.0
