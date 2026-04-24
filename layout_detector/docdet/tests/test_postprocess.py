"""
Tests for postprocess utilities: decode, NMS, and table dilation.
"""

from __future__ import annotations

import pytest
import torch

from layout_detector.docdet.model.postprocess import (
    class_aware_nms,
    decode_predictions,
    dilate_table_bboxes,
)


def _make_outputs(B: int = 1, C: int = 11) -> dict:
    """Construct synthetic per-level outputs that emit exactly one hit."""
    outputs = {}
    for level_name, (H, W) in [("p2", (4, 4)), ("p3", (2, 2)), ("p4", (1, 1))]:
        cls = torch.full((B, C, H, W), -10.0)
        reg = torch.full((B, 4, H, W), 1.5)
        cent = torch.full((B, 1, H, W), 5.0)
        # Force P3 (0, 0) to become a positive detection.
        if level_name == "p3":
            cls[0, 2, 0, 0] = 5.0  # class id 2, very high score
        outputs[level_name] = (cls, reg, cent)
    return outputs


def test_decode_returns_expected_shapes() -> None:
    outputs = _make_outputs(B=1)
    boxes, scores, labels = decode_predictions(outputs, score_threshold=0.01)
    assert len(boxes) == 1 and len(scores) == 1 and len(labels) == 1
    assert boxes[0].shape[1] == 4
    assert (scores[0] > 0).all()


def test_class_aware_nms_keeps_different_classes() -> None:
    """Overlapping boxes of DIFFERENT classes must survive NMS."""
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],  # identical geometry
    ])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 1])  # different classes
    keep_boxes, keep_scores, keep_labels = class_aware_nms(boxes, scores, labels)
    assert keep_boxes.shape[0] == 2


def test_class_aware_nms_suppresses_same_class() -> None:
    """Heavily overlapping boxes of the SAME class should be suppressed."""
    boxes = torch.tensor([
        [0.0, 0.0, 10.0, 10.0],
        [0.0, 0.0, 10.0, 10.0],
    ])
    scores = torch.tensor([0.9, 0.8])
    labels = torch.tensor([0, 0])
    keep_boxes, keep_scores, keep_labels = class_aware_nms(boxes, scores, labels)
    assert keep_boxes.shape[0] == 1
    assert float(keep_scores[0]) == pytest.approx(0.9, abs=1e-4)


def test_dilate_table_bboxes_only_tables() -> None:
    """Non-table boxes must stay exactly the same."""
    boxes = torch.tensor([
        [10.0, 10.0, 50.0, 50.0],  # Table (class id 8)
        [60.0, 60.0, 90.0, 90.0],  # Text (class id 9)
    ])
    labels = torch.tensor([8, 9])
    padded = dilate_table_bboxes(boxes, labels, table_class_id=8, pad_px=3)

    assert torch.allclose(padded[0], torch.tensor([7.0, 7.0, 53.0, 53.0]))
    assert torch.allclose(padded[1], boxes[1])


def test_dilate_table_bboxes_clamps_to_image() -> None:
    """Dilation near the border must not go below 0 or above image size."""
    boxes = torch.tensor([[1.0, 1.0, 99.0, 99.0]])
    labels = torch.tensor([8])
    padded = dilate_table_bboxes(
        boxes, labels, table_class_id=8, pad_px=5, image_size=(100, 100),
    )
    assert float(padded[0, 0]) == 0.0
    assert float(padded[0, 1]) == 0.0
    assert float(padded[0, 2]) == 100.0
    assert float(padded[0, 3]) == 100.0
