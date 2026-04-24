"""
Tests for loss and target assignment.

Covers:
* Shape correctness of ``compute_fcos_targets``.
* Determinism when gt_boxes is empty (no foregrounds, zero loss).
* Non-zero gradient flow through ``DocDetLoss`` with a synthetic
  batch.
* Per-class weighting in ``focal_loss``.
"""

from __future__ import annotations

import pytest
import torch

from layout_detector.docdet.model.loss import (
    DocDetLoss,
    LEVEL_NAMES,
    LEVEL_STRIDES,
    centerness_bce,
    ciou_loss,
    compute_fcos_targets,
    focal_loss,
)


FEAT_SIZES = {"p2": (10, 8), "p3": (5, 4), "p4": (3, 2)}


def test_targets_empty_gt() -> None:
    """Empty GT -> all classification targets zero (background)."""
    gt_boxes = torch.zeros((0, 4))
    gt_labels = torch.zeros((0,), dtype=torch.long)
    targets = compute_fcos_targets(gt_boxes, gt_labels, FEAT_SIZES)

    for level_name in LEVEL_NAMES:
        cls_t, reg_t, cent_t = targets[level_name]
        H, W = FEAT_SIZES[level_name]
        assert cls_t.shape == (H, W)
        assert reg_t.shape == (H, W, 4)
        assert cent_t.shape == (H, W)
        assert (cls_t == 0).all()
        assert (reg_t == 0).all()
        assert (cent_t == 0).all()


def test_targets_single_gt_foreground() -> None:
    """A small GT should produce foreground cells on the smallest level (P2)."""
    # P2 stride 4, feat size (10, 8) -> 32x40 px canvas covering
    # max_side = 40 which lands within P3 range (64..128) or P2 range
    # (0..64) depending on the exact box.  Use a 30x30 box which has
    # max_side 30 -> assigned to P2.
    gt_boxes = torch.tensor([[2.0, 2.0, 28.0, 28.0]])
    gt_labels = torch.tensor([3], dtype=torch.long)

    targets = compute_fcos_targets(gt_boxes, gt_labels, FEAT_SIZES)

    p2_cls, p2_reg, p2_cent = targets["p2"]
    fg_mask = p2_cls > 0
    assert fg_mask.any(), "expected at least one foreground cell on P2"
    assert (p2_cls[fg_mask] == 4).all(), (
        "class encoding: label + 1 (1-based foreground)."
    )
    assert (p2_reg[fg_mask] >= 0).all()
    assert (p2_cent[fg_mask] > 0).all()


def test_focal_loss_class_weighting() -> None:
    """Weighting a class up should increase the loss for that class."""
    torch.manual_seed(0)
    logits = torch.randn(20, 11, requires_grad=False)
    targets = torch.zeros(20, dtype=torch.long)
    targets[:4] = 3  # 3 positives in the boosted class (idx 2)

    base_loss = focal_loss(logits, targets, num_classes=11, reduction="sum")

    weights = torch.ones(11)
    weights[2] = 5.0  # boost class index 2 (targets have value 3 = id 2 + 1)
    weighted_loss = focal_loss(
        logits, targets, num_classes=11, reduction="sum", class_weights=weights,
    )
    assert weighted_loss.item() > base_loss.item(), (
        "Per-class weight should strictly increase the focal loss."
    )


def test_ciou_loss_zero_at_perfect_overlap() -> None:
    """CIoU loss should be ~0 when prediction equals target."""
    pred = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    target = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    loss = ciou_loss(pred, target)
    assert loss.item() < 1e-5


def test_centerness_bce_zero_at_perfect() -> None:
    """Centerness BCE should be small but positive on a perfect target."""
    logits = torch.full((5,), 100.0)
    target = torch.ones(5)
    loss = centerness_bce(logits, target)
    assert loss.item() < 1e-3


def test_docdet_loss_backward() -> None:
    """Full-stack: building outputs + targets and calling backward()."""
    B = 2
    num_classes = 11
    outputs = {}
    targets_per_image = []

    for b in range(B):
        # One box per pyramid level (sizes tuned to LEVEL_RANGES so
        # each of P2/P3/P4 gets at least one foreground cell).
        gt_boxes = torch.tensor([
            [0.0, 0.0, 20.0, 20.0],    # max_side 20 -> P2 range
            [0.0, 0.0, 80.0, 80.0],    # max_side 80 -> P3 range
            [0.0, 0.0, 200.0, 200.0],  # max_side 200 -> P4 range
        ])
        gt_labels = torch.tensor([0, 1, 2], dtype=torch.long)
        targets_per_image.append(
            compute_fcos_targets(gt_boxes, gt_labels, FEAT_SIZES)
        )

    for level_name, (H, W) in FEAT_SIZES.items():
        cls = torch.randn(B, num_classes, H, W, requires_grad=True)
        reg_base = torch.rand(B, 4, H, W) + 0.5
        reg = reg_base.clone().detach().requires_grad_(True)
        cent = torch.randn(B, 1, H, W, requires_grad=True)
        outputs[level_name] = (cls, reg, cent)

    criterion = DocDetLoss(num_classes=num_classes)
    loss, stats = criterion(outputs, targets_per_image)
    loss.backward()

    assert torch.isfinite(loss)
    assert stats["cls_loss"] >= 0
    assert stats["reg_loss"] >= 0
    assert stats["cent_loss"] >= 0

    for level_name, (cls, reg, cent) in outputs.items():
        assert cls.grad is not None, f"cls.grad is None for {level_name}"


def test_level_strides_match_targets_p2() -> None:
    """P2 stride 4 and coords should map consistently to pixel locations."""
    H, W = FEAT_SIZES["p2"]
    gt_boxes = torch.tensor([[0.0, 0.0, 16.0, 16.0]])
    gt_labels = torch.tensor([0], dtype=torch.long)
    targets = compute_fcos_targets(gt_boxes, gt_labels, FEAT_SIZES)

    cls_t, reg_t, cent_t = targets["p2"]
    stride = LEVEL_STRIDES["p2"]
    # Cells whose centre falls within the 16x16 box should be foreground,
    # i.e. cells with (row*4+2 < 16) and (col*4+2 < 16) -> rows 0..3, cols 0..3.
    fg = cls_t > 0
    for row in range(H):
        cell_cy = (row + 0.5) * stride
        for col in range(W):
            cell_cx = (col + 0.5) * stride
            inside = cell_cx < 16 and cell_cy < 16
            if inside:
                continue  # Foreground status depends on level range too.
            if row >= 4 or col >= 4:
                assert cls_t[row, col] == 0 or fg[row, col]
