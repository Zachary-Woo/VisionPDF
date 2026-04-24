"""
Smoke tests for the full DocDet model.

These tests intentionally use the MobileNetV3-Small backbone with
``pretrained=False`` so they run offline without downloading
weights, and use tiny input sizes so they complete within a few
seconds on CPU.
"""

from __future__ import annotations

import pytest
import torch

from layout_detector.docdet.model.model import DocDet


@pytest.fixture(scope="module")
def model() -> DocDet:
    """Lazy-build a single DocDet instance reused across tests."""
    m = DocDet(num_classes=11, backbone_name="mobilenetv3_small", pretrained=False)
    m.eval()
    return m


def test_forward_shapes(model: DocDet) -> None:
    """forward() should emit correctly-sized per-level tensors."""
    x = torch.zeros(1, 3, 320, 256)  # 20 x 16 P4 grid (stride 16)
    outputs = model(x)
    assert set(outputs.keys()) == {"p2", "p3", "p4"}

    expected = {"p2": (80, 64), "p3": (40, 32), "p4": (20, 16)}
    for name, (h, w) in expected.items():
        cls, reg, cent = outputs[name]
        assert cls.shape == (1, 11, h, w)
        assert reg.shape == (1, 4, h, w)
        assert cent.shape == (1, 1, h, w)
        assert torch.isfinite(cls).all()
        assert torch.isfinite(reg).all()
        assert torch.isfinite(cent).all()
        assert (reg > 0).all(), "Regression should be strictly positive (post-exp)"


def test_detect_returns_lists(model: DocDet) -> None:
    """detect() should return one list of detections per image."""
    x = torch.zeros(2, 3, 320, 256)
    result = model.detect(x, score_threshold=0.01)
    assert len(result) == 2
    for per_image in result:
        for det in per_image:
            label, x1, y1, x2, y2, score = det
            assert isinstance(label, str)
            assert x1 <= x2
            assert y1 <= y2
            assert 0.0 <= score <= 1.0


def test_num_parameters_reasonable(model: DocDet) -> None:
    """DocDet should be lean.  < 6M parameters keeps it edge-deployable."""
    total = model.num_parameters()
    assert 1_000_000 < total < 6_000_000, (
        f"DocDet parameter count {total} unexpected; spec budgets ~4M."
    )


def test_freeze_backbone_stages(model: DocDet) -> None:
    """freeze_backbone_stages should actually freeze the requested stages."""
    model.freeze_backbone_stages(2)
    frozen_counts = [
        all(not p.requires_grad for p in stage.parameters())
        for stage in (
            model.backbone.stage_p2,
            model.backbone.stage_p3,
            model.backbone.stage_p4,
        )
    ]
    assert frozen_counts == [True, True, False]

    model.unfreeze_backbone()
    assert all(p.requires_grad for p in model.backbone.parameters())
