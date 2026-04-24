"""
Dataclass configs for DocDet training phases.

Each phase script (``phase0_coco``, ``phase1_synth``, ``phase2_real``,
``phase4_targeted``) instantiates an appropriate config, potentially
overridden by CLI flags, and passes it to the ``Trainer`` class.

Keeping every hyperparameter in one place makes it trivial to
diff experiments, resume from the right checkpoint, and reproduce
runs between local machines and Colab.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass
class TrainerConfig:
    """
    Per-phase hyperparameters for the single-GPU Trainer.

    Defaults match the spec for Phase 2 (real-data fine-tuning); Phase
    scripts override the relevant fields.
    """

    epochs: int = 15
    batch_size: int = 24
    learning_rate: float = 2e-4
    min_learning_rate: float = 1e-6
    warmup_steps: int = 500
    weight_decay: float = 0.05
    grad_clip_max_norm: float = 10.0
    use_amp: bool = True
    num_workers: int = 4
    log_every_n_steps: int = 50
    checkpoint_every_n_steps: int = 2000
    eval_every_n_epochs: int = 1
    save_dir: Path = Path("layout_detector/weights")
    run_name: str = "docdet_run"
    resume_from: Optional[Path] = None
    freeze_backbone_stages: int = 0  # 0 = fully trainable, 2 = freeze P2+P3

    # Loss weights (fed to DocDetLoss).
    cls_weight: float = 1.0
    reg_weight: float = 1.0
    cent_weight: float = 1.0

    # Used by phase2 only.
    mixup_alpha: float = 0.0

    # Per-source weights for MultiSourceDataset (Phase 2 only).
    source_weights: Dict[str, float] = field(default_factory=dict)

    # Target image size (H, W).
    target_size: Tuple[int, int] = (1120, 800)


def phase1_config() -> TrainerConfig:
    """Spec 6 / Phase 1 defaults: DocSynth300K synthetic pretrain."""
    return TrainerConfig(
        epochs=20,
        batch_size=32,
        learning_rate=1e-3,
        min_learning_rate=1e-5,
        warmup_steps=1000,
        weight_decay=0.05,
        freeze_backbone_stages=2,
        run_name="docdet_phase1_synth",
        mixup_alpha=0.0,
    )


def phase2_config() -> TrainerConfig:
    """Spec 6 / Phase 2 defaults: multi-dataset weighted fine-tune."""
    return TrainerConfig(
        epochs=15,
        batch_size=24,
        learning_rate=2e-4,
        min_learning_rate=1e-6,
        warmup_steps=1000,
        weight_decay=0.05,
        freeze_backbone_stages=0,
        run_name="docdet_phase2_real",
        mixup_alpha=0.1,
        source_weights={
            "doclaynet": 3.0,
            "publaynet": 1.0,
            "tablebank": 2.0,
            "iiit_ar": 4.0,
        },
    )


def phase0_config() -> TrainerConfig:
    """Optional COCO 80-class warmup (natural-image figures)."""
    return TrainerConfig(
        epochs=12,
        batch_size=16,
        learning_rate=1e-3,
        min_learning_rate=1e-5,
        warmup_steps=1000,
        weight_decay=0.05,
        freeze_backbone_stages=0,
        run_name="docdet_phase0_coco",
        mixup_alpha=0.0,
        target_size=(640, 640),
    )


def phase4_config() -> TrainerConfig:
    """Phase 4: conditional targeted fine-tune at low LR."""
    return TrainerConfig(
        epochs=5,
        batch_size=24,
        learning_rate=5e-5,
        min_learning_rate=1e-6,
        warmup_steps=0,
        weight_decay=0.05,
        freeze_backbone_stages=0,
        run_name="docdet_phase4_targeted",
        mixup_alpha=0.0,
    )
