"""
Phase 2: multi-dataset real-document fine-tuning.

Runs ~15 epochs with the backbone fully unfrozen, sampling from a
weighted mixture of DocLayNet / PubLayNet / DocBank / TableBank /
IIIT-AR-13K (spec 6.2).  Picks up the Phase 1 checkpoint if
provided, otherwise starts from the ImageNet-only backbone + a
freshly initialised head.

Only the datasets actually present on disk are registered in the
mixture; missing datasets are logged and skipped so users without
Kaggle credentials or 500 GB of free space can still run the phase
with whatever subset they have.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..data.coco_source import CocoSource, docdet_collate
from ..data.label_map import NUM_DOCDET_CLASSES
from ..data.weighted_sampler import MultiSourceDataset
from ..model.loss import DocDetLoss
from ..model.model import DocDet
from .config import phase2_config
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _try_build_coco_source(
    name: str,
    annotation_file: Optional[Path],
    image_root: Optional[Path],
    train: bool,
) -> Optional[CocoSource]:
    """Build a CocoSource or return None (with a warning) if paths are missing."""
    if annotation_file is None or image_root is None:
        logger.warning("Skipping %s: paths not provided", name)
        return None
    if not Path(annotation_file).exists():
        logger.warning(
            "Skipping %s: annotation file not found at %s",
            name, annotation_file,
        )
        return None
    if not Path(image_root).exists():
        logger.warning(
            "Skipping %s: image root not found at %s", name, image_root,
        )
        return None
    return CocoSource(
        annotation_file=annotation_file,
        image_root=image_root,
        source_name=name,
        train=train,
    )


def _build_mixture(args) -> Tuple[MultiSourceDataset, List[str]]:
    """Assemble the weighted MultiSourceDataset from whatever is available."""
    candidates = {
        "doclaynet": (args.doclaynet_annotations, args.doclaynet_images),
        "publaynet": (args.publaynet_annotations, args.publaynet_images),
        "docbank": (args.docbank_annotations, args.docbank_images),
        "tablebank": (args.tablebank_annotations, args.tablebank_images),
        "iiit_ar": (args.iiit_ar_annotations, args.iiit_ar_images),
    }

    sources = {}
    for name, (ann, imgs) in candidates.items():
        src = _try_build_coco_source(name, ann, imgs, train=True)
        if src is not None:
            sources[name] = src
            logger.info("%s ready with %d images", name, len(src))

    if not sources:
        raise RuntimeError(
            "Phase 2 requires at least one dataset. Pass "
            "--doclaynet-annotations / --doclaynet-images (and the "
            "same for any other sources you have)."
        )

    cfg = phase2_config()
    default_weights = cfg.source_weights
    actual_weights = {
        name: default_weights.get(name, 1.0) for name in sources
    }
    mixture = MultiSourceDataset(sources=sources, weights=actual_weights)
    return mixture, list(sources.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocDet Phase 2: multi-dataset real-document fine-tuning.",
    )

    parser.add_argument("--doclaynet-annotations", type=Path, default=None)
    parser.add_argument("--doclaynet-images", type=Path, default=None)
    parser.add_argument("--publaynet-annotations", type=Path, default=None)
    parser.add_argument("--publaynet-images", type=Path, default=None)
    parser.add_argument("--docbank-annotations", type=Path, default=None)
    parser.add_argument("--docbank-images", type=Path, default=None)
    parser.add_argument("--tablebank-annotations", type=Path, default=None)
    parser.add_argument("--tablebank-images", type=Path, default=None)
    parser.add_argument("--iiit-ar-annotations", type=Path, default=None)
    parser.add_argument("--iiit-ar-images", type=Path, default=None)

    parser.add_argument("--phase1-checkpoint", type=Path, default=None,
                        help="Phase 1 checkpoint to warm-start from.")
    parser.add_argument("--resume-from", type=Path, default=None,
                        help="Resume a Phase 2 run (optimizer state included).")
    parser.add_argument("--save-dir", type=Path, default=Path("layout_detector/weights"))
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small",
                        choices=["mobilenetv3_small", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples-per-epoch", type=int, default=None,
                        help="Number of samples drawn per epoch from the mixture.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = phase2_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.save_dir = args.save_dir
    cfg.resume_from = args.resume_from

    logger.info("Phase 2 config: %s", cfg)

    mixture, active = _build_mixture(args)
    logger.info("Active sources: %s (sizes: %s)", active, mixture.source_sizes())

    num_samples = args.num_samples_per_epoch or len(mixture)
    sampler = mixture.build_sampler(num_samples=num_samples)

    loader = DataLoader(
        mixture,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=docdet_collate,
        pin_memory=True,
    )

    model = DocDet(
        num_classes=NUM_DOCDET_CLASSES,
        backbone_name=args.backbone,
        pretrained=True,
    )

    if args.phase1_checkpoint is not None:
        logger.info("Loading Phase 1 checkpoint %s", args.phase1_checkpoint)
        payload = torch.load(args.phase1_checkpoint, map_location="cpu")
        state = payload.get("model", payload)
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info("Phase 1 loaded (missing=%d unexpected=%d)", len(missing), len(unexpected))

    criterion = DocDetLoss(
        num_classes=NUM_DOCDET_CLASSES,
        cls_weight=cfg.cls_weight,
        reg_weight=cfg.reg_weight,
        cent_weight=cfg.cent_weight,
    )

    trainer = Trainer(model, criterion, loader, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
