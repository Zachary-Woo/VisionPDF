"""
Phase 2: multi-dataset real-document fine-tuning.

Runs ~15 epochs with the backbone fully unfrozen, sampling from a
weighted mixture of DocLayNet / PubLayNet / TableBank / IIIT-AR-13K
(spec 6.2).  Picks up the Phase 1 checkpoint if provided, otherwise
starts from the ImageNet-only backbone + a freshly initialised head.

Each dataset has its own native on-disk format, so this script
dispatches to the matching loader rather than forcing a COCO-JSON
intermediate:

    DocLayNet, PubLayNet, TableBank   ->  ParquetDetectionSource
    IIIT-AR-13K                       ->  IIITARVocSource
    (legacy COCO-JSON, any source)    ->  CocoSource

Only the datasets actually present on disk are registered in the
mixture; missing ones are logged and skipped so users with a partial
download can still run the phase.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..data.coco_source import CocoSource, docdet_collate
from ..data.iiitar_source import IIITARVocSource
from ..data.label_map import NUM_DOCDET_CLASSES
from ..data.parquet_source import ParquetDetectionSource
from ..data.weighted_sampler import MultiSourceDataset
from ..model.loss import DocDetLoss
from ..model.model import DocDet
from .config import phase2_config
from .trainer import Trainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source dataset factories
# ---------------------------------------------------------------------------

def _build_parquet_source(
    name: str,
    root: Optional[Path],
    ann_file: Optional[Path],
    img_root: Optional[Path],
    train: bool,
) -> Optional[Dataset]:
    """Return a Parquet- or COCO-backed source, whichever paths are given."""
    if root is not None and Path(root).exists():
        try:
            ds = ParquetDetectionSource(parquet_files=root, source_name=name)
            logger.info("%s (parquet): %d samples from %s", name, len(ds), root)
            return ds
        except Exception as e:
            logger.warning("%s parquet load failed (%s); trying COCO fallback", name, e)

    if ann_file and img_root and Path(ann_file).exists() and Path(img_root).exists():
        ds = CocoSource(
            annotation_file=ann_file,
            image_root=img_root,
            source_name=name,
            train=train,
        )
        logger.info("%s (coco): %d images from %s", name, len(ds), img_root)
        return ds

    logger.warning("Skipping %s: no usable parquet root or COCO paths provided", name)
    return None


def _build_iiit_ar_source(
    root: Optional[Path],
    ann_file: Optional[Path],
    img_root: Optional[Path],
    train: bool,
) -> Optional[Dataset]:
    """IIIT-AR-13K loader: VOC XML by default, COCO if --iiit-ar-annotations is given."""
    if root is not None and Path(root).exists():
        try:
            ds = IIITARVocSource(root=root, split="train" if train else "val")
            logger.info("iiit_ar (voc): %d images from %s", len(ds), root)
            return ds
        except Exception as e:
            logger.warning("iiit_ar VOC load failed (%s); trying COCO fallback", e)

    if ann_file and img_root and Path(ann_file).exists() and Path(img_root).exists():
        ds = CocoSource(
            annotation_file=ann_file,
            image_root=img_root,
            source_name="iiit_ar",
            train=train,
        )
        logger.info("iiit_ar (coco): %d images from %s", len(ds), img_root)
        return ds

    logger.warning("Skipping iiit_ar: no usable VOC root or COCO paths provided")
    return None


def _build_mixture(args) -> Tuple[MultiSourceDataset, List[str]]:
    """Assemble the weighted MultiSourceDataset from whatever is available."""
    sources = {}

    for name, root, ann, imgs in [
        ("doclaynet", args.doclaynet_root, args.doclaynet_annotations, args.doclaynet_images),
        ("publaynet", args.publaynet_root, args.publaynet_annotations, args.publaynet_images),
        ("tablebank", args.tablebank_root, args.tablebank_annotations, args.tablebank_images),
    ]:
        ds = _build_parquet_source(name, root, ann, imgs, train=True)
        if ds is not None:
            sources[name] = ds

    iiit = _build_iiit_ar_source(
        args.iiit_ar_root, args.iiit_ar_annotations, args.iiit_ar_images, train=True,
    )
    if iiit is not None:
        sources["iiit_ar"] = iiit

    if not sources:
        raise RuntimeError(
            "Phase 2 requires at least one dataset. Pass --doclaynet-root / "
            "--publaynet-root / --tablebank-root / --iiit-ar-root (or the "
            "legacy --*-annotations + --*-images COCO pair)."
        )

    cfg = phase2_config()
    default_weights = cfg.source_weights
    actual_weights = {name: default_weights.get(name, 1.0) for name in sources}
    mixture = MultiSourceDataset(sources=sources, weights=actual_weights)
    return mixture, list(sources.keys())


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocDet Phase 2: multi-dataset real-document fine-tuning.",
    )

    # Native-format roots (preferred): point at the dataset cache directory.
    parser.add_argument("--doclaynet-root", type=Path, default=None,
                        help="Root containing DocLayNet train parquet shards.")
    parser.add_argument("--publaynet-root", type=Path, default=None,
                        help="Root containing PubLayNet train parquet shards.")
    parser.add_argument("--tablebank-root", type=Path, default=None,
                        help="Root containing TableBank-Detection train parquets.")
    parser.add_argument("--iiit-ar-root", type=Path, default=None,
                        help="Root of the unpacked Kaggle IIIT-AR-13K dataset (VOC XML).")

    # Legacy COCO-JSON fallback (still honoured if both flags are passed).
    parser.add_argument("--doclaynet-annotations", type=Path, default=None)
    parser.add_argument("--doclaynet-images", type=Path, default=None)
    parser.add_argument("--publaynet-annotations", type=Path, default=None)
    parser.add_argument("--publaynet-images", type=Path, default=None)
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
