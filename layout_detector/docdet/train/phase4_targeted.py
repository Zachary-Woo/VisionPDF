"""
Phase 4: conditional targeted fine-tune.

The spec (section 6.4) describes Phase 4 as a *conditional* stage:
after Phase 3 evaluation on OmniDocBench, if one or two classes are
under-performing (most often Formula, Page-footer, or Table at high
IoU thresholds) we spend a handful of epochs upweighting those
classes on the Phase 2 training mixture at a low learning rate.

The implementation accepts:

- a Phase 2 checkpoint to warm-start from,
- the same per-source dataset flags as Phase 2,
- ``--boost-class CLASS`` (repeatable) to upweight a class,
- ``--boost-factor`` to control the per-class loss multiplier,

and trains for ``phase4_config().epochs`` epochs (default 5) with
LR 5e-5, no augmentation tweaks, and no backbone freezing.

If no boost class is specified the stage is effectively a very
gentle general-purpose polish (useful as a lightweight refresh).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from ..data.coco_source import CocoSource, docdet_collate
from ..data.label_map import DOCDET_CLASS_NAMES, NUM_DOCDET_CLASSES, canonical_id
from ..data.weighted_sampler import MultiSourceDataset
from ..model.loss import DocDetLoss
from ..model.model import DocDet
from .config import phase4_config
from .trainer import Trainer

logger = logging.getLogger(__name__)


def _maybe_source(
    name: str,
    annotation_file: Optional[Path],
    image_root: Optional[Path],
) -> Optional[CocoSource]:
    """Return a CocoSource if both paths exist, else None."""
    if annotation_file is None or image_root is None:
        return None
    if not Path(annotation_file).exists() or not Path(image_root).exists():
        logger.warning("Skipping %s: paths not found", name)
        return None
    return CocoSource(
        annotation_file=annotation_file,
        image_root=image_root,
        source_name=name,
        train=True,
    )


def _build_class_weights(
    boost_classes: List[str],
    boost_factor: float,
) -> Optional[torch.Tensor]:
    """
    Build a per-class scalar vector that multiplies the focal-loss
    classification contribution.

    Classes named in ``boost_classes`` receive ``boost_factor`` while
    the rest stay at 1.0.  Returns None if no boost is requested,
    telling the DocDetLoss to use uniform weighting.
    """
    if not boost_classes:
        return None

    weights = torch.ones(NUM_DOCDET_CLASSES, dtype=torch.float32)
    for cname in boost_classes:
        if cname not in DOCDET_CLASS_NAMES:
            raise ValueError(
                f"Unknown class '{cname}'. Valid: {list(DOCDET_CLASS_NAMES)}"
            )
        weights[canonical_id(cname)] = boost_factor

    logger.info(
        "Phase 4 class weights: %s",
        {n: float(weights[canonical_id(n)]) for n in DOCDET_CLASS_NAMES},
    )
    return weights


def _build_mixture(args) -> Tuple[MultiSourceDataset, List[str]]:
    """Assemble whatever subset of the Phase 2 mixture is available."""
    sources: Dict[str, CocoSource] = {}
    candidates = {
        "doclaynet": (args.doclaynet_annotations, args.doclaynet_images),
        "publaynet": (args.publaynet_annotations, args.publaynet_images),
        "docbank": (args.docbank_annotations, args.docbank_images),
        "tablebank": (args.tablebank_annotations, args.tablebank_images),
        "iiit_ar": (args.iiit_ar_annotations, args.iiit_ar_images),
    }
    for name, (ann, imgs) in candidates.items():
        src = _maybe_source(name, ann, imgs)
        if src is not None:
            sources[name] = src

    if not sources:
        raise RuntimeError(
            "Phase 4 requires at least one dataset; see Phase 2 flags."
        )

    # Use the Phase 2 weights by default unless the user overrides
    # by repeating --source-weight NAME FACTOR.
    from .config import phase2_config
    default_weights = phase2_config().source_weights
    actual_weights = {n: default_weights.get(n, 1.0) for n in sources}
    return MultiSourceDataset(sources, actual_weights), list(sources.keys())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocDet Phase 4: conditional targeted fine-tune."
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

    parser.add_argument("--phase2-checkpoint", type=Path, required=True,
                        help="Phase 2 checkpoint to warm-start from.")
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--save-dir", type=Path,
                        default=Path("layout_detector/weights"))
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small",
                        choices=["mobilenetv3_small", "efficientnet_b0"])

    parser.add_argument("--boost-class", action="append", default=[],
                        help="Class name to upweight. Repeatable.")
    parser.add_argument("--boost-factor", type=float, default=2.0,
                        help="Per-class loss multiplier for boost classes.")

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--num-samples-per-epoch", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = phase4_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.save_dir = args.save_dir
    cfg.resume_from = args.resume_from
    logger.info("Phase 4 config: %s", cfg)

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
        pretrained=False,  # we always warm-start from Phase 2 here
    )

    logger.info("Loading Phase 2 checkpoint %s", args.phase2_checkpoint)
    payload = torch.load(args.phase2_checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Phase 2 loaded (missing=%d unexpected=%d)",
                len(missing), len(unexpected))

    class_weights = _build_class_weights(args.boost_class, args.boost_factor)
    criterion = DocDetLoss(
        num_classes=NUM_DOCDET_CLASSES,
        cls_weight=cfg.cls_weight,
        reg_weight=cfg.reg_weight,
        cent_weight=cfg.cent_weight,
        class_weights=class_weights,
    )

    trainer = Trainer(model, criterion, loader, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
