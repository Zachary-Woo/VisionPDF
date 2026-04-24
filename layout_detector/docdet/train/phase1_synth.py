"""
Phase 1: synthetic-document pretraining on DocSynth300K.

Runs ~20 epochs with the backbone's first two stages frozen so the
neck and head learn document layout structure while the backbone's
ImageNet-derived visual primitives stay intact.  Writes a checkpoint
to ``layout_detector/weights/docdet_phase1_*.pt`` for Phase 2 to
pick up.

Disk use warning
----------------
The full DocSynth300K dataset is ~113 GB.  Pass
``--cleanup-after`` to ``shutil.rmtree`` the cache when training
completes.  Pass ``--stream`` to use HuggingFace streaming mode and
avoid the download entirely (slower; dependent on network).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from torch.utils.data import DataLoader

from ..data.coco_source import docdet_collate
from ..data.docsynth_source import DocSynthSource, DocSynthStreamingSource
from ..data.download import cleanup_docsynth, ensure_docsynth
from ..data.label_map import NUM_DOCDET_CLASSES
from ..model.loss import DocDetLoss
from ..model.model import DocDet
from .config import phase1_config
from .trainer import Trainer

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocDet Phase 1: DocSynth300K synthetic pretraining.",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream from HuggingFace Hub instead of downloading "
             "the full parquet cache locally.",
    )
    parser.add_argument(
        "--stream-take", type=int, default=None,
        help="When streaming, cap number of samples per epoch (for smoke tests).",
    )
    parser.add_argument(
        "--cache-root", type=Path, default=None,
        help="Optional override for the download cache location.",
    )
    parser.add_argument(
        "--cleanup-after", action="store_true",
        help="After training completes, remove the downloaded DocSynth cache.",
    )
    parser.add_argument(
        "--save-dir", type=Path,
        default=Path("layout_detector/weights"),
    )
    parser.add_argument(
        "--backbone", type=str, default="mobilenetv3_small",
        choices=["mobilenetv3_small", "efficientnet_b0"],
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument(
        "--coco-init-from", type=Path, default=None,
        help="Optional Phase 0 checkpoint to initialise from (head will be dropped).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    cfg = phase1_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.save_dir = args.save_dir
    cfg.resume_from = args.resume_from

    logger.info("Phase 1 config: %s", cfg)

    if args.stream:
        dataset = DocSynthStreamingSource(take=args.stream_take)
        # An IterableDataset is iterated by the DataLoader directly;
        # shuffle and sampler are both ignored.  We rely on the HF
        # streaming shuffle.
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            num_workers=0,  # streaming is CPU-single-threaded anyway
            collate_fn=docdet_collate,
            pin_memory=True,
        )
    else:
        cache_path = ensure_docsynth(args.cache_root)
        dataset = DocSynthSource(parquet_files=cache_path)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=docdet_collate,
            pin_memory=True,
        )
        logger.info("DocSynth300K dataset size: %d", len(dataset))

    model = DocDet(
        num_classes=NUM_DOCDET_CLASSES,
        backbone_name=args.backbone,
        pretrained=True,
    )

    if args.coco_init_from is not None:
        _load_backbone_and_neck_only(model, args.coco_init_from)

    criterion = DocDetLoss(
        num_classes=NUM_DOCDET_CLASSES,
        cls_weight=cfg.cls_weight,
        reg_weight=cfg.reg_weight,
        cent_weight=cfg.cent_weight,
    )

    trainer = Trainer(model, criterion, loader, cfg)
    trainer.train()

    if args.cleanup_after and not args.stream:
        cleanup_docsynth(args.cache_root)
        logger.info("DocSynth cache removed.")


def _load_backbone_and_neck_only(model: DocDet, ckpt_path: Path) -> None:
    """
    Load backbone + neck weights from a checkpoint, skipping the head.

    Used to migrate from Phase 0 (80-class COCO) to Phase 1 (11-class
    doc).  The head has different ``cls_pred`` shapes so would fail
    a full ``load_state_dict``; we filter it out instead.
    """
    import torch
    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload.get("model", payload)

    filtered = {
        k: v for k, v in state.items()
        if not k.startswith("head.cls_pred")
        and not k.startswith("head.reg_pred")
        and not k.startswith("head.centerness_pred")
    }
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    logger.info(
        "loaded %d weights from %s (skipped %d head weights)",
        len(filtered), ckpt_path, len(state) - len(filtered),
    )


if __name__ == "__main__":
    main()
