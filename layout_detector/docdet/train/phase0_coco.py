"""
Phase 0 (optional): COCO 80-class detection warmup for DocDet.

This phase is OPTIONAL and skipped by default.  It temporarily
re-heads the detector with ``num_classes=80``, trains briefly on
COCO natural images to sharpen the backbone and neck on figure-like
content, and produces a warm-started checkpoint that Phase 1 can
pick up (head-only, since DocLayNet's 11 classes don't overlap with
COCO's 80).

Only run this if:
* Your downstream documents contain many embedded natural images
  (product photos, screenshots, illustrations) where the ImageNet-
  pretrained backbone alone isn't sufficient.
* You have the time / compute budget; expected cost is ~1 GPU day
  on a single V100.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.coco_source import CocoSource, docdet_collate
from ..data.transforms import DocDetEvalTransform, DocDetTrainTransform
from ..model.loss import DocDetLoss
from ..model.model import DocDet
from .config import phase0_config
from .trainer import Trainer

logger = logging.getLogger(__name__)


def build_coco_source(
    annotations: Path, images: Path, train: bool
) -> CocoSource:
    """
    Thin wrapper around CocoSource that uses 80-class COCO IDs
    directly (no mapping through the DocDet label_map).

    We override ``source_name`` to a sentinel the label_map does not
    recognise, so the map returns None for every category; then we
    monkey-patch the dataset to use raw COCO IDs.  Alternative:
    could add 'coco' to the label_map, but that would pollute the
    document-only namespace.
    """
    target = (640, 640)
    transform = (
        DocDetTrainTransform(target_size=target)
        if train
        else DocDetEvalTransform(target_size=target)
    )

    class CocoDirectSource(CocoSource):
        """COCO source that keeps the original 0..79 category ids."""

        def _load_image_and_boxes(self, image_id):
            rec = self._images[image_id]
            path = self._resolve_image_path(rec)
            from PIL import Image
            import numpy as np
            image = Image.open(path).convert("RGB")
            anns = self._anns_by_image.get(image_id, [])
            if not anns:
                return (
                    image,
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                )
            boxes = []
            labels = []
            for a in anns:
                cat_id = int(a["category_id"])
                x, y, w, h = a["bbox"]
                if w <= 0 or h <= 0:
                    continue
                # COCO category_ids are 1..90 (not contiguous); map
                # to 0..79 by rank in the categories list.
                ordered = sorted(self._category_names.keys())
                if cat_id not in ordered:
                    continue
                labels.append(ordered.index(cat_id))
                boxes.append([x, y, x + w, y + h])
            if not boxes:
                return (
                    image,
                    np.zeros((0, 4), dtype=np.float32),
                    np.zeros((0,), dtype=np.int64),
                )
            return image, np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    return CocoDirectSource(
        annotation_file=annotations,
        image_root=images,
        source_name="coco",  # label_map will ignore
        transform=transform,
        train=train,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DocDet Phase 0: optional COCO warmup (80-class).",
    )
    parser.add_argument(
        "--coco-annotations", type=Path, required=True,
        help="Path to COCO train2017 instances JSON.",
    )
    parser.add_argument(
        "--coco-images", type=Path, required=True,
        help="Path to COCO train2017 image directory.",
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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    cfg = phase0_config()
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.save_dir = args.save_dir
    cfg.resume_from = args.resume_from

    logger.info("Phase 0 config: %s", cfg)

    train_source = build_coco_source(
        annotations=args.coco_annotations,
        images=args.coco_images,
        train=True,
    )
    loader = DataLoader(
        train_source,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=docdet_collate,
        pin_memory=True,
    )

    model = DocDet(num_classes=80, backbone_name=args.backbone, pretrained=True)
    criterion = DocDetLoss(
        num_classes=80,
        cls_weight=cfg.cls_weight,
        reg_weight=cfg.reg_weight,
        cent_weight=cfg.cent_weight,
    )
    trainer = Trainer(model, criterion, loader, cfg)
    trainer.train()


if __name__ == "__main__":
    main()
