"""
Training script for the SAM + FPN + FCOS detection head on DocLayNet.

The SAM ViT-B encoder is frozen (loaded from DeepSeek OCR 2 weights).
Only the FPN and FCOS head parameters are trained.

Usage:
    python -m benchmark.tier2_hybrid.training.train_sam_detector \
        --model-path deepseek-ai/DeepSeek-OCR-2 \
        --epochs 12 \
        --batch-size 4 \
        --lr 1e-4
"""

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import DEEPSEEK_OCR2_MODEL, PROJECT_ROOT
from benchmark.tier2_hybrid.training.doclaynet_dataset import DocLayNetDataset, collate_fn
from benchmark.tier2_hybrid.sam.encoder import load_sam_encoder
from benchmark.tier2_hybrid.sam.detector import (
    NUM_CLASSES,
    SAMDetector,
    compute_fcos_targets,
    focal_loss,
    giou_loss,
)


MODELS_DIR = PROJECT_ROOT / "models"


def train_one_epoch(
    model: SAMDetector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """
    Run one training epoch.  Returns average total loss.
    """
    model.fpn.train()
    model.head.train()

    total_loss_sum = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, dtype=torch.float16)
        gt_boxes_list = batch["boxes"]
        gt_labels_list = batch["labels"]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(images)

        batch_cls_loss = 0.0
        batch_reg_loss = 0.0
        batch_cent_loss = 0.0
        B = images.shape[0]

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)

            fpn_levels = {}
            for level_name, (cls_logits, bbox_pred, centerness) in outputs.items():
                _, C, H, W = cls_logits.shape
                fpn_levels[level_name] = (C, H, W)

            targets = compute_fcos_targets(
                gt_boxes, gt_labels, fpn_levels,
                SAMDetector.STRIDES, SAMDetector.LEVEL_RANGES,
            )

            for level_name in outputs:
                cls_logits, bbox_pred, centerness = outputs[level_name]
                cls_target, reg_target, cent_target = targets[level_name]

                cls_pred_flat = cls_logits[b].permute(1, 2, 0).reshape(-1, NUM_CLASSES)
                cls_target_flat = cls_target.reshape(-1)
                batch_cls_loss = batch_cls_loss + focal_loss(
                    cls_pred_flat.float(), cls_target_flat
                )

                fg_mask = cls_target.reshape(-1) > 0
                if fg_mask.any():
                    pred_ltrb = bbox_pred[b].permute(1, 2, 0).reshape(-1, 4)[fg_mask]
                    tgt_ltrb = reg_target.reshape(-1, 4)[fg_mask]
                    batch_reg_loss = batch_reg_loss + giou_loss(
                        pred_ltrb.float(), tgt_ltrb
                    )

                    pred_cent = centerness[b, 0].reshape(-1)[fg_mask]
                    tgt_cent = cent_target.reshape(-1)[fg_mask]
                    batch_cent_loss = batch_cent_loss + F.binary_cross_entropy_with_logits(
                        pred_cent.float(), tgt_cent, reduction="mean"
                    )

        loss = (batch_cls_loss + batch_reg_loss + batch_cent_loss) / B

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.fpn.parameters()) + list(model.head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        total_loss_sum += loss.item()
        num_batches += 1

        if batch_idx % 50 == 0:
            print(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={loss.item():.4f} "
                f"(cls={batch_cls_loss.item()/B:.4f} "
                f"reg={batch_reg_loss.item()/B if isinstance(batch_reg_loss, torch.Tensor) else 0:.4f} "
                f"cent={batch_cent_loss.item()/B if isinstance(batch_cent_loss, torch.Tensor) else 0:.4f})"
            )

    return total_loss_sum / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: SAMDetector,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """
    Simple validation pass computing average focal loss on the val set.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        images = batch["images"].to(device, dtype=torch.float16)
        gt_boxes_list = batch["boxes"]
        gt_labels_list = batch["labels"]

        with torch.cuda.amp.autocast(dtype=torch.float16):
            outputs = model(images)

        B = images.shape[0]
        batch_loss = 0.0

        for b in range(B):
            gt_boxes = gt_boxes_list[b].to(device)
            gt_labels = gt_labels_list[b].to(device)

            fpn_levels = {}
            for level_name, (cls_logits, bbox_pred, centerness) in outputs.items():
                _, C, H, W = cls_logits.shape
                fpn_levels[level_name] = (C, H, W)

            targets = compute_fcos_targets(
                gt_boxes, gt_labels, fpn_levels,
                SAMDetector.STRIDES, SAMDetector.LEVEL_RANGES,
            )

            for level_name in outputs:
                cls_logits, bbox_pred, centerness = outputs[level_name]
                cls_target, reg_target, cent_target = targets[level_name]
                cls_pred_flat = cls_logits[b].permute(1, 2, 0).reshape(-1, NUM_CLASSES)
                cls_target_flat = cls_target.reshape(-1)
                batch_loss = batch_loss + focal_loss(
                    cls_pred_flat.float(), cls_target_flat
                ).item()

        total_loss += batch_loss / B
        num_batches += 1

    return total_loss / max(num_batches, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=str, default=DEEPSEEK_OCR2_MODEL,
        help="HuggingFace model ID or local path for SAM encoder weights",
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument(
        "--save-path", type=Path, default=MODELS_DIR / "sam_doclaynet_head.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading SAM encoder from {args.model_path}...")
    sam_encoder = load_sam_encoder(args.model_path, device)

    detector = SAMDetector(sam_encoder).to(device)

    for param in detector.backbone.parameters():
        param.requires_grad = False

    trainable_params = (
        list(detector.fpn.parameters()) + list(detector.head.parameters())
    )
    num_params = sum(p.numel() for p in trainable_params if p.requires_grad)
    print(f"Trainable parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    print("Loading DocLayNet train/val splits...")
    train_ds = DocLayNetDataset("train", target_size=1024, augment=True, data_dir=args.data_dir)
    val_ds = DocLayNetDataset("val", target_size=1024, augment=False, data_dir=args.data_dir)
    print(f"  Train: {len(train_ds)} images")
    print(f"  Val:   {len(val_ds)} images")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_fn, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=collate_fn, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader),
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(detector, train_loader, optimizer, device, epoch)
        scheduler.step()
        val_loss = validate(detector, val_loader, device)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} -- "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"time={elapsed:.0f}s  lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                "epoch": epoch,
                "fpn_state_dict": detector.fpn.state_dict(),
                "head_state_dict": detector.head.state_dict(),
                "val_loss": val_loss,
            }
            torch.save(checkpoint, args.save_path)
            print(f"  Saved best checkpoint -> {args.save_path}")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}")
    print(f"Checkpoint: {args.save_path}")


if __name__ == "__main__":
    main()
