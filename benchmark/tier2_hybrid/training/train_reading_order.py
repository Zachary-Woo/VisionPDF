"""
Training script for the visual reading order head.

Uses DocLayNet images paired with reading order pseudo-labels generated
by LayoutReader (see generate_order_labels.py).  This gives the order
head real document images to learn visual features from, unlike the
original ReadingBank approach which lacked images.

The SAM encoder and detection head are frozen.  Only the reading
order transformer head is trained.

Prerequisites:
  - Generated order labels (run generate_order_labels.py first)
  - DocLayNet dataset (auto-downloads on first use)

Usage:
    python -m benchmark.tier2_hybrid.training.train_reading_order
    python -m benchmark.tier2_hybrid.training.train_reading_order --epochs 12 --batch-size 4
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import DEEPSEEK_OCR2_MODEL, DOCLAYNET_DIR, PROJECT_ROOT
from benchmark.tier2_hybrid.sam.encoder import load_sam_encoder
from benchmark.tier2_hybrid.sam.detector import MultiScaleSAMEncoder
from benchmark.tier2_hybrid.sam.order_head import ReadingOrderHead
from benchmark.tier2_hybrid.training.generate_order_labels import ORDER_LABELS_DIR

MODELS_DIR = PROJECT_ROOT / "models"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# DocLayNet + order labels dataset
# ---------------------------------------------------------------------------

class DocLayNetOrderDataset(Dataset):
    """
    Dataset that pairs DocLayNet page images with LayoutReader-generated
    reading order pseudo-labels for training the visual reading order head.

    Loads real document images so the SAM encoder can extract meaningful
    visual features, unlike position-only approaches.
    """

    def __init__(
        self,
        labels_path: Path,
        images_dir: Path,
        target_size: int = 1024,
        max_regions: int = 64,
    ):
        """
        Parameters
        ----------
        labels_path : Path
            JSON file from generate_order_labels.py.
        images_dir : Path
            DocLayNet PNG/ directory containing page images.
        target_size : int
            Image canvas size in pixels.
        max_regions : int
            Maximum number of regions per sample.
        """
        self.images_dir = images_dir
        self.target_size = target_size
        self.max_regions = max_regions

        if not labels_path.exists():
            raise FileNotFoundError(
                f"Order labels not found: {labels_path}\n"
                "Run generate_order_labels.py first:\n"
                "  python -m benchmark.tier2_hybrid.training.generate_order_labels"
            )

        with open(labels_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

        print(f"DocLayNetOrder ({labels_path.stem}): {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        file_name = sample["file_name"]
        boxes_xyxy = sample["boxes"]
        order = sample["order"]

        img_path = self.images_dir / file_name
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.target_size, self.target_size), (255, 255, 255))

        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        canvas.paste(image, (0, 0))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1)

        n = min(len(boxes_xyxy), self.max_regions)
        scaled_boxes = []
        scaled_order = []
        for i in range(n):
            bx = [c * scale for c in boxes_xyxy[i]]
            scaled_boxes.append(bx)
            scaled_order.append(order[i])

        if not scaled_boxes:
            scaled_boxes = [[0, 0, 1, 1]]
            scaled_order = [0]

        boxes_t = torch.tensor(scaled_boxes, dtype=torch.float32)
        orders_t = torch.tensor(scaled_order, dtype=torch.long)

        return {
            "image": img_tensor,
            "boxes": boxes_t,
            "orders": orders_t,
        }


def collate_order_fn(batch):
    """Custom collate for variable-length region lists."""
    images = torch.stack([b["image"] for b in batch])
    boxes = [b["boxes"] for b in batch]
    orders = [b["orders"] for b in batch]
    return {"images": images, "boxes": boxes, "orders": orders}


# ---------------------------------------------------------------------------
# Pairwise ranking loss
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(scores: torch.Tensor, gt_order: torch.Tensor) -> torch.Tensor:
    """
    Given predicted scores (N,) and ground-truth order indices (N,),
    penalise pairs where the score ordering disagrees with gt ordering.

    Uses a margin-based ranking loss: for every pair (i, j) where
    gt_order[i] < gt_order[j], we want scores[i] < scores[j].
    """
    N = scores.shape[0]
    if N <= 1:
        return torch.tensor(0.0, device=scores.device, requires_grad=True)

    idx_i, idx_j = torch.triu_indices(N, N, offset=1, device=scores.device)
    order_i = gt_order[idx_i]
    order_j = gt_order[idx_j]

    should_i_first = (order_i < order_j).float()
    should_j_first = (order_j < order_i).float()

    score_diff = scores[idx_i] - scores[idx_j]

    margin = 0.5
    loss_i_first = torch.clamp(margin + score_diff, min=0) * should_i_first
    loss_j_first = torch.clamp(margin - score_diff, min=0) * should_j_first

    loss = (loss_i_first + loss_j_first).mean()
    return loss


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    backbone: MultiScaleSAMEncoder,
    order_head: ReadingOrderHead,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> float:
    """Train the reading order head for one epoch."""
    order_head.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(loader):
        images = batch["images"].to(device, dtype=torch.float16)
        boxes_list = batch["boxes"]
        orders_list = batch["orders"]
        B = images.shape[0]

        with torch.no_grad():
            features = backbone(images)
        p3 = features["p3"].float()

        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for b in range(B):
            boxes = boxes_list[b].to(device)
            gt_order = orders_list[b].to(device)

            if boxes.shape[0] < 2:
                continue

            scores = order_head(
                p3[b:b+1], boxes, img_h=1024, img_w=1024,
            )
            loss = pairwise_ranking_loss(scores, gt_order)
            batch_loss = batch_loss + loss

        batch_loss = batch_loss / B

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(order_head.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        num_batches += 1

        if batch_idx % 100 == 0:
            print(f"  Epoch {epoch} [{batch_idx}/{len(loader)}] loss={batch_loss.item():.4f}")

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate_order(
    backbone: MultiScaleSAMEncoder,
    order_head: ReadingOrderHead,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Validate the order head.  Returns (avg_loss, kendall_tau_accuracy).
    """
    order_head.eval()
    total_loss = 0.0
    total_correct_pairs = 0
    total_pairs = 0
    num_batches = 0

    for batch in loader:
        images = batch["images"].to(device, dtype=torch.float16)
        boxes_list = batch["boxes"]
        orders_list = batch["orders"]
        B = images.shape[0]

        features = backbone(images)
        p3 = features["p3"].float()

        for b in range(B):
            boxes = boxes_list[b].to(device)
            gt_order = orders_list[b].to(device)

            if boxes.shape[0] < 2:
                continue

            scores = order_head(p3[b:b+1], boxes, 1024, 1024)
            loss = pairwise_ranking_loss(scores, gt_order)
            total_loss += loss.item()
            num_batches += 1

            pred_order = scores.argsort()
            N = len(pred_order)
            for i in range(N):
                for j in range(i + 1, N):
                    total_pairs += 1
                    gt_before = gt_order[i] < gt_order[j]
                    pred_before = scores[i] < scores[j]
                    if gt_before == pred_before:
                        total_correct_pairs += 1

    avg_loss = total_loss / max(num_batches, 1)
    pair_acc = total_correct_pairs / max(total_pairs, 1)
    return avg_loss, pair_acc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels-dir", type=Path, default=ORDER_LABELS_DIR,
        help="Directory with order label JSONs from generate_order_labels.py",
    )
    parser.add_argument(
        "--doclaynet-dir", type=Path, default=DOCLAYNET_DIR,
        help="Path to DocLayNet dataset root (for images)",
    )
    parser.add_argument(
        "--sam-model-path", type=str, default=DEEPSEEK_OCR2_MODEL,
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument(
        "--save-path", type=Path,
        default=MODELS_DIR / "sam_reading_order.pt",
    )
    args = parser.parse_args()

    train_labels = args.labels_dir / "train.json"
    val_labels = args.labels_dir / "val.json"
    images_dir = args.doclaynet_dir / "PNG"

    if not train_labels.exists():
        print(
            f"Order labels not found at {args.labels_dir}\n"
            "Generate them first:\n"
            "  python -m benchmark.tier2_hybrid.training.generate_order_labels"
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SAM encoder from {args.sam_model_path}...")
    sam_encoder = load_sam_encoder(args.sam_model_path, device)
    backbone = MultiScaleSAMEncoder(sam_encoder).to(device)
    backbone.eval()

    order_head = ReadingOrderHead().to(device)
    num_params = sum(p.numel() for p in order_head.parameters() if p.requires_grad)
    print(f"Reading order head parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    print("Loading DocLayNet order-labeled dataset...")
    train_ds = DocLayNetOrderDataset(train_labels, images_dir)
    val_ds = DocLayNetOrderDataset(val_labels, images_dir)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, collate_fn=collate_order_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_order_fn, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(
        order_head.parameters(), lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    best_pair_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            backbone, order_head, train_loader, optimizer, device, epoch,
        )
        scheduler.step()
        val_loss, pair_acc = validate_order(
            backbone, order_head, val_loader, device,
        )
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} -- "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"pair_acc={pair_acc:.4f}  time={elapsed:.0f}s"
        )

        if pair_acc > best_pair_acc:
            best_pair_acc = pair_acc
            torch.save(
                {"model_state_dict": order_head.state_dict(), "pair_acc": pair_acc},
                args.save_path,
            )
            print(f"  Saved best checkpoint -> {args.save_path}")

    print(f"\nTraining complete. Best pair accuracy: {best_pair_acc:.4f}")


if __name__ == "__main__":
    main()
