"""
Training script for the visual reading order head.

Uses ReadingBank (Microsoft) which provides word-level reading order
annotations for ~500K document pages.  Words are grouped into
paragraph-level blocks by spatial proximity, and the block-level
reading order is derived from the word sequence.

The SAM encoder and detection head are frozen.  Only the reading
order transformer head is trained.

Prerequisites:
  - ReadingBank dataset downloaded.  Get it from:
    https://aka.ms/readingbank
    Expected structure:
      ReadingBank/
        train/
          *.jsonl   (each line: {"src": image_path, "tgt": word_list})
        val/
          *.jsonl

  - A trained SAM detection head checkpoint (from train_sam_detector.py)

Usage:
    python -m benchmark.tier2_hybrid.training.train_reading_order \
        --readingbank-dir ReadingBank \
        --sam-model-path deepseek-ai/DeepSeek-OCR-2 \
        --epochs 8
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import DEEPSEEK_OCR2_MODEL, PROJECT_ROOT
from benchmark.tier2_hybrid.sam.encoder import load_sam_encoder
from benchmark.tier2_hybrid.sam.detector import MultiScaleSAMEncoder
from benchmark.tier2_hybrid.sam.order_head import ReadingOrderHead

MODELS_DIR = PROJECT_ROOT / "models"

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# ReadingBank dataset -- groups words into blocks for region-level order
# ---------------------------------------------------------------------------

def _group_words_to_blocks(
    words: List[Dict],
    y_tolerance: float = 10.0,
    x_gap_tolerance: float = 50.0,
) -> List[Dict]:
    """
    Group word-level annotations into paragraph blocks by spatial proximity.

    Each word dict has: {"text": str, "bbox": [x1, y1, x2, y2]}
    Returns blocks with: {"bbox": [x1, y1, x2, y2], "order": int}
    where order is the position of the block's first word in the
    reading sequence.
    """
    if not words:
        return []

    lines: List[List[Dict]] = []
    current_line = [words[0]]

    for w in words[1:]:
        prev = current_line[-1]
        prev_cy = (prev["bbox"][1] + prev["bbox"][3]) / 2
        cur_cy = (w["bbox"][1] + w["bbox"][3]) / 2
        if abs(cur_cy - prev_cy) < y_tolerance:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
    lines.append(current_line)

    blocks: List[List[List[Dict]]] = [[lines[0]]]
    for line in lines[1:]:
        prev_block_last_line = blocks[-1][-1]
        prev_bottom = max(w["bbox"][3] for w in prev_block_last_line)
        cur_top = min(w["bbox"][1] for w in line)
        gap = cur_top - prev_bottom

        line_height = max(
            max(w["bbox"][3] - w["bbox"][1] for w in line),
            max(w["bbox"][3] - w["bbox"][1] for w in prev_block_last_line),
        )
        if gap < line_height * 1.5:
            blocks[-1].append(line)
        else:
            blocks.append([line])

    result = []
    for block_idx, block_lines in enumerate(blocks):
        all_words_in_block = [w for line in block_lines for w in line]
        x1 = min(w["bbox"][0] for w in all_words_in_block)
        y1 = min(w["bbox"][1] for w in all_words_in_block)
        x2 = max(w["bbox"][2] for w in all_words_in_block)
        y2 = max(w["bbox"][3] for w in all_words_in_block)
        result.append({"bbox": [x1, y1, x2, y2], "order": block_idx})

    return result


class ReadingBankDataset(Dataset):
    """
    Dataset that loads ReadingBank JSONL files and produces
    (image_tensor, block_boxes, block_order) tuples for training
    the reading order head.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        target_size: int = 1024,
        max_blocks: int = 64,
    ):
        self.target_size = target_size
        self.max_blocks = max_blocks
        self.samples: List[Dict] = []

        split_dir = data_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(
                f"ReadingBank split not found: {split_dir}\n"
                f"Download from https://aka.ms/readingbank"
            )

        for jsonl_file in sorted(split_dir.glob("*.jsonl")):
            with open(jsonl_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.samples.append(json.loads(line))

        if not self.samples:
            raise RuntimeError(f"No samples found in {split_dir}")

        print(f"ReadingBank {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        words = sample.get("tgt", [])
        if isinstance(words, str):
            try:
                words = json.loads(words)
            except json.JSONDecodeError:
                words = []

        parsed_words = []
        for w in words:
            if isinstance(w, dict) and "bbox" in w:
                parsed_words.append(w)
            elif isinstance(w, (list, tuple)) and len(w) >= 5:
                parsed_words.append({
                    "text": str(w[0]),
                    "bbox": [float(w[1]), float(w[2]), float(w[3]), float(w[4])],
                })

        blocks = _group_words_to_blocks(parsed_words)

        img_path = sample.get("src", "")
        try:
            from PIL import Image
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (1024, 1024), (255, 255, 255))

        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        from PIL import Image as PILImage
        canvas = PILImage.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        canvas.paste(image, (0, 0))

        arr = np.array(canvas, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        img_tensor = torch.from_numpy(arr).permute(2, 0, 1)

        block_boxes = []
        block_orders = []
        for block in blocks[:self.max_blocks]:
            bx = [c * scale for c in block["bbox"]]
            block_boxes.append(bx)
            block_orders.append(block["order"])

        if not block_boxes:
            block_boxes = [[0, 0, 1, 1]]
            block_orders = [0]

        boxes_t = torch.tensor(block_boxes, dtype=torch.float32)
        orders_t = torch.tensor(block_orders, dtype=torch.long)

        return {
            "image": img_tensor,
            "boxes": boxes_t,
            "orders": orders_t,
        }


def collate_order_fn(batch):
    """Custom collate for variable-length block lists."""
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
        "--readingbank-dir", type=Path,
        default=PROJECT_ROOT / "ReadingBank",
        help="Path to ReadingBank dataset root",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SAM encoder from {args.sam_model_path}...")
    sam_encoder = load_sam_encoder(args.sam_model_path, device)
    backbone = MultiScaleSAMEncoder(sam_encoder).to(device)
    backbone.eval()

    order_head = ReadingOrderHead().to(device)
    num_params = sum(p.numel() for p in order_head.parameters() if p.requires_grad)
    print(f"Reading order head parameters: {num_params:,} ({num_params/1e6:.1f}M)")

    print(f"Loading ReadingBank from {args.readingbank_dir}...")
    train_ds = ReadingBankDataset(args.readingbank_dir, split="train")
    val_ds = ReadingBankDataset(args.readingbank_dir, split="val")

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
        optimizer, T_max=args.epochs * len(train_loader),
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
