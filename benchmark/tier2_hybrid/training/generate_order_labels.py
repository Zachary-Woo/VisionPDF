"""
Generate reading order pseudo-labels for DocLayNet using LayoutReader.

Takes DocLayNet COCO annotations (bounding boxes per page), runs the
pre-trained LayoutReader model to predict reading order, and caches
the results as JSON files for use by the visual reading order head
training script.

This eliminates the need for the ReadingBank dataset -- DocLayNet
provides real document images and LayoutReader provides reading order
supervision.  The visual order head then learns whether SAM encoder
features add value beyond what positional information alone gives.

Prerequisites:
  - DocLayNet dataset (auto-downloads ~28 GB on first run)
  - LayoutReader model (auto-downloads from hantian/layoutreader)

Usage:
    python -m benchmark.tier2_hybrid.training.generate_order_labels
    python -m benchmark.tier2_hybrid.training.generate_order_labels --splits train val
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import LayoutLMv3ForTokenClassification

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import (
    DOCLAYNET_DIR,
    DOCLAYNET_LABELS,
    LAYOUTREADER_MODEL,
    MODELS_DIR,
    PROJECT_ROOT,
)
from benchmark.tier2_hybrid.training.doclaynet_dataset import (
    _load_coco_annotations,
    download_doclaynet,
)
from benchmark.tier2_hybrid.shared import (
    _boxes2inputs,
    _parse_logits,
    _prepare_inputs,
)

ORDER_LABELS_DIR = MODELS_DIR / "order_labels"

_LABEL_TO_IDX = {name: i for i, name in enumerate(DOCLAYNET_LABELS)}


def _coco_to_xyxy(ann: Dict) -> List[float]:
    """Convert COCO [x, y, w, h] bbox to [x1, y1, x2, y2]."""
    x, y, w, h = ann["bbox"]
    return [x, y, x + w, y + h]


def _scale_boxes_to_1000(
    boxes_xyxy: List[List[float]], img_w: int, img_h: int,
) -> List[List[int]]:
    """Scale pixel-coordinate boxes to the 0-1000 range LayoutReader expects."""
    sx = 1000.0 / img_w
    sy = 1000.0 / img_h
    scaled = []
    for x1, y1, x2, y2 in boxes_xyxy:
        scaled.append([
            max(0, min(1000, round(x1 * sx))),
            max(0, min(1000, round(y1 * sy))),
            max(0, min(1000, round(x2 * sx))),
            max(0, min(1000, round(y2 * sy))),
        ])
    return scaled


def generate_labels_for_split(
    split: str,
    data_dir: Path,
    lr_model: LayoutLMv3ForTokenClassification,
    output_dir: Path,
) -> Path:
    """
    Generate reading order labels for one DocLayNet split.

    For each image, collects all annotation bounding boxes, runs
    LayoutReader, and saves the predicted order alongside the boxes
    and class labels.

    Returns the path to the output JSON file.
    """
    json_path = data_dir / "COCO" / f"{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"DocLayNet annotation not found: {json_path}\n"
            f"Run download_doclaynet() or place the dataset under {data_dir}"
        )

    images, cat_id_to_name, img_id_to_anns = _load_coco_annotations(json_path)

    cat_id_to_idx = {}
    for coco_id, name in cat_id_to_name.items():
        if name in _LABEL_TO_IDX:
            cat_id_to_idx[coco_id] = _LABEL_TO_IDX[name]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split}.json"

    results = []
    skipped = 0

    for img_info in tqdm(images, desc=f"Labeling {split}"):
        img_id = img_info["id"]
        file_name = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        anns = img_id_to_anns.get(img_id, [])

        boxes_xyxy = []
        labels = []
        for ann in anns:
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in cat_id_to_idx:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes_xyxy.append([x, y, x + w, y + h])
            labels.append(cat_id_to_idx[coco_cat_id])

        if len(boxes_xyxy) < 2:
            skipped += 1
            continue

        boxes_1000 = _scale_boxes_to_1000(boxes_xyxy, img_w, img_h)

        truncated = boxes_1000[:510]
        inputs = _boxes2inputs(truncated)
        inputs = _prepare_inputs(inputs, lr_model)

        with torch.no_grad():
            logits = lr_model(**inputs).logits.cpu().squeeze(0)
        order = _parse_logits(logits, len(truncated))

        if len(boxes_xyxy) > 510:
            order.extend(range(len(order), len(boxes_xyxy)))

        results.append({
            "file_name": file_name,
            "img_w": img_w,
            "img_h": img_h,
            "boxes": boxes_xyxy,
            "labels": labels,
            "order": order,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    print(f"  {split}: {len(results)} images labeled, {skipped} skipped (<2 regions)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--doclaynet-dir", type=Path, default=DOCLAYNET_DIR,
        help="Path to DocLayNet dataset root",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ORDER_LABELS_DIR,
        help="Directory to write order label JSON files",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Which DocLayNet splits to label (default: train val)",
    )
    parser.add_argument(
        "--layoutreader-model", type=str, default=LAYOUTREADER_MODEL,
        help="LayoutReader HuggingFace model ID or local path",
    )
    args = parser.parse_args()

    if not (args.doclaynet_dir / "COCO" / "train.json").exists():
        print("DocLayNet not found, downloading...")
        download_doclaynet(args.doclaynet_dir)

    print(f"Loading LayoutReader from {args.layoutreader_model}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.layoutreader_model
    ).to(device)
    lr_model.eval()
    print(f"  Device: {device}")

    for split in args.splits:
        print(f"Generating order labels for {split}...")
        out_path = generate_labels_for_split(
            split, args.doclaynet_dir, lr_model, args.output_dir,
        )
        print(f"  Saved to {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
