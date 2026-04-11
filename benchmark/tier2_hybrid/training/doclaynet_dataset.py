"""
DocLayNet COCO-format dataset for training the SAM detection head.

Downloads the DocLayNet dataset from HuggingFace (ds4sd/DocLayNet) and
provides a PyTorch Dataset with standard detection augmentations.

DocLayNet COCO format:
  - 11 categories matching benchmark.config.DOCLAYNET_LABELS
  - Annotations have "bbox" in [x, y, w, h] format and "category_id"
  - Images are document page PNGs at ~1025x1025

Usage:
    from benchmark.tier2_hybrid.training.doclaynet_dataset import DocLayNetDataset
    ds = DocLayNetDataset("train")
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from benchmark.config import DOCLAYNET_LABELS, PROJECT_ROOT

DOCLAYNET_DIR = PROJECT_ROOT / "DocLayNet"

_LABEL_TO_IDX = {name: i for i, name in enumerate(DOCLAYNET_LABELS)}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def download_doclaynet(target_dir: Path) -> Path:
    """
    Download DocLayNet COCO dataset from HuggingFace if not already present.

    Expected structure after download:
      DocLayNet/
        COCO/
          train.json
          val.json
          test.json
        PNG/
          <image files>
    """
    if (target_dir / "COCO" / "train.json").exists():
        print(f"DocLayNet already present at {target_dir}")
        return target_dir

    print("Downloading DocLayNet from HuggingFace (ds4sd/DocLayNet)...")
    print("This may take a while (~30GB for full dataset).")
    print("Alternatively, download manually and place under:", target_dir)

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            "ds4sd/DocLayNet",
            repo_type="dataset",
            local_dir=str(target_dir),
            allow_patterns=["COCO/*.json", "PNG/**"],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download DocLayNet: {e}\n"
            f"Please download manually from https://huggingface.co/datasets/ds4sd/DocLayNet\n"
            f"and place under {target_dir}"
        ) from e

    return target_dir


def _load_coco_annotations(json_path: Path) -> Tuple[List[Dict], Dict[int, str], Dict[int, List[Dict]]]:
    """
    Parse a COCO annotation JSON and return:
      - images: list of image dicts
      - cat_id_to_name: mapping from COCO category_id to name
      - img_id_to_anns: mapping from image_id to list of annotation dicts
    """
    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cat_id_to_name = {}
    for cat in coco.get("categories", []):
        cat_id_to_name[cat["id"]] = cat["name"]

    img_id_to_anns: Dict[int, List[Dict]] = {}
    for ann in coco.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    return coco["images"], cat_id_to_name, img_id_to_anns


class DocLayNetDataset(Dataset):
    """
    PyTorch Dataset for DocLayNet detection training.

    Parameters
    ----------
    split : str
        One of "train", "val", "test".
    target_size : int
        Images are resized to (target_size, target_size) with letterboxing.
    augment : bool
        Apply training augmentations (random resize, flip, colour jitter).
    data_dir : Path
        Root directory containing DocLayNet COCO/ and PNG/ folders.
    """

    def __init__(
        self,
        split: str = "train",
        target_size: int = 1024,
        augment: bool = True,
        data_dir: Optional[Path] = None,
    ):
        self.target_size = target_size
        self.augment = augment and (split == "train")

        data_dir = data_dir or DOCLAYNET_DIR
        json_path = data_dir / "COCO" / f"{split}.json"

        if not json_path.exists():
            download_doclaynet(data_dir)

        if not json_path.exists():
            raise FileNotFoundError(
                f"DocLayNet annotation not found: {json_path}\n"
                f"Run download_doclaynet() or place the dataset under {data_dir}"
            )

        self.images_dir = data_dir / "PNG"
        self.image_list, self.cat_id_to_name, self.img_id_to_anns = (
            _load_coco_annotations(json_path)
        )

        self.cat_id_to_idx = {}
        for coco_id, name in self.cat_id_to_name.items():
            if name in _LABEL_TO_IDX:
                self.cat_id_to_idx[coco_id] = _LABEL_TO_IDX[name]

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict with keys:
          image  : (3, target_size, target_size) float32 tensor, normalised
          boxes  : (N, 4) float32 tensor in xyxy pixel coords (in target_size space)
          labels : (N,) long tensor of 0-based class indices
          scale  : float -- resize scale applied to original image
        """
        img_info = self.image_list[idx]
        img_id = img_info["id"]
        file_name = img_info["file_name"]

        image = Image.open(self.images_dir / file_name).convert("RGB")

        anns = self.img_id_to_anns.get(img_id, [])
        boxes = []
        labels = []
        for ann in anns:
            coco_cat_id = ann["category_id"]
            if coco_cat_id not in self.cat_id_to_idx:
                continue
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(self.cat_id_to_idx[coco_cat_id])

        boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.array(labels, dtype=np.int64)

        if self.augment:
            image, boxes = self._augment(image, boxes)

        image, boxes, scale = self._resize_and_pad(image, boxes)

        arr = np.array(image, dtype=np.float32) / 255.0
        arr = (arr - _MEAN) / _STD
        tensor = torch.from_numpy(arr).permute(2, 0, 1)

        boxes_t = torch.from_numpy(boxes).float()
        labels_t = torch.from_numpy(labels).long()

        return {
            "image": tensor,
            "boxes": boxes_t,
            "labels": labels_t,
            "scale": scale,
        }

    def _augment(
        self, image: Image.Image, boxes: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray]:
        """Random horizontal flip and colour jitter."""
        w, h = image.size

        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if boxes.shape[0] > 0:
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes[:, 0] = x1
                boxes[:, 2] = x2

        arr = np.array(image, dtype=np.float32)
        brightness = random.uniform(0.8, 1.2)
        contrast = random.uniform(0.8, 1.2)
        arr = arr * contrast + (brightness - 1.0) * 128
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr)

        return image, boxes

    def _resize_and_pad(
        self, image: Image.Image, boxes: np.ndarray
    ) -> Tuple[Image.Image, np.ndarray, float]:
        """Resize maintaining aspect ratio and pad to target_size square."""
        w, h = image.size
        scale = self.target_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (self.target_size, self.target_size), (0, 0, 0))
        canvas.paste(image, (0, 0))

        if boxes.shape[0] > 0:
            boxes = boxes * scale
            boxes[:, 0] = np.clip(boxes[:, 0], 0, new_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, new_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, new_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, new_h)

        return canvas, boxes, scale


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate for variable-number-of-boxes per image.

    Returns
    -------
    dict with:
      images : (B, 3, H, W) stacked tensor
      boxes  : list of (Ni, 4) tensors
      labels : list of (Ni,) tensors
      scales : (B,) tensor
    """
    images = torch.stack([item["image"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    scales = torch.tensor([item["scale"] for item in batch])
    return {"images": images, "boxes": boxes, "labels": labels, "scales": scales}
