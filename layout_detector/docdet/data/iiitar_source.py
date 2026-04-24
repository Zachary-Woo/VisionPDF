"""
IIIT-AR-13K detection dataset source (Pascal VOC layout).

The Kaggle mirror of IIIT-AR-13K ships annotations as one Pascal VOC
``<filename>.xml`` per image plus a sibling images directory.  The
exact directory layout varies slightly between Kaggle revisions, so
this loader auto-discovers ``Annotations/`` and ``Images/`` folders
underneath the user-supplied root.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_map import map_class_with_logging
from .transforms import DocDetTrainTransform

logger = logging.getLogger(__name__)

_TARGET_SIZE: Tuple[int, int] = (1120, 800)
_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


def _find_subdir(root: Path, *names: str) -> Optional[Path]:
    """Case-insensitive search for any of ``names`` underneath ``root``."""
    targets = {n.lower() for n in names}
    for child in root.rglob("*"):
        if child.is_dir() and child.name.lower() in targets:
            return child
    return None


class IIITARVocSource(Dataset):
    """
    Pascal-VOC formatted IIIT-AR-13K loader.

    Parameters
    ----------
    root      : path to the unpacked Kaggle dataset (any depth).  The
                loader will locate the ``Annotations`` and ``Images``
                directories itself.
    split     : "train" | "val" | "test" - filters via the
                ``ImageSets/Main/<split>.txt`` file when present;
                otherwise loads every annotation found.
    transform : DocDet augmentation pipeline (defaults to train).
    """

    SOURCE_NAME = "iiit_ar"

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"IIIT-AR root not found: {root}")

        ann_dir = _find_subdir(root, "Annotations", "annotations", "xml")
        img_dir = _find_subdir(root, "Images", "images", "JPEGImages")
        if ann_dir is None or img_dir is None:
            raise FileNotFoundError(
                f"Could not auto-discover Annotations/ and Images/ under {root}. "
                f"Found ann={ann_dir}, img={img_dir}"
            )

        self.ann_dir = ann_dir
        self.img_dir = img_dir
        self.transform = transform or DocDetTrainTransform(target_size=_TARGET_SIZE)

        # Resolve split membership from ImageSets/Main/<split>.txt if present.
        split_ids: Optional[set] = None
        sets_dir = _find_subdir(root, "ImageSets", "imagesets")
        if sets_dir is not None:
            cand = sets_dir / "Main" / f"{split}.txt"
            if not cand.exists():
                cand = sets_dir / f"{split}.txt"
            if cand.exists():
                split_ids = {
                    line.strip().split()[0]
                    for line in cand.read_text().splitlines()
                    if line.strip()
                }

        # Build the (xml_path, image_path) pair list once at init.
        self._items: List[Tuple[Path, Path]] = []
        for xml_path in sorted(self.ann_dir.glob("*.xml")):
            stem = xml_path.stem
            if split_ids is not None and stem not in split_ids:
                continue
            img_path = self._resolve_image(stem)
            if img_path is None:
                continue
            self._items.append((xml_path, img_path))

        if not self._items:
            raise RuntimeError(
                f"IIITARVocSource(split={split}) resolved 0 items under {root}"
            )

    def _resolve_image(self, stem: str) -> Optional[Path]:
        """Find ``stem.<ext>`` in the images dir for any known extension."""
        for ext in _IMAGE_EXTS:
            cand = self.img_dir / f"{stem}{ext}"
            if cand.exists():
                return cand
        return None

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Dict:
        xml_path, img_path = self._items[index]
        boxes_np, labels_np = self._parse_voc(xml_path)
        image = Image.open(img_path).convert("RGB")
        tensor, box_tensor, meta = self.transform(image, boxes_np)

        if box_tensor.shape[0] != labels_np.shape[0]:
            labels_np = labels_np[: box_tensor.shape[0]]

        return {
            "image": tensor,
            "boxes": box_tensor,
            "labels": torch.from_numpy(np.asarray(labels_np, dtype=np.int64)).long(),
            "image_id": index,
            "source": self.SOURCE_NAME,
            "meta": meta,
        }

    @staticmethod
    def _parse_voc(xml_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse a single Pascal VOC XML file into (boxes xyxy, labels)."""
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes: List[List[float]] = []
        labels: List[int] = []
        for obj in root.findall("object"):
            name_node = obj.find("name")
            bnd = obj.find("bndbox")
            if name_node is None or bnd is None:
                continue
            try:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
            except ValueError:
                continue
            if xmax <= xmin or ymax <= ymin:
                continue
            mapped = map_class_with_logging("iiit_ar", name_node.text or "")
            if mapped is None:
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(mapped)

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        return (
            np.asarray(boxes, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
        )
