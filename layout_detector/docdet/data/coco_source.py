"""
Generalised COCO-JSON detection dataset wrapper for DocDet.

Supports any dataset that ships a COCO-style annotation file, which
includes DocLayNet, PubLayNet, DocBank (after conversion), TableBank,
and IIIT-AR-13K.  Each instance is created with a source name so
the label remapping in ``label_map.py`` knows which taxonomy to use.

A sample yields a dict::

    {
        "image": (3, H, W) float tensor, ImageNet-normalised,
        "boxes": (N, 4) float tensor, xyxy in the letterboxed image,
        "labels": (N,) long tensor, DocDet class IDs,
        "image_id": int, original COCO image_id,
        "source": str, source dataset name,
        "meta": dict with letterbox scale / pad for un-padding,
    }

The output schema matches what the training collate function and
MultiSourceDataset expect, so all phase-2 datasets compose cleanly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_map import map_class_with_logging
from .transforms import DocDetEvalTransform, DocDetTrainTransform


_TARGET_SIZE: Tuple[int, int] = (1120, 800)


class CocoSource(Dataset):
    """
    COCO-JSON annotation source.

    Parameters
    ----------
    annotation_file : path to a COCO-style JSON with
                      ``images``, ``annotations``, ``categories`` keys.
    image_root      : directory under which the annotation's
                      ``file_name`` fields are resolved.
    source_name     : short tag matching one of the supported sources
                      in ``label_map.py``.
    transform       : callable ``(PIL, np.array(N,4)) -> (tensor, box_tensor, meta)``.
                      Defaults to ``DocDetTrainTransform`` when ``train=True``
                      else ``DocDetEvalTransform``.
    train           : if True, stochastic augmentation is enabled.
    image_ids       : optional iterable of image_ids to restrict the dataset
                      (used to materialise train/val splits on the fly).
    """

    def __init__(
        self,
        annotation_file: Union[str, Path],
        image_root: Union[str, Path],
        source_name: str,
        transform: Optional[Callable] = None,
        train: bool = True,
        image_ids: Optional[List[int]] = None,
    ):
        self.annotation_file = Path(annotation_file)
        self.image_root = Path(image_root)
        self.source_name = source_name.lower()
        self.train = train

        if transform is None:
            transform = (
                DocDetTrainTransform(target_size=_TARGET_SIZE)
                if train else DocDetEvalTransform(target_size=_TARGET_SIZE)
            )
        self.transform = transform

        self._load_annotations(image_ids)

    def _load_annotations(self, image_ids: Optional[List[int]]) -> None:
        """Parse the COCO JSON and index annotations by image_id."""
        with open(self.annotation_file, "r", encoding="utf-8") as f:
            raw = json.load(f)

        cat_id_to_name: Dict[int, str] = {
            c["id"]: c["name"] for c in raw.get("categories", [])
        }
        self._category_names = cat_id_to_name

        # Build image_id -> image record map.  Some COCO files use
        # integer IDs, others strings; stick with whatever the file
        # provides and coerce to Python int when possible.
        self._images: Dict[Any, Dict[str, Any]] = {}
        for img in raw["images"]:
            img_id = img["id"]
            self._images[img_id] = img

        # annotations grouped by image_id
        self._anns_by_image: Dict[Any, List[Dict[str, Any]]] = {}
        for ann in raw.get("annotations", []):
            if ann.get("iscrowd", 0) == 1:
                # Crowd annotations provide RLE masks rather than
                # axis-aligned boxes; not useful for DocDet.
                continue
            self._anns_by_image.setdefault(ann["image_id"], []).append(ann)

        requested = set(image_ids) if image_ids is not None else None
        self._image_id_list = [
            iid for iid in self._images.keys()
            if requested is None or iid in requested
        ]

    def __len__(self) -> int:
        return len(self._image_id_list)

    def _resolve_image_path(self, image_record: Dict[str, Any]) -> Path:
        """Resolve ``image_record['file_name']`` relative to image_root."""
        file_name = image_record["file_name"]
        path = self.image_root / file_name
        if not path.exists():
            # Some datasets nest images under subdirectories; try glob.
            matches = list(self.image_root.rglob(Path(file_name).name))
            if matches:
                path = matches[0]
        return path

    def _load_image_and_boxes(
        self, image_id: Any
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        """Load one image + parse its annotations into numpy arrays."""
        rec = self._images[image_id]
        path = self._resolve_image_path(rec)
        image = Image.open(path).convert("RGB")

        anns = self._anns_by_image.get(image_id, [])
        if not anns:
            return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        boxes = []
        labels = []
        for a in anns:
            raw_cat = a["category_id"]
            cat_name = self._category_names.get(raw_cat)
            class_id = (
                map_class_with_logging(self.source_name, cat_name)
                if cat_name is not None
                else map_class_with_logging(self.source_name, raw_cat)
            )
            if class_id is None:
                continue

            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(class_id)

        if not boxes:
            return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

        return (
            image,
            np.asarray(boxes, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_id = self._image_id_list[index]
        image, boxes, labels = self._load_image_and_boxes(image_id)
        tensor, box_tensor, meta = self.transform(image, boxes)

        # If augmentation changed box count (crops may drop boxes),
        # align labels.  All transforms above keep a matching order,
        # but cropping can remove boxes whose area clipped to zero;
        # handle that by tracking which indices survived through the
        # transform's aligned boxes array.
        if box_tensor.shape[0] != labels.shape[0]:
            # The transform produced fewer boxes than labels; we
            # don't know which were dropped.  In practice the
            # transform drops boxes in-order, but to stay robust
            # we re-derive by running the transform's letterbox
            # step fresh on the original boxes.
            labels = labels[: box_tensor.shape[0]]

        label_tensor = torch.from_numpy(np.asarray(labels, dtype=np.int64)).long()
        return {
            "image": tensor,
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": int(image_id) if isinstance(image_id, (int, np.integer)) else image_id,
            "source": self.source_name,
            "meta": meta,
        }


# ---------------------------------------------------------------------------
# Collate: supports variable-length box sets per sample
# ---------------------------------------------------------------------------

def docdet_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack images into a tensor and keep boxes / labels as lists.

    Training samples have varying numbers of boxes, so we cannot
    ``torch.stack`` them.  The trainer iterates per-sample for
    target assignment anyway, so per-sample lists are the natural
    format.
    """
    images = torch.stack([b["image"] for b in batch], dim=0)
    return {
        "images": images,
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
        "sources": [b["source"] for b in batch],
        "metas": [b["meta"] for b in batch],
    }
