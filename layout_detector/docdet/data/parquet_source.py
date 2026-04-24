"""
Parquet-backed detection dataset sources for DocDet Phase 2.

The HuggingFace mirrors of DocLayNet, PubLayNet, and TableBank ship
their images and annotations as parquet columns rather than as
COCO-style ``annotations.json`` + ``images/`` directories.  This
module provides a single generic ``ParquetDetectionSource`` that
reads any of those parquets, delegating schema interpretation to a
small per-dataset adapter function.

Sample schema produced by every adapter matches ``CocoSource``::

    {
        "image"    : (3, H, W) float tensor,
        "boxes"    : (N, 4)    float tensor, xyxy in the letterboxed image,
        "labels"   : (N,)      long  tensor, DocDet class IDs,
        "image_id" : int,
        "source"   : str,
        "meta"     : dict,
    }
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .label_map import map_class_with_logging
from .transforms import DocDetTrainTransform

logger = logging.getLogger(__name__)

_TARGET_SIZE: Tuple[int, int] = (1120, 800)


# ---------------------------------------------------------------------------
# Image decoding helper
# ---------------------------------------------------------------------------

def _decode_image_cell(cell: Any) -> Image.Image:
    """
    Decode a parquet image cell into a PIL RGB image.

    HuggingFace stores images in a few different shapes:
      - raw bytes (most TableBank rows),
      - a dict ``{"bytes": <bytes>, "path": <str|None>}`` (PubLayNet /
        DocLayNet-v1.2 via the Image feature),
      - a plain base64 string in legacy dumps.
    """
    if isinstance(cell, dict):
        cell = cell.get("bytes") or cell.get("path")
    if cell is None:
        raise ValueError("Empty image cell in parquet row")
    if isinstance(cell, (bytes, bytearray)):
        return Image.open(io.BytesIO(cell)).convert("RGB")
    if isinstance(cell, str):
        # Could be a file path or a base64 string.  Try path first.
        p = Path(cell)
        if p.exists():
            return Image.open(p).convert("RGB")
        import base64
        return Image.open(io.BytesIO(base64.b64decode(cell))).convert("RGB")
    raise TypeError(f"Unsupported image cell type {type(cell)}")


# ---------------------------------------------------------------------------
# Per-dataset adapters
# ---------------------------------------------------------------------------
#
# Each adapter takes a dict {column_name: value} for one parquet row and
# returns (PIL image, boxes xyxy float32, labels int64).  They all share
# the DocDet class-id remapping via ``label_map.map_class_with_logging``.
# ---------------------------------------------------------------------------


def _adapter_doclaynet(row: Dict[str, Any]) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """docling-project/DocLayNet-v1.2 parquet schema.

    Columns used: ``image``, ``bboxes`` (list of [x,y,w,h]),
    ``category_id`` (list of int, 1..11 per DocLayNet taxonomy).
    """
    image = _decode_image_cell(row["image"])
    bboxes = row.get("bboxes") or []
    cat_ids = row.get("category_id") or []

    boxes: List[List[float]] = []
    labels: List[int] = []
    for xywh, cid in zip(bboxes, cat_ids):
        if xywh is None or len(xywh) != 4:
            continue
        x, y, w, h = (float(v) for v in xywh)
        if w <= 0 or h <= 0:
            continue
        mapped = map_class_with_logging("doclaynet", int(cid))
        if mapped is None:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(mapped)

    if not boxes:
        return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return image, np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _adapter_publaynet(row: Dict[str, Any]) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """jordanparker6/publaynet parquet schema.

    Columns used: ``image`` (Image feature), ``annotations`` (list of
    dicts with COCO-style ``bbox`` [x,y,w,h] and ``category_id``
    integer 1..5 matching text/title/list/table/figure).
    """
    image = _decode_image_cell(row["image"])
    anns = row.get("annotations") or []

    boxes: List[List[float]] = []
    labels: List[int] = []
    for a in anns:
        if a is None:
            continue
        if int(a.get("iscrowd", 0)) == 1:
            continue
        bbox = a.get("bbox")
        if bbox is None or len(bbox) != 4:
            continue
        x, y, w, h = (float(v) for v in bbox)
        if w <= 0 or h <= 0:
            continue
        mapped = map_class_with_logging("publaynet", int(a["category_id"]))
        if mapped is None:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(mapped)

    if not boxes:
        return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return image, np.asarray(boxes, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _adapter_tablebank(row: Dict[str, Any]) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    """deepcopy/TableBank-Detection parquet schema.

    Each row is a single (image, table-annotation) pair with columns
    ``image``, ``bbox`` ([x,y,w,h]), ``name`` ("table").  Pages with
    multiple tables appear as repeated rows; we treat each as its own
    sample.  For Phase-2 weighted sampling this is a known but minor
    approximation (weight=2.0, single-class supervision).
    """
    image = _decode_image_cell(row["image"])
    bbox = row.get("bbox")
    if bbox is None or len(bbox) != 4:
        return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    x, y, w, h = (float(v) for v in bbox)
    if w <= 0 or h <= 0:
        return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    raw_name = row.get("name") or row.get("category") or "table"
    mapped = map_class_with_logging("tablebank", str(raw_name))
    if mapped is None:
        return image, np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return (
        image,
        np.asarray([[x, y, x + w, y + h]], dtype=np.float32),
        np.asarray([mapped], dtype=np.int64),
    )


# Registry so callers can pick an adapter by source name.
ADAPTERS: Dict[str, Callable[[Dict[str, Any]], Tuple[Image.Image, np.ndarray, np.ndarray]]] = {
    "doclaynet": _adapter_doclaynet,
    "publaynet": _adapter_publaynet,
    "tablebank": _adapter_tablebank,
}


# ---------------------------------------------------------------------------
# Generic parquet detection source
# ---------------------------------------------------------------------------

class ParquetDetectionSource(Dataset):
    """
    Map-style detection dataset backed by one or more parquet shards.

    Parameters
    ----------
    parquet_files : a file path, directory path, or iterable of paths
                    to parquet files.  Directories are globbed for
                    ``*.parquet``.
    source_name   : one of the keys in :data:`ADAPTERS`.  Controls
                    both the per-row schema adapter and the class-ID
                    remapping via :mod:`label_map`.
    transform     : augmentation pipeline callable.  Defaults to the
                    DocDet training transform at 1120x800.
    split_filter  : optional iterable of allowed ``split`` column
                    values (e.g. ``{"train"}``).  Some HF mirrors
                    interleave train/val in a single parquet.
    """

    def __init__(
        self,
        parquet_files: Union[str, Path, Iterable[Union[str, Path]]],
        source_name: str,
        transform: Optional[Callable] = None,
        split_filter: Optional[Iterable[str]] = None,
    ):
        import pyarrow.parquet as pq

        key = source_name.lower()
        if key not in ADAPTERS:
            raise ValueError(
                f"Unknown parquet source '{source_name}'. "
                f"Supported: {sorted(ADAPTERS)}"
            )
        self.source_name = key
        self._adapter = ADAPTERS[key]
        self.transform = transform or DocDetTrainTransform(target_size=_TARGET_SIZE)

        paths = _expand_parquet_paths(parquet_files)
        if not paths:
            raise FileNotFoundError(
                f"No parquet files resolved from {parquet_files}"
            )

        # Prefer reading the train split only; most HF mirrors place
        # train shards under a ``train/`` sub-directory we can detect
        # by path, but a split column filter is also supported.
        self._tables = []
        self._cumulative: List[int] = []
        running = 0
        for p in paths:
            tbl = pq.read_table(p)
            if split_filter is not None and "split" in tbl.column_names:
                mask = np.isin(
                    tbl.column("split").to_pylist(),
                    list(split_filter),
                )
                if not mask.any():
                    continue
                tbl = tbl.filter(mask.tolist())
            self._tables.append(tbl)
            running += tbl.num_rows
            self._cumulative.append(running)

        self._total = running
        if self._total == 0:
            raise RuntimeError(
                f"ParquetDetectionSource({source_name}) resolved 0 rows. "
                f"Check parquet files or split_filter."
            )

    def __len__(self) -> int:
        return self._total

    def _resolve(self, index: int) -> Tuple[int, int]:
        """Map a flat index to (table_idx, row_idx)."""
        if index < 0 or index >= self._total:
            raise IndexError(index)
        prev = 0
        for t_idx, cum in enumerate(self._cumulative):
            if index < cum:
                return t_idx, index - prev
            prev = cum
        raise IndexError(index)

    def _read_row(self, index: int) -> Dict[str, Any]:
        t_idx, r_idx = self._resolve(index)
        table = self._tables[t_idx]
        row: Dict[str, Any] = {}
        for col in table.column_names:
            row[col] = table.column(col)[r_idx].as_py()
        return row

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self._read_row(index)
        image, boxes, labels = self._adapter(row)
        tensor, box_tensor, meta = self.transform(image, boxes)

        # Augmentations may drop boxes whose area collapsed to zero;
        # align labels to however many boxes survived (in-order drop).
        if box_tensor.shape[0] != labels.shape[0]:
            labels = labels[: box_tensor.shape[0]]

        return {
            "image": tensor,
            "boxes": box_tensor,
            "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)).long(),
            "image_id": index,
            "source": self.source_name,
            "meta": meta,
        }


def _expand_parquet_paths(
    parquet_files: Union[str, Path, Iterable[Union[str, Path]]],
) -> List[Path]:
    """Normalise the ``parquet_files`` arg into a flat list of Paths."""
    if isinstance(parquet_files, (str, Path)):
        candidates: List[Path] = [Path(parquet_files)]
    else:
        candidates = [Path(p) for p in parquet_files]

    resolved: List[Path] = []
    for c in candidates:
        if c.is_dir():
            # Prefer a ``train/`` sub-directory if it exists, otherwise
            # fall through to globbing the whole tree.  This matches
            # HF's ``data/train-*.parquet`` layout for most mirrors.
            train_subdir = c / "data"
            if train_subdir.is_dir():
                train_shards = sorted(train_subdir.glob("train-*.parquet"))
                if train_shards:
                    resolved.extend(train_shards)
                    continue
            resolved.extend(sorted(c.rglob("*.parquet")))
        elif c.is_file():
            resolved.append(c)
    return resolved
