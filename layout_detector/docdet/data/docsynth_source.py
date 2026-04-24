"""
DocSynth300K parquet dataset wrapper for DocDet Phase 1 pretraining.

Layout of the HuggingFace ``juliozhao/DocSynth300K`` parquet shards:

    * image_data : base64-encoded PNG bytes (string column)
    * anno_string: space-delimited annotations with 9 numbers per
                   polygon: ``class_id x1 y1 x2 y2 x3 y3 x4 y4``.
                   Coordinates are normalised (0..1 of page size)
                   and describe the four corners of a quadrilateral
                   which we reduce to its axis-aligned bbox.

Each row thus represents one synthetic document page with a variable
number of region annotations.  Category IDs follow the DocSynth
ordering handled by ``label_map.map_class(source='docsynth', ...)``.

Design notes
------------
* We decode the base64 image on the fly, paying a small CPU cost per
  sample.  DataLoader workers mask the latency when training on GPU.
* The parquet file system supports random access so we can build a
  proper map-style ``Dataset`` (``__getitem__`` by index).  For
  HuggingFace streaming mode, use ``DocSynthStreamingSource`` which
  wraps a ``datasets.IterableDataset``.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from .label_map import map_class_with_logging
from .transforms import DocDetTrainTransform


_TARGET_SIZE: Tuple[int, int] = (1120, 800)
_SOURCE_NAME = "docsynth"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_image(image_data: Union[str, bytes]) -> Image.Image:
    """Decode base64 (or already-bytes) into a PIL RGB image."""
    if isinstance(image_data, str):
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def _parse_anno_string(
    anno_string: str, image_w: int, image_h: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a DocSynth-style anno_string into (boxes, class_ids).

    Annotation format per region (9 floats separated by spaces):
        class_id x1 y1 x2 y2 x3 y3 x4 y4

    Coordinates are normalised to [0, 1].  We produce the axis-
    aligned bbox by taking min/max over the four polygon corners and
    then multiplying by the image resolution.
    """
    if not anno_string or anno_string.isspace():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    tokens = anno_string.strip().split()
    if len(tokens) % 9 != 0:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    n = len(tokens) // 9
    data = np.asarray(tokens, dtype=np.float32).reshape(n, 9)
    raw_cls = data[:, 0].astype(np.int64)
    xs = data[:, [1, 3, 5, 7]] * image_w
    ys = data[:, [2, 4, 6, 8]] * image_h
    boxes = np.stack([xs.min(1), ys.min(1), xs.max(1), ys.max(1)], axis=1)

    mapped_labels: List[int] = []
    keep_mask = np.zeros(n, dtype=bool)
    for i, raw in enumerate(raw_cls):
        docdet_id = map_class_with_logging(_SOURCE_NAME, int(raw))
        if docdet_id is None:
            continue
        mapped_labels.append(docdet_id)
        keep_mask[i] = True

    boxes = boxes[keep_mask].astype(np.float32)
    return boxes, np.asarray(mapped_labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# Map-style dataset (reads local parquet files)
# ---------------------------------------------------------------------------

class DocSynthSource(Dataset):
    """
    Map-style DocSynth300K dataset backed by local parquet files.

    Parameters
    ----------
    parquet_files : list (or single) of parquet file paths or a
                    directory that contains them.
    transform     : augmentation pipeline callable.  Defaults to the
                    training transform with target size 1120x800.
    image_column  : parquet column name for the image blob.  The HF
                    repo uses 'image_data'; override if a local dump
                    renames it.
    anno_column   : parquet column name for the annotation string.
    """

    def __init__(
        self,
        parquet_files: Union[str, Path, Iterable[Union[str, Path]]],
        transform: Optional[Callable] = None,
        image_column: str = "image_data",
        anno_column: str = "anno_string",
    ):
        import pyarrow.parquet as pq

        paths = _expand_parquet_paths(parquet_files)
        if not paths:
            raise FileNotFoundError(
                f"No parquet files resolved from {parquet_files}"
            )

        self._tables = [pq.read_table(p) for p in paths]
        self._cumulative: List[int] = []
        running = 0
        for t in self._tables:
            running += t.num_rows
            self._cumulative.append(running)
        self._total = running

        self._image_column = image_column
        self._anno_column = anno_column
        self.transform = transform or DocDetTrainTransform(target_size=_TARGET_SIZE)

    def __len__(self) -> int:
        return self._total

    def _resolve(self, index: int) -> Tuple[int, int]:
        """Map a flat index to (table_idx, row_idx)."""
        if index < 0 or index >= self._total:
            raise IndexError(index)
        # Linear scan across a small list of tables is cheap; avoids
        # pulling in bisect.
        prev = 0
        for t_idx, cum in enumerate(self._cumulative):
            if index < cum:
                return t_idx, index - prev
            prev = cum
        raise IndexError(index)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        t_idx, r_idx = self._resolve(index)
        table = self._tables[t_idx]
        image_data = table.column(self._image_column)[r_idx].as_py()
        anno_string = table.column(self._anno_column)[r_idx].as_py()

        image = _decode_image(image_data)
        boxes, labels = _parse_anno_string(anno_string, image.width, image.height)

        tensor, box_tensor, meta = self.transform(image, boxes)

        # Augmentations may change box count (crops can drop some);
        # align labels defensively.
        if box_tensor.shape[0] != labels.shape[0]:
            labels = labels[: box_tensor.shape[0]]

        return {
            "image": tensor,
            "boxes": box_tensor,
            "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)).long(),
            "image_id": index,
            "source": _SOURCE_NAME,
            "meta": meta,
        }


def _expand_parquet_paths(
    parquet_files: Union[str, Path, Iterable[Union[str, Path]]]
) -> List[Path]:
    """Normalise the parquet_files argument to a flat list of Paths."""
    if isinstance(parquet_files, (str, Path)):
        candidates = [Path(parquet_files)]
    else:
        candidates = [Path(p) for p in parquet_files]

    resolved: List[Path] = []
    for c in candidates:
        if c.is_dir():
            resolved.extend(sorted(c.rglob("*.parquet")))
        elif c.is_file():
            resolved.append(c)
    return resolved


# ---------------------------------------------------------------------------
# Streaming source (for low-disk-space Colab runs)
# ---------------------------------------------------------------------------

class DocSynthStreamingSource(IterableDataset):
    """
    Iterable dataset streaming DocSynth300K from HuggingFace Hub.

    Used when there is not enough local disk to download the full
    ~113 GB parquet shards.  Stream mode uses HuggingFace's
    ``datasets.load_dataset(..., streaming=True)`` under the hood.

    Parameters
    ----------
    repo_id   : HuggingFace repo, default ``juliozhao/DocSynth300K``.
    transform : augmentation pipeline callable.
    split     : dataset split name, default ``'train'``.
    take      : optional int to cap the number of yielded samples
                (useful for smoke tests / subsampled training).
    """

    def __init__(
        self,
        repo_id: str = "juliozhao/DocSynth300K",
        transform: Optional[Callable] = None,
        split: str = "train",
        take: Optional[int] = None,
    ):
        self.repo_id = repo_id
        self.split = split
        self.take = take
        self.transform = transform or DocDetTrainTransform(target_size=_TARGET_SIZE)

    def __iter__(self):
        from datasets import load_dataset

        ds = load_dataset(self.repo_id, split=self.split, streaming=True)
        if self.take is not None:
            ds = ds.take(self.take)

        for i, row in enumerate(ds):
            image = _decode_image(row["image_data"])
            boxes, labels = _parse_anno_string(
                row["anno_string"], image.width, image.height
            )
            tensor, box_tensor, meta = self.transform(image, boxes)
            if box_tensor.shape[0] != labels.shape[0]:
                labels = labels[: box_tensor.shape[0]]
            yield {
                "image": tensor,
                "boxes": box_tensor,
                "labels": torch.from_numpy(np.asarray(labels, dtype=np.int64)).long(),
                "image_id": i,
                "source": _SOURCE_NAME,
                "meta": meta,
            }
