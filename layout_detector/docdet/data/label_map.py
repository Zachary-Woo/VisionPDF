"""
Per-source label remapping to the canonical DocDet class index.

DocDet uses the 11-class DocLayNet taxonomy defined in
``benchmark.config.DOCLAYNET_LABELS``.  Training datasets expose
different category sets: some use integer IDs, others use strings,
some cover a subset (TableBank is single-class), others a superset
(DocBank has 13 classes).  This module centralises all mappings
and logs unknown categories exactly once.

The canonical class indices used throughout DocDet::

    0 Caption
    1 Footnote
    2 Formula
    3 List-item
    4 Page-footer
    5 Page-header
    6 Picture
    7 Section-header
    8 Table
    9 Text
    10 Title

These match ``DOCLAYNET_LABELS`` order so the benchmark integration
can feed DocDet predictions straight into the existing text-assembly
and reading-order code with zero remapping.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical DocDet taxonomy
# ---------------------------------------------------------------------------

DOCDET_CLASS_NAMES = (
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
)

NUM_DOCDET_CLASSES = len(DOCDET_CLASS_NAMES)

_NAME_TO_ID: Dict[str, int] = {name: i for i, name in enumerate(DOCDET_CLASS_NAMES)}


def canonical_id(class_name: str) -> int:
    """Return the DocDet class ID for a canonical DocLayNet name."""
    return _NAME_TO_ID[class_name]


# ---------------------------------------------------------------------------
# Per-source mapping tables
# ---------------------------------------------------------------------------

# DocLayNet-v1.2 (docling-project).  Category IDs 1..11 map 1:1 to
# DocDet names.  COCO-style IDs start at 1 (0 reserved for bg).
_DOCLAYNET_ID_TO_DOCDET: Dict[int, int] = {
    1: canonical_id("Caption"),
    2: canonical_id("Footnote"),
    3: canonical_id("Formula"),
    4: canonical_id("List-item"),
    5: canonical_id("Page-footer"),
    6: canonical_id("Page-header"),
    7: canonical_id("Picture"),
    8: canonical_id("Section-header"),
    9: canonical_id("Table"),
    10: canonical_id("Text"),
    11: canonical_id("Title"),
}

# PubLayNet.  5 categories mapped to the closest DocDet class.
_PUBLAYNET_NAME_TO_DOCDET: Dict[str, int] = {
    "text": canonical_id("Text"),
    "title": canonical_id("Title"),
    "list": canonical_id("List-item"),
    "table": canonical_id("Table"),
    "figure": canonical_id("Picture"),
}

# DocBank original label names (see https://huggingface.co/datasets/maveriq/DocBank).
# Categories absent in DocDet are either remapped to the nearest
# class (e.g. 'reference' -> Text) or dropped (returned None).
_DOCBANK_NAME_TO_DOCDET: Dict[str, Optional[int]] = {
    "abstract": canonical_id("Text"),
    "author": canonical_id("Text"),
    "caption": canonical_id("Caption"),
    "equation": canonical_id("Formula"),
    "figure": canonical_id("Picture"),
    "footer": canonical_id("Page-footer"),
    "list": canonical_id("List-item"),
    "paragraph": canonical_id("Text"),
    "reference": canonical_id("Text"),
    "section": canonical_id("Section-header"),
    "table": canonical_id("Table"),
    "title": canonical_id("Title"),
    "date": canonical_id("Text"),
}

# TableBank is a single-class detection dataset (only tables).  Any
# category ID is mapped to the Table class.
_TABLEBANK_NAME_TO_DOCDET: Dict[str, int] = {
    "table": canonical_id("Table"),
    "Table": canonical_id("Table"),
}

# IIIT-AR-13K (annual reports).  Names per the official documentation.
_IIITAR_NAME_TO_DOCDET: Dict[str, Optional[int]] = {
    "table": canonical_id("Table"),
    "figure": canonical_id("Picture"),
    "natural_image": canonical_id("Picture"),
    "logo": canonical_id("Picture"),
    "signature": canonical_id("Picture"),
}

# DocSynth300K exposes integer category IDs inside the anno_string.
# The dataset README documents these IDs (see
# https://huggingface.co/datasets/juliozhao/DocSynth300K).  Unknown
# IDs are dropped; callers should log them via the warning path.
_DOCSYNTH_ID_TO_DOCDET: Dict[int, Optional[int]] = {
    0: canonical_id("Title"),
    1: canonical_id("Text"),
    2: canonical_id("List-item"),
    3: canonical_id("Table"),
    4: canonical_id("Picture"),
    5: canonical_id("Formula"),
    6: canonical_id("Caption"),
    7: canonical_id("Page-header"),
    8: canonical_id("Page-footer"),
    9: canonical_id("Footnote"),
    10: canonical_id("Section-header"),
}

# OmniDocBench uses free-form string category_type strings.
# None means "drop this annotation" (masks, abandon regions,
# un-evaluated content).
_OMNIDOCBENCH_NAME_TO_DOCDET: Dict[str, Optional[int]] = {
    "title": canonical_id("Title"),
    "text_block": canonical_id("Text"),
    "code_txt": canonical_id("Text"),
    "reference": canonical_id("Text"),
    "figure": canonical_id("Picture"),
    "figure_caption": canonical_id("Caption"),
    "figure_footnote": canonical_id("Footnote"),
    "table": canonical_id("Table"),
    "table_caption": canonical_id("Caption"),
    "table_footnote": canonical_id("Footnote"),
    "equation_isolated": canonical_id("Formula"),
    "equation_caption": canonical_id("Caption"),
    "header": canonical_id("Page-header"),
    "footer": canonical_id("Page-footer"),
    "page_footnote": canonical_id("Footnote"),
    "page_number": canonical_id("Page-footer"),
    "abandon": None,
    "need_mask": None,
    "text_mask": None,
    "table_mask": None,
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

_UNKNOWN_SEEN: Dict[str, set] = {}


def _log_unknown_once(source: str, raw: Union[str, int]) -> None:
    """Print a one-off warning per (source, raw) pair."""
    seen = _UNKNOWN_SEEN.setdefault(source, set())
    key = str(raw).lower()
    if key not in seen:
        seen.add(key)
        logger.warning(
            "label_map: dropping unknown '%s' category '%s' (not mapped "
            "to any DocDet class)", source, raw,
        )


def map_class(source: str, raw: Union[str, int]) -> Optional[int]:
    """
    Map a raw category (ID or name) from ``source`` to a DocDet ID.

    Parameters
    ----------
    source : "doclaynet" | "publaynet" | "docbank" | "tablebank" |
             "iiit_ar" | "docsynth"
    raw    : integer category ID or string category name, whichever
             the source dataset uses.

    Returns
    -------
    DocDet class ID in ``[0, NUM_DOCDET_CLASSES)``, or ``None`` if the
    category is unknown / intentionally dropped (caller should skip
    that annotation).
    """
    source = source.lower()
    if source == "doclaynet":
        if isinstance(raw, str):
            return _NAME_TO_ID.get(raw)
        return _DOCLAYNET_ID_TO_DOCDET.get(int(raw))

    if source == "publaynet":
        if isinstance(raw, int):
            # COCO-style category_ids in PubLayNet start at 1.
            names = ("text", "title", "list", "table", "figure")
            idx = int(raw) - 1
            if 0 <= idx < len(names):
                return _PUBLAYNET_NAME_TO_DOCDET.get(names[idx])
            return None
        return _PUBLAYNET_NAME_TO_DOCDET.get(str(raw).lower())

    if source == "docbank":
        if isinstance(raw, int):
            names = tuple(_DOCBANK_NAME_TO_DOCDET.keys())
            idx = int(raw)
            if 0 <= idx < len(names):
                return _DOCBANK_NAME_TO_DOCDET.get(names[idx])
            return None
        return _DOCBANK_NAME_TO_DOCDET.get(str(raw).lower())

    if source == "tablebank":
        if isinstance(raw, int):
            return canonical_id("Table")
        return _TABLEBANK_NAME_TO_DOCDET.get(str(raw).lower())

    if source == "iiit_ar":
        if isinstance(raw, int):
            names = tuple(_IIITAR_NAME_TO_DOCDET.keys())
            idx = int(raw)
            if 0 <= idx < len(names):
                return _IIITAR_NAME_TO_DOCDET.get(names[idx])
            return None
        return _IIITAR_NAME_TO_DOCDET.get(str(raw).lower())

    if source == "docsynth":
        return _DOCSYNTH_ID_TO_DOCDET.get(int(raw))

    if source == "omnidocbench":
        if isinstance(raw, int):
            names = tuple(_OMNIDOCBENCH_NAME_TO_DOCDET.keys())
            idx = int(raw)
            if 0 <= idx < len(names):
                return _OMNIDOCBENCH_NAME_TO_DOCDET.get(names[idx])
            return None
        return _OMNIDOCBENCH_NAME_TO_DOCDET.get(str(raw).lower())

    raise ValueError(
        f"Unknown dataset source '{source}'. "
        f"Supported: doclaynet, publaynet, docbank, tablebank, iiit_ar, docsynth, omnidocbench"
    )


def map_class_with_logging(source: str, raw: Union[str, int]) -> Optional[int]:
    """Same as ``map_class`` but logs unknown categories once."""
    result = map_class(source, raw)
    if result is None:
        _log_unknown_once(source, raw)
    return result
