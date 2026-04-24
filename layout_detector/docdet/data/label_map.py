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

# DocSynth300K uses the M6Doc 74-class taxonomy (its element pool was
# fragmented from M6Doc test images - see DocLayout-YOLO arxiv:2410.12628
# section 3.1).  Official ID -> name table:
#   doclayout_yolo/cfg/datasets/docsynth300k.yaml lines 17-91.
#
# We map each M6Doc category to the nearest DocDet class and drop
# categories that have no reasonable DocDet analogue (None) so the
# DocSynth300K pretraining signal aligns with the downstream 11-class
# head.  Without this remap, ~99% of the 300K samples' boxes would be
# silently discarded and Phase 1 would degenerate into noise.
_DOCSYNTH_ID_TO_DOCDET: Dict[int, Optional[int]] = {
    0:  None,                              # QR code
    1:  None,                              # advertisement
    2:  None,                              # algorithm
    3:  None,                              # answer (test-paper only)
    4:  canonical_id("Text"),              # author
    5:  None,                              # barcode
    6:  None,                              # bill
    7:  None,                              # blank
    8:  None,                              # bracket
    9:  None,                              # breakout (magazine pull-quote)
    10: canonical_id("Text"),              # byline
    11: canonical_id("Caption"),           # caption
    12: canonical_id("List-item"),         # catalogue
    13: canonical_id("Section-header"),    # chapter title
    14: canonical_id("Text"),              # code
    15: None,                              # correction
    16: canonical_id("Text"),              # credit
    17: canonical_id("Text"),              # dateline
    18: None,                              # drop cap (decorative)
    19: canonical_id("Text"),              # editor's note
    20: canonical_id("Footnote"),          # endnote
    21: canonical_id("Text"),              # examinee information
    22: canonical_id("Section-header"),    # fifth-level title
    23: canonical_id("Picture"),           # figure
    24: canonical_id("Text"),              # first-level question number
    25: canonical_id("Title"),             # first-level title
    26: None,                              # flag (newspaper masthead)
    27: canonical_id("Page-footer"),       # folio
    28: canonical_id("Page-footer"),       # footer
    29: canonical_id("Footnote"),          # footnote
    30: canonical_id("Formula"),           # formula
    31: canonical_id("Section-header"),    # fourth-level section title
    32: canonical_id("Section-header"),    # fourth-level title
    33: canonical_id("Page-header"),       # header
    34: canonical_id("Title"),             # headline
    35: canonical_id("List-item"),         # index
    36: None,                              # inside (ambiguous)
    37: canonical_id("Text"),              # institute
    38: canonical_id("Text"),              # jump line
    39: canonical_id("Text"),              # kicker
    40: canonical_id("Text"),              # lead
    41: canonical_id("Footnote"),          # marginal note
    42: None,                              # matching (test-paper only)
    43: canonical_id("Picture"),           # mugshot
    44: canonical_id("List-item"),         # option (multiple-choice)
    45: canonical_id("List-item"),         # ordered list
    46: canonical_id("Text"),              # other question number
    47: canonical_id("Page-footer"),       # page number
    48: canonical_id("Text"),              # paragraph
    49: canonical_id("Section-header"),    # part
    50: canonical_id("Text"),              # play
    51: canonical_id("Text"),              # poem
    52: canonical_id("Text"),              # reference
    53: None,                              # sealing line (test-paper only)
    54: canonical_id("Text"),              # second-level question number
    55: canonical_id("Section-header"),    # second-level title
    56: canonical_id("Section-header"),    # section
    57: canonical_id("Section-header"),    # section title
    58: canonical_id("Text"),              # sidebar
    59: canonical_id("Section-header"),    # sub section title
    60: canonical_id("Section-header"),    # subhead
    61: canonical_id("Section-header"),    # subsub section title
    62: canonical_id("Footnote"),          # supplementary note
    63: canonical_id("Table"),             # table
    64: canonical_id("Caption"),           # table caption
    65: canonical_id("Footnote"),          # table note
    66: canonical_id("Text"),              # teasers
    67: canonical_id("Text"),              # third-level question number
    68: canonical_id("Section-header"),    # third-level title
    69: canonical_id("Title"),             # title
    70: canonical_id("Text"),              # translator
    71: None,                              # underscore (formatting)
    72: canonical_id("List-item"),         # unordered list
    73: None,                              # weather forecast
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


def _is_known(source: str, raw: Union[str, int]) -> bool:
    """Return True if ``raw`` is present in the source's mapping table.

    Used to distinguish intentional drops (key present, value None) from
    truly unknown categories (key missing) so logging can stay quiet on
    the former.
    """
    source = source.lower()
    if source == "doclaynet":
        if isinstance(raw, str):
            return raw in _NAME_TO_ID
        return int(raw) in _DOCLAYNET_ID_TO_DOCDET
    if source == "publaynet":
        if isinstance(raw, int):
            return 1 <= int(raw) <= 5
        return str(raw).lower() in _PUBLAYNET_NAME_TO_DOCDET
    if source == "docbank":
        if isinstance(raw, int):
            return 0 <= int(raw) < len(_DOCBANK_NAME_TO_DOCDET)
        return str(raw).lower() in _DOCBANK_NAME_TO_DOCDET
    if source == "tablebank":
        if isinstance(raw, int):
            return True
        return str(raw).lower() in _TABLEBANK_NAME_TO_DOCDET
    if source == "iiit_ar":
        if isinstance(raw, int):
            return 0 <= int(raw) < len(_IIITAR_NAME_TO_DOCDET)
        return str(raw).lower() in _IIITAR_NAME_TO_DOCDET
    if source == "docsynth":
        return int(raw) in _DOCSYNTH_ID_TO_DOCDET
    if source == "omnidocbench":
        if isinstance(raw, int):
            return 0 <= int(raw) < len(_OMNIDOCBENCH_NAME_TO_DOCDET)
        return str(raw).lower() in _OMNIDOCBENCH_NAME_TO_DOCDET
    return False


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
    """Same as ``map_class`` but logs truly-unknown categories once.

    Categories that are present in the source's mapping table but
    intentionally set to ``None`` (e.g. DocSynth's QR codes, drop caps,
    exam-paper artifacts) are silently dropped without a warning, since
    those drops are deliberate design choices rather than missing data.
    """
    result = map_class(source, raw)
    if result is None and not _is_known(source, raw):
        _log_unknown_once(source, raw)
    return result
