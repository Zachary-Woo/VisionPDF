"""
Tests for per-source label remapping.

Every mapping source must return IDs within ``[0, NUM_DOCDET_CLASSES)``
for known inputs and ``None`` for intentionally dropped categories.
"""

from __future__ import annotations

from layout_detector.docdet.data.label_map import (
    DOCDET_CLASS_NAMES,
    NUM_DOCDET_CLASSES,
    canonical_id,
    map_class,
)


def test_canonical_ids_contiguous() -> None:
    ids = [canonical_id(n) for n in DOCDET_CLASS_NAMES]
    assert ids == list(range(NUM_DOCDET_CLASSES))


def test_doclaynet_ids() -> None:
    # DocLayNet category IDs 1..11 cover the full class set.
    assert map_class("doclaynet", 9) == canonical_id("Table")
    assert map_class("doclaynet", 11) == canonical_id("Title")
    assert map_class("doclaynet", 99) is None


def test_publaynet_ids_and_names() -> None:
    assert map_class("publaynet", "table") == canonical_id("Table")
    assert map_class("publaynet", 4) == canonical_id("Table")
    assert map_class("publaynet", 999) is None


def test_docbank_ids() -> None:
    assert map_class("docbank", "paragraph") == canonical_id("Text")
    assert map_class("docbank", "equation") == canonical_id("Formula")
    assert map_class("docbank", "unknown-class") is None


def test_tablebank_always_table() -> None:
    assert map_class("tablebank", "table") == canonical_id("Table")
    assert map_class("tablebank", 0) == canonical_id("Table")


def test_iiit_ar_mapping() -> None:
    assert map_class("iiit_ar", "figure") == canonical_id("Picture")
    assert map_class("iiit_ar", "table") == canonical_id("Table")


def test_docsynth_ids() -> None:
    assert map_class("docsynth", 3) == canonical_id("Table")
    assert map_class("docsynth", 99) is None


def test_omnidocbench_mapping() -> None:
    assert map_class("omnidocbench", "table") == canonical_id("Table")
    assert map_class("omnidocbench", "text_block") == canonical_id("Text")
    assert map_class("omnidocbench", "abandon") is None
