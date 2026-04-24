"""
DocDet data loading and preparation.

Modules:
    label_map        per-source class-name -> DocDet id remapping
    transforms       document-safe augmentations (no h-flip, no mosaic)
    coco_source      generalised COCO-JSON dataset wrapper
    docsynth_source  DocSynth300K parquet dataset wrapper
    parquet_source   DocLayNet/PubLayNet/TableBank parquet dataset wrapper
    iiitar_source    IIIT-AR-13K Pascal VOC dataset wrapper
    weighted_sampler multi-source dataset + weighted random sampler
    download         per-phase dataset download + cleanup helpers
"""

from .parquet_source import ParquetDetectionSource
from .iiitar_source import IIITARVocSource

__all__ = [
    "ParquetDetectionSource",
    "IIITARVocSource",
]
