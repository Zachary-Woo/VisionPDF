"""
DocDet layout detector wrapped as a Docling BaseLayoutModel.

This module lets the DocDet layout detector slot into Docling's
standard PDF pipeline in place of the built-in RT-DETR "Layout
Heron" or the Ultralytics YOLO wrapper.  Docling handles everything
downstream (cell assignment, overlap resolution, reading order,
table structure, markdown serialisation).

DocDet is a license-clean, in-house FCOS-style detector trained on
DocLayNet / PubLayNet / DocBank / TableBank / IIIT-AR-13K.  It
accepts either a .pt checkpoint or a .onnx file; the correct
runtime is selected automatically by DocDetPredictor.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import List, Optional

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorOptions
from docling.datamodel.base_models import (
    BoundingBox,
    Cluster,
    LayoutPrediction,
    Page,
)
from docling.datamodel.document import ConversionResult
from docling.datamodel.pipeline_options import LayoutOptions
from docling.datamodel.settings import settings
from docling.models.base_layout_model import BaseLayoutModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.layout_postprocessor import LayoutPostprocessor
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import CoordOrigin, DocItemLabel

from benchmark.config import DOCDET_MODEL, DOCLAYNET_LABELS
from layout_detector.docdet.infer.predict import DocDetPredictor

# Module-level override for the DocDet weights path.  The benchmark
# extract script sets this before constructing the Docling
# DocumentConverter so --docdet-weights propagates all the way to
# DocDetLayoutModel without extending Docling's LayoutOptions schema.
DOCDET_WEIGHTS_OVERRIDE: Optional[Path] = None

_LABEL_MAP = {
    "Caption": DocItemLabel.CAPTION,
    "Footnote": DocItemLabel.FOOTNOTE,
    "Formula": DocItemLabel.FORMULA,
    "List-item": DocItemLabel.LIST_ITEM,
    "Page-footer": DocItemLabel.PAGE_FOOTER,
    "Page-header": DocItemLabel.PAGE_HEADER,
    "Picture": DocItemLabel.PICTURE,
    "Section-header": DocItemLabel.SECTION_HEADER,
    "Table": DocItemLabel.TABLE,
    "Text": DocItemLabel.TEXT,
    "Title": DocItemLabel.TITLE,
}


class DocDetLayoutModel(BaseLayoutModel):
    """
    BaseLayoutModel implementation backed by DocDet weights.

    Drop-in replacement for
    ``docling.models.stages.layout.layout_model.LayoutModel``.  The
    DocDet model is loaded once in ``__init__`` and reused across
    every page.  Inference runs on the page image rendered at
    ``IMAGE_SCALE`` (same convention as the YOLO wrapper so the
    benchmark's timings are comparable).
    """

    IMAGE_SCALE = 2.0

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
        enable_remote_services: bool = False,
        model_path: Optional[Path] = None,
        score_threshold: float = 0.3,
    ):
        _ = artifacts_path
        _ = enable_remote_services

        self.options = options
        self.device = decide_device(accelerator_options.device)

        path = model_path or self._resolve_weights()
        self.predictor = DocDetPredictor(
            model_path=path,
            device=self.device,
            score_threshold=score_threshold,
            input_size=(1120, 800),
        )

    @classmethod
    def get_options_type(cls) -> type[LayoutOptions]:
        return LayoutOptions

    @staticmethod
    def _resolve_weights() -> Path:
        """Return a local path to DocDet weights, else raise."""
        if DOCDET_WEIGHTS_OVERRIDE is not None and Path(DOCDET_WEIGHTS_OVERRIDE).exists():
            return Path(DOCDET_WEIGHTS_OVERRIDE)
        if DOCDET_MODEL.exists():
            return DOCDET_MODEL
        raise FileNotFoundError(
            f"DocDet weights not found at {DOCDET_MODEL}. Train via "
            f"`python -m layout_detector.docdet.train.phase2_real "
            f"...` or point --docdet-weights at a .pt/.onnx file."
        )

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        """
        Render each valid page, run DocDet, build Docling Cluster
        objects in PDF-point space, then apply
        ``LayoutPostprocessor`` to merge clusters and assign cells.
        """
        pages = list(pages)
        predictions: List[LayoutPrediction] = []

        valid_pages: List[Page] = []
        valid_images = []
        for page in pages:
            if page._backend is None or not page._backend.is_valid():
                continue
            assert page.size is not None
            image = page.get_image(scale=self.IMAGE_SCALE)
            if image is None:
                continue
            valid_pages.append(page)
            valid_images.append(image)

        per_page_detections = []
        if valid_images:
            with TimeRecorder(conv_res, "layout"):
                for img in valid_images:
                    per_page_detections.append(self.predictor.predict(img))

        valid_idx = 0
        for page in pages:
            if page._backend is None or not page._backend.is_valid():
                existing = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = existing
                predictions.append(existing)
                continue

            dets = per_page_detections[valid_idx]
            valid_idx += 1

            clusters = self._detections_to_clusters(dets)
            processed_clusters, processed_cells = LayoutPostprocessor(
                page, clusters, self.options
            ).postprocess()

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "Mean of empty slice|invalid value encountered in scalar divide",
                    RuntimeWarning,
                    "numpy",
                )
                conv_res.confidence.pages[page.page_no].layout_score = float(
                    np.mean([c.confidence for c in processed_clusters])
                    if processed_clusters
                    else 0.0
                )
                conv_res.confidence.pages[page.page_no].ocr_score = float(
                    np.mean(
                        [c.confidence for c in processed_cells if c.from_ocr]
                    )
                    if processed_cells
                    else 0.0
                )

            prediction = LayoutPrediction(clusters=processed_clusters)
            page.predictions.layout = prediction
            predictions.append(prediction)

        _ = settings
        return predictions

    def _detections_to_clusters(self, detections) -> List[Cluster]:
        """Convert DocDetPredictor output into Docling Cluster objects."""
        clusters: List[Cluster] = []
        if not detections:
            return clusters

        for ix, det in enumerate(detections):
            doc_label = _LABEL_MAP.get(det.label)
            if doc_label is None:
                continue
            x1, y1, x2, y2 = det.bbox
            l = x1 / self.IMAGE_SCALE
            t = y1 / self.IMAGE_SCALE
            r = x2 / self.IMAGE_SCALE
            b = y2 / self.IMAGE_SCALE

            bbox = BoundingBox(
                l=l, t=t, r=r, b=b, coord_origin=CoordOrigin.TOPLEFT
            )
            clusters.append(
                Cluster(
                    id=ix,
                    label=doc_label,
                    confidence=float(det.score),
                    bbox=bbox,
                    cells=[],
                )
            )
        return clusters
