"""
YOLO-DocLayNet layout detector wrapped as a Docling BaseLayoutModel.

This module lets the YOLO layout detector slot into Docling's standard
PDF pipeline in place of the built-in RT-DETR "Layout Heron" model.
Docling handles everything downstream -- cell assignment, overlap
resolution, reading order, table structure, markdown serialisation --
so we no longer maintain custom text assembly or reading-order logic.

Integration point:
  * predict_layout(...) renders each page, runs YOLO, builds Docling
    Cluster objects, then hands them to LayoutPostprocessor (the same
    step Docling's built-in layout model uses) for cluster cleanup and
    cell-to-cluster assignment.
  * A LayoutPrediction is returned per page, which Docling then feeds
    to TableStructureModel, PageAssembleModel, and ReadingOrderModel.
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

from benchmark.config import DOCLAYNET_LABELS

# Mapping from DocLayNet class names (produced by hantian/yolo-doclaynet)
# to Docling's DocItemLabel enum.  Docling reasons about clusters via
# this enum, so every label YOLO can emit must appear here.
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


class YoloLayoutModel(BaseLayoutModel):
    """
    BaseLayoutModel implementation backed by Ultralytics YOLO weights
    trained on DocLayNet.  Drop-in replacement for
    docling.models.stages.layout.layout_model.LayoutModel.

    The YOLO model is loaded once in __init__ and reused across every
    page the pipeline processes.  Inference is done at an image scale
    of 2.0 (so a 72dpi page is rendered at 144dpi before detection)
    which matches how the DocLayNet weights were trained and gives
    noticeably sharper bounding boxes than the 1.0 scale that RT-DETR
    uses.  Coordinates are divided by the same factor before being
    handed to Docling so the resulting clusters live in PDF-point space.
    """

    IMAGE_SCALE = 2.0

    def __init__(
        self,
        artifacts_path: Optional[Path],
        accelerator_options: AcceleratorOptions,
        options: LayoutOptions,
        enable_remote_services: bool = False,
    ):
        from ultralytics import YOLO

        # artifacts_path / enable_remote_services are accepted to keep
        # the ctor signature interchangeable with LayoutModel, even
        # though YOLO fetches its own weights.
        _ = artifacts_path
        _ = enable_remote_services

        self.options = options
        self.device = decide_device(accelerator_options.device)

        weights = self._resolve_weights()
        self.yolo = YOLO(str(weights))

    @classmethod
    def get_options_type(cls) -> type[LayoutOptions]:
        return LayoutOptions

    @staticmethod
    def _resolve_weights() -> Path:
        """
        Return a local path to the DocLayNet YOLO weights, downloading
        from HuggingFace Hub on first use.  We import lazily so the
        huggingface_hub dependency is only required when the file is
        missing.
        """
        from benchmark.config import YOLO_MODEL, YOLO_MODEL_FILE, YOLO_MODEL_REPO

        if YOLO_MODEL.exists():
            return YOLO_MODEL

        from huggingface_hub import hf_hub_download

        local_pt = hf_hub_download(YOLO_MODEL_REPO, filename=YOLO_MODEL_FILE)
        return Path(local_pt)

    def predict_layout(
        self,
        conv_res: ConversionResult,
        pages: Sequence[Page],
    ) -> Sequence[LayoutPrediction]:
        """
        For each valid page, render a page image, run YOLO, build
        Cluster objects in PDF-point space, then apply Docling's
        LayoutPostprocessor to merge/clean the clusters and assign
        text cells.  Invalid pages pass through unchanged.
        """
        pages = list(pages)
        predictions: List[LayoutPrediction] = []

        # Batch YOLO inference across all valid pages in one call --
        # matches the built-in LayoutModel's batching behaviour.
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

        batch_results = []
        if valid_images:
            with TimeRecorder(conv_res, "layout"):
                batch_results = self.yolo(
                    valid_images,
                    verbose=False,
                    device=self.device,
                )

        valid_idx = 0
        for page in pages:
            if page._backend is None or not page._backend.is_valid():
                existing = page.predictions.layout or LayoutPrediction()
                page.predictions.layout = existing
                predictions.append(existing)
                continue

            result = batch_results[valid_idx]
            valid_idx += 1

            clusters = self._result_to_clusters(result)

            # Mirror the built-in model: run postprocessing (drops
            # low-confidence clusters, resolves overlaps, assigns
            # cells), then record the layout confidence.
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

        _ = settings  # kept for parity with built-in model's debug hooks

        return predictions

    def _result_to_clusters(self, result) -> List[Cluster]:
        """
        Convert one Ultralytics Results object into Docling Cluster
        objects in PDF-point space.  Unknown YOLO labels are skipped
        rather than silently mis-labelled.
        """
        clusters: List[Cluster] = []
        if result.boxes is None or len(result.boxes) == 0:
            return clusters

        for ix, box in enumerate(result.boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())

            if not (0 <= cls_id < len(DOCLAYNET_LABELS)):
                continue
            raw_label = DOCLAYNET_LABELS[cls_id]
            doc_label = _LABEL_MAP.get(raw_label)
            if doc_label is None:
                continue

            # YOLO coords are pixels in the scaled image; divide to get
            # PDF points, which is the coordinate system Docling's
            # cells and downstream models expect.
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
                    confidence=conf,
                    bbox=bbox,
                    cells=[],
                )
            )
        return clusters
