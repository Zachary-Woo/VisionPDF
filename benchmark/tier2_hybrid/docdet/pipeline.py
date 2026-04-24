"""
Docling StandardPdfPipeline variant that uses DocDet for layout detection.

Mirrors ``YoloStandardPdfPipeline`` but swaps the layout model for
the license-clean DocDet detector.  Every other stage (preprocess,
OCR, table structure, page assembly, reading order, enrichment) is
the same, so we inherit Docling's battle-tested text assembly.
"""

from __future__ import annotations

import warnings

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.models.stages.code_formula.code_formula_model import (
    CodeFormulaModel,
    CodeFormulaModelOptions,
)
from docling.models.stages.page_assemble.page_assemble_model import (
    PageAssembleModel,
    PageAssembleOptions,
)
from docling.models.stages.page_preprocessing.page_preprocessing_model import (
    PagePreprocessingModel,
    PagePreprocessingOptions,
)
from docling.models.stages.reading_order.readingorder_model import (
    ReadingOrderModel,
    ReadingOrderOptions,
)
from docling.models.factories import get_table_structure_factory
from docling.pipeline.legacy_standard_pdf_pipeline import LegacyStandardPdfPipeline

from benchmark.tier2_hybrid.docdet.docdet_layout_model import DocDetLayoutModel


class DocDetStandardPdfPipeline(LegacyStandardPdfPipeline):
    """Standard Docling pipeline with DocDetLayoutModel swapped in."""

    def __init__(self, pipeline_options: PdfPipelineOptions):
        super(LegacyStandardPdfPipeline, self).__init__(pipeline_options)
        self.pipeline_options: PdfPipelineOptions = pipeline_options

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            self.keep_images = (
                self.pipeline_options.generate_page_images
                or self.pipeline_options.generate_picture_images
                or self.pipeline_options.generate_table_images
            )

        self.reading_order_model = ReadingOrderModel(options=ReadingOrderOptions())

        ocr_model = self.get_ocr_model(artifacts_path=self.artifacts_path)

        layout_model = DocDetLayoutModel(
            artifacts_path=self.artifacts_path,
            accelerator_options=pipeline_options.accelerator_options,
            options=pipeline_options.layout_options,
            enable_remote_services=pipeline_options.enable_remote_services,
        )

        table_factory = get_table_structure_factory(
            allow_external_plugins=self.pipeline_options.allow_external_plugins
        )
        table_model = table_factory.create_instance(
            options=pipeline_options.table_structure_options,
            enabled=pipeline_options.do_table_structure,
            artifacts_path=self.artifacts_path,
            accelerator_options=pipeline_options.accelerator_options,
            enable_remote_services=pipeline_options.enable_remote_services,
        )

        self.build_pipe = [
            PagePreprocessingModel(
                options=PagePreprocessingOptions(
                    images_scale=pipeline_options.images_scale,
                )
            ),
            ocr_model,
            layout_model,
            table_model,
            PageAssembleModel(options=PageAssembleOptions()),
        ]

        self.enrichment_pipe = [
            CodeFormulaModel(
                enabled=pipeline_options.do_code_enrichment
                or pipeline_options.do_formula_enrichment,
                artifacts_path=self.artifacts_path,
                options=CodeFormulaModelOptions(
                    do_code_enrichment=pipeline_options.do_code_enrichment,
                    do_formula_enrichment=pipeline_options.do_formula_enrichment,
                ),
                accelerator_options=pipeline_options.accelerator_options,
            ),
            *self.enrichment_pipe,
        ]

        if (
            self.pipeline_options.do_formula_enrichment
            or self.pipeline_options.do_code_enrichment
            or self.pipeline_options.do_picture_classification
            or self.pipeline_options.do_picture_description
        ):
            self.keep_backend = True
