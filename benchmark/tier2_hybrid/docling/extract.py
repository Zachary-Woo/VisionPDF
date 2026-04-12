"""
Tier 2 -- Docling RT-DETR pipeline (text layer + vision layout).

Uses Docling's StandardPdfPipeline with OCR disabled so that all text
comes from the native PDF text layer.  Layout detection is handled by
Docling's built-in RT-DETR "Layout Heron" model (ResNet50, trained on
DocLayNet), and reading order is predicted by Docling's own
ReadingOrderModel.

This is the most mature hybrid pipeline in the benchmark and serves as
the primary Tier 2 baseline.

Usage:
    python -m benchmark.tier2_hybrid.docling.extract [--input-dir path]
                                                      [--output-dir path]
"""

import argparse
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR, find_pdfs
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier2_docling"


def build_converter():
    """
    Build a Docling DocumentConverter configured for text-layer-only
    extraction with full layout detection and table structure recovery.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    return converter


def run(input_dir: Path, output_base: Path):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = find_pdfs(input_dir)

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print("Building Docling converter (RT-DETR layout, no OCR)...")
    converter = build_converter()

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            result = converter.convert(str(pdf_path))
            markdown = result.document.export_to_markdown()

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(
            timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds
        )

    write_summary(out_dir, METHOD_NAME, {"total_pages": len(pdf_files)})
    print(f"{METHOD_NAME}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
