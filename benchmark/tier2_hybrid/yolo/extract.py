"""
Tier 2 -- YOLO layout detection feeding Docling's assembly pipeline.

This replaces the old hand-rolled geometric / LayoutReader assemblers.
YOLO detects layout regions (trained on DocLayNet), and Docling takes
care of everything else: cell assignment, overlap resolution, reading
order (via its ReadingOrderModel), table structure (via TableFormer),
and markdown serialisation.

Usage:
    python -m benchmark.tier2_hybrid.yolo.extract [--input-dir path]
                                                  [--output-dir path]
                                                  [--overwrite]
"""

from __future__ import annotations

import argparse
import pathlib
import platform
import sys
from pathlib import Path

from tqdm import tqdm

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR, find_pdfs
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.tier2_hybrid.docling_export import markdown_with_html_tables
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier2_yolo"


def build_converter():
    """
    Build a Docling DocumentConverter whose PDF pipeline uses YOLO
    for layout detection.  OCR is disabled -- text comes from the
    embedded PDF text layer.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    from benchmark.tier2_hybrid.yolo.pipeline import YoloStandardPdfPipeline

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=True,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=YoloStandardPdfPipeline,
                pipeline_options=pipeline_options,
            ),
        }
    )
    return converter


def run(input_dir: Path, output_base: Path, overwrite: bool = False):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = find_pdfs(input_dir)

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print("Building Docling converter (YOLO layout, no OCR)...")
    converter = build_converter()

    processed = 0
    skipped = 0
    failed: list = []

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem
        out_md = out_dir / f"{page_id}.md"
        if out_md.exists() and not overwrite:
            skipped += 1
            continue

        try:
            with timed(use_cuda=True) as t:
                result = converter.convert(str(pdf_path))
                markdown = markdown_with_html_tables(result.document)

            write_page_markdown(out_dir, page_id, markdown)
            append_timing_row(
                timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds
            )
            processed += 1
        except Exception as exc:
            tqdm.write(
                f"[{METHOD_NAME}] FAILED {page_id}: {type(exc).__name__}: {exc}"
            )
            failed.append(page_id)
            continue

    write_summary(out_dir, METHOD_NAME, {
        "total_pages": len(pdf_files),
        "processed": processed,
        "skipped_existing": skipped,
        "failed": len(failed),
    })
    print(
        f"{METHOD_NAME}: processed={processed} skipped={skipped} "
        f"failed={len(failed)} -> {out_dir}"
    )
    if failed:
        print(
            f"  failed page_ids ({len(failed)}): {', '.join(failed[:10])}"
            + (" ..." if len(failed) > 10 else "")
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract PDFs whose output .md already exists.  Default "
             "is to skip them, which makes the run resumable after an "
             "interruption (just rerun the same command).",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
