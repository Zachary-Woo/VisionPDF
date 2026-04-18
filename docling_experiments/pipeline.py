"""
Lightweight hybrid PDF extraction pipeline.

Uses Docling's layout model (Heron RT-DETR) for structural detection
and reading order, with all character content pulled directly from
the PDF text layer.  Post-processing rules clean up common layout-
model mistakes (margin line numbers, over-segmented paragraphs,
text misidentified as tables).

Routing:
  - Text layer present -> layout model + text layer (no OCR)
  - No text layer      -> layout model + OCR fallback

Usage:
    python -m yolo_experiments.pipeline --input path/to/doc.pdf
    python -m yolo_experiments.pipeline --input path/to/pdfs/
    python -m yolo_experiments.pipeline --input doc.pdf --force-ocr
    python -m yolo_experiments.pipeline --input doc.pdf --table-mode fast
    python -m yolo_experiments.pipeline --input doc.pdf --no-postprocess
    python -m yolo_experiments.pipeline --input doc.pdf --no-visualize
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

import pypdfium2 as pdfium

from yolo_experiments.config import (
    DEFAULT_OUTPUT_DIR,
    DO_CELL_MATCHING,
    DO_POSTPROCESS,
    TABLE_MODE,
    TEXT_LAYER_MIN_CHARS,
    TEXT_LAYER_MIN_PAGE_RATIO,
    TEXT_LAYER_SAMPLE_PAGES,
)
from yolo_experiments.assembly import assemble_markdown
from yolo_experiments.postprocess import postprocess

log = logging.getLogger(__name__)


# =========================================================================
# Text-layer detection
# =========================================================================

def has_text_layer(pdf_path: str, min_chars: int = TEXT_LAYER_MIN_CHARS,
                   sample_pages: int = TEXT_LAYER_SAMPLE_PAGES,
                   min_ratio: float = TEXT_LAYER_MIN_PAGE_RATIO) -> bool:
    """
    Check whether a PDF has a usable native text layer.

    Opens the PDF with pypdfium2 and extracts raw text from a sample
    of pages.  If a sufficient fraction of sampled pages contain at
    least *min_chars* non-whitespace characters, the PDF is considered
    to have a text layer.

    Args:
        pdf_path:     Path to the PDF file.
        min_chars:    Minimum non-whitespace characters per page.
        sample_pages: Maximum number of pages to sample.
        min_ratio:    Fraction of sampled pages that must pass.

    Returns:
        True if the PDF has a usable text layer.
    """
    doc = pdfium.PdfDocument(pdf_path)
    total_pages = len(doc)
    pages_to_check = min(total_pages, sample_pages)

    passes = 0
    for i in range(pages_to_check):
        page = doc[i]
        text_page = page.get_textpage()
        text = text_page.get_text_bounded()
        text_page.close()
        page.close()

        non_ws = len(text.replace(" ", "").replace("\n", "").replace("\t", ""))
        if non_ws >= min_chars:
            passes += 1

    doc.close()

    ratio = passes / max(pages_to_check, 1)
    result = ratio >= min_ratio
    log.info("Text-layer check: %d/%d pages passed (%.0f%%) -> %s",
             passes, pages_to_check, ratio * 100,
             "text-layer" if result else "OCR needed")
    return result


# =========================================================================
# Docling converter construction
# =========================================================================

def build_converter(use_ocr: bool = False, table_mode: str = TABLE_MODE,
                    cell_matching: bool = DO_CELL_MATCHING):
    """
    Build a Docling DocumentConverter with the appropriate settings.

    Args:
        use_ocr:       Enable OCR (for scanned / image-only PDFs).
        table_mode:    "accurate" or "fast" -- TableFormer quality mode.
        cell_matching: Map table structure predictions back to PDF
                       text cells (True) or use model-predicted cells
                       (False).

    Returns:
        A configured ``DocumentConverter`` instance.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableFormerMode,
        TableStructureOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    mode = (TableFormerMode.ACCURATE if table_mode == "accurate"
            else TableFormerMode.FAST)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(
        do_cell_matching=cell_matching,
        mode=mode,
    )

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            ),
        }
    )
    return converter


# =========================================================================
# Core conversion
# =========================================================================

def convert_pdf(pdf_path: str, output_dir: str,
                force_ocr: bool = False,
                table_mode: str = TABLE_MODE,
                cell_matching: bool = DO_CELL_MATCHING,
                export_tables: bool = True,
                visualize: bool = True,
                do_postprocess: bool = DO_POSTPROCESS) -> dict:
    """
    Convert a single PDF through the hybrid pipeline.

    Layout detection and reading order come from Docling's Heron
    model.  Character content comes from the PDF text layer (or OCR
    if no text layer exists).  Post-processing rules clean up common
    layout-model mistakes before the final markdown is written.

    Args:
        pdf_path:       Path to the PDF file.
        output_dir:     Directory for output files.
        force_ocr:      Skip text-layer detection and always enable OCR.
        table_mode:     "accurate" or "fast" TableFormer mode.
        cell_matching:  Use PDF text cells for table content.
        export_tables:  Write per-table HTML and CSV files.
        visualize:      Save annotated layout PNG(s) showing detected regions.
        do_postprocess: Apply post-processing rules to clean up markdown.

    Returns:
        A dict with conversion metadata (path, mode, timing, table count).
    """
    pdf_path = str(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(pdf_path).stem

    # --- Detect text layer ---
    if force_ocr:
        use_ocr = True
        log.info("OCR forced on for %s", stem)
    else:
        use_ocr = not has_text_layer(pdf_path)

    mode_label = "ocr" if use_ocr else "text_layer"
    log.info("Converting %s (mode: %s, table: %s)", stem, mode_label, table_mode)

    # --- Build converter and run ---
    converter = build_converter(
        use_ocr=use_ocr,
        table_mode=table_mode,
        cell_matching=cell_matching,
    )

    start = time.perf_counter()
    result = converter.convert(pdf_path)
    elapsed = time.perf_counter() - start

    doc = result.document

    # --- Export main markdown ---
    # Custom assembler extracts text directly from the PDF text layer
    # for each detected region, bypassing Docling's lossy text assembly.
    assembled_markdown = assemble_markdown(result, pdf_path)

    if do_postprocess:
        markdown = postprocess(assembled_markdown)
    else:
        markdown = assembled_markdown

    md_path = output_dir / f"{stem}.md"
    md_path.write_text(markdown, encoding="utf-8")

    # Keep the raw Docling built-in export for comparison / debugging.
    raw_docling = doc.export_to_markdown()
    raw_path = output_dir / f"{stem}_docling_raw.md"
    raw_path.write_text(raw_docling, encoding="utf-8")

    # Also keep the pre-postprocess assembled output.
    assembled_path = output_dir / f"{stem}_assembled.md"
    assembled_path.write_text(assembled_markdown, encoding="utf-8")

    # --- Export tables ---
    table_count = 0
    if export_tables:
        for idx, table in enumerate(doc.tables):
            table_count += 1

            html = table.export_to_html(doc=doc)
            html_path = output_dir / f"{stem}_table_{idx + 1}.html"
            html_path.write_text(html, encoding="utf-8")

            try:
                import pandas as pd
                df = table.export_to_dataframe(doc=doc)
                csv_path = output_dir / f"{stem}_table_{idx + 1}.csv"
                df.to_csv(csv_path, index=False)
            except ImportError:
                log.debug("pandas not available; skipping CSV export")

    # --- Layout visualization ---
    layout_images = []
    if visualize:
        from yolo_experiments.visualize import save_layout_images
        layout_images = save_layout_images(result, pdf_path, str(output_dir))
        for img_path in layout_images:
            log.info("Saved layout image: %s", img_path.name)

    # --- Write metadata ---
    meta = {
        "source": pdf_path,
        "mode": mode_label,
        "table_mode": table_mode,
        "cell_matching": cell_matching,
        "tables_found": table_count,
        "layout_images": [str(p) for p in layout_images],
        "elapsed_seconds": round(elapsed, 3),
    }
    meta_path = output_dir / f"{stem}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Done: %s (%.2fs, %d tables) -> %s",
             stem, elapsed, table_count, output_dir)
    return meta


def convert_directory(input_dir: str, output_dir: str,
                      force_ocr: bool = False,
                      table_mode: str = TABLE_MODE,
                      cell_matching: bool = DO_CELL_MATCHING,
                      export_tables: bool = True,
                      visualize: bool = True,
                      do_postprocess: bool = DO_POSTPROCESS) -> List[dict]:
    """
    Convert all PDFs in a directory.

    Each PDF gets its own subdirectory under *output_dir* named after
    the PDF stem.

    Returns:
        A list of per-PDF metadata dicts.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        log.warning("No PDFs found in %s", input_dir)
        return []

    log.info("Found %d PDFs in %s", len(pdf_files), input_dir)
    results = []

    for pdf_path in pdf_files:
        pdf_out = output_dir / pdf_path.stem
        meta = convert_pdf(
            str(pdf_path), str(pdf_out),
            force_ocr=force_ocr,
            table_mode=table_mode,
            cell_matching=cell_matching,
            export_tables=export_tables,
            visualize=visualize,
            do_postprocess=do_postprocess,
        )
        results.append(meta)

    # Write a summary of the batch
    summary_path = output_dir / "batch_summary.json"
    summary = {
        "total_pdfs": len(results),
        "total_tables": sum(r["tables_found"] for r in results),
        "total_elapsed": round(sum(r["elapsed_seconds"] for r in results), 3),
        "files": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Batch complete: %d PDFs, %d tables, %.2fs total",
             summary["total_pdfs"], summary["total_tables"],
             summary["total_elapsed"])
    return results


# =========================================================================
# CLI entrypoint
# =========================================================================

def main(argv: Optional[List[str]] = None):
    """Command-line interface for the Docling extraction pipeline."""
    parser = argparse.ArgumentParser(
        description="Docling PDF extraction pipeline with automatic "
                    "text-layer detection and cell-level table extraction.",
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to a single PDF or a directory of PDFs.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory (default: results/docling_pipeline).",
    )
    parser.add_argument(
        "--force-ocr", action="store_true",
        help="Always enable OCR regardless of text-layer detection.",
    )
    parser.add_argument(
        "--table-mode", choices=["accurate", "fast"], default=TABLE_MODE,
        help="TableFormer quality mode (default: accurate).",
    )
    parser.add_argument(
        "--no-tables", action="store_true",
        help="Skip per-table HTML/CSV export (tables still appear "
             "in the markdown output).",
    )
    parser.add_argument(
        "--no-postprocess", action="store_true",
        help="Skip post-processing cleanup rules.",
    )
    parser.add_argument(
        "--no-visualize", action="store_true",
        help="Skip layout visualization PNG output.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(name)s  %(message)s",
    )

    input_path: Path = args.input
    export_tables = not args.no_tables
    do_postprocess = not args.no_postprocess
    visualize = not args.no_visualize

    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        convert_pdf(
            str(input_path), str(args.output),
            force_ocr=args.force_ocr,
            table_mode=args.table_mode,
            export_tables=export_tables,
            visualize=visualize,
            do_postprocess=do_postprocess,
        )
    elif input_path.is_dir():
        convert_directory(
            str(input_path), str(args.output),
            force_ocr=args.force_ocr,
            table_mode=args.table_mode,
            export_tables=export_tables,
            visualize=visualize,
            do_postprocess=do_postprocess,
        )
    else:
        parser.error(f"Input must be a PDF file or directory: {input_path}")


if __name__ == "__main__":
    main()
