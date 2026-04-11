"""
Tier 1 -- Native text-layer extraction using PyMuPDF (fitz).

Provides two extraction modes:
  - "raw"      : page.get_text("text") -- plain token stream
  - "markdown" : pymupdf4llm.to_markdown() -- basic structural reconstruction
                 (headings, bold/italic, table heuristics)

Usage:
    python -m benchmark.tier1_text_layer.extract_pymupdf [--input-dir path]
                                                          [--output-dir path]
                                                          [--mode raw|markdown]
"""

import argparse
import sys
from pathlib import Path

import pymupdf  # fitz
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.timing import append_timing_row, timed

METHOD_RAW = "tier1_pymupdf_raw"
METHOD_MD = "tier1_pymupdf_md"


def extract_raw(pdf_path: str, page_index: int = 0) -> str:
    """
    Plain text extraction via PyMuPDF.  Returns the raw text string
    in whatever order MuPDF's layout engine produces.
    """
    doc = pymupdf.open(pdf_path)
    page = doc[page_index]
    text = page.get_text("text")
    doc.close()
    return text


def extract_markdown(pdf_path: str, page_index: int = 0) -> str:
    """
    Markdown extraction via pymupdf4llm.  Attempts basic structural
    recovery (headings, bold/italic, tables).
    """
    import pymupdf4llm

    md = pymupdf4llm.to_markdown(
        pdf_path,
        pages=[page_index],
        show_progress=False,
    )
    return md


def run(input_dir: Path, output_base: Path, mode: str):
    """
    Process every single-page PDF in *input_dir*.
    """
    if mode == "markdown":
        method_name = METHOD_MD
        extractor = extract_markdown
    else:
        method_name = METHOD_RAW
        extractor = extract_raw

    out_dir = method_output_dir(output_base, method_name)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    for pdf_path in tqdm(pdf_files, desc=method_name):
        page_id = pdf_path.stem

        with timed() as t:
            text = extractor(str(pdf_path))

        write_page_markdown(out_dir, page_id, text)
        append_timing_row(timing_csv, method_name, page_id, t.wall_seconds)

    total_pages = len(pdf_files)
    write_summary(out_dir, method_name, {"total_pages": total_pages, "mode": mode})
    print(f"{method_name}: processed {total_pages} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=OMNIDOCBENCH_PDFS,
        help="Directory of single-page PDFs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base output directory",
    )
    parser.add_argument(
        "--mode",
        choices=["raw", "markdown"],
        default="raw",
        help="Extraction mode: 'raw' for plain text, 'markdown' for pymupdf4llm",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.mode)


if __name__ == "__main__":
    main()
