"""
Tier 1 -- Native text-layer extraction using pypdfium2.

Extracts raw text from each PDF page via PDFium's text API.  No vision
model is involved; structural ordering is whatever the PDF's internal
token stream happens to produce.

Usage:
    python -m benchmark.tier1_text_layer.extract_pypdfium2 [--input-dir path]
                                                            [--output-dir path]
"""

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier1_pypdfium2"


def extract_page(pdf_path: str, page_index: int = 0) -> str:
    """
    Extract all text from a single PDF page using pypdfium2's text API.

    Returns the raw text string.
    """
    doc = pdfium.PdfDocument(pdf_path)
    page = doc[page_index]
    text_page = page.get_textpage()
    text = text_page.get_text_range()
    text_page.close()
    page.close()
    doc.close()
    return text


def run(input_dir: Path, output_base: Path):
    """
    Process every single-page PDF in *input_dir*, write per-page markdown
    and timing CSV to the output directory.
    """
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed() as t:
            text = extract_page(str(pdf_path))

        write_page_markdown(out_dir, page_id, text)
        append_timing_row(timing_csv, METHOD_NAME, page_id, t.wall_seconds)

    total_pages = len(pdf_files)
    write_summary(out_dir, METHOD_NAME, {"total_pages": total_pages})
    print(f"{METHOD_NAME}: processed {total_pages} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=OMNIDOCBENCH_PDFS,
        help="Directory of single-page PDFs (default: OmniDocBench/pdfs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Base output directory (default: results/)",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
