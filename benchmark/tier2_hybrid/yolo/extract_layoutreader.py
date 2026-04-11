"""
Tier 2 -- YOLO-DocLayNet + LayoutReader for learned reading order.

Pipeline:
  1. Render PDF page to image.
  2. Run YOLO (trained on DocLayNet) to detect layout regions.
  3. Feed the detected bounding boxes into LayoutReader (LayoutLMv3)
     to predict the correct reading order.
  4. Reorder regions according to LayoutReader's prediction.
  5. Extract text from the native text layer for each region.
  6. Reconstruct markdown.

This extends the basic YOLO hybrid by replacing naive geometric
sorting with a learned reading-order model.

Usage:
    python -m benchmark.tier2_hybrid.yolo.extract_layoutreader [--input-dir path]
                                                                [--output-dir path]
"""

import argparse
import sys
from pathlib import Path

import pypdfium2 as pdfium
import torch
from tqdm import tqdm
from transformers import LayoutLMv3ForTokenClassification
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import (
    LAYOUTREADER_MODEL,
    OMNIDOCBENCH_PDFS,
    OUTPUT_DIR,
    RENDER_DPI,
    YOLO_MODEL,
)
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import render_page
from benchmark.tier2_hybrid.yolo.extract_geometric import detect_regions
from benchmark.tier2_hybrid.shared import (
    predict_reading_order,
    regions_to_markdown,
)
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier2_yolo_layoutreader"


# ── Main pipeline ────────────────────────────────────────────────────────

def run(input_dir: Path, output_base: Path):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading YOLO model: {YOLO_MODEL}")
    try:
        yolo_model = YOLO(YOLO_MODEL)
    except Exception:
        from huggingface_hub import hf_hub_download
        local_pt = hf_hub_download(YOLO_MODEL, filename="yolov8x-doclaynet.pt")
        yolo_model = YOLO(local_pt)

    print(f"Loading LayoutReader: {LAYOUTREADER_MODEL}")
    lr_model = (
        LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTREADER_MODEL)
        .to(dtype=torch.bfloat16, device=device)
        .eval()
    )

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            image, scale = render_page(str(pdf_path), dpi=RENDER_DPI)
            regions = detect_regions(yolo_model, image, scale)

            doc = pdfium.PdfDocument(str(pdf_path))
            page = doc[0]
            page_width, page_height = page.get_size()
            page.close()
            doc.close()

            regions = predict_reading_order(lr_model, regions, page_width, page_height)
            markdown = regions_to_markdown(regions, str(pdf_path))

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds)

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
