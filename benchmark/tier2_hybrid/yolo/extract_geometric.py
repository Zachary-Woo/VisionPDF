"""
Tier 2 -- YOLO-DocLayNet bounding-box detection + text-layer extraction.

Pipeline:
  1. Render PDF page to image.
  2. Run YOLO (trained on DocLayNet) to detect layout regions.
  3. Map pixel bounding boxes back to PDF coordinates.
  4. Extract text from the native text layer within each region.
  5. Order regions top-to-bottom, left-to-right (naive geometric sort).
  6. Reconstruct markdown from the labelled, ordered regions.

This is the simplest hybrid approach: vision supplies region labels,
the text layer supplies character content.

Usage:
    python -m benchmark.tier2_hybrid.yolo.extract_geometric [--input-dir path]
                                                             [--output-dir path]
"""

import argparse
import sys
from pathlib import Path
from typing import List

from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import (
    DOCLAYNET_LABELS,
    OMNIDOCBENCH_PDFS,
    OUTPUT_DIR,
    RENDER_DPI,
    YOLO_MODEL,
)
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import pixel_to_pdf_coords, render_page
from benchmark.tier2_hybrid.shared import (
    Region,
    regions_to_markdown,
    sort_regions_geometric,
)
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier2_yolo_geometric"


# ── YOLO detection ───────────────────────────────────────────────────────

def detect_regions(model: YOLO, image, scale: float) -> List[Region]:
    """
    Run YOLO on *image* and return a list of Regions whose coordinates
    are in PDF-point space (divided by *scale*).
    """
    results = model(image, verbose=False)[0]
    regions: List[Region] = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        px1, py1, px2, py2 = pixel_to_pdf_coords(x1, y1, x2, y2, scale)
        label = DOCLAYNET_LABELS[cls_id] if cls_id < len(DOCLAYNET_LABELS) else "Text"
        regions.append(Region(label, px1, py1, px2, py2, conf))
    return regions


# ── Main pipeline ────────────────────────────────────────────────────────

def run(input_dir: Path, output_base: Path):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Loading YOLO model: {YOLO_MODEL}")
    try:
        model = YOLO(YOLO_MODEL)
    except Exception:
        from huggingface_hub import hf_hub_download
        local_pt = hf_hub_download(YOLO_MODEL, filename="yolov8x-doclaynet.pt")
        model = YOLO(local_pt)

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            image, scale = render_page(str(pdf_path), dpi=RENDER_DPI)
            regions = detect_regions(model, image, scale)
            regions = sort_regions_geometric(regions)
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
