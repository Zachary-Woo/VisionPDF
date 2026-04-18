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
import pathlib
import platform
import sys
from pathlib import Path
from typing import List

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

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
    YOLO_MODEL_FILE,
    YOLO_MODEL_REPO,
    find_pdfs,
)
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import render_page
from benchmark.tier2_hybrid.yolo.extract_geometric import detect_regions
from benchmark.tier2_hybrid.shared import (
    Region,
    detect_page_columns,
    sort_columns_reading_order,
    sort_regions_geometric,
    sort_single_column,
)
from benchmark.tier2_hybrid.reading_order import predict_reading_order
from benchmark.tier2_hybrid.yolo.text_assembly import regions_to_markdown
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier2_yolo_layoutreader"

PAGE_SEPARATOR = "\n\n---\n\n"

# Columns with fewer regions than this threshold are considered
# "trivial" and can be sorted geometrically instead of through
# LayoutReader -- the cost of tokenising + running LR is not worth it
# when the column has only 1-2 items and the vertical order is
# unambiguous.
_TRIVIAL_COLUMN_LEN = 3


# ── Reading-order dispatch ───────────────────────────────────────────────

def _resolve_reading_order(
    lr_model,
    regions: List[Region],
    page_width: float,
    page_height: float,
) -> List[Region]:
    """
    Pick the cheapest reading-order strategy that still produces a
    correct ordering for this page's layout.

    Three tiers, in order of cost:
      1. Single column and no spanning elements -> geometric sort only.
         LayoutReader is skipped entirely; on OmniDocBench this path
         covers the majority of pages.
      2. Multi-column but every column has fewer than _TRIVIAL_COLUMN_LEN
         regions and no spanning elements -> per-column geometric sort
         plus left-to-right column merge.
      3. Anything with spanning regions or dense columns -> run
         LayoutReader once per column and interleave with spanning
         elements the same way as before.
    """
    columns, spanning = detect_page_columns(regions)

    if len(columns) == 1 and not spanning:
        return sort_regions_geometric(regions)

    if not spanning and all(len(col) < _TRIVIAL_COLUMN_LEN for col in columns):
        ordered_cols = [sort_single_column(col) for col in columns]
        return sort_columns_reading_order(ordered_cols, spanning)

    ordered_cols = [
        predict_reading_order(lr_model, col, page_width, page_height)
        for col in columns
    ]
    return sort_columns_reading_order(ordered_cols, spanning)


# ── Main pipeline ────────────────────────────────────────────────────────

def run(input_dir: Path, output_base: Path, enable_tables: bool = True,
        table_mode: str = "accurate"):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = find_pdfs(input_dir)

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading YOLO model: {YOLO_MODEL_FILE}")
    if YOLO_MODEL.exists():
        yolo_model = YOLO(str(YOLO_MODEL))
    else:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading {YOLO_MODEL_FILE} from {YOLO_MODEL_REPO}...")
        local_pt = hf_hub_download(YOLO_MODEL_REPO, filename=YOLO_MODEL_FILE)
        yolo_model = YOLO(local_pt)

    print(f"Loading LayoutReader: {LAYOUTREADER_MODEL}")
    lr_model = (
        LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTREADER_MODEL)
        .to(dtype=torch.bfloat16, device=device)
        .eval()
    )

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        doc = pdfium.PdfDocument(str(pdf_path))
        n_pages = len(doc)
        page_sizes = []
        for pi in range(n_pages):
            pg = doc[pi]
            page_sizes.append(pg.get_size())
            pg.close()
        doc.close()

        page_parts: List[str] = []
        phase_totals = {
            "t_render": 0.0,
            "t_detect": 0.0,
            "t_reading_order": 0.0,
            "t_assemble": 0.0,
        }
        cuda_total = 0.0
        cuda_seen = False

        for pi in range(n_pages):
            page_width, page_height = page_sizes[pi]

            with timed(use_cuda=True) as t_r:
                image, scale = render_page(
                    str(pdf_path), page_index=pi, dpi=RENDER_DPI,
                )
            with timed(use_cuda=True) as t_d:
                regions = detect_regions(yolo_model, image, scale)
            with timed(use_cuda=True) as t_o:
                regions = _resolve_reading_order(
                    lr_model, regions, page_width, page_height,
                )
            with timed(use_cuda=False) as t_a:
                page_md = regions_to_markdown(
                    regions, str(pdf_path), page_index=pi,
                    page_image=image,
                    enable_tables=enable_tables,
                    table_mode=table_mode,
                )

            page_parts.append(page_md)
            phase_totals["t_render"] += t_r.wall_seconds
            phase_totals["t_detect"] += t_d.wall_seconds
            phase_totals["t_reading_order"] += t_o.wall_seconds
            phase_totals["t_assemble"] += t_a.wall_seconds
            for phase_timer in (t_r, t_d, t_o):
                if phase_timer.cuda_seconds is not None:
                    cuda_total += phase_timer.cuda_seconds
                    cuda_seen = True

        markdown = PAGE_SEPARATOR.join(page_parts)
        wall_total = sum(phase_totals.values())
        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(
            timing_csv, METHOD_NAME, page_id,
            wall_total, cuda_total if cuda_seen else None,
            breakdown=phase_totals,
        )

    write_summary(out_dir, METHOD_NAME, {"total_pages": len(pdf_files)})
    print(f"{METHOD_NAME}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--tables", dest="enable_tables", action="store_true",
        default=True,
        help="Run TableFormer on YOLO Table regions to produce HTML "
             "cell structure (default: on).",
    )
    parser.add_argument(
        "--no-tables", dest="enable_tables", action="store_false",
        help="Emit [Table] placeholder instead of running TableFormer "
             "(faster, avoids the TableFormer weight download).",
    )
    parser.add_argument(
        "--table-mode", choices=["accurate", "fast"], default="accurate",
        help="TableFormer quality mode (default: accurate).",
    )
    args = parser.parse_args()
    run(
        args.input_dir, args.output_dir,
        enable_tables=args.enable_tables,
        table_mode=args.table_mode,
    )


if __name__ == "__main__":
    main()
