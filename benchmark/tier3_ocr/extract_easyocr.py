"""
Tier 3 -- EasyOCR traditional deep-learning OCR.

Renders each PDF page to an image, then runs EasyOCR (CRAFT text
detection + CRNN recognition) to extract text.  Detected text regions
are sorted top-to-bottom, left-to-right and concatenated into markdown.

EasyOCR has no structural understanding (no table detection, no heading
classification), so it is expected to score lowest on structural metrics
but provides a useful throughput/accuracy baseline.

Usage:
    python -m benchmark.tier3_ocr.extract_easyocr [--input-dir path]
                                                    [--output-dir path]
                                                    [--languages en]
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR, RENDER_DPI
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import render_page
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier3_easyocr"


def sort_detections(
    detections: List[Tuple],
    line_tolerance: float = 15.0,
) -> List[Tuple]:
    """
    Sort EasyOCR detections into reading order (top-to-bottom,
    left-to-right), grouping detections on roughly the same line.

    Each detection is (bbox_points, text, confidence).
    bbox_points is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]].
    """
    if not detections:
        return detections

    def center(det):
        pts = det[0]
        cy = (pts[0][1] + pts[2][1]) / 2.0
        cx = (pts[0][0] + pts[2][0]) / 2.0
        return cy, cx

    detections = sorted(detections, key=lambda d: center(d))

    lines: List[List[Tuple]] = []
    current_line: List[Tuple] = [detections[0]]
    current_y = center(detections[0])[0]

    for det in detections[1:]:
        cy, _ = center(det)
        if abs(cy - current_y) <= line_tolerance:
            current_line.append(det)
        else:
            lines.append(current_line)
            current_line = [det]
            current_y = cy
    lines.append(current_line)

    ordered: List[Tuple] = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda d: center(d)[1]))
    return ordered


def detections_to_markdown(detections: List[Tuple]) -> str:
    """
    Join sorted EasyOCR detections into a plain-text markdown string.
    Detections on the same line are space-separated; lines are separated
    by newlines; paragraphs (larger vertical gaps) get blank lines.
    """
    if not detections:
        return ""

    GAP_THRESHOLD = 30.0

    def top_y(det):
        return det[0][0][1]

    parts: List[str] = []
    prev_bottom = None

    for det in detections:
        text = det[1].strip()
        if not text:
            continue

        cur_top = top_y(det)
        if prev_bottom is not None and (cur_top - prev_bottom) > GAP_THRESHOLD:
            parts.append("")

        parts.append(text)
        prev_bottom = det[0][2][1]

    return "\n".join(parts)


def run(input_dir: Path, output_base: Path, languages: List[str]):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    import easyocr

    print(f"Initialising EasyOCR reader (languages={languages})...")
    reader = easyocr.Reader(languages, gpu=True)

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            image, _ = render_page(str(pdf_path), dpi=RENDER_DPI)
            img_array = np.array(image)

            detections = reader.readtext(img_array)
            detections = sort_detections(detections)
            markdown = detections_to_markdown(detections)

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(
            timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds
        )

    write_summary(
        out_dir, METHOD_NAME,
        {"total_pages": len(pdf_files), "languages": languages},
    )
    print(f"{METHOD_NAME}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en"],
        help="Language codes for EasyOCR (default: en)",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.languages)


if __name__ == "__main__":
    main()
