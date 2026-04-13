"""
Tier 2 -- SAM encoder + trained FCOS detection head (DocLayNet).

Uses the frozen SAM ViT-B encoder from DeepSeek OCR 2 with a lightweight
FPN + FCOS head trained on DocLayNet to detect labelled layout regions.
Text is extracted from the native PDF text layer within each region.

Supports two reading order modes:
  --order geometric     Naive top-to-bottom, left-to-right sort
  --order layoutreader  LayoutReader (LayoutLMv3) learned reading order

This allows direct comparison against YOLO-based methods using the
same reading order models.

Usage:
    python -m benchmark.tier2_hybrid.sam.extract_det \
        --order geometric --input-dir OmniDocBench/pdfs
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pypdfium2 as pdfium
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from benchmark.config import (
    DEEPSEEK_OCR2_MODEL,
    OMNIDOCBENCH_PDFS,
    OUTPUT_DIR,
    RENDER_DPI,
    PROJECT_ROOT,
    find_pdfs,
)
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import pixel_to_pdf_coords, render_page
from benchmark.tier2_hybrid.sam.encoder import load_sam_encoder, preprocess_image
from benchmark.tier2_hybrid.shared import (
    Region,
    regions_to_markdown,
    sort_regions_geometric,
)
from benchmark.tier2_hybrid.sam.detector import SAMDetector
from benchmark.timing import append_timing_row, timed

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_DETECTOR_CKPT = MODELS_DIR / "sam_doclaynet_head.pt"


def load_detector(
    sam_model_path: str,
    detector_ckpt: Path,
    device: torch.device,
) -> SAMDetector:
    """
    Load the SAM encoder and trained FPN+FCOS head from checkpoints.
    """
    sam_encoder = load_sam_encoder(sam_model_path, device)
    detector = SAMDetector(sam_encoder).to(device)

    ckpt = torch.load(str(detector_ckpt), map_location=device)
    detector.fpn.load_state_dict(ckpt["fpn_state_dict"])
    detector.head.load_state_dict(ckpt["head_state_dict"])
    detector.eval()

    return detector


def detect_page(
    detector: SAMDetector,
    image_tensor: torch.Tensor,
    img_scale: float,
    render_scale: float,
    score_threshold: float = 0.3,
) -> List[Region]:
    """
    Run detector on a preprocessed image tensor and return Regions
    in PDF-point coordinates.

    The coordinate chain is:
      detector pixel coords (1024x1024 space)
      -> original rendered image coords (undo img_scale)
      -> PDF point coords (undo render_scale)
    """
    detections = detector.detect(image_tensor, score_threshold=score_threshold)
    regions: List[Region] = []

    for label, x1, y1, x2, y2, score in detections[0]:
        rx1 = x1 / img_scale
        ry1 = y1 / img_scale
        rx2 = x2 / img_scale
        ry2 = y2 / img_scale
        px1, py1, px2, py2 = pixel_to_pdf_coords(rx1, ry1, rx2, ry2, render_scale)
        regions.append(Region(label, px1, py1, px2, py2, score))

    return regions


def run(
    input_dir: Path,
    output_base: Path,
    order_mode: str,
    sam_model_path: str,
    detector_ckpt: Path,
):
    method_name = f"tier2_sam_det_{order_mode}"
    out_dir = method_output_dir(output_base, method_name)
    timing_csv = out_dir / "timing.csv"
    pdf_files = find_pdfs(input_dir)

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading SAM detector (encoder: {sam_model_path})...")
    print(f"  Detection head: {detector_ckpt}")
    detector = load_detector(sam_model_path, detector_ckpt, device)

    lr_model = None

    if order_mode == "layoutreader":
        from transformers import LayoutLMv3ForTokenClassification
        from benchmark.config import LAYOUTREADER_MODEL
        print(f"Loading LayoutReader: {LAYOUTREADER_MODEL}")
        lr_model = (
            LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTREADER_MODEL)
            .to(dtype=torch.bfloat16, device=device)
            .eval()
        )

    for pdf_path in tqdm(pdf_files, desc=method_name):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            image, render_scale = render_page(str(pdf_path), dpi=RENDER_DPI)
            tensor, img_scale, new_w, new_h = preprocess_image(image, target_size=1024)
            tensor = tensor.to(device=device, dtype=torch.float16)

            regions = detect_page(
                detector, tensor, img_scale, render_scale
            )

            if order_mode == "geometric":
                regions = sort_regions_geometric(regions)

            elif order_mode == "layoutreader":
                from benchmark.tier2_hybrid.shared import predict_reading_order
                doc = pdfium.PdfDocument(str(pdf_path))
                page = doc[0]
                page_width, page_height = page.get_size()
                page.close()
                doc.close()
                regions = predict_reading_order(lr_model, regions, page_width, page_height)

            markdown = regions_to_markdown(regions, str(pdf_path))

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(timing_csv, method_name, page_id, t.wall_seconds, t.cuda_seconds)

    write_summary(out_dir, method_name, {
        "total_pages": len(pdf_files),
        "order_mode": order_mode,
    })
    print(f"{method_name}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--order", choices=["geometric", "layoutreader"],
        default="geometric",
        help="Reading order method",
    )
    parser.add_argument(
        "--sam-model-path", type=str, default=DEEPSEEK_OCR2_MODEL,
        help="HuggingFace model ID or local path for SAM encoder",
    )
    parser.add_argument(
        "--detector-ckpt", type=Path, default=DEFAULT_DETECTOR_CKPT,
        help="Path to trained FPN+FCOS checkpoint",
    )
    args = parser.parse_args()
    run(
        args.input_dir, args.output_dir, args.order,
        args.sam_model_path, args.detector_ckpt,
    )


if __name__ == "__main__":
    main()
