"""
Visualize YOLO and LayoutReader region detection on a single PDF page.

Produces annotated images, markdown output, and ground truth reference:
  - {stem}_yolo_raw.png          -- all YOLO detections before merging
  - {stem}_yolo_geometric.png    -- merged boxes in geometric reading order
  - {stem}_yolo_layoutreader.png -- merged boxes reordered by LayoutReader
  - {stem}_yolo_geometric.md     -- extracted markdown (geometric)
  - {stem}_yolo_layoutreader.md  -- extracted markdown (layoutreader)
  - {stem}_ground_truth.md       -- OmniDocBench ground truth for comparison

Reading-order numbers are drawn on each box so you can see how the two
ordering strategies differ.

Usage:
    python demo/visualize_regions.py <path_to_pdf>
    python demo/visualize_regions.py OmniDocBench/ori_pdfs/docstructbench_00039896.1983.10545823.pdf_1.pdf
"""

import argparse
import json
import pathlib
import platform
import sys
from pathlib import Path
from typing import Optional

if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pypdfium2 as pdfium
import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import LayoutLMv3ForTokenClassification
from ultralytics import YOLO

from benchmark.config import (
    DOCLAYNET_LABELS,
    LAYOUTREADER_MODEL,
    OMNIDOCBENCH_JSON,
    RENDER_DPI,
    YOLO_MODEL,
    YOLO_MODEL_FILE,
    YOLO_MODEL_REPO,
)
from benchmark.pdf_render import render_page
from benchmark.tier2_hybrid.shared import (
    Region,
    merge_overlapping_regions,
    predict_reading_order,
    regions_to_markdown,
    sort_regions_geometric,
)

LABEL_COLORS = {
    "Caption":        (255, 165,   0),
    "Footnote":       (128, 128, 128),
    "Formula":        (148,   0, 211),
    "List-item":      (  0, 128, 128),
    "Page-footer":    (169, 169, 169),
    "Page-header":    (169, 169, 169),
    "Picture":        ( 65, 105, 225),
    "Section-header": (220,  20,  60),
    "Table":          ( 34, 139,  34),
    "Text":           ( 30, 144, 255),
    "Title":          (255,  50,  50),
}


def _get_font(size: int = 14):
    """Load a basic font, falling back to the default bitmap font."""
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except OSError:
            return ImageFont.load_default()


def draw_regions(
    image: Image.Image,
    regions: list,
    scale: float,
    title: str,
    show_order: bool = True,
) -> Image.Image:
    """
    Draw labelled bounding boxes on a copy of the rendered page image.

    Each box gets a coloured outline, a label tag, and (optionally) a
    large reading-order number in its top-left corner.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    title_font = _get_font(16)
    label_font = _get_font(11)
    order_font = _get_font(22)

    for idx, region in enumerate(regions):
        px1 = region.x1 * scale
        py1 = region.y1 * scale
        px2 = region.x2 * scale
        py2 = region.y2 * scale

        color = LABEL_COLORS.get(region.label, (200, 200, 200))

        draw.rectangle([px1, py1, px2, py2], outline=color, width=2)

        tag = f"{region.label} {region.confidence:.0%}"
        tag_y = max(py1 - 14, 0)
        draw.rectangle(
            [px1, tag_y, px1 + len(tag) * 6.5 + 4, tag_y + 13],
            fill=color,
        )
        draw.text((px1 + 2, tag_y), tag, fill="white", font=label_font)

        if show_order:
            num = str(idx + 1)
            draw.rectangle(
                [px1 + 1, py1 + 1, px1 + 26, py1 + 26],
                fill=color,
            )
            draw.text((px1 + 4, py1 + 1), num, fill="white", font=order_font)

    draw.rectangle([4, 4, len(title) * 10 + 12, 24], fill="white")
    draw.text((8, 5), title, fill="black", font=title_font)

    return img


def load_ground_truth_for_page(
    page_stem: str, gt_json: Path,
) -> Optional[str]:
    """
    Look up the OmniDocBench ground truth for a page and return it as
    formatted markdown text.  Returns None if the page isn't found.

    The page_stem is the PDF filename without extension (e.g.
    ``docstructbench_00039896.1983.10545823.pdf_1``).  GT entries are
    keyed by the image filename stem which matches the PDF stem.
    """
    if not gt_json.exists():
        return None

    with open(gt_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data:
        image_path = entry.get("page_info", {}).get("image_path", "")
        if Path(image_path).stem == page_stem:
            layout_dets = entry.get("layout_dets", [])
            ordered = sorted(
                layout_dets,
                key=lambda d: (d.get("order") is None, d.get("order", 0)),
            )

            lines = []
            for det in ordered:
                text = det.get("text", "")
                if not isinstance(text, str) or not text.strip():
                    continue
                cat = det.get("category_type", "")
                order = det.get("order")
                lines.append(f"[order={order} | {cat}]")
                lines.append(text.strip())
                lines.append("")
            return "\n".join(lines) if lines else None

    return None


def detect_regions(model: YOLO, image: Image.Image, scale: float) -> list:
    """Run YOLO on the rendered image and return Regions in PDF-point space."""
    results = model(image, verbose=False)[0]
    regions = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        px1 = x1 / scale
        py1 = y1 / scale
        px2 = x2 / scale
        py2 = y2 / scale
        label = DOCLAYNET_LABELS[cls_id] if cls_id < len(DOCLAYNET_LABELS) else "Text"
        regions.append(Region(label, px1, py1, px2, py2, conf))
    return regions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path, help="Path to a single-page PDF")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parent,
        help="Directory for output files (default: demo/)",
    )
    parser.add_argument("--dpi", type=int, default=RENDER_DPI)
    args = parser.parse_args()

    pdf_path = args.pdf
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = pdf_path.stem

    print(f"Rendering {pdf_path} at {args.dpi} DPI...")
    image, scale = render_page(str(pdf_path), dpi=args.dpi)

    print(f"Loading YOLO model: {YOLO_MODEL_FILE}")
    if YOLO_MODEL.exists():
        yolo_model = YOLO(str(YOLO_MODEL))
    else:
        from huggingface_hub import hf_hub_download
        print(f"  Downloading {YOLO_MODEL_FILE} from {YOLO_MODEL_REPO}...")
        local_pt = hf_hub_download(YOLO_MODEL_REPO, filename=YOLO_MODEL_FILE)
        yolo_model = YOLO(local_pt)

    print("Detecting regions...")
    raw_regions = detect_regions(yolo_model, image, scale)
    print(f"  YOLO raw detections: {len(raw_regions)}")

    # -- Raw YOLO detections (before merge, no ordering) --
    raw_img = draw_regions(
        image, raw_regions, scale,
        f"YOLO Raw ({len(raw_regions)} detections)", show_order=False,
    )
    raw_img_path = out_dir / f"{stem}_yolo_raw.png"
    raw_img.save(str(raw_img_path))
    print(f"  Saved: {raw_img_path}")

    # -- Merge overlapping regions --
    merged = merge_overlapping_regions(raw_regions)
    print(f"  After merge: {len(merged)} regions")

    # -- Geometric ordering --
    geo_regions = sort_regions_geometric(list(merged))
    geo_img = draw_regions(
        image, geo_regions, scale,
        f"Geometric Order ({len(geo_regions)} regions)",
    )
    geo_img_path = out_dir / f"{stem}_yolo_geometric.png"
    geo_img.save(str(geo_img_path))
    print(f"  Saved: {geo_img_path}")

    geo_md = regions_to_markdown(geo_regions, str(pdf_path))
    geo_md_path = out_dir / f"{stem}_yolo_geometric.md"
    geo_md_path.write_text(geo_md, encoding="utf-8")
    print(f"  Saved: {geo_md_path}")

    # -- LayoutReader ordering --
    print("Loading LayoutReader model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr_model = (
        LayoutLMv3ForTokenClassification.from_pretrained(LAYOUTREADER_MODEL)
        .to(dtype=torch.bfloat16, device=device)
        .eval()
    )

    doc = pdfium.PdfDocument(str(pdf_path))
    page = doc[0]
    page_width, page_height = page.get_size()
    page.close()
    doc.close()

    lr_regions = predict_reading_order(lr_model, list(merged), page_width, page_height)
    lr_img = draw_regions(
        image, lr_regions, scale,
        f"LayoutReader Order ({len(lr_regions)} regions)",
    )
    lr_img_path = out_dir / f"{stem}_yolo_layoutreader.png"
    lr_img.save(str(lr_img_path))
    print(f"  Saved: {lr_img_path}")

    lr_md = regions_to_markdown(lr_regions, str(pdf_path))
    lr_md_path = out_dir / f"{stem}_yolo_layoutreader.md"
    lr_md_path.write_text(lr_md, encoding="utf-8")
    print(f"  Saved: {lr_md_path}")

    # -- Ground truth --
    gt_text = load_ground_truth_for_page(stem, OMNIDOCBENCH_JSON)
    if gt_text:
        gt_path = out_dir / f"{stem}_ground_truth.md"
        gt_path.write_text(gt_text, encoding="utf-8")
        print(f"  Saved: {gt_path}")
    else:
        print(f"  Ground truth not found for '{stem}' in {OMNIDOCBENCH_JSON}")

    print(f"\nDone. All outputs in {out_dir}/")
    print(f"  Raw detections:     {raw_img_path.name}")
    print(f"  Geometric order:    {geo_img_path.name} + .md")
    print(f"  LayoutReader order: {lr_img_path.name} + .md")
    if gt_text:
        print(f"  Ground truth:       {stem}_ground_truth.md")


if __name__ == "__main__":
    main()
