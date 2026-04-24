"""
Command-line entry point for DocDet inference.

Exposed as ``python -m layout_detector.docdet.infer.cli``.

Supported operations:
- Single image   -> prints JSON detections to stdout (or
                    ``--output file.json``)
- Directory      -> batches every image into one .jsonl file
- Visualisation  -> optional ``--visualize OUT.png`` dumps a
                    matplotlib rendering of the boxes on top of the
                    input image (only when matplotlib is installed).

The CLI is kept intentionally narrow.  Production integrations (the
Docling pipeline, the benchmark scripts) import ``DocDetPredictor``
directly instead of shelling out.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

from PIL import Image

from .predict import DocDetPredictor, LayoutDetection

logger = logging.getLogger(__name__)

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _discover_images(root: Path) -> List[Path]:
    """Return image paths under ``root``, or [root] if it is a file."""
    if root.is_file():
        return [root]
    paths: List[Path] = []
    for ext in _IMAGE_EXTENSIONS:
        paths.extend(sorted(root.rglob(f"*{ext}")))
    return paths


def _detections_to_dict(page_id: str, dets: Iterable[LayoutDetection]) -> dict:
    """Convert LayoutDetection list to a JSON-serialisable dict."""
    return {
        "page_id": page_id,
        "detections": [asdict(d) for d in dets],
    }


def _maybe_visualize(
    image_path: Path,
    detections: List[LayoutDetection],
    output_path: Path,
) -> None:
    """Draw detections over the input image and save as ``output_path``."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        logger.warning(
            "matplotlib not installed; skipping visualisation. "
            "Install with `pip install matplotlib`."
        )
        return

    pil = Image.open(image_path).convert("RGB")
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(pil)
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.5, edgecolor="red", facecolor="none",
        )
        ax.add_patch(rect)
        ax.text(
            x1, max(y1 - 4, 0),
            f"{d.label} {d.score:.2f}",
            color="white",
            fontsize=7,
            bbox=dict(facecolor="red", edgecolor="none", pad=1.0),
        )
    ax.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True,
                        help=".pt, .pth, or .onnx DocDet weights.")
    parser.add_argument("--input", type=Path, required=True,
                        help="Image file or directory of images.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Write JSON(L) detections here.  Default: stdout.")
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--input-height", type=int, default=1120)
    parser.add_argument("--input-width", type=int, default=800)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"])
    parser.add_argument("--dilate-tables-px", type=int, default=2,
                        help="Pixels to grow Table bboxes by (0 to disable).")
    parser.add_argument("--visualize-dir", type=Path, default=None,
                        help="Directory to dump annotated PNGs.")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    predictor = DocDetPredictor(
        model_path=args.model,
        device=args.device,
        input_size=(args.input_height, args.input_width),
        score_threshold=args.score_threshold,
        nms_threshold=args.nms_threshold,
        dilate_tables_px=args.dilate_tables_px,
    )

    paths = _discover_images(args.input)
    if not paths:
        logger.error("No images found under %s", args.input)
        return 1

    write_handle = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_handle = args.output.open("w", encoding="utf-8")

    try:
        for image_path in paths:
            detections = predictor(image_path)
            page_id = image_path.stem
            record = _detections_to_dict(page_id, detections)

            line = json.dumps(record, ensure_ascii=False)
            if write_handle is not None:
                write_handle.write(line + "\n")
            else:
                sys.stdout.write(line + "\n")

            if args.visualize_dir is not None:
                vis_path = args.visualize_dir / f"{page_id}.png"
                _maybe_visualize(image_path, detections, vis_path)
    finally:
        if write_handle is not None:
            write_handle.close()

    logger.info("Processed %d image(s)", len(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
