"""
Export a trained DocDet checkpoint to ONNX.

Design choices (per spec 9.1):

1. Opset 17 is chosen because it is the oldest opset that
   reliably supports all the tensor ops DocDet emits (including
   ``torch.scatter`` patterns used inside the head's bias init and
   ``torch.where`` with broadcast) while still being supported by
   ONNX Runtime's CPU EP.
2. Dynamic axes are enabled on the batch dim AND the spatial dims,
   so a single .onnx file can be served at 800x1120 (docs) and
   640x640 (natural-image debugging) without re-exporting.
3. Only the FCOS head's raw outputs are exported. NMS is kept in
   Python because ONNX's NonMaxSuppression is limited (single IoU
   threshold, no class-aware batching) and documents tolerate the
   small post-process cost.
4. We DO export the per-level Scale + exp so the consumer gets
   stride-normalised LTRB distances ready for decoding.

Usage
-----
python -m layout_detector.docdet.export.export_onnx \\
    --checkpoint layout_detector/weights/phase2/last.pt \\
    --output layout_detector/weights/docdet.onnx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn

from ..data.label_map import NUM_DOCDET_CLASSES
from ..model.model import DocDet

logger = logging.getLogger(__name__)


class _DocDetExportWrapper(nn.Module):
    """
    Wrap ``DocDet.forward`` so ONNX export gets a flat tuple output.

    ``DocDet.forward`` returns a dict (keyed by level name) of
    3-tuples.  ONNX export prefers plain tensors or nested tuples.
    This wrapper flattens everything into a tuple of 9 tensors
    ordered (cls_p2, reg_p2, cent_p2, cls_p3, ..., cent_p4) so the
    consumer can unpack by index.
    """

    def __init__(self, model: DocDet):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        return (
            outputs["p2"][0], outputs["p2"][1], outputs["p2"][2],
            outputs["p3"][0], outputs["p3"][1], outputs["p3"][2],
            outputs["p4"][0], outputs["p4"][1], outputs["p4"][2],
        )


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    backbone_name: str = "mobilenetv3_small",
    input_height: int = 1120,
    input_width: int = 800,
    opset: int = 17,
    dynamic_axes: bool = True,
) -> Path:
    """
    Load ``checkpoint_path`` and write an ONNX model to ``output_path``.

    Returns
    -------
    The resolved ``output_path`` (useful for chaining with quantize).
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    backbone = payload.get("backbone_name", backbone_name)

    model = DocDet(
        num_classes=NUM_DOCDET_CLASSES,
        backbone_name=backbone,
        pretrained=False,
    )
    state = payload.get("model", payload)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        logger.warning("Missing keys during load: %d", len(missing))
    if unexpected:
        logger.warning("Unexpected keys during load: %d", len(unexpected))

    model.eval()

    wrapper = _DocDetExportWrapper(model).eval()
    example = torch.zeros(1, 3, input_height, input_width)

    output_names = [
        "cls_p2", "reg_p2", "cent_p2",
        "cls_p3", "reg_p3", "cent_p3",
        "cls_p4", "reg_p4", "cent_p4",
    ]

    dynamic_axes_dict = None
    if dynamic_axes:
        dynamic_axes_dict = {"images": {0: "batch", 2: "H", 3: "W"}}
        for name in output_names:
            dynamic_axes_dict[name] = {0: "batch", 2: "H", 3: "W"}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Exporting ONNX to %s (opset=%d)", output_path, opset)
    torch.onnx.export(
        wrapper,
        example,
        str(output_path),
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes_dict,
    )
    logger.info("ONNX export complete: %s", output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DocDet to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", type=str, default="mobilenetv3_small",
                        choices=["mobilenetv3_small", "efficientnet_b0"])
    parser.add_argument("--input-height", type=int, default=1120)
    parser.add_argument("--input-width", type=int, default=800)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--static-shapes", action="store_true",
                        help="Disable dynamic axes (fixed H and W).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        backbone_name=args.backbone,
        input_height=args.input_height,
        input_width=args.input_width,
        opset=args.opset,
        dynamic_axes=not args.static_shapes,
    )


if __name__ == "__main__":
    main()
