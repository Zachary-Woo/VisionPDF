"""
Shared configuration for the PDF extraction benchmark.

Central location for dataset paths, model identifiers, rendering
constants, and output directories used by every extraction script.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project root (parent of this file's directory)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# OmniDocBench dataset paths
# Download from: https://huggingface.co/datasets/opendatalab/OmniDocBench
# Expected structure:
#   OmniDocBench/
#     images/       <- page images (jpg)
#     pdfs/         <- single-page PDFs
#     OmniDocBench.json  <- ground-truth annotations
# ---------------------------------------------------------------------------
OMNIDOCBENCH_DIR = PROJECT_ROOT / "OmniDocBench"
OMNIDOCBENCH_IMAGES = OMNIDOCBENCH_DIR / "images"
OMNIDOCBENCH_PDFS = OMNIDOCBENCH_DIR / "pdfs"
OMNIDOCBENCH_JSON = OMNIDOCBENCH_DIR / "OmniDocBench.json"

# ---------------------------------------------------------------------------
# Output directory for extraction results (per-method sub-folders)
# ---------------------------------------------------------------------------
OUTPUT_DIR = PROJECT_ROOT / "results"

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------
RENDER_DPI = 144  # balance between detail and speed; 72 * 2

# ---------------------------------------------------------------------------
# Model identifiers (Hugging Face repos / local paths)
# ---------------------------------------------------------------------------
# The hantian/yolo-doclaynet HF repo hosts multiple model sizes.
# Ultralytics needs a direct path to a .pt file.  Download first:
#   huggingface-cli download hantian/yolo-doclaynet yolov8x-doclaynet.pt --local-dir models
# Then set YOLO_MODEL to the local path, or leave the HF URI below
# which works with ultralytics >= 8.3 when the repo contains a single
# matching file for the requested variant.
YOLO_MODEL = "hantian/yolo-doclaynet"

LAYOUTREADER_MODEL = "hantian/layoutreader"

DEEPSEEK_OCR2_MODEL = "deepseek-ai/DeepSeek-OCR-2"

# ---------------------------------------------------------------------------
# Trained model checkpoints (created by training scripts)
# ---------------------------------------------------------------------------
MODELS_DIR = PROJECT_ROOT / "models"
SAM_DETECTOR_CHECKPOINT = MODELS_DIR / "sam_doclaynet_head.pt"
SAM_ORDER_CHECKPOINT = MODELS_DIR / "sam_reading_order.pt"

# ---------------------------------------------------------------------------
# Dataset paths for training
# ---------------------------------------------------------------------------
DOCLAYNET_DIR = PROJECT_ROOT / "DocLayNet"
READINGBANK_DIR = PROJECT_ROOT / "ReadingBank"

# ---------------------------------------------------------------------------
# DocLayNet label map (shared by YOLO-based scripts)
# Order matches the hantian/yolo-doclaynet training config.
# ---------------------------------------------------------------------------
DOCLAYNET_LABELS = [
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Page-footer",
    "Page-header",
    "Picture",
    "Section-header",
    "Table",
    "Text",
    "Title",
]

# Mapping from DocLayNet labels to markdown formatting helpers.
LABEL_TO_MD = {
    "Title": "# ",
    "Section-header": "## ",
    "Caption": "*",
    "Footnote": "> ",
    "List-item": "- ",
}

# Labels whose text content should be extracted from the text layer.
TEXT_LABELS = {
    "Caption",
    "Footnote",
    "Formula",
    "List-item",
    "Section-header",
    "Text",
    "Title",
}

# Labels that are non-text regions (skip text extraction).
NON_TEXT_LABELS = {"Picture", "Page-footer", "Page-header", "Table"}
