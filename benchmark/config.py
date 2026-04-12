"""
Shared configuration for the PDF extraction benchmark.

Central location for dataset paths, model identifiers, rendering
constants, output directories, and dataset download helpers used
by every extraction script.
"""

from pathlib import Path
from typing import Optional

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
#   hf download hantian/yolo-doclaynet yolov8x-doclaynet.pt --local-dir models
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


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------

# The v1.5 update (2025-09-25) removed pdfs/ and ori_pdfs/ from the HF repo
# and only re-uploaded images/.  This older revision still has all three.
_OMNIDOCBENCH_PDF_REVISION = "f5f559bddf50e36f7f9899d842d0006f13ce8afc"


def ensure_omnidocbench(pdf_dir: Optional[Path] = None) -> Path:
    """
    Download the OmniDocBench dataset from HuggingFace if not already
    present, including single-page PDFs with real text layers.

    The current ``main`` branch of opendatalab/OmniDocBench only has
    images and the JSON annotation file (the pdfs/ and ori_pdfs/ folders
    were deleted during the v1.5 restructure).  This function pulls:

      1. **pdfs/ and images/** from an older revision that still has
         the original single-page PDFs with embedded text layers.
      2. **OmniDocBench.json** from ``main`` so you get the latest
         ground-truth annotations.

    Returns the path to the pdfs/ directory.
    """
    pdf_dir = pdf_dir or OMNIDOCBENCH_PDFS

    if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
        return pdf_dir

    from huggingface_hub import snapshot_download

    # Step 1: PDFs and images from the older revision that still has them.
    if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
        print("Downloading OmniDocBench PDFs + images (older revision)...")
        print(f"  Target: {OMNIDOCBENCH_DIR}")
        try:
            snapshot_download(
                "opendatalab/OmniDocBench",
                repo_type="dataset",
                revision=_OMNIDOCBENCH_PDF_REVISION,
                local_dir=str(OMNIDOCBENCH_DIR),
                allow_patterns=["pdfs/**", "images/**"],
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download OmniDocBench PDFs: {e}\n"
                "Please download manually:\n"
                "  hf download opendatalab/OmniDocBench"
                f" --repo-type dataset --revision {_OMNIDOCBENCH_PDF_REVISION}"
                " --local-dir OmniDocBench\n"
                f"and place under {OMNIDOCBENCH_DIR}"
            ) from e

    # Step 2: Latest ground-truth JSON from main.
    if not OMNIDOCBENCH_JSON.exists():
        print("Downloading latest OmniDocBench.json from main...")
        try:
            snapshot_download(
                "opendatalab/OmniDocBench",
                repo_type="dataset",
                local_dir=str(OMNIDOCBENCH_DIR),
                allow_patterns=["OmniDocBench.json"],
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to download OmniDocBench.json: {e}\n"
                "Please download manually:\n"
                "  hf download opendatalab/OmniDocBench"
                " OmniDocBench.json --repo-type dataset"
                " --local-dir OmniDocBench"
            ) from e

    count = len(list(pdf_dir.glob("*.pdf"))) if pdf_dir.exists() else 0
    print(f"OmniDocBench ready: {count} PDFs in {pdf_dir}")
    return pdf_dir


def find_pdfs(input_dir: Path) -> list:
    """
    Glob for PDFs in *input_dir*.  If *input_dir* is the default
    OmniDocBench path and no PDFs exist yet, trigger an automatic
    download and image-to-PDF conversion first.

    Returns a sorted list of Path objects (may be empty if a
    non-default directory has no PDFs).
    """
    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files and input_dir == OMNIDOCBENCH_PDFS:
        ensure_omnidocbench(input_dir)
        pdf_files = sorted(input_dir.glob("*.pdf"))
    return pdf_files
