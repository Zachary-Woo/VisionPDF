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
# Download from: https://huggingface.co/datasets/samiuc/omnidocbench
# Expected structure:
#   OmniDocBench/
#     images/       <- page images (jpg/png)
#     ori_pdfs/     <- original single-page PDFs with embedded text layers
#     pdfs/         <- image-only PDFs (not used -- no text layer)
#     OmniDocBench.json  <- ground-truth annotations
# ---------------------------------------------------------------------------
OMNIDOCBENCH_DIR = PROJECT_ROOT / "OmniDocBench"
OMNIDOCBENCH_IMAGES = OMNIDOCBENCH_DIR / "images"
OMNIDOCBENCH_PDFS = OMNIDOCBENCH_DIR / "ori_pdfs"
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
MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# YOLO layout detection model (trained on DocLayNet).
# The hantian/yolo-doclaynet HF repo hosts v8/v10/v11/v12/v26 weights.
# Download with:
#   huggingface-cli download hantian/yolo-doclaynet yolo26l-doclaynet.pt --local-dir models
# ---------------------------------------------------------------------------
YOLO_MODEL_REPO = "hantian/yolo-doclaynet"
YOLO_MODEL_FILE = "yolo26l-doclaynet.pt"
YOLO_MODEL = MODELS_DIR / YOLO_MODEL_FILE

# ---------------------------------------------------------------------------
# DocDet layout detection model (license-clean, trained in-house).
# Lives under ``layout_detector/weights`` rather than ``models`` so
# training artefacts from ``layout_detector/docdet/train/*.py`` land
# in the same place the benchmark reads from.  The default filename
# mirrors the Phase 2 checkpoint name emitted by the trainer.
# ---------------------------------------------------------------------------
DOCDET_WEIGHTS_DIR = PROJECT_ROOT / "layout_detector" / "weights"
DOCDET_MODEL_FILE = "docdet.pt"
DOCDET_MODEL = DOCDET_WEIGHTS_DIR / DOCDET_MODEL_FILE
DOCDET_ONNX_FILE = "docdet.onnx"
DOCDET_ONNX = DOCDET_WEIGHTS_DIR / DOCDET_ONNX_FILE
DOCDET_INT8_FILE = "docdet_int8.onnx"
DOCDET_INT8 = DOCDET_WEIGHTS_DIR / DOCDET_INT8_FILE

DEEPSEEK_OCR2_MODEL = "deepseek-ai/DeepSeek-OCR-2"

# ---------------------------------------------------------------------------
# DocLayNet label map (used to decode YOLO class ids into region names).
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


# ---------------------------------------------------------------------------
# Dataset download helpers
# ---------------------------------------------------------------------------

_OMNIDOCBENCH_HF_REPO = "samiuc/omnidocbench"


def ensure_omnidocbench(pdf_dir: Optional[Path] = None) -> Path:
    """
    Download the OmniDocBench dataset from HuggingFace if not already
    present, including original single-page PDFs with real text layers.

    Uses samiuc/omnidocbench which keeps ori_pdfs/, images/, and
    OmniDocBench.json together on main.  We download ori_pdfs/ (the
    originals with embedded text layers) rather than pdfs/ (which are
    just images wrapped in PDF format).

    Returns the path to the ori_pdfs/ directory.
    """
    pdf_dir = pdf_dir or OMNIDOCBENCH_PDFS

    if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
        return pdf_dir

    from huggingface_hub import snapshot_download

    print(f"Downloading OmniDocBench from {_OMNIDOCBENCH_HF_REPO}...")
    print(f"  Target: {OMNIDOCBENCH_DIR}")
    try:
        snapshot_download(
            _OMNIDOCBENCH_HF_REPO,
            repo_type="dataset",
            local_dir=str(OMNIDOCBENCH_DIR),
            allow_patterns=[
                "OmniDocBench.json",
                "ori_pdfs/**",
                "images/**",
            ],
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to download OmniDocBench: {e}\n"
            "Please download manually:\n"
            f"  huggingface-cli download {_OMNIDOCBENCH_HF_REPO}"
            " --repo-type dataset --local-dir OmniDocBench\n"
            f"and place under {OMNIDOCBENCH_DIR}"
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
