"""
Self-contained configuration for the Docling extraction pipeline.

Philosophy: lightweight hybrid -- the layout model (Heron RT-DETR)
provides structural bounding boxes and reading order, while all
character content comes directly from the PDF text layer (no OCR,
no VLM text prediction).  Post-processing rules clean up common
layout-model mistakes before the final markdown is written.

For scanned / image-only PDFs that lack a text layer, the pipeline
falls back to Docling with OCR enabled.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "docling_pipeline"

# ---------------------------------------------------------------------------
# Pipeline defaults
# ---------------------------------------------------------------------------

# TableFormer mode: "accurate" is slower but handles complex tables
# (multi-level headers, spans) better than "fast".
TABLE_MODE = "accurate"

# When True, TableFormer predictions are mapped back to native PDF
# text cells so character content comes from the text layer rather
# than being re-predicted by the model.
DO_CELL_MATCHING = True

# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

# Apply post-processing rules to clean up layout-model mistakes.
DO_POSTPROCESS = True

# ---------------------------------------------------------------------------
# Text-layer detection thresholds
# ---------------------------------------------------------------------------

# Minimum non-whitespace characters on a page to consider the PDF as
# having a usable text layer.
TEXT_LAYER_MIN_CHARS = 20

# How many pages to sample when checking for a text layer.
TEXT_LAYER_SAMPLE_PAGES = 3

# Fraction of sampled pages that must pass the character threshold
# for the PDF to be classified as "has text layer".
TEXT_LAYER_MIN_PAGE_RATIO = 0.5
