"""
yolo_experiments -- Docling-based PDF extraction pipeline.

Provides automatic text-layer detection and routes PDFs through
Docling's StandardPdfPipeline with cell-level table extraction.
"""


def __getattr__(name):
    """Lazy imports to avoid RuntimeWarning when running as ``python -m``."""
    if name == "convert_pdf":
        from yolo_experiments.pipeline import convert_pdf
        return convert_pdf
    if name == "has_text_layer":
        from yolo_experiments.pipeline import has_text_layer
        return has_text_layer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["convert_pdf", "has_text_layer"]
