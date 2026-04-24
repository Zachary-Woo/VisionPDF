"""
DocDet: lightweight document layout detection model.

A YOLO-inspired but from-scratch anchor-free detector implemented
with a MobileNetV3-Small backbone, PANet neck, and FCOS detection
head.  Target: 11-class DocLayNet taxonomy for digitally-born PDF
pages, deployable on edge CPUs after INT8 quantization.

Package layout:
    model/      backbone, neck, head, losses, postprocess, assembly
    data/       datasets, transforms, label remapping, download helpers
    train/      trainer + per-phase training scripts
    eval/       mAP metrics + OmniDocBench evaluation
    export/     ONNX export + INT8 quantization
    infer/      inference entry point + CLI
    tests/      unit tests (pytest)
"""

__version__ = "0.1.0"
