"""
DocDet model components.

Modules:
    backbone    feature extractor (MobileNetV3-Small / EfficientNet-B0)
    blocks      building blocks (CSPBlock, Scale layer)
    neck        PANet feature pyramid fusion
    head        FCOS anchor-free detection head
    loss        focal + CIoU + centerness losses, FCOS target assignment
    postprocess decoding + NMS + table bbox dilation
    model       top-level DocDet assembly
"""
