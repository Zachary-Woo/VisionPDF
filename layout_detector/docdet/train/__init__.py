"""
DocDet training scripts.

Modules:
    config            phase config dataclasses
    trainer           single-GPU Trainer class (AMP, cosine LR, checkpoint)
    phase0_coco       optional COCO 80-class warmup
    phase1_synth      DocSynth300K synthetic pretraining
    phase2_real       multi-dataset weighted fine-tuning
    phase4_targeted   conditional targeted fine-tune after eval
"""
