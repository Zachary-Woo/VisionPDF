All Commands
Run all of these from the project root (c:\Users\zackw\OneDrive\Desktop\VisionPDF).

================================================================================
0. Environment Setup
================================================================================

# Install PyTorch first (pick your CUDA version from https://pytorch.org/get-started/locally/)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all benchmark dependencies
pip install -r benchmark/requirements.txt

# Download YOLO weights
hf download hantian/yolo-doclaynet yolo26l-doclaynet.pt --local-dir models 

# (Optional) PaddleOCR -- requires separate install:
#   CPU:  pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
#   GPU:  pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
#   Then: pip install "paddleocr>=3.2"

================================================================================
1. Dataset Downloads
================================================================================

OmniDocBench and DocLayNet download automatically on first use.
If you want to trigger them manually:

# OmniDocBench (PDFs with text layers + images + ground truth JSON)
#   Auto-downloads when any extraction script runs.
#   Manual alternative:
hf download samiuc/omnidocbench --repo-type dataset --local-dir OmniDocBench

# DocLayNet (~28 GB, needed for SAM detector training only)
#   Auto-downloads from IBM S3 when train_sam_detector runs.

================================================================================
2. Training -- SAM Detection Head
================================================================================

python -m benchmark.tier2_hybrid.training.train_sam_detector --model-path deepseek-ai/DeepSeek-OCR-2 --epochs 12 --batch-size 16 --lr 4e-4

# Freezes SAM ViT-B encoder, trains FPN + FCOS head on DocLayNet.
# DocLayNet auto-downloads if not present (~28 GB).
# Saves checkpoint to models/sam_doclaynet_head.pt

================================================================================
3. Extraction -- Tier 1 (Text Layer)
================================================================================

# pypdfium2 (raw text API)
python -m benchmark.tier1_text_layer.extract_pypdfium2

# PyMuPDF raw text
python -m benchmark.tier1_text_layer.extract_pymupdf --mode raw

# PyMuPDF via pymupdf4llm (markdown output)
python -m benchmark.tier1_text_layer.extract_pymupdf --mode markdown

# With custom PDF directory (any tier):
python -m benchmark.tier1_text_layer.extract_pypdfium2 --input-dir path/to/your/pdfs

================================================================================
4. Extraction -- Tier 2 (Hybrid: Vision Layout + Text Layer)
================================================================================

# YOLO + geometric reading order
python -m benchmark.tier2_hybrid.yolo.extract_geometric

# YOLO + LayoutReader reading order
python -m benchmark.tier2_hybrid.yolo.extract_layoutreader

# SAM detector + geometric order
python -m benchmark.tier2_hybrid.sam.extract_det --order geometric

# SAM detector + LayoutReader order
python -m benchmark.tier2_hybrid.sam.extract_det --order layoutreader

# SAM unsupervised clustering baseline
python -m benchmark.tier2_hybrid.sam.extract_clustering

# Docling (RT-DETR pipeline)
python -m benchmark.tier2_hybrid.docling.extract

================================================================================
5. Extraction -- Tier 3 (Full OCR / VLM)
================================================================================

# EasyOCR
python -m benchmark.tier3_ocr.extract_easyocr

# DeepSeek OCR 2 (high VRAM -- needs ~24GB+)
python -m benchmark.tier3_ocr.extract_deepseek_ocr2

# MonkeyOCR (requires separate install of MonkeyOCR/magic_pdf)
python -m benchmark.tier3_ocr.extract_monkeyocr

# PaddleOCR (requires separate PaddlePaddle install, see above)
python -m benchmark.tier3_ocr.extract_paddleocr

================================================================================
6. Evaluation
================================================================================

# Self-contained evaluation (normalised edit distance + throughput)
python -m benchmark.evaluate.run_eval

# With explicit paths
python -m benchmark.evaluate.run_eval --results-dir results --gt-json OmniDocBench/OmniDocBench.json

# With full OmniDocBench evaluation suite (TEDS + CDM)
python -m benchmark.evaluate.run_eval --omnidocbench-repo path/to/OmniDocBench-repo

================================================================================
Execution Order
================================================================================

Step  What                        Prerequisite
----  --------------------------  --------------------------------------------
0     Install dependencies        None
1     Download datasets           Automatic on first use (or manual, see above)
2     Train SAM detector          DocLayNet (auto-downloads)
3     Run Tier 1 extractions      OmniDocBench PDFs (auto-downloads)
4     Run Tier 2 extractions      YOLO weights; Step 2 for SAM methods
5     Run Tier 3 extractions      Model downloads (automatic for most)
6     Run evaluation              At least one method's results + OmniDocBench.json

Tiers 1, 2, and 3 are independent -- run in any order or skip entirely.
Only hard chain: train SAM detector -> SAM-based extraction methods.
All extraction scripts accept --input-dir to use your own PDFs.
