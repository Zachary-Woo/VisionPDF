# PDF Text Layer Repair -- Extraction Benchmark

Benchmark comparing speed + accuracy of PDF text extraction across three tiers:
pure text-layer parsing, hybrid vision-corrected reconstruction, and full OCR/VLM methods.

Core thesis: enterprise PDFs are digitally generated, not scanned. Text layer exists but structure is wrong.
Problem is not character recognition -- problem is recovering correct reading order, columns, tables, hierarchy.
Most benchmarks compare OCR accuracy. None compare **hybrid** extraction methods that use vision to guide
text-layer reconstruction. This benchmark fills that gap.

---

## Problem

PDF format encodes drawing instructions, not logical structure.
Two documents rendering identically can have completely different internal token streams.

Common structural failures in digitally generated PDFs:
- Multi-column text interleaved in token stream
- Broken reading order
- Tables flattened into linear text
- Lost indentation hierarchy
- Header/footer pollution across pages
- Character-per-line fragmentation artifacts

Traditional pipelines fall into two camps:
1. **Text-layer parsing** -- fast, structurally unreliable
2. **Full OCR reconstruction** -- visually robust, expensive, sometimes lossy

Hybrid approach: use vision models to detect layout, pull text from native PDF layer.
Gets structure from vision + speed from text layer. Best of both worlds -- if it works.
This benchmark measures whether it works.

---

## Benchmark Tiers

### Tier 1 -- Pure Text Layer

No vision. Extract text directly from PDF internals.
Structural ordering = whatever PDF token stream produces.

| Method | Script | Description |
|--------|--------|-------------|
| **pypdfium2** | `tier1_text_layer/extract_pypdfium2.py` | PDFium text API. Raw token stream. |
| **PyMuPDF** | `tier1_text_layer/extract_pymupdf.py` | MuPDF engine. Two modes: `raw` (plain text) and `markdown` (pymupdf4llm structural recovery). |

### Tier 2 -- Hybrid Reconstruction (Detector + Reading Order)

Vision model detects layout regions. Text extracted from native PDF layer within each region.
Reading order determined by geometric heuristics or learned models.

| Method | Script | Detection | Reading Order |
|--------|--------|-----------|---------------|
| **YOLO + Geometric** | `tier2_hybrid/yolo/extract_geometric.py` | YOLO (DocLayNet) | Top-to-bottom, left-to-right |
| **YOLO + LayoutReader** | `tier2_hybrid/yolo/extract_layoutreader.py` | YOLO (DocLayNet) | LayoutReader (LayoutLMv3) |
| **Docling RT-DETR** | `tier2_hybrid/docling/extract.py` | RT-DETR "Layout Heron" (DocLayNet) | Docling built-in ReadingOrderModel |

### Tier 3 -- Full OCR / VLM

Complete visual analysis + text recognition from page image. No text layer used.

| Method | Script | Notes |
|--------|--------|-------|
| **DeepSeek OCR 2** | `tier3_ocr/extract_deepseek_ocr2.py` | Full 14B+ VLM. Visual + Qwen2 causal-flow encoder + LM decoder. Needs >= 24GB VRAM + flash-attn. |
| **MonkeyOCR** | `tier3_ocr/extract_monkeyocr.py` | SRR paradigm (Structure-Recognition-Relation). 1.2-3B params. DocLayoutYOLO + own LayoutReader. |
| **PaddleOCR** | `tier3_ocr/extract_paddleocr.py` | PP-StructureV3 (default) or PP-OCRv5. Separate install from PaddlePaddle index. |
| **EasyOCR** | `tier3_ocr/extract_easyocr.py` | CRAFT detection + CRNN recognition. No structural understanding. Throughput baseline. |

---

## Directory Structure

```
benchmark/
    config.py                           Shared config: paths, model IDs, label maps
    timing.py                           Wall-clock + CUDA timing context manager
    output.py                           Per-page markdown + summary JSON writers
    pdf_render.py                       pypdfium2 page render + coord conversion
    requirements.txt                    Python dependencies

    tier1_text_layer/
        extract_pypdfium2.py            pypdfium2 raw extraction
        extract_pymupdf.py              PyMuPDF raw/markdown extraction

    tier2_hybrid/
        shared.py                       Region class, geometric sort, text extraction,
                                        markdown reconstruction, LayoutReader helpers
        yolo/
            extract_geometric.py        YOLO + geometric sort
            extract_layoutreader.py     YOLO + LayoutReader
        docling/
            extract.py                  Docling RT-DETR pipeline

    tier3_ocr/
        extract_deepseek_ocr2.py        DeepSeek OCR 2 full VLM
        extract_monkeyocr.py            MonkeyOCR SRR pipeline
        extract_paddleocr.py            PaddleOCR PP-StructureV3 / PP-OCRv5
        extract_easyocr.py              EasyOCR CRAFT+CRNN baseline

    evaluate/
        run_eval.py                     Edit distance, TEDS, throughput metrics
```

---

## Datasets

### OmniDocBench (Evaluation)

Primary evaluation benchmark. Download from [HuggingFace](https://huggingface.co/datasets/opendatalab/OmniDocBench).

Expected structure:
```
OmniDocBench/
    images/              Page images (jpg)
    pdfs/                Single-page PDFs
    OmniDocBench.json    Ground-truth annotations
```

Place in project root. Path configured in `config.py` as `OMNIDOCBENCH_DIR`.

### DocLayNet (taxonomy)

Public 11-class document layout taxonomy. Pre-trained YOLO DocLayNet weights
use this label set; see `benchmark/config.py` for the class list and markdown
prefix mapping.

---

## Installation

### Base Dependencies

```bash
# Install PyTorch first (match CUDA version to your system)
# See: https://pytorch.org/get-started/locally/
pip install torch torchvision

# Install benchmark dependencies
pip install -r benchmark/requirements.txt
```

### PaddleOCR (Separate Install)

PaddlePaddle uses its own package index. Cannot `pip install` from standard PyPI.

```bash
# CPU (simplest on Windows):
pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
pip install "paddleocr>=3.2"

# GPU (CUDA 11.8):
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
pip install "paddleocr>=3.2"
```

**Windows notes for PaddleOCR:**
- Torch + PaddlePaddle have DLL conflict on Windows. Script handles this by importing torch first.
- RTX 50-series (Blackwell) needs special PaddlePaddle wheels. See [PaddleOCR install docs](https://paddlepaddle.github.io/PaddleOCR/v3.3.2/en/version3.x/installation.html).
- If GPU fails, script auto-falls back to CPU.

### MonkeyOCR (Separate Install)

```bash
git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
cd MonkeyOCR && pip install -e .
```

### DeepSeek OCR 2

Needs `flash-attn` and >= 24GB VRAM. Model auto-downloads from HuggingFace.

```bash
pip install flash-attn --no-build-isolation
```

### YOLO DocLayNet Weights

Ultralytics needs direct `.pt` file. Download first:

```bash
hf download hantian/yolo-doclaynet yolov8x-doclaynet.pt --local-dir models
```

Scripts include fallback that auto-downloads via `hf_hub_download` if bare repo ID fails.

---

## Usage

Every extraction script follows same pattern:
- Takes `--input-dir` (directory of single-page PDFs, default: `OmniDocBench/pdfs`)
- Takes `--output-dir` (base results directory, default: `results/`)
- Writes one `.md` file per page + `timing.csv` + `summary.json`

### Tier 1

```bash
python -m benchmark.tier1_text_layer.extract_pypdfium2
python -m benchmark.tier1_text_layer.extract_pymupdf --mode raw
python -m benchmark.tier1_text_layer.extract_pymupdf --mode markdown
```

### Tier 2

```bash
# YOLO-based
python -m benchmark.tier2_hybrid.yolo.extract_geometric
python -m benchmark.tier2_hybrid.yolo.extract_layoutreader

# Docling
python -m benchmark.tier2_hybrid.docling.extract
```

### Tier 3

```bash
python -m benchmark.tier3_ocr.extract_deepseek_ocr2
python -m benchmark.tier3_ocr.extract_monkeyocr
python -m benchmark.tier3_ocr.extract_paddleocr --mode structure
python -m benchmark.tier3_ocr.extract_paddleocr --mode ocr
python -m benchmark.tier3_ocr.extract_easyocr
```

### Evaluation

```bash
# Lightweight eval (edit distance + throughput)
python -m benchmark.evaluate.run_eval

# With full OmniDocBench eval suite (if repo cloned)
python -m benchmark.evaluate.run_eval --omnidocbench-repo path/to/OmniDocBench
```

Evaluation finds all method output dirs under `results/` that contain `summary.json`,
computes metrics against ground truth, writes `benchmark_results.json`, prints comparison table.

---

## Evaluation Metrics

### Lightweight Evaluation (built-in)

- **Normalised Edit Distance**: Character-level Levenshtein distance / max(pred_len, gt_len).
  Range [0, 1]. Lower = better. Converted to accuracy: `(1 - edit_dist) * 100`.
- **Throughput**: Pages per second from timing CSVs.

### Full OmniDocBench Evaluation (optional)

If OmniDocBench eval repo available:
- **Text Edit Distance**: Per-block text accuracy
- **Table TEDS**: Tree edit distance for table structure recovery
- **Reading Order Edit Distance**: Measures order correctness specifically
- **Overall Score**: `((1 - text_edit_dist) * 100 + table_TEDS + formula_CDM) / 3`

---

## Output Format

Every method produces per-page markdown files matching OmniDocBench expected format:

```
results/
    <method_name>/
        <page_id>.md          Extracted markdown for one page
        timing.csv            method, page_id, wall_seconds, cuda_seconds
        summary.json          Method metadata
```

Markdown formatting guided by DocLayNet labels:
- Title -> `# `
- Section-header -> `## `
- Caption -> `*...*`
- Footnote -> `> `
- List-item -> `- `
- Picture -> `[Image]`
- Table -> raw text content

---

## Configuration

All paths and model IDs centralized in `benchmark/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `OMNIDOCBENCH_DIR` | `PROJECT_ROOT/OmniDocBench` | Evaluation dataset |
| `OUTPUT_DIR` | `PROJECT_ROOT/results` | Extraction output |
| `RENDER_DPI` | 144 | PDF render resolution (72 * 2) |
| `YOLO_MODEL` | `hantian/yolo-doclaynet` | YOLO weights (HF repo) |
| `LAYOUTREADER_MODEL` | `hantian/layoutreader` | LayoutReader weights |
| `DEEPSEEK_OCR2_MODEL` | `deepseek-ai/DeepSeek-OCR-2` | DeepSeek OCR 2 weights |
| `MODELS_DIR` | `PROJECT_ROOT/models` | Local model checkpoints (e.g. YOLO `.pt`) |

---

## Hardware Requirements

| Method | GPU VRAM | Notes |
|--------|----------|-------|
| Tier 1 (text layer) | None | CPU only |
| YOLO hybrid | ~4 GB | Standard YOLO inference |
| Docling | ~4 GB | RT-DETR inference |
| DeepSeek OCR 2 | >= 24 GB | Full 14B+ VLM. Needs flash-attn |
| MonkeyOCR | >= 12 GB | 1.2-3B multimodal model |
| PaddleOCR | ~2-4 GB | Lightweight. CPU fallback available |
| EasyOCR | ~2 GB | CRAFT + CRNN. CPU fallback available |

---

## Background

### Why This Benchmark Exists

No existing benchmark compares hybrid extraction methods. Current landscape:
- OCR benchmarks measure character recognition accuracy (not relevant for digital PDFs)
- Document parsing benchmarks assume image-only input
- Enterprise PDFs have valid text layers -- just structurally broken

Gap: how well can vision-guided methods reconstruct structure from PDFs that already have text?

### Why DeepSeek OCR 2 Encoder

DeepSeek OCR 2 paper introduces "visual causal flow encoder" -- Qwen2-0.5B repurposed as
second-stage vision encoder that creates causally ordered visual tokens. Reading order emerges
from architecture, not explicit module.

Cannot extract reading order component directly because:
- D2E is ~500M param model acting as encoder, not a simple ordering function
- Order is emergent from causal attention mask + flow queries
- Output is dense feature vectors consumed by 14B+ decoder, not order indices
- Deeply integrated -- not separable without the full model

Tier 2 in this repo compares fast layout detectors (YOLO DocLayNet, Docling) plus geometric
versus learned reading order (LayoutReader), without running the full DeepSeek VLM stack.

### EasyOCR vs PaddleOCR

Both traditional deep-learning OCR. Key differences:
- **EasyOCR**: CRAFT detection + CRNN recognition. Simple, widely used. No layout understanding.
- **PaddleOCR**: PP-OCRv5 (detection + recognition) and PP-StructureV3 (full document parsing
  with layout detection, table recognition, formula recognition, markdown output).
  More capable but PaddlePaddle framework has Windows compatibility friction.

Both included for completeness. PaddleOCR PP-StructureV3 is more comparable to
DeepSeek/MonkeyOCR. EasyOCR serves as throughput/accuracy floor baseline.
