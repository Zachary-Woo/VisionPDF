# Vision-Guided Structural Correction for Enterprise PDF Extraction

Three-tier benchmark comparing speed and accuracy of PDF text extraction across pure
text-layer parsing, hybrid vision-corrected reconstruction, and full OCR/VLM pipelines.

Core thesis: enterprise PDFs are digitally generated, not scanned. Text layer exists
but structure is wrong. The problem is not character recognition -- it is recovering
correct reading order, columns, tables, and hierarchy. Most benchmarks measure OCR
accuracy on scanned documents. None compare hybrid extraction methods that use vision
to guide text-layer reconstruction. This benchmark fills that gap.

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
1. Text-layer parsing -- fast, structurally unreliable
2. Full OCR reconstruction -- visually robust, expensive, sometimes lossy

Hybrid approach: use vision models to detect layout, pull text from native PDF layer.
Gets structure from vision and character fidelity from the text layer, without the
cost of full OCR. The idea is straightforward, but no existing benchmark measures
whether it actually works.

---

## Benchmark Tiers

### Tier 1 -- Pure Text Layer

No vision. Extract text directly from PDF internals.
Structural ordering = whatever the PDF token stream produces.

| Method | Script | Description |
|--------|--------|-------------|
| pypdfium2 | `tier1_text_layer/extract_pypdfium2.py` | PDFium text API. Raw token stream. |
| PyMuPDF (raw) | `tier1_text_layer/extract_pymupdf.py` | MuPDF engine, raw plain text mode. |

### Tier 2 -- Hybrid Reconstruction (Detector + Docling Assembly)

A lightweight vision model detects layout regions from a rendered page image.
Docling's assembly pipeline handles the downstream structural recovery:
ReadingOrderModel sequences the detected blocks, TableFormer reconstructs table
structure, and all character content is pulled from the native PDF text layer
without OCR. The only axis of variation is which layout detector feeds the pipeline.

| Method | Script | Detection | Assembly |
|--------|--------|-----------|----------|
| YOLO + Docling | `tier2_hybrid/yolo/extract.py` | YOLO (DocLayNet) | Docling (ReadingOrderModel + TableFormer) |
| Docling RT-DETR | `tier2_hybrid/docling/extract.py` | RT-DETR "Layout Heron" (DocLayNet) | Docling (ReadingOrderModel + TableFormer) |

### Tier 3 -- Full OCR / VLM

Complete visual analysis and text recognition from page image. The existing PDF
text layer is discarded entirely.

| Method | Script | Notes |
|--------|--------|-------|
| PaddleOCR | `tier3_ocr/extract_paddleocr.py` | PP-StructureV3 (default) or PP-OCRv5. Full document parsing with layout detection, table recognition, and markdown output. |
| EasyOCR | `tier3_ocr/extract_easyocr.py` | CRAFT detection + CRNN recognition. No structural understanding. Throughput and accuracy floor baseline. |

---

## Results

Evaluated on the full OmniDocBench suite (981 annotated pages across nine document
types).

### Structural accuracy and throughput

| Method | Tier | Edit Dist. | Text Acc. (%) | Block Rec. (%) | Composite (%) | Pages/s |
|--------|------|------------|---------------|----------------|---------------|---------|
| PyMuPDF (raw) | 1 | 0.480 | 52.1 | 39.7 | 53.3 | 191.8 |
| pypdfium2 | 1 | 0.480 | 52.1 | 49.9 | 57.0 | 159.1 |
| Docling (RT-DETR) | 2 | 0.472 | 52.8 | 49.3 | 56.2 | 2.0 |
| YOLO Hybrid | 2 | 0.493 | 50.7 | 55.3 | 58.3 | 1.7 |
| EasyOCR | 3 | 0.751 | 24.9 | 25.3 | 26.8 | 0.47 |
| PaddleOCR (PP-StructV3) | 3 | 0.204 | 79.6 | 66.5 | 77.7 | 0.03 |

### LLM judge evaluation (20-page subsample, DeepSeek v3.2 judge)

| Method | Tier | Overall | Info Compl. | Read. Order | Struct. Bound. | Table/Data |
|--------|------|---------|-------------|-------------|----------------|------------|
| PaddleOCR (PP-StructV3) | 3 | 4.15 | 4.25 | 4.55 | 4.90 | 4.95 |
| Docling (RT-DETR) | 2 | 3.45 | 3.75 | 3.45 | 4.00 | 4.80 |
| pypdfium2 | 1 | 3.35 | 3.70 | 3.20 | 3.75 | 4.80 |
| PyMuPDF (raw) | 1 | 3.25 | 3.85 | 2.95 | 3.75 | 5.00 |
| YOLO Hybrid | 2 | 2.84 | 3.26 | 2.95 | 3.26 | 4.63 |
| EasyOCR | 3 | 1.45 | 1.75 | 1.90 | 2.15 | 4.25 |

### Key findings

- The YOLO Hybrid achieves the highest block recovery among non-PaddleOCR methods
  (55.3%), gaining +5.4 points over pypdfium2 and +15.6 points over PyMuPDF raw.
- PaddleOCR is the accuracy leader (66.5% block recovery) but at 0.03 pages/sec it
  is 57x slower than the YOLO Hybrid and 5,300x slower than pypdfium2.
- Tier 1 and Tier 2 text accuracy clusters in a narrow band of 50.7-52.8%,
  confirming that layout detection does not alter the underlying character stream.
- On academic literature, the YOLO Hybrid achieves 80.9% block recovery vs 59.7%
  for pypdfium2 (+21.2 points). On newspapers, Docling reaches 81.6% vs 52.8%
  (+28.8 points). Multi-column documents gain the most from vision.
- Notes (handwritten, image-only) expose the boundary: all Tier 1 and Tier 2
  methods produce near-zero block recovery because there is no embedded text layer.
  PaddleOCR recovers 56.5% on this category by working entirely from pixels.
- EasyOCR (24.9% text accuracy) confirms that OCR without structure is
  counterproductive on digitally generated PDFs: re-recognizing characters from
  pixels introduces errors on text the PDF layer already contains perfectly.

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
        extract_pymupdf.py              PyMuPDF raw extraction

    tier2_hybrid/
        docling_export.py               Markdown serialiser (HTML tables) used by both detectors
        yolo/
            yolo_layout_model.py        YOLO detector wrapped as a Docling BaseLayoutModel
            pipeline.py                 LegacyStandardPdfPipeline subclass using YoloLayoutModel
            extract.py                  Entry point: YOLO layout + Docling assembly
        docling/
            extract.py                  Entry point: RT-DETR layout + Docling assembly

    tier3_ocr/
        extract_paddleocr.py            PaddleOCR PP-StructureV3 / PP-OCRv5
        extract_easyocr.py              EasyOCR CRAFT+CRNN baseline

    evaluate/
        run_eval.py                     Edit distance, block recovery, throughput metrics
        llm_judge.py                    LLM-based usability evaluation (DeepSeek v3.2)
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

### DocLayNet (Taxonomy)

Public 11-class document layout taxonomy. Pre-trained YOLO DocLayNet
weights use this label set. `benchmark/config.py` holds the class list,
and `yolo_layout_model.py` maps each DocLayNet label to the corresponding
`DocItemLabel` that Docling reasons about.

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

Windows notes for PaddleOCR:
- Torch + PaddlePaddle have a DLL conflict on Windows. Script handles this by importing torch first.
- RTX 50-series (Blackwell) needs special PaddlePaddle wheels. See [PaddleOCR install docs](https://paddlepaddle.github.io/PaddleOCR/v3.3.2/en/version3.x/installation.html).
- If GPU fails, the script auto-falls back to CPU.

### YOLO DocLayNet Weights

Ultralytics needs the direct `.pt` file. Download first:

```bash
hf download hantian/yolo-doclaynet yolov8x-doclaynet.pt --local-dir models
```

Scripts include a fallback that auto-downloads via `hf_hub_download` if the bare repo ID fails.

### LLM Judge (Optional)

The LLM-based evaluation uses OpenRouter to call DeepSeek v3.2. Set:

```bash
export OPENROUTER_API_KEY=your_key_here
```

---

## Usage

Every extraction script follows the same pattern:
- Takes `--input-dir` (directory of single-page PDFs, default: `OmniDocBench/pdfs`)
- Takes `--output-dir` (base results directory, default: `results/`)
- Writes one `.md` file per page plus `timing.csv` and `summary.json`

### Tier 1

```bash
python -m benchmark.tier1_text_layer.extract_pypdfium2
python -m benchmark.tier1_text_layer.extract_pymupdf --mode raw
```

### Tier 2

```bash
# YOLO layout detector fed into Docling's assembly pipeline
python -m benchmark.tier2_hybrid.yolo.extract

# Docling (built-in RT-DETR layout)
python -m benchmark.tier2_hybrid.docling.extract
```

### Tier 3

```bash
python -m benchmark.tier3_ocr.extract_paddleocr --mode structure
python -m benchmark.tier3_ocr.extract_paddleocr --mode ocr
python -m benchmark.tier3_ocr.extract_easyocr
```

### Evaluation

```bash
# Lightweight eval (edit distance, block recovery, throughput)
python -m benchmark.evaluate.run_eval

# With full OmniDocBench eval suite (if repo cloned)
python -m benchmark.evaluate.run_eval --omnidocbench-repo path/to/OmniDocBench

# LLM judge evaluation (requires OPENROUTER_API_KEY)
python -m benchmark.evaluate.llm_judge --samples 20 --seed 7
```

Evaluation finds all method output dirs under `results/` that contain `summary.json`,
computes metrics against ground truth, writes `benchmark_results.json`, and prints
a comparison table.

---

## Evaluation Metrics

Both prediction and ground truth are first passed through `normalize_text()`, which
strips markdown syntax, HTML tags and entities, LaTeX delimiters, image placeholders,
and soft-hyphen artifacts, then collapses whitespace. All metrics below are computed
on the normalized strings so that scores reflect content fidelity rather than
formatting differences.

### Automated Metrics (built-in)

- Normalized Edit Distance: `Lev(N(p), N(g)) / max(|N(p)|, |N(g)|)`. Range [0, 1],
  lower is better. Converted to text accuracy: `(1 - NED) * 100`.
- Token Recall: Multiset overlap divided by ground-truth token count.
  Tokens are extracted after punctuation removal and per-character splitting of CJK text.
- Token Precision: Multiset overlap divided by predicted token count.
- Token F1: Harmonic mean of token precision and recall.
- Block Recovery: For each annotated ground-truth block, the lowest edit distance
  found among the predicted chunks (single chunks, adjacent chunk pairs, and the
  full chunk concatenation), then `1 - mean(NED)` averaged across all blocks.
  Continuous score in [0, 1] -- partial matches receive partial credit.
- Composite Score: Weighted blend `0.30 * TextAcc + 0.35 * TokenF1 + 0.35 * BlockRec`.
- Throughput: Pages per second from timing CSVs.

### Full OmniDocBench Evaluation (optional)

If the OmniDocBench eval repo is available:
- Text Edit Distance: Per-block text accuracy
- Table TEDS: Tree edit distance for table structure recovery
- Reading Order Edit Distance: Measures order correctness specifically

### LLM Judge Evaluation

`benchmark/evaluate/llm_judge.py` uses DeepSeek v3.2 (via OpenRouter) to score
both ground truth and extraction independently on five axes (1--5 scale):
information completeness, reading order, structural boundaries, table/data
preservation, and overall usability. Used to capture qualitative usability
that automated metrics may miss.

---

## Output Format

Every method produces per-page markdown files matching the OmniDocBench expected format:

```
results/
    <method_name>/
        <page_id>.md          Extracted markdown for one page
        timing.csv            method, page_id, wall_seconds, cuda_seconds
        summary.json          Method metadata
```

For Tier 2 methods (`tier2_yolo`, `tier2_docling`) markdown is produced by Docling's
`export_to_markdown`, with table blocks rewritten to HTML so table syntax matches
OmniDocBench ground truth. Other tiers use their own serialisers.