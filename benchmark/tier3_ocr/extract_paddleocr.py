"""
Tier 3 -- PaddleOCR (PP-OCRv5 / PP-StructureV3) document parsing.

PaddleOCR 3.x provides two relevant pipelines:
  - "ocr"       : PP-OCRv5 text detection + recognition (basic OCR, no layout)
  - "structure"  : PP-StructureV3 full document parsing with layout detection,
                   table recognition, formula recognition, and markdown output

The default mode is "structure" because it produces structured markdown
comparable to other Tier 3 methods (DeepSeek OCR 2, MonkeyOCR).

Windows compatibility notes:
  - PaddlePaddle and PaddleOCR must be installed separately from the rest
    of the benchmark dependencies (they ship via PaddlePaddle's own index).
    See the install instructions below.
  - On Windows, torch and PaddlePaddle have a known DLL conflict.  This
    script imports torch BEFORE paddleocr to work around it.
  - NVIDIA RTX 50-series (Blackwell) on Windows requires special
    PaddlePaddle wheels.  See PaddleOCR docs for the latest links.
  - If GPU inference fails, the script falls back to CPU mode.

Install (CPU -- simplest on Windows):
    pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
    pip install "paddleocr>=3.2"

Install (GPU -- CUDA 11.8):
    pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
    pip install "paddleocr>=3.2"

Usage:
    python -m benchmark.tier3_ocr.extract_paddleocr [--input-dir path]
                                                     [--output-dir path]
                                                     [--mode structure|ocr]
                                                     [--lang en]
"""

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

# On Windows, torch must be imported BEFORE paddleocr to avoid DLL conflicts.
# See: https://github.com/PaddlePaddle/PaddleOCR/issues/14979
try:
    import torch  # noqa: F401
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR, RENDER_DPI
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import render_page
from benchmark.timing import append_timing_row, timed

METHOD_OCR = "tier3_paddleocr"
METHOD_STRUCTURE = "tier3_paddleocr_structure"


def _check_paddle_installed() -> bool:
    """Verify PaddlePaddle and PaddleOCR are importable."""
    try:
        import paddle  # noqa: F401
        import paddleocr  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PP-StructureV3 mode (full document parsing -> markdown)
# ---------------------------------------------------------------------------

def run_structure(input_dir: Path, output_base: Path, lang: str):
    """
    Use PP-StructureV3 for full document parsing.  This detects layout
    regions (text, tables, figures, formulas), recognises text, and
    produces structured markdown output.
    """
    from paddleocr import PPStructureV3

    method_name = METHOD_STRUCTURE
    out_dir = method_output_dir(output_base, method_name)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print("Initialising PP-StructureV3 pipeline...")
    try:
        pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    except Exception as e:
        print(f"Failed to initialise PP-StructureV3: {e}")
        print("Falling back to CPU mode...")
        pipeline = PPStructureV3(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            device="cpu",
        )

    for pdf_path in tqdm(pdf_files, desc=method_name):
        page_id = pdf_path.stem

        image, _ = render_page(str(pdf_path), dpi=RENDER_DPI)
        tmp_img = Path(out_dir) / f"_tmp_{page_id}.png"
        image.save(str(tmp_img))

        with timed(use_cuda=False) as t:
            results = pipeline.predict(input=str(tmp_img))

            # PP-StructureV3 returns a list of result objects with
            # save_to_markdown capability.  Extract markdown text directly.
            markdown_parts = []
            for res in results:
                # The result object has a .markdown property or can be
                # saved to markdown.  Try the dict/json route first.
                try:
                    md_path = Path(out_dir) / f"_tmp_md_{page_id}"
                    md_path.mkdir(parents=True, exist_ok=True)
                    res.save_to_markdown(save_path=str(md_path))
                    # Read back the generated markdown file
                    md_files = list(md_path.glob("*.md"))
                    for mf in md_files:
                        markdown_parts.append(mf.read_text(encoding="utf-8"))
                        mf.unlink(missing_ok=True)
                    # Clean up temp dir
                    import shutil
                    shutil.rmtree(str(md_path), ignore_errors=True)
                except Exception:
                    # Fallback: extract text from result dict
                    res_dict = res.json if hasattr(res, "json") else {}
                    if isinstance(res_dict, dict):
                        for block in res_dict.get("layout_det", []):
                            text = block.get("text", "")
                            if text:
                                markdown_parts.append(text)

            markdown = "\n\n".join(markdown_parts)

        tmp_img.unlink(missing_ok=True)
        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(timing_csv, method_name, page_id, t.wall_seconds)

    write_summary(out_dir, method_name, {
        "total_pages": len(pdf_files),
        "mode": "structure",
        "lang": lang,
    })
    print(f"{method_name}: processed {len(pdf_files)} pages -> {out_dir}")


# ---------------------------------------------------------------------------
# PP-OCRv5 mode (basic text detection + recognition)
# ---------------------------------------------------------------------------

def run_ocr(input_dir: Path, output_base: Path, lang: str):
    """
    Use PP-OCRv5 for basic text detection and recognition.  Produces
    plain text output sorted by detected text line positions.
    """
    from paddleocr import PaddleOCR

    method_name = METHOD_OCR
    out_dir = method_output_dir(output_base, method_name)
    timing_csv = out_dir / "timing.csv"
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Initialising PaddleOCR (PP-OCRv5, lang={lang})...")
    try:
        ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as e:
        print(f"Failed to initialise PaddleOCR with GPU: {e}")
        print("Falling back to CPU mode...")
        ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
        )

    for pdf_path in tqdm(pdf_files, desc=method_name):
        page_id = pdf_path.stem

        with timed(use_cuda=False) as t:
            image, _ = render_page(str(pdf_path), dpi=RENDER_DPI)
            img_array = np.array(image)

            results = ocr.predict(input=img_array)

            # Extract text lines from PaddleOCR 3.x result format.
            # Each result has rec_texts and dt_polys/rec_boxes for positions.
            text_lines = []
            for res in results:
                res_dict = res.json if hasattr(res, "json") else {}
                if isinstance(res_dict, dict) and "rec_texts" in res_dict:
                    rec_texts = res_dict["rec_texts"]
                    rec_boxes = res_dict.get("rec_boxes", [])
                    # Pair text with vertical position for sorting
                    for i, txt in enumerate(rec_texts):
                        if not txt.strip():
                            continue
                        y_pos = 0
                        if i < len(rec_boxes):
                            box = rec_boxes[i]
                            y_pos = box[1] if len(box) >= 2 else 0
                        text_lines.append((y_pos, txt.strip()))

            # Sort by vertical position
            text_lines.sort(key=lambda x: x[0])
            markdown = "\n".join(line[1] for line in text_lines)

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(timing_csv, method_name, page_id, t.wall_seconds)

    write_summary(out_dir, method_name, {
        "total_pages": len(pdf_files),
        "mode": "ocr",
        "lang": lang,
    })
    print(f"{method_name}: processed {len(pdf_files)} pages -> {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(input_dir: Path, output_base: Path, mode: str, lang: str):
    if not _check_paddle_installed():
        print(
            "PaddleOCR is not installed.\n"
            "\n"
            "PaddlePaddle and PaddleOCR must be installed separately because\n"
            "they use their own package index.  Run these commands:\n"
            "\n"
            "  # CPU (simplest on Windows):\n"
            "  pip install paddlepaddle==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/\n"
            "  pip install \"paddleocr>=3.2\"\n"
            "\n"
            "  # GPU (CUDA 11.8):\n"
            "  pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/\n"
            "  pip install \"paddleocr>=3.2\"\n"
            "\n"
            "Windows notes:\n"
            "  - If you also have PyTorch installed, PaddleOCR must be imported\n"
            "    AFTER torch (this script handles that automatically).\n"
            "  - RTX 50-series GPUs need special PaddlePaddle wheels.\n"
            "    See: https://paddlepaddle.github.io/PaddleOCR/v3.3.2/en/version3.x/installation.html\n"
        )
        return

    if mode == "structure":
        run_structure(input_dir, output_base, lang)
    else:
        run_ocr(input_dir, output_base, lang)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=["ocr", "structure"],
        default="structure",
        help="'ocr' for PP-OCRv5 basic text recognition, "
             "'structure' for PP-StructureV3 full document parsing (default)",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code for PaddleOCR (default: en)",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.mode, args.lang)


if __name__ == "__main__":
    main()
