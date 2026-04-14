"""
Tier 3 -- Full DeepSeek OCR 2 VLM inference.

Runs the complete DeepSeek OCR 2 model (SAM encoder + Qwen2 causal-flow
encoder + language model decoder) via the HuggingFace ``AutoModel`` path.
This represents the heavyweight VLM approach where the model performs
both layout understanding and text recognition from the page image.

Known issue:
  The DeepSeek-OCR-2 remote model code imports ``LlamaFlashAttention2`` from
  ``transformers``, which was removed in transformers >= 4.46. If you hit an
  ``ImportError`` for that symbol, patch the HuggingFace-cached file
  ``~/.cache/huggingface/modules/transformers_modules/deepseek_hyphen_ai/
  DeepSeek_hyphen_OCR_hyphen_2/<revision>/modeling_deepseekv2.py`` so the
  import falls back to ``LlamaAttention`` (which handles all attention
  backends in modern transformers).

Prerequisites:
  - CUDA GPU with >= 24 GB VRAM (tested on RTX 4090)
  - flash-attn installed
  - Model downloaded: deepseek-ai/DeepSeek-OCR-2

Usage:
    python -m benchmark.tier3_ocr.extract_deepseek_ocr2 [--input-dir path]
                                                         [--output-dir path]
"""

import argparse
import re
import sys
from pathlib import Path

import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import DEEPSEEK_OCR2_MODEL, OMNIDOCBENCH_PDFS, OUTPUT_DIR, RENDER_DPI, find_pdfs
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.pdf_render import render_page
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier3_deepseek_ocr2"

PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "


def load_model(model_name: str):
    """
    Load the full DeepSeek OCR 2 model and tokenizer.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        _attn_implementation="eager", # default is flash_attention_2 - changed to eager to avoid ImportError
        trust_remote_code=True,
        use_safetensors=True,
    )
    model = model.eval().cuda().to(torch.bfloat16)
    return model, tokenizer


def run(input_dir: Path, output_base: Path, model_name: str):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = find_pdfs(input_dir)

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Loading DeepSeek OCR 2 model: {model_name}")
    model, tokenizer = load_model(model_name)

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        image, _ = render_page(str(pdf_path), dpi=RENDER_DPI)
        tmp_img = Path(out_dir) / f"_tmp_{page_id}.png"
        image.save(str(tmp_img))

        with timed(use_cuda=True) as t:
            result = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=str(tmp_img),
                output_path=str(out_dir),
                base_size=1024,
                image_size=768,
                crop_mode=True,
                save_results=False,
            )

        if isinstance(result, str):
            markdown = result
        elif isinstance(result, (list, tuple)):
            markdown = result[0] if result else ""
        else:
            markdown = str(result)

        markdown = re.sub(r"<\|ref\|>.*?<\|/ref\|>", "", markdown)
        markdown = re.sub(r"<\|det\|>.*?<\|/det\|>", "", markdown)

        write_page_markdown(out_dir, page_id, markdown)
        append_timing_row(
            timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds
        )

        tmp_img.unlink(missing_ok=True)

    write_summary(out_dir, METHOD_NAME, {"total_pages": len(pdf_files)})
    print(f"{METHOD_NAME}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEEPSEEK_OCR2_MODEL,
        help="HuggingFace model ID or local path",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.model_name)


if __name__ == "__main__":
    main()
