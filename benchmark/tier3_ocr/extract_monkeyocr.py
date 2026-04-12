"""
Tier 3 -- MonkeyOCR (SRR paradigm) document parsing.

MonkeyOCR decomposes document parsing into Structure detection,
Recognition, and Relation prediction using a lightweight 1.2-3B
parameter multimodal model.  It uses DocLayoutYOLO for layout detection
and its own LayoutReader for reading order.

Prerequisites:
  - Clone MonkeyOCR: git clone https://github.com/Yuliang-Liu/MonkeyOCR.git
  - Install:  cd MonkeyOCR && pip install -e .
  - GPU with >= 12 GB VRAM (RTX 3090 / 4090)

The script wraps MonkeyOCR's built-in parsing pipeline and collects the
output markdown for each page.

Usage:
    python -m benchmark.tier3_ocr.extract_monkeyocr [--input-dir path]
                                                     [--output-dir path]
                                                     [--config path]
"""

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_PDFS, OUTPUT_DIR, find_pdfs
from benchmark.output import method_output_dir, write_page_markdown, write_summary
from benchmark.timing import append_timing_row, timed

METHOD_NAME = "tier3_monkeyocr"


def _try_import_monkeyocr():
    """
    Import MonkeyOCR's core components.  Returns None if not installed.
    """
    try:
        from magic_pdf.model.custom_model import MonkeyOCR
        from magic_pdf.data.data_reader_writer import (
            FileBasedDataReader,
            FileBasedDataWriter,
        )
        from magic_pdf.data.dataset import PymuDocDataset
        from magic_pdf.model.doc_analyze_by_custom_model_llm import doc_analyze_llm
        return MonkeyOCR, FileBasedDataReader, FileBasedDataWriter, PymuDocDataset, doc_analyze_llm
    except ImportError:
        return None


def run(input_dir: Path, output_base: Path, config_path: str):
    out_dir = method_output_dir(output_base, METHOD_NAME)
    timing_csv = out_dir / "timing.csv"
    pdf_files = find_pdfs(input_dir)

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    imports = _try_import_monkeyocr()
    if imports is None:
        print(
            "MonkeyOCR is not installed.  Please install it first:\n"
            "  git clone https://github.com/Yuliang-Liu/MonkeyOCR.git\n"
            "  cd MonkeyOCR && pip install -e .\n"
        )
        return

    MonkeyOCR_cls, FileBasedDataReader, FileBasedDataWriter, PymuDocDataset, doc_analyze_llm = imports

    print(f"Loading MonkeyOCR model (config: {config_path})...")
    model = MonkeyOCR_cls(config_path)

    tmp_output = out_dir / "_monkeyocr_raw"
    tmp_output.mkdir(parents=True, exist_ok=True)

    reader = FileBasedDataReader()

    for pdf_path in tqdm(pdf_files, desc=METHOD_NAME):
        page_id = pdf_path.stem

        with timed(use_cuda=True) as t:
            file_bytes = reader.read(str(pdf_path))
            ds = PymuDocDataset(file_bytes)

            infer_result = ds.apply(doc_analyze_llm, MonkeyOCR_model=model)

            page_out = tmp_output / page_id
            page_out.mkdir(parents=True, exist_ok=True)
            img_dir = page_out / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            image_writer = FileBasedDataWriter(str(img_dir))
            md_writer = FileBasedDataWriter(str(page_out))

            pipe_result = infer_result.pipe_ocr_mode(image_writer, debug_mode=False)
            md_content = pipe_result.get_markdown(image_dir="images")

        write_page_markdown(out_dir, page_id, md_content)
        append_timing_row(
            timing_csv, METHOD_NAME, page_id, t.wall_seconds, t.cuda_seconds
        )

    shutil.rmtree(str(tmp_output), ignore_errors=True)

    write_summary(out_dir, METHOD_NAME, {"total_pages": len(pdf_files)})
    print(f"{METHOD_NAME}: processed {len(pdf_files)} pages -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=OMNIDOCBENCH_PDFS)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--config",
        type=str,
        default="model_configs.yaml",
        help="Path to MonkeyOCR config YAML (default: model_configs.yaml in MonkeyOCR repo)",
    )
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.config)


if __name__ == "__main__":
    main()
