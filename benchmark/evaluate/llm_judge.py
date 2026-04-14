import os
import json
import random
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List
import requests
from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmark.config import OMNIDOCBENCH_JSON, OMNIDOCBENCH_IMAGES, OUTPUT_DIR
from benchmark.evaluate.run_eval import load_ground_truth

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "deepseek/deepseek-v3.2"

def evaluate_with_llm(gt_text: str, pred_text: str) -> dict:
    """
    Calls OpenRouter to evaluate the extraction quality.
    """
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables.")

    system_prompt = """You are an expert AI evaluator tasked with grading the usability of RAW document text extractions for downstream LLM tasks.
IMPORTANT CONTEXT: This is the FIRST data extraction step prior to any post-processing. You must evaluate the RAW extraction quality. Do NOT penalize the extraction for lacking perfectly polished Markdown (like `##` for headers or `|---|` for tables). 

You will be provided with:
1. GROUND TRUTH TEXT: An annotated representation of the page. Note that the ground truth might use custom tags (like [order=2 | title]) and might omit table bodies or flatten math.
2. EXTRACTED TEXT: The raw output from the PDF extraction pipeline.

Since the Ground Truth itself might not be perfectly formatted for an LLM (e.g., using custom tags instead of Markdown), a "blind test" of only the extraction is unfair. Therefore, you must evaluate BOTH the Ground Truth and the Extracted Text independently on a 1-5 scale across these four axes:

1. Information Completeness (Recall): Does the text capture all the core body text, headings, and metadata? (For the extraction, ignore minor OCR errors or typos).
2. Reading Order / Flow: Does the text flow logically? If headers/footers are interleaved, do they break the flow of the main paragraphs? (Moving headers/footers to the top or bottom is acceptable).
3. Structural Boundaries (formerly Formatting): Are the raw textual elements distinct? Focus on whether structural boundaries (like line breaks between paragraphs or sections) exist. Do NOT penalize for lack of Markdown tags. A raw block of text separated by newlines is a perfect 5 if it reflects the document's chunks.
4. Table/Data Preservation: Is the raw text of the table extracted so that no data is lost? It does NOT need to be a perfectly formatted Markdown/HTML table. If there are no tables in the document, score this a 5.

Provide scores for both the Ground Truth and the Extraction. For each axis, include a justification (1-2 sentences). For any score <= 3, specifically name what is missing, wrong, or broken. Keep justifications concise -- no more than 2 sentences each.
"""

    MAX_INPUT_CHARS = 12000
    gt_display = gt_text[:MAX_INPUT_CHARS] + "\n[...TRUNCATED...]" if len(gt_text) > MAX_INPUT_CHARS else gt_text
    pred_display = pred_text[:MAX_INPUT_CHARS] + "\n[...TRUNCATED...]" if len(pred_text) > MAX_INPUT_CHARS else pred_text

    user_prompt = f"""
=== GROUND TRUTH TEXT ===
{gt_display}

=== EXTRACTED TEXT ===
{pred_display}
"""

    score_obj = {
        "type": "object",
        "properties": {
            "gt_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Score 1-5 for GT"},
            "pred_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Score 1-5 for Extraction"},
            "justification": {"type": "string", "description": "Why these scores? For any score <= 3, list the specific problems found (missing text, scrambled paragraphs, etc)."}
        },
        "required": ["gt_score", "pred_score", "justification"],
        "additionalProperties": False
    }

    schema = {
        "type": "object",
        "properties": {
            "information_completeness": score_obj,
            "reading_order": score_obj,
            "structural_boundaries": score_obj,
            "table_data_preservation": score_obj,
            "gt_overall_usability_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Overall score 1-5 for GT"},
            "pred_overall_usability_score": {"type": "integer", "minimum": 1, "maximum": 5, "description": "Overall score 1-5 for Extraction"},
            "overall_justification": {"type": "string", "description": "Summary of the key strengths and weaknesses of the extraction."}
        },
        "required": [
            "information_completeness", "reading_order",
            "structural_boundaries", "table_data_preservation",
            "gt_overall_usability_score", "pred_overall_usability_score",
            "overall_justification"
        ],
        "additionalProperties": False
    }

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/visionpdf",
        "X-Title": "VisionPDF Benchmark"
    }

    example_1_user = """
=== GROUND TRUTH TEXT ===
[order=1 | title]
Introduction

[order=2 | text_block]
This is a test document.

=== EXTRACTED TEXT ===
Introduction

This is a test document.
"""

    example_1_assistant = json.dumps({
        "information_completeness": {
            "gt_score": 5, "pred_score": 5,
            "justification": "Both texts capture all information: the title 'Introduction' and the body text. Nothing is missing from either."
        },
        "reading_order": {
            "gt_score": 5, "pred_score": 5,
            "justification": "Both present the title before the body text, which is the correct reading order."
        },
        "structural_boundaries": {
            "gt_score": 3, "pred_score": 5,
            "justification": "The GT uses custom [order | type] tags which are not standard and would confuse an LLM parser. The Extraction uses a clean blank line between the title and body, clearly separating the two blocks."
        },
        "table_data_preservation": {
            "gt_score": 5, "pred_score": 5,
            "justification": "No tables are present in either text."
        },
        "gt_overall_usability_score": 4,
        "pred_overall_usability_score": 5,
        "overall_justification": "The Extraction is a perfect raw capture: all text present, correct order, clean paragraph separation. The GT is slightly less usable due to its custom annotation tags."
    })

    example_2_user = """
=== GROUND TRUTH TEXT ===
[order=1 | title]
Results

[order=2 | table_caption]
Table 1: Data

[order=3 | text_block]
The results show a 20% increase.

=== EXTRACTED TEXT ===
The results show a 20% increase.

Results
"""

    example_2_assistant = json.dumps({
        "information_completeness": {
            "gt_score": 5, "pred_score": 3,
            "justification": "The GT contains all three blocks: title, table caption, and body text. The Extraction is missing the table caption 'Table 1: Data' entirely."
        },
        "reading_order": {
            "gt_score": 5, "pred_score": 2,
            "justification": "The GT flows correctly: title -> caption -> body. The Extraction has the body text first and the title last, which inverts the document structure."
        },
        "structural_boundaries": {
            "gt_score": 3, "pred_score": 3,
            "justification": "The GT uses custom tags (not ideal). The Extraction has a blank line between the body and the misplaced title, so boundaries exist but are applied to wrongly-ordered content."
        },
        "table_data_preservation": {
            "gt_score": 5, "pred_score": 5,
            "justification": "No actual table data is present in either text (only a caption reference)."
        },
        "gt_overall_usability_score": 4,
        "pred_overall_usability_score": 2,
        "overall_justification": "The Extraction loses a table caption and inverts the reading order. An LLM reading this would see a body paragraph first with no context, then a floating title. Usable but structurally broken."
    })

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example_1_user},
            {"role": "assistant", "content": example_1_assistant},
            {"role": "user", "content": example_2_user},
            {"role": "assistant", "content": example_2_assistant},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "extraction_evaluation",
                "strict": True,
                "schema": schema
            }
        },
        "temperature": 0.0,
        "max_tokens": 4096
    }

    MAX_RETRIES = 2
    for attempt in range(MAX_RETRIES + 1):
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload
        )

        if response.status_code != 200:
            raise Exception(f"OpenRouter API Error: {response.status_code} - {response.text}")

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if attempt < MAX_RETRIES:
                print(f"    [Retry {attempt + 1}/{MAX_RETRIES}] JSON parse failed, retrying...")
                continue
            raise

def _format_review_md(page_id: str, data_source: str, gt_text: str,
                      pred_text: str, eval_result: dict, image_name: str) -> str:
    """
    Build a human-readable Markdown review for a single evaluated page.
    """
    lines = []
    lines.append(f"# Review: {page_id}")
    lines.append(f"**Data source:** {data_source}\n")

    if image_name:
        lines.append(f"![Page image]({image_name})\n")

    lines.append("---")
    lines.append("## Scores\n")
    lines.append("| Axis | GT | Pred | Justification |")
    lines.append("|------|----|------|---------------|")
    for axis in ("information_completeness", "reading_order",
                 "structural_boundaries", "table_data_preservation"):
        entry = eval_result.get(axis, {})
        label = axis.replace("_", " ").title()
        gt_s = entry.get("gt_score", "?")
        pred_s = entry.get("pred_score", "?")
        just = entry.get("justification", "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {label} | {gt_s} | {pred_s} | {just} |")

    gt_overall = eval_result.get("gt_overall_usability_score", "?")
    pred_overall = eval_result.get("pred_overall_usability_score", "?")
    lines.append(f"\n**Overall Usability:** GT {gt_overall}/5  |  Pred {pred_overall}/5\n")

    overall_just = eval_result.get("overall_justification", "")
    if overall_just:
        lines.append(f"> {overall_just}\n")

    lines.append("---")
    lines.append("## Ground Truth\n")
    lines.append("```")
    lines.append(gt_text)
    lines.append("```\n")

    lines.append("---")
    lines.append("## Extraction\n")
    lines.append("```")
    lines.append(pred_text)
    lines.append("```\n")

    return "\n".join(lines)


def run_llm_eval(
    results_dir: Path,
    gt_json: Path,
    method_name: str,
    num_samples: int = 10,
    seed: int = 42,
    review: bool = False,
    lang: str = None
):
    print(f"Loading ground truth from {gt_json}...")
    gt_pages = load_ground_truth(gt_json)
    gt_by_id = {p["page_id"]: p for p in gt_pages}

    method_dir = results_dir / method_name
    if not method_dir.exists() or not method_dir.is_dir():
        print(f"Error: Method directory {method_dir} does not exist.")
        return

    md_files = sorted(method_dir.glob("*.md"))
    if not md_files:
        print(f"No markdown files found in {method_dir}")
        return

    valid_files = [f for f in md_files if gt_by_id.get(f.stem) and gt_by_id[f.stem]["full_text"]]

    if lang:
        before = len(valid_files)
        valid_files = [f for f in valid_files if gt_by_id[f.stem].get("language") == lang]
        print(f"Language filter '{lang}': {before} -> {len(valid_files)} pages")

    print(f"Found {len(valid_files)} valid pages for {method_name}.")
    
    # Sample to save API costs
    random.seed(seed)
    if len(valid_files) > num_samples:
        sample_files = random.sample(valid_files, num_samples)
        print(f"Sampling {num_samples} pages for LLM evaluation...")
    else:
        sample_files = valid_files
        print(f"Evaluating all {len(valid_files)} pages...")

    review_dir = None
    if review:
        review_dir = results_dir / f"llm_review_{method_name}"
        review_dir.mkdir(parents=True, exist_ok=True)
        print(f"Review artifacts will be saved to {review_dir}")

    results = []

    for i, md_file in enumerate(sample_files, 1):
        page_id = md_file.stem
        gt_page = gt_by_id[page_id]

        print(f"[{i}/{len(sample_files)}] Evaluating {page_id}...")

        gt_formatted_blocks = []
        for block in gt_page["blocks"]:
            order = block.get("order", "None")
            cat = block.get("category_type", "unknown")
            gt_formatted_blocks.append(f"[order={order} | {cat}]\n{block['text']}")
        gt_text = "\n\n".join(gt_formatted_blocks)

        pred_text = md_file.read_text(encoding="utf-8")

        try:
            eval_result = evaluate_with_llm(gt_text, pred_text)
            eval_result["page_id"] = page_id
            eval_result["data_source"] = gt_page["data_source"]
            results.append(eval_result)
            print(f"  -> GT Usability: {eval_result['gt_overall_usability_score']}/5 | Pred Usability: {eval_result['pred_overall_usability_score']}/5")

            if review_dir:
                page_review_dir = review_dir / page_id
                page_review_dir.mkdir(parents=True, exist_ok=True)

                image_name = ""
                for ext in (".jpg", ".png", ".jpeg"):
                    src_img = OMNIDOCBENCH_IMAGES / f"{page_id}{ext}"
                    if src_img.exists():
                        shutil.copy2(src_img, page_review_dir / src_img.name)
                        image_name = src_img.name
                        break

                (page_review_dir / "ground_truth.md").write_text(gt_text, encoding="utf-8")
                (page_review_dir / "extraction.md").write_text(pred_text, encoding="utf-8")

                review_md = _format_review_md(
                    page_id, gt_page["data_source"],
                    gt_text, pred_text, eval_result, image_name
                )
                (page_review_dir / "review.md").write_text(review_md, encoding="utf-8")

                (page_review_dir / "scores.json").write_text(
                    json.dumps(eval_result, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        except Exception as e:
            print(f"  -> Error evaluating {page_id}: {e}")

    if not results:
        print("No successful evaluations.")
        return

    n = len(results)
    avg_gt_completeness = sum(r["information_completeness"]["gt_score"] for r in results) / n
    avg_pred_completeness = sum(r["information_completeness"]["pred_score"] for r in results) / n
    
    avg_gt_reading_order = sum(r["reading_order"]["gt_score"] for r in results) / n
    avg_pred_reading_order = sum(r["reading_order"]["pred_score"] for r in results) / n
    
    avg_gt_formatting = sum(r["structural_boundaries"]["gt_score"] for r in results) / n
    avg_pred_formatting = sum(r["structural_boundaries"]["pred_score"] for r in results) / n
    
    avg_gt_table = sum(r["table_data_preservation"]["gt_score"] for r in results) / n
    avg_pred_table = sum(r["table_data_preservation"]["pred_score"] for r in results) / n
    
    avg_gt_overall = sum(r["gt_overall_usability_score"] for r in results) / n
    avg_pred_overall = sum(r["pred_overall_usability_score"] for r in results) / n

    print("\n" + "="*65)
    print(f"LLM Evaluation Summary for {method_name}")
    print("="*65)
    print(f"Pages evaluated: {len(results)}")
    print(f"{'Metric':<30} | {'GT Score':<10} | {'Pred Score':<10}")
    print("-" * 65)
    print(f"{'Information Completeness':<30} | {avg_gt_completeness:<10.2f} | {avg_pred_completeness:<10.2f}")
    print(f"{'Reading Order / Flow':<30} | {avg_gt_reading_order:<10.2f} | {avg_pred_reading_order:<10.2f}")
    print(f"{'Structural Boundaries':<30} | {avg_gt_formatting:<10.2f} | {avg_pred_formatting:<10.2f}")
    print(f"{'Table/Data Preservation':<30} | {avg_gt_table:<10.2f} | {avg_pred_table:<10.2f}")
    print("-" * 65)
    print(f"{'OVERALL USABILITY':<30} | {avg_gt_overall:<10.2f} | {avg_pred_overall:<10.2f}")
    print("="*65)

    # Save results
    out_file = results_dir / f"llm_eval_{method_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "method": method_name,
            "samples": len(results),
            "averages": {
                "gt": {
                    "information_completeness": avg_gt_completeness,
                    "reading_order": avg_gt_reading_order,
                    "structural_boundaries": avg_gt_formatting,
                    "table_data_preservation": avg_gt_table,
                    "overall_usability": avg_gt_overall
                },
                "pred": {
                    "information_completeness": avg_pred_completeness,
                    "reading_order": avg_pred_reading_order,
                    "structural_boundaries": avg_pred_formatting,
                    "table_data_preservation": avg_pred_table,
                    "overall_usability": avg_pred_overall
                }
            },
            "page_results": results
        }, f, indent=2)
    print(f"Detailed results saved to {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate extraction quality using an LLM-as-a-judge via OpenRouter.")
    parser.add_argument("method_name", type=str, help="Name of the method directory in results/ (e.g., tier2_yolo_layoutreader)")
    parser.add_argument("--samples", type=int, default=10, help="Number of pages to sample for evaluation")
    parser.add_argument("--results-dir", type=Path, default=OUTPUT_DIR, help="Directory containing per-method output folders")
    parser.add_argument("--gt-json", type=Path, default=OMNIDOCBENCH_JSON, help="Path to OmniDocBench.json ground truth")
    parser.add_argument("--review", action="store_true", help="Save review artifacts (page image, GT, extraction, scores) per page into results/llm_review_{method}/")
    parser.add_argument("--lang", type=str, default=None, choices=["english", "simplified_chinese", "en_ch_mixed"], help="Only evaluate pages in this language")

    args = parser.parse_args()

    run_llm_eval(
        results_dir=args.results_dir,
        gt_json=args.gt_json,
        method_name=args.method_name,
        num_samples=args.samples,
        review=args.review,
        lang=args.lang
    )
