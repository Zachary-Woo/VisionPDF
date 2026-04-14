"""
Evaluation harness for the PDF extraction benchmark.

Measures extraction quality on two axes:

  Text fidelity --  normalised edit distance on the full page text after
                    stripping formatting (markdown, HTML, LaTeX) from both
                    the prediction and the ground truth.

  Block recovery -- for each annotated GT block (title, text_block, table,
                    caption ...) find the best-matching chunk in the
                    prediction and score it.  This shows whether the
                    content of each semantic unit was preserved regardless
                    of ordering.

Results are segmented by:
  - Document type   (page_attribute.data_source)
  - Block category  (category_type)

Usage:
    python -m benchmark.evaluate.run_eval [--results-dir path] [--gt-json path]
"""

import argparse
import csv
import html
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_JSON, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_IMG_PLACEHOLDER = re.compile(r"\[Image\]", re.IGNORECASE)
_MD_HEADER = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_EMPHASIS = re.compile(r"\*+")
_MD_BLOCKQUOTE = re.compile(r"^>\s*", re.MULTILINE)
_MD_LIST = re.compile(r"^[-*+]\s+", re.MULTILINE)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_CODE_FENCE = re.compile(r"^```[^\n]*$", re.MULTILINE)
_MD_HRULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_MD_TABLE_SEP = re.compile(r"^\|?[-:| ]+\|[-:| ]*$", re.MULTILINE)
_MD_TABLE_PIPE = re.compile(r"\|")
_LATEX_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$\s*\^?\{?([^}$]*)\}?\s*\$")
_SOFT_HYPHEN = re.compile(r"[\u00ad\ufffe\uffff]")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Strip formatting from both GT and prediction texts so that edit
    distance measures content accuracy, not format differences.

    Handles HTML (tags + comments + entities), markdown (headers,
    emphasis, blockquotes, lists, links, code fences, horizontal
    rules, table separators), LaTeX (inline + display), image
    placeholders, and soft-hyphen / replacement characters.
    """
    text = _HTML_COMMENT.sub("", text)
    text = _HTML_TAG.sub("", text)
    text = _IMG_PLACEHOLDER.sub("", text)
    text = _MD_CODE_FENCE.sub("", text)
    text = _MD_HRULE.sub("", text)
    text = _MD_TABLE_SEP.sub("", text)
    text = _MD_TABLE_PIPE.sub(" ", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_EMPHASIS.sub("", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _MD_LIST.sub("", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _LATEX_DISPLAY.sub("", text)
    text = _LATEX_INLINE.sub(r"\1", text)
    text = html.unescape(text)
    text = _SOFT_HYPHEN.sub("", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Ground truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(gt_json_path: Path) -> List[Dict[str, Any]]:
    """
    Load OmniDocBench ground truth and return a list of page records.

    Each record contains:
        page_id      -- stem of the GT image filename
        data_source  -- document category (academic_literature, PPT2PDF, ...)
        blocks       -- list of {text, category_type, order} with normalised text
        full_text    -- all block texts joined (normalised), for whole-page metric
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    pages: List[Dict[str, Any]] = []

    for entry in data:
        page_info = entry.get("page_info", {})
        image_path = page_info.get("image_path", "")
        page_id = Path(image_path).stem
        page_attr = page_info.get("page_attribute", {})
        data_source = page_attr.get("data_source", "unknown")
        language = page_attr.get("language", "unknown")

        layout_dets = entry.get("layout_dets", [])
        ordered = sorted(
            layout_dets,
            key=lambda d: (d.get("order") is None, d.get("order", 0)),
        )

        blocks: List[Dict[str, Any]] = []
        block_texts: List[str] = []

        for det in ordered:
            raw = det.get("text", "")
            if not isinstance(raw, str) or not raw.strip():
                continue
            norm = normalize_text(raw)
            if not norm:
                continue
            blocks.append({
                "text": norm,
                "category_type": det.get("category_type", ""),
                "order": det.get("order"),
            })
            block_texts.append(norm)

        pages.append({
            "page_id": page_id,
            "data_source": data_source,
            "language": language,
            "blocks": blocks,
            "full_text": " ".join(block_texts),
        })

    return pages


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def normalised_edit_distance(pred: str, gt: str) -> float:
    """
    Normalised Levenshtein distance in [0, 1].
    0 = identical, 1 = completely different.
    """
    import editdistance

    if not gt and not pred:
        return 0.0
    dist = editdistance.eval(pred, gt)
    return dist / max(len(pred), len(gt), 1)


def token_overlap(pred: str, gt: str) -> Tuple[float, float, float]:
    """
    Order-insensitive bag-of-words overlap between *pred* and *gt*
    (both assumed to be already normalised / whitespace-collapsed).

    Returns (precision, recall, f1).
      - recall  = fraction of GT tokens recovered in the prediction
      - precision = fraction of prediction tokens that appear in the GT
      - f1 = harmonic mean
    """
    import string
    import re

    # Remove punctuation so "word," matches "word"
    translator = str.maketrans('', '', string.punctuation + '，。！？；：【】（）《》“”‘’')
    pred = pred.translate(translator)
    gt = gt.translate(translator)

    # Add spaces around CJK characters so they are treated as individual tokens
    pred = re.sub(r'([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', r' \1 ', pred)
    gt = re.sub(r'([\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af])', r' \1 ', gt)

    pred_tokens = pred.split()
    gt_tokens = gt.split()

    if not gt_tokens and not pred_tokens:
        return 1.0, 1.0, 1.0
    if not gt_tokens:
        return 0.0, 1.0, 0.0
    if not pred_tokens:
        return 1.0, 0.0, 0.0

    overlap = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


def _split_into_chunks(raw_text: str) -> List[str]:
    """
    Split raw prediction markdown into normalised paragraph chunks for
    block-level matching.

    Splits on blank-line boundaries first, then normalises each chunk
    independently.  Very long single paragraphs are further split on
    newlines so that blocks aren't lost inside a wall of text.
    """
    raw_paragraphs = re.split(r"\n\s*\n", raw_text)
    chunks: List[str] = []

    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) > 600:
            for line in para.split("\n"):
                norm = normalize_text(line)
                if norm:
                    chunks.append(norm)
        else:
            norm = normalize_text(para)
            if norm:
                chunks.append(norm)

    return chunks


def _best_block_match(
    block_text: str,
    pred_chunks: List[str],
) -> float:
    """
    Score a single GT block against the prediction chunks.

    Strategy (fast):
      1. Try each individual chunk.
      2. Try each pair of adjacent chunks (handles splits across two
         paragraphs).
      3. Try the full concatenation of all chunks (handles a GT block
         that spans most of the page, e.g. a long body paragraph that
         the prediction split into many lines).

    Returns the lowest normalised edit distance found.
    """
    if not pred_chunks:
        return 1.0

    best = 1.0

    for i, chunk in enumerate(pred_chunks):
        ed = normalised_edit_distance(chunk, block_text)
        if ed < best:
            best = ed
        if best == 0.0:
            return 0.0
        if i + 1 < len(pred_chunks):
            merged = chunk + " " + pred_chunks[i + 1]
            ed = normalised_edit_distance(merged, block_text)
            if ed < best:
                best = ed

    if len(pred_chunks) > 2:
        full = " ".join(pred_chunks)
        ed = normalised_edit_distance(full, block_text)
        if ed < best:
            best = ed

    return best


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def load_timing(timing_csv: Path) -> Dict[str, float]:
    """Read timing CSV into {page_id: wall_seconds}."""
    timing: Dict[str, float] = {}
    if not timing_csv.exists():
        return timing
    with open(timing_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            timing[row["page_id"]] = float(row["wall_seconds"])
    return timing


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

COMPOSITE_WEIGHT_ED = 0.30
COMPOSITE_WEIGHT_TOKEN_F1 = 0.35
COMPOSITE_WEIGHT_BLOCK = 0.35


def evaluate_method(
    method_dir: Path,
    gt_pages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate a single extraction method against all GT pages.

    Returns a dict with:
      - text accuracy (normalised edit distance, whole-page)
      - token recall / precision / F1 (order-insensitive)
      - per-page mean block recovery
      - composite weighted score combining all three axes
      - per-data_source and per-category_type breakdowns
      - per-page detail
    """
    gt_by_id = {p["page_id"]: p for p in gt_pages}
    md_files = sorted(method_dir.glob("*.md"))
    if not md_files:
        return {"error": "no markdown files found", "method": method_dir.name}

    timing = load_timing(method_dir / "timing.csv")

    page_results: List[Dict[str, Any]] = []
    total_page_ed = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1 = 0.0
    total_composite = 0.0
    total_pages = 0

    by_source: Dict[str, Dict] = defaultdict(
        lambda: {"ed_sum": 0.0, "recall_sum": 0.0, "f1_sum": 0.0,
                 "composite_sum": 0.0, "count": 0}
    )
    by_category: Dict[str, Dict] = defaultdict(
        lambda: {"ed_sum": 0.0, "count": 0}
    )

    for md_file in md_files:
        page_id = md_file.stem
        gt_page = gt_by_id.get(page_id)
        if not gt_page or not gt_page["full_text"]:
            continue

        raw_pred = md_file.read_text(encoding="utf-8")
        pred_norm = normalize_text(raw_pred)

        page_ed = normalised_edit_distance(pred_norm, gt_page["full_text"])
        prec, rec, f1 = token_overlap(pred_norm, gt_page["full_text"])

        total_page_ed += page_ed
        total_recall += rec
        total_precision += prec
        total_f1 += f1
        total_pages += 1

        pred_chunks = _split_into_chunks(raw_pred)

        block_scores: List[Dict[str, Any]] = []
        block_ed_sum = 0.0
        for block in gt_page["blocks"]:
            best_ed = _best_block_match(block["text"], pred_chunks)
            cat = block["category_type"]
            by_category[cat]["ed_sum"] += best_ed
            by_category[cat]["count"] += 1
            block_scores.append({
                "category": cat,
                "edit_distance": round(best_ed, 4),
            })
            block_ed_sum += best_ed

        n_blocks = len(gt_page["blocks"])
        page_block_sim = (
            1.0 - block_ed_sum / n_blocks if n_blocks > 0 else (1.0 - page_ed)
        )

        page_composite = (
            COMPOSITE_WEIGHT_ED * (1.0 - page_ed)
            + COMPOSITE_WEIGHT_TOKEN_F1 * f1
            + COMPOSITE_WEIGHT_BLOCK * page_block_sim
        )
        total_composite += page_composite

        src = gt_page["data_source"]
        by_source[src]["ed_sum"] += page_ed
        by_source[src]["recall_sum"] += rec
        by_source[src]["f1_sum"] += f1
        by_source[src]["composite_sum"] += page_composite
        by_source[src]["count"] += 1

        page_results.append({
            "page_id": page_id,
            "data_source": src,
            "page_edit_distance": round(page_ed, 4),
            "token_precision": round(prec, 4),
            "token_recall": round(rec, 4),
            "token_f1": round(f1, 4),
            "block_recovery": round(page_block_sim, 4),
            "composite": round(page_composite, 4),
            "block_scores": block_scores,
            "wall_seconds": timing.get(page_id),
        })

    avg_page_ed = total_page_ed / max(total_pages, 1)
    avg_recall = total_recall / max(total_pages, 1)
    avg_precision = total_precision / max(total_pages, 1)
    avg_f1 = total_f1 / max(total_pages, 1)
    avg_composite = total_composite / max(total_pages, 1)
    total_time = sum(v for v in timing.values())
    throughput = total_pages / total_time if total_time > 0 else 0.0

    source_summary: Dict[str, Any] = {}
    for src, acc in sorted(by_source.items()):
        n = max(acc["count"], 1)
        source_summary[src] = {
            "accuracy_pct": round((1.0 - acc["ed_sum"] / n) * 100, 2),
            "token_recall_pct": round(acc["recall_sum"] / n * 100, 2),
            "token_f1_pct": round(acc["f1_sum"] / n * 100, 2),
            "composite_pct": round(acc["composite_sum"] / n * 100, 2),
            "pages": acc["count"],
        }

    category_summary: Dict[str, Any] = {}
    for cat, acc in sorted(by_category.items()):
        avg = acc["ed_sum"] / max(acc["count"], 1)
        category_summary[cat] = {
            "accuracy_pct": round((1.0 - avg) * 100, 2),
            "blocks": acc["count"],
        }

    return {
        "method": method_dir.name,
        "total_pages": total_pages,
        "avg_edit_distance": round(avg_page_ed, 4),
        "text_accuracy_pct": round((1.0 - avg_page_ed) * 100, 2),
        "token_recall_pct": round(avg_recall * 100, 2),
        "token_precision_pct": round(avg_precision * 100, 2),
        "token_f1_pct": round(avg_f1 * 100, 2),
        "composite_pct": round(avg_composite * 100, 2),
        "throughput_pages_per_sec": round(throughput, 2),
        "total_time_seconds": round(total_time, 2),
        "by_data_source": source_summary,
        "by_category": category_summary,
        "page_results": page_results,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_eval_guide() -> None:
    """
    Print a short legend so console output reads like a figure caption.
    """
    print("\n" + "-" * 72)
    print("How to read the tables below")
    print("-" * 72)
    print(
        "Edit Dist % = (1 - normalised Levenshtein) on the full page "
        "(order-sensitive). Token Recall % = fraction of ground-truth "
        "tokens found in the prediction (order-insensitive). Token F1 % "
        "= harmonic mean of token precision and recall. Composite % = "
        f"weighted blend ({COMPOSITE_WEIGHT_ED:.0%} edit-distance "
        f"+ {COMPOSITE_WEIGHT_TOKEN_F1:.0%} token-F1 "
        f"+ {COMPOSITE_WEIGHT_BLOCK:.0%} block-recovery). "
        "All metrics are higher-is-better."
    )
    print(
        "Pages/sec is wall-clock throughput from timing.csv (higher is faster). "
        "Time (s) is the sum of per-page wall times for that method."
    )
    print(
        "Methods are sorted by Composite % (best first)."
    )
    print("-" * 72)


def _sort_key(r: Dict[str, Any]) -> float:
    """Sort methods best-composite-first (lower is worse, so negate)."""
    return -(r.get("composite_pct", 0))


def _print_overall_table(all_results: List[Dict[str, Any]]):
    """Print the top-level accuracy / throughput comparison."""
    try:
        from tabulate import tabulate
    except ImportError:
        return

    rows = []
    for r in sorted(all_results, key=_sort_key):
        rows.append([
            r["method"],
            r.get("composite_pct", "N/A"),
            r.get("text_accuracy_pct", "N/A"),
            r.get("token_recall_pct", "N/A"),
            r.get("token_f1_pct", "N/A"),
            r.get("throughput_pages_per_sec", "N/A"),
            r.get("total_pages", 0),
            r.get("total_time_seconds", "N/A"),
        ])
    print("\n=== Table 1: Overall ===")
    print(
        "Caption: Composite % is the headline metric (weighted blend of "
        "edit-distance similarity, token F1, and block recovery). "
        "Edit Dist % is order-sensitive full-page Levenshtein similarity. "
        "Token Recall % and Token F1 % are order-insensitive bag-of-words "
        "measures showing content completeness."
    )
    print(tabulate(
        rows,
        headers=["Method", "Composite %", "Edit Dist %",
                 "Tok Recall %", "Tok F1 %",
                 "Pages/sec", "Pages", "Time (s)"],
        tablefmt="grid",
    ))


def _print_source_table(all_results: List[Dict[str, Any]]):
    """Print composite score broken down by document type."""
    try:
        from tabulate import tabulate
    except ImportError:
        return

    all_sources = sorted({
        src
        for r in all_results
        for src in r.get("by_data_source", {})
    })
    if not all_sources:
        return

    headers = ["Method"] + [s[:16] for s in all_sources]
    rows = []
    for r in sorted(all_results, key=_sort_key):
        row = [r["method"]]
        for src in all_sources:
            info = r.get("by_data_source", {}).get(src)
            row.append(f"{info['composite_pct']}" if info else "-")
        rows.append(row)

    print("\n=== Table 2: Composite % by document type ===")
    print(
        "Caption: Each cell is the composite score for pages of that "
        "document type. Columns are OmniDocBench page_attribute.data_source."
    )
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def _print_category_table(all_results: List[Dict[str, Any]]):
    """Print block-level accuracy broken down by GT category."""
    try:
        from tabulate import tabulate
    except ImportError:
        return

    all_cats = sorted({
        cat
        for r in all_results
        for cat in r.get("by_category", {})
    })
    if not all_cats:
        return

    headers = ["Method"] + [c[:14] for c in all_cats]
    rows = []
    for r in sorted(all_results, key=_sort_key):
        row = [r["method"]]
        for cat in all_cats:
            info = r.get("by_category", {}).get(cat)
            row.append(f"{info['accuracy_pct']}" if info else "-")
        rows.append(row)

    print("\n=== Table 3: Block recovery by GT category (%) ===")
    print(
        "Caption: For each ground-truth layout block (title, text_block, "
        "table_caption, ...), we score the best-matching paragraph in the "
        "prediction, then average within that category. Good: high scores on "
        "text_block and title when you care about body copy and headings. Bad: "
        "very low page_number or header can be noisy (short strings, many "
        "blocks); large gaps between methods on text_block/title usually "
        "matter more than small gaps on rare categories."
    )
    print(tabulate(rows, headers=headers, tablefmt="grid"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    results_dir: Path,
    gt_json: Path,
    exclude_sources: Optional[List[str]] = None,
    only_sources: Optional[List[str]] = None,
):
    print(f"Loading ground truth from {gt_json}...")
    gt_pages = load_ground_truth(gt_json)
    print(f"  Loaded {len(gt_pages)} ground-truth pages")

    if exclude_sources:
        before = len(gt_pages)
        gt_pages = [
            p for p in gt_pages if p["data_source"] not in exclude_sources
        ]
        print(f"  Excluded {before - len(gt_pages)} pages from: {', '.join(exclude_sources)}")
        print(f"  Evaluating {len(gt_pages)} pages")
    elif only_sources:
        before = len(gt_pages)
        gt_pages = [
            p for p in gt_pages if p["data_source"] in only_sources
        ]
        print(f"  Filtered to {len(gt_pages)} pages from: {', '.join(only_sources)}")

    method_dirs = sorted(
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "summary.json").exists()
    )
    if not method_dirs:
        print(f"No method output directories found in {results_dir}")
        return

    all_results: List[Dict[str, Any]] = []

    for method_dir in method_dirs:
        print(f"\nEvaluating: {method_dir.name}")
        result = evaluate_method(method_dir, gt_pages)
        all_results.append(result)

        print(f"  Pages evaluated:  {result.get('total_pages', 0)}")
        print(f"  Composite:        {result.get('composite_pct', 'N/A')}%")
        print(f"  Edit distance:    {result.get('text_accuracy_pct', 'N/A')}%")
        print(f"  Token recall:     {result.get('token_recall_pct', 'N/A')}%")
        print(f"  Token F1:         {result.get('token_f1_pct', 'N/A')}%")
        print(f"  Throughput:       {result.get('throughput_pages_per_sec', 'N/A')} pages/sec")

    if not all_results:
        return

    summary_path = results_dir / "benchmark_results.json"
    out_data = []
    for r in all_results:
        compact = {k: v for k, v in r.items() if k != "page_results"}
        out_data.append(compact)
    summary_path.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
    print(f"\nResults written to {summary_path}")

    detail_path = results_dir / "benchmark_results_detail.json"
    detail_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Detailed per-page results written to {detail_path}")

    _print_eval_guide()
    _print_overall_table(all_results)
    _print_source_table(all_results)
    _print_category_table(all_results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory containing per-method output folders",
    )
    parser.add_argument(
        "--gt-json",
        type=Path,
        default=OMNIDOCBENCH_JSON,
        help="Path to OmniDocBench.json ground truth",
    )
    parser.add_argument(
        "--exclude-sources",
        nargs="+",
        default=None,
        help="Document types to exclude (e.g. note exam_paper)",
    )
    parser.add_argument(
        "--only-sources",
        nargs="+",
        default=None,
        help="Only evaluate these document types (e.g. academic_literature PPT2PDF)",
    )
    args = parser.parse_args()
    run(args.results_dir, args.gt_json, args.exclude_sources, args.only_sources)


if __name__ == "__main__":
    main()
