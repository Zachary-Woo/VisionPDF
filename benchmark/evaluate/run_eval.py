"""
Evaluation harness for the PDF extraction benchmark.

Computes metrics against OmniDocBench ground truth:
  - Text Edit Distance (character-level accuracy, lower is better)
  - Table TEDS          (table structure recovery, higher is better)
  - Throughput          (pages per second from timing CSVs)
  - Overall score       ((1 - text_edit_dist) * 100 + table_TEDS + formula_CDM) / 3

For the lightweight self-contained evaluation (without the full OmniDocBench
evaluation codebase), this script computes normalised edit distance between
the extracted markdown and the ground-truth text for each page.

If the full OmniDocBench repo is available, set --omnidocbench-repo to use
their evaluation suite instead.

Usage:
    python -m benchmark.evaluate.run_eval [--results-dir path] [--gt-json path]
                                           [--omnidocbench-repo path]
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from benchmark.config import OMNIDOCBENCH_JSON, OUTPUT_DIR


# ---------------------------------------------------------------------------
# Lightweight edit-distance evaluation
# ---------------------------------------------------------------------------

def normalised_edit_distance(pred: str, gt: str) -> float:
    """
    Compute normalised Levenshtein edit distance in [0, 1].
    0 = perfect match, 1 = completely different.
    """
    import editdistance

    if not gt and not pred:
        return 0.0
    dist = editdistance.eval(pred, gt)
    return dist / max(len(pred), len(gt), 1)


def load_ground_truth(gt_json_path: Path) -> Dict[str, str]:
    """
    Load OmniDocBench ground-truth JSON and build a mapping from
    page_id to ground-truth text.

    OmniDocBench format: a list of dicts, each with ``page_info.image_path``
    and ``layout_dets`` containing annotated content.  We extract a
    concatenated plain-text representation for comparison.
    """
    with open(gt_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    gt_map: Dict[str, str] = {}
    for entry in data:
        # page_id derived from the image filename (without extension)
        page_info = entry.get("page_info", {})
        image_path = page_info.get("image_path", "")
        page_id = Path(image_path).stem

        # Concatenate all text content from layout_dets
        texts: List[str] = []
        for det in entry.get("layout_dets", []):
            content = det.get("content", "")
            if isinstance(content, str) and content.strip():
                texts.append(content.strip())
        gt_map[page_id] = "\n".join(texts)

    return gt_map


def load_timing(timing_csv: Path) -> Dict[str, float]:
    """
    Read a timing CSV and return a mapping from page_id to wall_seconds.
    """
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

def evaluate_method(
    method_dir: Path,
    gt_map: Dict[str, str],
) -> Dict[str, Any]:
    """
    Evaluate a single extraction method's output directory.

    Returns a dict with per-page and aggregate metrics.
    """
    md_files = sorted(method_dir.glob("*.md"))
    if not md_files:
        return {"error": "no markdown files found"}

    timing = load_timing(method_dir / "timing.csv")

    page_results: List[Dict[str, Any]] = []
    total_edit_dist = 0.0
    total_pages = 0

    for md_file in md_files:
        page_id = md_file.stem
        pred_text = md_file.read_text(encoding="utf-8")
        gt_text = gt_map.get(page_id, "")

        if not gt_text:
            continue

        ed = normalised_edit_distance(pred_text, gt_text)
        total_edit_dist += ed
        total_pages += 1

        page_results.append({
            "page_id": page_id,
            "edit_distance": round(ed, 4),
            "wall_seconds": timing.get(page_id),
        })

    avg_edit_dist = total_edit_dist / max(total_pages, 1)
    total_time = sum(t for t in timing.values() if t is not None)
    throughput = total_pages / total_time if total_time > 0 else 0.0

    return {
        "method": method_dir.name,
        "total_pages": total_pages,
        "avg_edit_distance": round(avg_edit_dist, 4),
        "text_accuracy_pct": round((1.0 - avg_edit_dist) * 100, 2),
        "throughput_pages_per_sec": round(throughput, 2),
        "total_time_seconds": round(total_time, 2),
        "page_results": page_results,
    }


# ---------------------------------------------------------------------------
# Optional: OmniDocBench full evaluation bridge
# ---------------------------------------------------------------------------

def run_omnidocbench_eval(
    omnidocbench_repo: Path,
    method_dir: Path,
    gt_json: Path,
):
    """
    If the OmniDocBench evaluation repo is available, run their full
    end-to-end evaluation (edit distance + TEDS + CDM).

    This writes a temporary config YAML and invokes their pdf_validation.py.
    """
    import subprocess
    import tempfile
    import yaml

    config = {
        "end2end_eval": {
            "metrics": {
                "text_block": {"metric": ["Edit_dist"]},
                "table": {"metric": ["TEDS", "Edit_dist"]},
                "reading_order": {"metric": ["Edit_dist"]},
            },
            "dataset": {
                "dataset_name": "end2end_dataset",
                "ground_truth": {"data_path": str(gt_json)},
                "prediction": {
                    "data_path": str(method_dir),
                    "match_method": "quick_match",
                },
            },
        }
    }

    config_path = Path(tempfile.mktemp(suffix=".yaml"))
    config_path.write_text(yaml.dump(config), encoding="utf-8")

    cmd = [
        sys.executable,
        str(omnidocbench_repo / "pdf_validation.py"),
        "--config", str(config_path),
    ]
    print(f"Running OmniDocBench eval: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(omnidocbench_repo), check=True)
    config_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    results_dir: Path,
    gt_json: Path,
    omnidocbench_repo: Optional[Path],
):
    print(f"Loading ground truth from {gt_json}...")
    gt_map = load_ground_truth(gt_json)
    print(f"  Loaded {len(gt_map)} ground-truth pages")

    # Find all method output directories
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

        if omnidocbench_repo and omnidocbench_repo.exists():
            run_omnidocbench_eval(omnidocbench_repo, method_dir, gt_json)
        else:
            result = evaluate_method(method_dir, gt_map)
            all_results.append(result)

            print(f"  Pages evaluated:  {result.get('total_pages', 0)}")
            print(f"  Avg edit dist:    {result.get('avg_edit_distance', 'N/A')}")
            print(f"  Text accuracy:    {result.get('text_accuracy_pct', 'N/A')}%")
            print(f"  Throughput:       {result.get('throughput_pages_per_sec', 'N/A')} pages/sec")

    # Write aggregate results
    if all_results:
        summary_path = results_dir / "benchmark_results.json"
        summary_path.write_text(
            json.dumps(all_results, indent=2), encoding="utf-8"
        )
        print(f"\nResults written to {summary_path}")

        # Print comparison table
        try:
            from tabulate import tabulate
            table_data = []
            for r in sorted(all_results, key=lambda x: x.get("avg_edit_distance", 1.0)):
                table_data.append([
                    r["method"],
                    r.get("text_accuracy_pct", "N/A"),
                    r.get("throughput_pages_per_sec", "N/A"),
                    r.get("total_pages", 0),
                    r.get("total_time_seconds", "N/A"),
                ])
            print("\n" + tabulate(
                table_data,
                headers=["Method", "Accuracy %", "Pages/sec", "Pages", "Total Time (s)"],
                tablefmt="grid",
            ))
        except ImportError:
            pass


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
        "--omnidocbench-repo",
        type=Path,
        default=None,
        help="Optional path to cloned OmniDocBench repo for full evaluation",
    )
    args = parser.parse_args()
    run(args.results_dir, args.gt_json, args.omnidocbench_repo)


if __name__ == "__main__":
    main()
