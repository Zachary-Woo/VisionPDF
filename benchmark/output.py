"""
Standardised output helpers for the benchmark.

Every extraction script writes one ``.md`` file per page (the format
OmniDocBench evaluation expects) plus an optional summary JSON that
records metadata like method name, model, and aggregate timing.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def method_output_dir(base_output_dir: Path, method_name: str) -> Path:
    """
    Return (and create) the output directory for a given extraction method.
    """
    d = base_output_dir / method_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_page_markdown(
    output_dir: Path,
    page_id: str,
    markdown: str,
) -> Path:
    """
    Write a single page's extracted markdown to *output_dir/<page_id>.md*.

    Returns the path of the written file.
    """
    out_path = output_dir / f"{page_id}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path


def write_summary(
    output_dir: Path,
    method_name: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Write a summary JSON to *output_dir/summary.json* containing the
    method name and any extra metadata the caller supplies.
    """
    payload: Dict[str, Any] = {"method": method_name}
    if extra:
        payload.update(extra)
    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
