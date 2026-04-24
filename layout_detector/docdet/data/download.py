"""
Per-dataset download + cleanup helpers for DocDet training phases.

Each helper resolves a local path to the dataset, downloading it
on demand via ``huggingface_hub.snapshot_download`` (or the Kaggle
API in the IIIT-AR case).  A paired ``cleanup_*`` function removes
the local copy after the phase finishes to free disk space.

Design notes
------------
* Downloads are idempotent: if the target directory already exists
  and contains at least one expected file, we skip the network call.
* Cleanup is explicit.  We never silently purge datasets; callers
  opt-in via the phase script's ``--cleanup-after`` flag.
* Kaggle access is optional; if the credentials file is missing we
  print a descriptive error so the user can configure it without
  having to read the docs.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Default local cache layout
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CACHE_ROOT = _PROJECT_ROOT / "layout_detector" / "data_cache"


def _resolve_cache_root(user_path: Optional[Path]) -> Path:
    """Return the user-specified cache root or the package default."""
    root = Path(user_path) if user_path is not None else _DEFAULT_CACHE_ROOT
    root.mkdir(parents=True, exist_ok=True)
    return root


def _already_populated(path: Path, patterns: tuple = ("*.parquet", "*.json", "*.png")) -> bool:
    """Return True if any of ``patterns`` matches inside ``path``."""
    if not path.exists():
        return False
    for pat in patterns:
        if any(path.rglob(pat)):
            return True
    return False


# ---------------------------------------------------------------------------
# DocSynth300K (Phase 1)
# ---------------------------------------------------------------------------

DOCSYNTH_REPO = "juliozhao/DocSynth300K"


def ensure_docsynth(cache_root: Optional[Path] = None) -> Path:
    """
    Snapshot-download DocSynth300K parquet shards to the cache.

    Warning: the full dataset is ~113 GB.  Callers on constrained
    disks should use ``DocSynthStreamingSource`` instead.
    """
    root = _resolve_cache_root(cache_root) / "docsynth"
    if _already_populated(root, patterns=("*.parquet",)):
        logger.info("DocSynth300K already at %s - skipping download", root)
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading DocSynth300K to %s ...", root)
    snapshot_download(
        repo_id=DOCSYNTH_REPO,
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=["*.parquet", "*.json"],
    )
    logger.info("DocSynth300K ready at %s", root)
    return root


def cleanup_docsynth(cache_root: Optional[Path] = None) -> None:
    """Delete the local DocSynth300K cache."""
    root = _resolve_cache_root(cache_root) / "docsynth"
    if root.exists():
        logger.info("Removing %s ...", root)
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# DocLayNet (Phase 2 - already present under ./DocLayNet or downloadable)
# ---------------------------------------------------------------------------

DOCLAYNET_REPO = "docling-project/DocLayNet-v1.2"


def ensure_doclaynet(cache_root: Optional[Path] = None) -> Path:
    """Download DocLayNet-v1.2 if not already present at the root."""
    local_fallback = _PROJECT_ROOT / "DocLayNet"
    if local_fallback.exists() and any(local_fallback.rglob("*.json")):
        return local_fallback

    root = _resolve_cache_root(cache_root) / "doclaynet"
    if _already_populated(root, patterns=("*.json", "*.png", "*.parquet")):
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading DocLayNet to %s ...", root)
    snapshot_download(
        repo_id=DOCLAYNET_REPO,
        repo_type="dataset",
        local_dir=str(root),
    )
    logger.info("DocLayNet ready at %s", root)
    return root


def cleanup_doclaynet(cache_root: Optional[Path] = None) -> None:
    """Remove the downloaded DocLayNet cache (leaves ./DocLayNet alone)."""
    root = _resolve_cache_root(cache_root) / "doclaynet"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# PubLayNet (Phase 2)
# ---------------------------------------------------------------------------

PUBLAYNET_REPO = "jordanparker6/publaynet"


def ensure_publaynet(cache_root: Optional[Path] = None) -> Path:
    """Download PubLayNet to the cache."""
    root = _resolve_cache_root(cache_root) / "publaynet"
    if _already_populated(root, patterns=("*.json", "*.tar", "*.png")):
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading PubLayNet to %s ...", root)
    snapshot_download(
        repo_id=PUBLAYNET_REPO,
        repo_type="dataset",
        local_dir=str(root),
    )
    return root


def cleanup_publaynet(cache_root: Optional[Path] = None) -> None:
    """Remove the PubLayNet cache."""
    root = _resolve_cache_root(cache_root) / "publaynet"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# DocBank (Phase 2)
# ---------------------------------------------------------------------------

DOCBANK_REPO = "maveriq/DocBank"


def ensure_docbank(cache_root: Optional[Path] = None) -> Path:
    """Download DocBank to the cache."""
    root = _resolve_cache_root(cache_root) / "docbank"
    if _already_populated(root):
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading DocBank to %s ...", root)
    snapshot_download(
        repo_id=DOCBANK_REPO,
        repo_type="dataset",
        local_dir=str(root),
    )
    return root


def cleanup_docbank(cache_root: Optional[Path] = None) -> None:
    """Remove the DocBank cache."""
    root = _resolve_cache_root(cache_root) / "docbank"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# TableBank (Phase 2)
# ---------------------------------------------------------------------------

TABLEBANK_REPO = "deepcopy/TableBank-Detection"


def ensure_tablebank(cache_root: Optional[Path] = None) -> Path:
    """Download TableBank to the cache."""
    root = _resolve_cache_root(cache_root) / "tablebank"
    if _already_populated(root):
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading TableBank to %s ...", root)
    snapshot_download(
        repo_id=TABLEBANK_REPO,
        repo_type="dataset",
        local_dir=str(root),
    )
    return root


def cleanup_tablebank(cache_root: Optional[Path] = None) -> None:
    """Remove the TableBank cache."""
    root = _resolve_cache_root(cache_root) / "tablebank"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# IIIT-AR-13K (Phase 2) - requires Kaggle API credentials
# ---------------------------------------------------------------------------

IIIT_AR_KAGGLE_SLUG = "gabrieletazza/iiitar13k"


def ensure_iiit_ar(cache_root: Optional[Path] = None) -> Path:
    """
    Download IIIT-AR-13K from Kaggle.

    Requires ``kaggle.json`` credentials either at
    ``~/.kaggle/kaggle.json`` or via environment variables
    ``KAGGLE_USERNAME`` + ``KAGGLE_KEY``.  The kaggle CLI is
    imported lazily so DocDet does not force a kaggle dependency
    for users who skip IIIT-AR.
    """
    root = _resolve_cache_root(cache_root) / "iiit_ar"
    if _already_populated(root):
        return root

    try:
        from kaggle import api as kaggle_api
    except ImportError as e:
        raise ImportError(
            "IIIT-AR-13K requires the 'kaggle' pip package. "
            "Install via `pip install kaggle` and configure "
            "credentials at ~/.kaggle/kaggle.json."
        ) from e

    if not os.environ.get("KAGGLE_USERNAME") and not (
        Path.home() / ".kaggle" / "kaggle.json"
    ).exists():
        raise RuntimeError(
            "Kaggle credentials not configured. Place kaggle.json in "
            "~/.kaggle/ or set KAGGLE_USERNAME / KAGGLE_KEY env vars."
        )

    root.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading IIIT-AR-13K from Kaggle to %s ...", root)
    kaggle_api.authenticate()
    kaggle_api.dataset_download_files(
        IIIT_AR_KAGGLE_SLUG, path=str(root), unzip=True,
    )
    return root


def cleanup_iiit_ar(cache_root: Optional[Path] = None) -> None:
    """Remove the IIIT-AR cache."""
    root = _resolve_cache_root(cache_root) / "iiit_ar"
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# OmniDocBench (Phase 3 - never train on this)
# ---------------------------------------------------------------------------

def ensure_omnidocbench(cache_root: Optional[Path] = None) -> Path:
    """
    Locate OmniDocBench.  Prefers the existing top-level
    ``./OmniDocBench/`` folder (already populated by the benchmark
    code) and falls back to a fresh HF snapshot to the cache root.
    """
    top_level = _PROJECT_ROOT / "OmniDocBench"
    if top_level.exists() and any(top_level.rglob("*.json")):
        return top_level

    root = _resolve_cache_root(cache_root) / "omnidocbench"
    if _already_populated(root, patterns=("*.json", "*.jpg", "*.png")):
        return root

    from huggingface_hub import snapshot_download

    logger.info("Downloading OmniDocBench to %s ...", root)
    snapshot_download(
        repo_id="samiuc/omnidocbench",
        repo_type="dataset",
        local_dir=str(root),
        allow_patterns=[
            "OmniDocBench.json",
            "images/**",
        ],
    )
    return root
