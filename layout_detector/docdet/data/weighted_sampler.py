"""
Multi-source dataset + weighted sampling for DocDet Phase 2.

Phase 2 trains jointly on five real-document datasets with per-source
importance weights (spec 6.2).  We do not want to physically
oversample files on disk; instead we keep each underlying dataset
intact and use a ``WeightedRandomSampler`` with replacement so that
each training step draws from the weighted mixture.

Usage
-----
>>> mixture = MultiSourceDataset(
...     sources={
...         "doclaynet": doclaynet_dataset,
...         "publaynet": publaynet_dataset,
...     },
...     weights={"doclaynet": 3.0, "publaynet": 1.0},
... )
>>> sampler = mixture.build_sampler(num_samples=100_000)
>>> loader = DataLoader(mixture, sampler=sampler, batch_size=24,
...                     collate_fn=docdet_collate)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler

# Default Phase 2 mixing weights per spec 6.2.  Callers can pass a
# different dict to ``MultiSourceDataset.__init__`` if they disable
# one of the sub-datasets (e.g. IIIT-AR if no Kaggle access).
DEFAULT_PHASE2_WEIGHTS: Dict[str, float] = {
    "doclaynet": 3.0,
    "publaynet": 1.0,
    "tablebank": 2.0,
    "iiit_ar": 4.0,
}


class MultiSourceDataset(Dataset):
    """
    Concatenate multiple sub-datasets with per-source weights.

    Parameters
    ----------
    sources : dict source_name -> torch Dataset.  All sub-datasets
              must yield the same sample schema (see ``CocoSource``).
    weights : dict source_name -> float.  Must have a value for each
              key in ``sources``.  Higher weight = sampled more often.
    """

    def __init__(
        self,
        sources: Dict[str, Dataset],
        weights: Optional[Dict[str, float]] = None,
    ):
        if not sources:
            raise ValueError("MultiSourceDataset needs at least one source")
        self.sources: Dict[str, Dataset] = dict(sources)
        self.names: List[str] = list(sources.keys())

        if weights is None:
            weights = {name: 1.0 for name in self.names}
        missing = set(self.names) - set(weights.keys())
        if missing:
            raise ValueError(
                f"Missing weights for sources: {sorted(missing)}"
            )
        self.weights = {name: float(weights[name]) for name in self.names}

        # Pre-compute global->local index mapping once.
        self._offsets: List[int] = []
        running = 0
        for name in self.names:
            self._offsets.append(running)
            running += len(self.sources[name])
        self._total = running

        # Per-sample weight in the concatenated space.  Sampled with
        # probability proportional to (source weight / source size)
        # so all sources hit the requested importance exactly,
        # irrespective of their raw sizes.
        self._sample_weights: torch.Tensor = torch.zeros(self._total, dtype=torch.double)
        for name, offset in zip(self.names, self._offsets):
            sz = len(self.sources[name])
            if sz == 0:
                continue
            self._sample_weights[offset : offset + sz] = self.weights[name] / sz

    def __len__(self) -> int:
        return self._total

    def _locate(self, global_index: int) -> Tuple[str, int]:
        """Map a global index back to (source_name, local_index)."""
        if global_index < 0 or global_index >= self._total:
            raise IndexError(global_index)
        prev = 0
        for name, offset in zip(self.names, self._offsets):
            sz = len(self.sources[name])
            if global_index < offset + sz:
                return name, global_index - offset
            prev = offset + sz
        raise IndexError(global_index)

    def __getitem__(self, index: int):
        name, local_idx = self._locate(index)
        return self.sources[name][local_idx]

    def build_sampler(
        self,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> WeightedRandomSampler:
        """
        Construct a ``WeightedRandomSampler`` consistent with the
        per-source weights.

        Parameters
        ----------
        num_samples : samples to draw per epoch; defaults to
                      ``len(self)`` which gives each "epoch" the same
                      size as iterating over all underlying data once.
        replacement : MUST be True for weighted sampling (torch
                      requirement).
        """
        if num_samples is None:
            num_samples = self._total
        return WeightedRandomSampler(
            weights=self._sample_weights,
            num_samples=num_samples,
            replacement=replacement,
            generator=generator,
        )

    def source_sizes(self) -> Dict[str, int]:
        """Diagnostic helper: {source_name: len(sub_dataset)}."""
        return {name: len(self.sources[name]) for name in self.names}
