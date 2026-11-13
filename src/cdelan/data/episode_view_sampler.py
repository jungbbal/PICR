"""Two-views-per-episode batch sampler for the label-free InfoNCE.

A plain shuffled batch almost never contains two windows of the same episode, so a
same-episode InfoNCE would see no positives. This sampler builds each batch from
``batch_size // 2`` episodes and draws 2 windows from each, so every anchor has
exactly one positive and ``2*(B/2 - 1)`` negatives.

The two views are required to start at least ``min_view_gap`` steps apart so they
share no history frames -- the "augmentation" is temporal, and overlapping windows
would make the task trivially solvable from the shared frames rather than from the
dynamics. Episodes that cannot supply such a pair fall back to any two distinct
windows, and episodes with a single window are skipped.

Nothing here reads the payload mass. Episode identity is the only supervision.
"""

from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np


class EpisodeTwoViewBatchSampler:
    """Yields batches of window indices, paired as (episode, 2 views)."""

    def __init__(
        self,
        episode_window_indices: Sequence[np.ndarray],
        window_start: np.ndarray,
        batch_size: int,
        num_batches: int,
        min_view_gap: int = 12,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        if batch_size < 4 or batch_size % 2 != 0:
            raise ValueError(f"batch_size must be an even number >= 4; got {batch_size}")
        self.pairs_per_batch = batch_size // 2
        self.num_batches = int(num_batches)
        self.min_view_gap = int(min_view_gap)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.epoch = 0

        self.episodes = [np.asarray(idx, dtype=np.int64) for idx in episode_window_indices if len(idx) >= 2]
        if len(self.episodes) < self.pairs_per_batch:
            raise RuntimeError(
                f"Only {len(self.episodes)} episodes have >=2 windows; need {self.pairs_per_batch} per batch."
            )
        self.starts = [np.asarray(window_start, dtype=np.int64)[idx] for idx in self.episodes]
        self.n_fallback_pairs = 0

    def __len__(self) -> int:
        return self.num_batches

    def _draw_pair(self, rng: np.random.Generator, e: int) -> tuple[int, int]:
        windows, starts = self.episodes[e], self.starts[e]
        i = int(rng.integers(len(windows)))
        far = np.flatnonzero(np.abs(starts - starts[i]) >= self.min_view_gap)
        if far.size == 0:  # short episode: accept any other window
            self.n_fallback_pairs += 1
            far = np.flatnonzero(np.arange(len(windows)) != i)
        j = int(far[rng.integers(far.size)])
        return int(windows[i]), int(windows[j])

    def __iter__(self) -> Iterator[list[int]]:
        epoch = 0 if self.deterministic else self.epoch
        rng = np.random.default_rng(self.seed + 1000 * epoch)
        for _ in range(self.num_batches):
            chosen = rng.choice(len(self.episodes), size=self.pairs_per_batch, replace=False)
            batch: list[int] = []
            for e in chosen:
                a, b = self._draw_pair(rng, int(e))
                batch.append(a)
                batch.append(b)
            yield batch
        self.epoch = epoch + 1
