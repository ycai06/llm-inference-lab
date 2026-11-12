"""Descriptive statistics over recorded repetitions, never mixed across cases."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(
    values: Sequence[float], bootstrap_samples: int = 2000, seed: int = 42
) -> dict[str, Any]:
    """Return a deterministic percentile-bootstrap 95% CI for the sample median.

    The CI is omitted below five repetitions; the empirical p95 is omitted below
    twenty. Neither policy turns a small sample into a population guarantee.
    Standard deviation uses Bessel's correction and is unavailable for n=1.
    """
    samples = [float(value) for value in values]
    if not samples or any(not math.isfinite(value) for value in samples):
        raise ValueError("Statistics require at least one finite value and no NaN/Infinity")
    if (
        isinstance(bootstrap_samples, bool)
        or not isinstance(bootstrap_samples, int)
        or bootstrap_samples < 1
    ):
        raise ValueError("bootstrap_samples must be a positive integer")
    count = len(samples)
    ordered = sorted(samples)
    interval: list[float] | None = None
    if count >= 5:
        rng = random.Random(seed)
        medians = sorted(
            statistics.median(rng.choices(samples, k=count)) for _ in range(bootstrap_samples)
        )
        interval = [_percentile(medians, 0.025), _percentile(medians, 0.975)]
    return {
        "count": count,
        "mean": statistics.mean(samples),
        "median": statistics.median(samples),
        "stdev": statistics.stdev(samples) if count >= 2 else None,
        "p95": _percentile(ordered, 0.95) if count >= 20 else None,
        "median_ci95": interval,
    }
