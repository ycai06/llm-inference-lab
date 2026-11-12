import math

import pytest

from backend.metrics.statistics import summarize


def test_known_distribution() -> None:
    result = summarize([1, 2, 3, 4, 5], bootstrap_samples=500)
    assert result["count"] == 5
    assert result["mean"] == 3
    assert result["median"] == 3
    assert result["stdev"] == pytest.approx(math.sqrt(2.5))
    assert result["p95"] is None
    assert result["median_ci95"] is not None
    assert result["median_ci95"][0] <= 3 <= result["median_ci95"][1]


def test_small_sample_suppresses_tail_and_interval() -> None:
    result = summarize([10])
    assert result == {
        "count": 1,
        "mean": 10,
        "median": 10,
        "stdev": None,
        "p95": None,
        "median_ci95": None,
    }
    assert summarize([1, 2, 3, 4])["median_ci95"] is None


def test_empirical_p95_linear_interpolation_at_twenty_samples() -> None:
    assert summarize(list(range(1, 21)))["p95"] == pytest.approx(19.05)


def test_bootstrap_is_reproducible_and_does_not_change_global_rng() -> None:
    import random

    before = random.getstate()
    first = summarize([1, 3, 5, 7, 9], seed=67)
    assert first == summarize([1, 3, 5, 7, 9], seed=67)
    assert random.getstate() == before


def test_constant_samples_have_zero_width_interval() -> None:
    result = summarize([7] * 5)
    assert result["stdev"] == 0
    assert result["median_ci95"] == [7, 7]


@pytest.mark.parametrize("values", [[], [float("nan")], [1, float("inf")], [float("-inf")]])
def test_reject_invalid_values(values: list[float]) -> None:
    with pytest.raises(ValueError, match="finite value"):
        summarize(values)


@pytest.mark.parametrize("count", [0, -1, True])
def test_reject_invalid_bootstrap_count(count: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        summarize([1, 2, 3, 4, 5], bootstrap_samples=count)
