"""v0.4 inference: Ledoit-Wolf Sharpe-difference test, minimum detectable effect, deflated
Sharpe ratio and probability of backtest overfitting."""
import numpy as np
import pytest
from costrank.metrics import deflated_sharpe, minimum_detectable, pbo_cscv, sharpe_difference_test


def test_ledoit_wolf_detects_a_large_difference_and_not_an_identical_pair():
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 1200)
    x = common + rng.normal(0.0015, 0.005, 1200)
    y = common + rng.normal(0.0, 0.005, 1200)
    result = sharpe_difference_test(x, y, samples=999, seed=1)
    assert result["sharpe_difference_annual"] > 1 and result["p_value"] < 0.01
    same = sharpe_difference_test(x, x + 0.0, samples=199, seed=1)
    assert same["sharpe_difference_annual"] == 0 and same["p_value"] > 0.5


def test_a_series_without_variance_gives_an_undefined_test_not_a_tiny_p_value():
    x = np.random.default_rng(1).normal(0.001, 0.01, 400)
    result = sharpe_difference_test(x, np.zeros(400), samples=99, seed=1)
    assert np.isnan(result["p_value"]) and np.isnan(result["sharpe_difference_annual"])


def test_decision_condition_one_needs_a_positive_difference():
    import sys
    sys.path.insert(0, "scripts")
    from compare_v04 import primary_passes
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.01, 1200)
    better, worse = common + rng.normal(0.0015, 0.005, 1200), common + rng.normal(0.0, 0.005, 1200)
    assert primary_passes(sharpe_difference_test(better, worse, samples=199, seed=1))
    # Significantly worse is significant too, but never a reason to adopt S4P4.
    assert not primary_passes(sharpe_difference_test(worse, better, samples=199, seed=1))
    assert not primary_passes({"p_value": float("nan"), "sharpe_difference_annual": 1.0})


def test_ledoit_wolf_size_under_the_null_is_not_wildly_off():
    rng = np.random.default_rng(5)
    rejections = 0
    for _ in range(40):
        common = rng.normal(0, 0.01, 600)
        x, y = common + rng.normal(0.0005, 0.006, 600), common + rng.normal(0.0005, 0.006, 600)
        rejections += sharpe_difference_test(x, y, samples=299, seed=2)["p_value"] < 0.05
    assert rejections <= 8  # nominal 2 of 40


def test_minimum_detectable_effect():
    assert minimum_detectable(0.3) == pytest.approx((1.959964 + 0.841621) * 0.3, rel=1e-5)


def test_deflated_sharpe_falls_with_more_trials():
    r = np.random.default_rng(3).normal(0.001, 0.01, 1250)
    few, many = deflated_sharpe(r, 5, 0.5 / 252), deflated_sharpe(r, 500, 0.5 / 252)
    assert 0 <= many["deflated_sharpe"] < few["deflated_sharpe"] <= 1
    assert many["expected_max_sharpe_annual"] > few["expected_max_sharpe_annual"] > 0


def test_pbo_low_for_a_dominant_strategy_and_near_half_for_noise():
    values = []
    for seed in range(5):
        noise = np.random.default_rng(seed).normal(0, 0.01, (800, 6))
        values.append(pbo_cscv(noise, segments=8)["pbo"])
        good = noise.copy()
        good[:, 0] += 0.004
        assert pbo_cscv(good, segments=8)["pbo"] < 0.05
    assert 0.3 <= np.mean(values) <= 0.7  # one noise matrix varies a lot; the average sits near one half
