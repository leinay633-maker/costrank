"""v0.3: hold band at the top-k boundary."""
from dataclasses import replace
import pandas as pd
import pytest
from costrank.config import Config
from costrank.portfolio import allocate

SCORES = pd.Series({"a": 5., "b": 4., "c": 3., "d": 2., "e": 1.})
SIGMA = pd.Series(.02, index=list("abcde"))
HELD = pd.Series({"c": .45, "e": .45})  # c ranks third, inside a band of 3; e ranks fifth, outside


def config(hold_rank):
    return replace(Config(), topk=2, hold_rank=hold_rank, universe_size=5, max_weight=.5, invested=.9)


@pytest.mark.parametrize("hold_rank,chosen", [(0, {"a", "b"}), (3, {"a", "c"}), (5, {"c", "e"})])
def test_topk_hold_band_keeps_holdings_inside_the_band_and_fills_with_new_names(hold_rank, chosen):
    w, _ = allocate(SCORES, HELD, SIGMA, config(hold_rank), "topk", "2024-01-02")
    assert set(w[w > 0].index) == chosen
    assert w[w > 0].round(6).eq(.45).all()


def test_lp_hold_band_keeps_positive_utility_for_banded_holdings():
    w, info = allocate(SCORES, HELD, SIGMA, config(3), "risk", "2024-01-02")
    assert info["status"] == "optimal"
    assert w["c"] > 0 and w["e"] == pytest.approx(0) and w.get("b", 0) == pytest.approx(0)
    w0, _ = allocate(SCORES, HELD, SIGMA, config(0), "risk", "2024-01-02")
    assert w0["c"] == pytest.approx(0)


@pytest.mark.parametrize("hold_rank", [1, 6, -1, True])
def test_hold_rank_must_sit_between_topk_and_universe(hold_rank):
    with pytest.raises(ValueError, match="hold_rank"):
        config(hold_rank)
