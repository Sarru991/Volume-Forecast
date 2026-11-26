import numpy as np
import pandas as pd
import pytest

import generate_data
from model import add_features, bias, walk_forward, wape
from staffing import summarise


@pytest.fixture(scope="module")
def data():
    return generate_data.build()


@pytest.fixture(scope="module")
def featured(data):
    return add_features(data)


@pytest.fixture(scope="module")
def results(featured):
    return walk_forward(featured, folds=4)


def test_dataset_shape_and_repeatable(data):
    assert len(data) == 608 * 32
    assert data["offered"].min() >= 0
    assert generate_data.build().equals(data)


def test_wape_and_bias_maths():
    assert wape([100, 100], [90, 110]) == pytest.approx(0.10)
    assert bias([100, 100], [90, 90]) == pytest.approx(-0.10)


def test_no_future_leakage(featured):
    """Every lag feature must equal the real value from exactly k*7 days earlier."""
    idx = featured.set_index(["date", "slot"])["offered"]
    sample = featured.sample(200, random_state=1)
    for _, r in sample.iterrows():
        for k in (7, 14, 21, 28):
            past = r["date"] - pd.Timedelta(days=k)
            if (past, r["slot"]) in idx.index:
                assert r[f"lag{k}"] == idx[(past, r["slot"])]


def test_walk_forward_trains_only_on_the_past(featured):
    res = walk_forward(featured, folds=3)
    assert res["date"].min() > featured["date"].min()
    assert res.groupby("fold")["date"].min().is_monotonic_increasing
    assert res["ml"].min() >= 0


def test_ml_beats_simple_methods_overall(results):
    assert wape(results.offered, results.ml) < wape(results.offered, results.seasonal)
    assert wape(results.offered, results.seasonal) < wape(results.offered, results.naive)


def test_ml_wins_clearly_on_special_days(results):
    sp = results[results.special == 1]
    assert wape(sp.offered, sp.ml) < 0.75 * wape(sp.offered, sp.seasonal)


def test_staffing_summary_is_consistent(results):
    s = summarise(results)
    assert set(s["method"]) == {"naive", "seasonal", "ml"}
    assert (s["within_2_agents_pct"].between(0, 100)).all()
    ml = s[s.method == "ml"].iloc[0]
    naive = s[s.method == "naive"].iloc[0]
    assert ml["within_2_agents_pct"] > naive["within_2_agents_pct"]
