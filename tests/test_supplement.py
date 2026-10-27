"""RESEARCH section 11: scripts/supplement_run.py follows experiment.run's holdout path over another
period. Over the holdout's own period it must reproduce the opened run file for file; the report
scripts read its run directory; neutral_ic removes what the exposures explain."""
import gzip
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from costrank.experiment import freeze, run
from test_v04_pipeline import schema3, small
from test_v04_signals import library_bundle

sys.path.insert(0, "scripts")
HOLDOUT, END = "2019-01-02", "2019-03-22"


def content(path):
    data = Path(path).read_bytes()
    return gzip.decompress(data) if path.name.endswith(".gz") else data


@pytest.fixture(scope="module")
def opened(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("supplement")
    bundle = library_bundle(tmp / "src")
    _, development_data = schema3(tmp, bundle=bundle, before=HOLDOUT, name="dev")
    _, final_data = schema3(tmp, bundle=bundle, name="final")
    cfg = small(dev_start="2018-07-02", dev_end="2018-12-31", holdout_start=HOLDOUT, holdout_end=END,
                models=("factor", "ridge", "lightgbm"), policies=("cost",))
    development = run(cfg, development_data, tmp / "runs", allow_proxy=True)
    lock = freeze(tmp / "protocol" / "freeze.json", cfg, development_data, development)
    final = run(cfg, final_data, tmp / "final", holdout=True, freeze_file=lock, allow_proxy=True)
    return tmp, final_data, development, lock, final


def test_over_the_holdout_period_it_reproduces_the_opened_run(opened):
    from supplement_run import supplement
    tmp, data, _, lock, final = opened
    protocol = sorted(p.name for p in Path(lock).parent.iterdir())
    again = supplement(lock, data, tmp / "again", HOLDOUT, END, allow_proxy=True)
    assert sorted(p.name for p in Path(lock).parent.iterdir()) == protocol  # no receipt, nothing reopened
    manifest = json.loads((again / "manifest.json").read_text(encoding="utf-8"))
    original = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["phase"] == "supplement"
    assert manifest["prediction_hash"] == original["prediction_hash"] and manifest["development_prefix_matches"]
    compared = [n for n in original["result_files"] if n not in ("REPORT.md", "events.jsonl")]
    assert len(compared) > 20 and set(compared) <= set(manifest["result_files"])
    for name in compared:
        assert content(again / name) == content(final / name), name


def test_a_later_period_and_its_reports(opened):
    from supplement_run import supplement
    from supplement_report import report
    tmp, data, development, lock, _ = opened
    later = supplement(lock, data, tmp / "later", "2019-01-14", END, allow_proxy=True)
    folds = json.loads((later / "folds.json").read_text(encoding="utf-8"))
    assert len(folds) == 1 and folds[0]["test_start"] >= "2019-01-14" and folds[0]["test_end"] <= END
    daily = pd.read_csv(later / "lightgbm__cost__slip_10bps" / "daily.csv.gz", parse_dates=["date"])
    assert daily.date.min() >= pd.Timestamp("2019-01-14") and daily.date.max() <= pd.Timestamp(END)
    from audit_run import audit_run
    audited = audit_run(later)
    assert audited["passed"] and audited["scenarios_audited"] == 6  # 3 models x (zero costs, 10 bps)
    r = report(later, development, samples=99)
    assert set(r["models"]) == {"factor", "ridge", "lightgbm"}
    assert r["forward_window"]["models"]["lightgbm"]["window"]["sessions"] == len(daily) - 1
    assert set(r["models"]["lightgbm"]["comparisons"]) >= {"eligible-pool benchmark", "wide equal weight"}
    with pytest.raises(ValueError, match="holdout"):
        supplement(lock, data, tmp / "early", "2018-12-03", END, allow_proxy=True)


def test_neutral_ic_reproduces_the_rank_ic_and_its_period(opened):
    from neutral_ic import neutral_ic
    tmp, data, _, _, final = opened
    result, daily = neutral_ic(final, data)
    assert result["phase"] == "holdout" and result["raw_vs_rank_ic_csv_max_abs_gap"] < 1e-9
    assert daily.date.min() >= pd.Timestamp(HOLDOUT) and daily.date.max() <= pd.Timestamp(END)
    assert set(result["models"]) == {"factor", "ridge", "lightgbm"}


def test_neutralizing_removes_what_the_exposures_explain():
    from neutral_ic import session_ics
    rng = np.random.default_rng(5)
    n = 400
    names = [f"s{i}" for i in range(n)]
    size, alpha = rng.normal(size=n), rng.normal(size=n)
    groups = rng.integers(0, 4, n)
    exposures = pd.DataFrame({"size": size, "beta": rng.normal(size=n), "value": 0.0}, index=names)
    for g in range(4):
        exposures[f"ind_{g}"] = (groups == g).astype(float)
    exposures["ind_empty"] = 0.0
    labels = pd.Series(size + 0.5 * alpha + rng.normal(size=n), index=names)
    scores = pd.DataFrame({"style": size + 0.01 * rng.normal(size=n), "alpha": alpha}, index=names)
    ics = session_ics(scores, labels, exposures)
    assert ics["style"]["raw"] > 0.5 and abs(ics["style"]["industry_style"]) < 0.1
    assert ics["style"]["industry"] > 0.5  # industry alone does not explain a size score
    assert ics["style"]["score_r2"] > 0.99 and ics["alpha"]["score_r2"] < 0.05
    assert ics["alpha"]["raw"] > 0.2 and abs(ics["alpha"]["industry_style"] - ics["alpha"]["raw"]) < 0.05
    expected = scores["alpha"].rank().corr(labels.rank())
    assert abs(ics["alpha"]["raw"] - expected) < 1e-12
