"""The one-time final path of RESEARCH section 9 (supplement six) on synthetic schema-3 data, for
both freeze candidates: develop on a snapshot sealed before the holdout, freeze, open the holdout
on the extended snapshot exactly once, and report it."""
import json
import sys
from pathlib import Path
import pandas as pd
import pytest
from costrank.data import read_bundle
from costrank.experiment import freeze, run
from test_v04_pipeline import schema3, small
from test_v04_signals import library_bundle

HOLDOUT, END = "2019-01-02", "2019-03-22"


@pytest.mark.parametrize("cell", [dict(policies=("cost",)),
                                  dict(policies=("mvo",), feature_set="v04", label_mode="neutral_rank")],
                         ids=["s2p2", "s4p4"])
def test_freeze_then_open_the_holdout_once(tmp_path, cell):
    bundle = library_bundle(tmp_path / "src")
    _, development_data = schema3(tmp_path, bundle=bundle, before=HOLDOUT, name="dev")
    _, final_data = schema3(tmp_path, bundle=bundle, name="final")
    assert (read_bundle(development_data, allow_after=HOLDOUT).prefix_hash(HOLDOUT)
            == read_bundle(final_data).prefix_hash(HOLDOUT))
    cfg = small(dev_start="2018-07-02", dev_end="2018-12-31", holdout_start=HOLDOUT, holdout_end=END,
                models=("factor", "ridge", "lightgbm"), **cell)
    with pytest.raises(Exception, match="holdout"):
        run(cfg, final_data, tmp_path / "early", allow_proxy=True)
    development = run(cfg, development_data, tmp_path / "runs", allow_proxy=True)
    superseded = freeze(tmp_path / "protocol" / "freeze-0.json", cfg, development_data, development)
    lock = freeze(tmp_path / "protocol" / "freeze.json", cfg, development_data, development)
    final = run(cfg, final_data, tmp_path / "final", holdout=True, freeze_file=lock, allow_proxy=True)
    manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete" and manifest["phase"] == "holdout"
    assert manifest["data_prefix_hash"] == json.loads(Path(lock).read_text(encoding="utf-8"))["development_prefix_hash"]
    assert Path(str(lock) + ".opened.json").exists()
    policy = cell["policies"][0]
    daily = pd.read_csv(final / f"lightgbm__{policy}__slip_10bps" / "daily.csv.gz", parse_dates=["date"])
    assert daily.date.min() >= pd.Timestamp(HOLDOUT) and daily.date.max() <= pd.Timestamp(END)
    folds = json.loads((final / "folds.json").read_text(encoding="utf-8"))
    assert len(folds) == 1 and folds[0]["test_start"] >= HOLDOUT
    sys.path.insert(0, "scripts")
    from holdout_report import report
    r = report(final, development=development, samples=99)
    assert set(r["models"]) == {"factor", "ridge", "lightgbm"} and r["development_lightgbm_ir_by_year"]
    with pytest.raises(Exception):
        run(cfg, final_data, tmp_path / "again", holdout=True, freeze_file=lock, allow_proxy=True)
    # Spent for every other record too: one written before the opening, and one written after it (for
    # forward trading) even when moved out of the protocol directory.
    later = freeze(tmp_path / "protocol" / "freeze-2.json", cfg, development_data, development)
    assert json.loads(Path(later).read_text(encoding="utf-8"))["holdout_opened_before"][0]["run"] == str(final.resolve())
    moved = tmp_path / "elsewhere" / "freeze-2.json"
    moved.parent.mkdir()
    moved.write_bytes(Path(later).read_bytes())
    for record in (superseded, later, moved):
        with pytest.raises(RuntimeError, match="spent"):
            run(cfg, final_data, tmp_path / "again", holdout=True, freeze_file=record, allow_proxy=True)
        assert not Path(str(record) + ".opened.json").exists()
