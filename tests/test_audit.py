import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import pandas as pd
from costrank.config import Config
from costrank.data import synthetic
from costrank.experiment import run

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def edit_csv(path, mask, column, change):
    frame = pd.read_csv(path, dtype=str)
    i = frame.index[mask(frame)][0]
    frame.loc[i, column] = str(change(float(frame.loc[i, column])))
    frame.to_csv(path, index=False)


def test_run_audit_passes_clean_ledgers_and_flags_tampering(tmp_path):
    audit_run = load("audit_run").audit_run
    synthetic(tmp_path / "development", days=260, symbols=8)
    cfg = Config(dev_start="2018-10-01", dev_end="2018-10-31", holdout_start="2019-01-01",
        holdout_end="2019-01-31", train_days=100, valid_days=30, test_days=12, horizon=2,
        universe_size=8, topk=3, max_weight=.4, trees=30, threads=1, models=("factor",),
        policies=("topk", "cost"), slippage_grid_bps=(5.0,), bootstrap_samples=20)
    root = run(cfg, tmp_path / "development", tmp_path / "runs")
    clean = audit_run(root)
    assert clean["passed"] and clean["scenarios_audited"] == 4
    assert clean["worst_case"]["cash_identity_max_abs_cny"] < 1e-6
    daily = Path("factor__topk__slip_5bps") / "daily.csv.gz"
    orders = Path("factor__cost__zero_all_costs") / "orders.csv.gz"
    edit_csv(root / daily, lambda f: f.index == 5, "cash", lambda x: x + 1)
    edit_csv(root / orders, lambda f: f.side.eq("buy") & f.status.eq("filled"), "filled_qty", lambda x: x + 50)
    tampered = audit_run(root)
    assert not tampered["passed"]
    assert tampered["result_files_changed"] == sorted([str(daily), str(orders)])
    assert tampered["scenarios_failed"] == ["factor__cost__zero_all_costs", "factor__topk__slip_5bps"]
    assert tampered["scenarios"]["factor__cost__zero_all_costs"]["failures"] == ["odd_lot_buys"]
    assert {"cash_identity_max_abs_cny", "nav_identity_max_abs_cny"} <= set(
        tampered["scenarios"]["factor__topk__slip_5bps"]["failures"])


def test_snapshot_audit_offline_on_collector_output(tmp_path, monkeypatch):
    from costrank.providers import collect_baostock
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    master = [["sh.600000", "A", "2010-01-01", "", "1", "1"],
              ["sz.000001", "B", "2010-01-01", dates[2], "1", "0"]]
    class Result:
        def __init__(self, fields, rows):
            self.fields, self.rows, self.error_code, self.error_msg = fields.split(","), iter(rows), "0", "ok"
        def next(self):
            self.row = next(self.rows, None)
            return self.row is not None
        def get_row_data(self):
            return self.row
    def history(code, fields, **kwargs):
        rows = [[d, code, "10", "10", "10", "10", "10", "10000", "100000", "1", "0"] for d in dates]
        if code == "sz.000001":
            rows[2] = [dates[2], code, "10", "10", "10", "10", "10", "", "", "0", "0"]
        return Result(fields, rows)
    def listing(day):
        alive = [m[0] for m in master if m[2] <= day and (not m[3] or m[3] > day)]
        return Result("code,tradeStatus,code_name", [[c, "1", "x"] for c in alive])
    fake = SimpleNamespace(login=lambda: SimpleNamespace(error_code="0"), logout=lambda: None,
        query_trade_dates=lambda **k: Result("calendar_date,is_trading_day", [[d, "1"] for d in dates]),
        query_stock_basic=lambda **k: Result("code,code_name,ipoDate,outDate,type,status", master),
        query_all_stock=lambda day: listing(day), query_history_k_data_plus=history)
    monkeypatch.setitem(sys.modules, "baostock", fake)
    monkeypatch.setattr("costrank.providers.time.sleep", lambda _: None)
    collect_baostock(tmp_path / "snap", dates[0], dates[-1], pause=0)
    report = load("audit_snapshot").audit_snapshot(tmp_path / "snap", Config(), dividend_symbols=0)
    assert report["validate"]["bars"] == 5 and report["validate"]["instruments"] == 2
    assert report["sample_coverage"]["delisted_within_range"] == 1
    assert report["suspension_st"]["suspended_rows"] == 0
    universe = report["provider_universe"]["snapshots_vs_master_alive"]
    assert set(universe) == {"2020-01-02", "2020-01-06"}
    assert all(v["snapshot_only_count"] == v["master_only_count"] == 0 for v in universe.values())
    assert "dividend_reconciliation" not in report
