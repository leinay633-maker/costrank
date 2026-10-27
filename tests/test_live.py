"""Forward paper trading (docs/PAPER.md): day-by-day runs on a synthetic schema-3 snapshot that
grows one session at a time. Scores are stored once and never change, the official ledger equals a
one-shot replay, a failed data gate holds the book, and each close leaves the next session's orders."""
import json
import re
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from costrank.backtest import execution_views, simulate
from costrank.data import Bundle, TABLES3
from costrank.experiment import benchmark_returns
from costrank.features import prepare
from costrank.live import (GateFailed, MissingRecord, Protocol, differing_dates, evaluation_benchmark, first_execution,
                           forward_folds, monitor, read_record, record_session, run_day, shortfall, weekly_report)
from test_v04_pipeline import schema3, small

PASSED = {"passed": True, "checks": {"synthetic": True}, "examples": {}, "rows": 0}
FAILED = {"passed": False, "checks": {"row_count": False}, "examples": {}, "rows": 0}


def through(bundle, day):
    """The snapshot as it stood after `day`'s close."""
    day = pd.Timestamp(day)
    tables = {}
    for name, frame in bundle.tables.items():
        when = TABLES3.get(f"{name}.parquet", (None, "date"))[1]
        tables[name] = frame[frame[when] <= day] if when in frame else frame
    return Bundle(bundle.bars[bundle.bars.date <= day], bundle.instruments, bundle.calendar[bundle.calendar <= day],
                  dict(bundle.metadata, end=day.strftime("%Y-%m-%d")), bundle.actions[bundle.actions.date <= day],
                  bundle.delistings, tables)


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    bundle, _ = schema3(tmp_path_factory.mktemp("live"))
    return bundle


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """A development run on the same synthetic snapshot, for the weekly report's ranges."""
    from costrank.experiment import run
    _, root = schema3(tmp_path_factory.mktemp("reference"))
    return run(small(models=("ridge",), policies=("cost",)), root, tmp_path_factory.mktemp("runs"), allow_proxy=True)


def test_forward_folds_train_before_each_block():
    cal = pd.bdate_range("2020-01-01", periods=400)
    cfg = small()
    blocks = forward_folds(cal, cfg, cal[300], cal[345], 20)
    assert [b.test_start for b in blocks] == [str(cal[i].date()) for i in (300, 320, 340)]
    assert blocks[-1].test_end == str(cal[345].date())
    for b in blocks:
        assert b.validation_start < b.test_start and b.train_label_before <= b.validation_start
        assert b.validation_label_before < b.test_start


def test_day_by_day_runs_match_a_one_shot_replay(world, reference, tmp_path):
    root = tmp_path / "raw-cache"
    root.mkdir()
    (root / "collection-plan.json").write_text(json.dumps({"provider": "baostock", "schema": 3,
                                                           "start": "2018-01-02", "end": "2019-03-25"}))
    cfg = small(models=("ridge",), policies=("cost",))
    days = [d.strftime("%Y-%m-%d") for d in world.calendar[230:262]]
    protocol = Protocol(forward_start=days[0], model="ridge", policy="cost", refit_every=20, slippage_bps=10.0,
                        reference_run=str(reference))
    out = tmp_path / "paper"
    failed = days[10]  # a rebalance day (every 5th forward session) whose data gate fails
    for i, day in enumerate(days[:-1]):
        if day == failed:
            # A failed gate records nothing; once a person accepts it, the session is booked as no signal.
            with pytest.raises(GateFailed):
                run_day(root, cfg, protocol, out, day, days[i + 1], bundle=through(world, day), today_gate=FAILED)
            assert not (out / "ledger" / "daily" / f"{day}.csv").exists()
            with pytest.raises(MissingRecord):
                run_day(root, cfg, protocol, out, days[i + 1], days[i + 2], bundle=through(world, days[i + 1]),
                        today_gate=PASSED)
            (out / "gates" / f"{day}.accept").write_text("accepted: synthetic test", encoding="utf-8")
        result = run_day(root, cfg, protocol, out, day, days[i + 1], bundle=through(world, day),
                         today_gate=FAILED if day == failed else PASSED)
        assert result["revisions"] == {}, (day, result["revisions"])
        assert (out / "orders" / f"{days[i + 1]}.csv").exists() and (out / "reports" / f"{day}.md").exists()
    stored, recorded = read_record(out / "scores")
    assert failed not in recorded and len(recorded) == len(days) - 2
    # Every stored score is what the last day's data would also give: no session used later data.
    last = days[-2]
    full = prepare(through(world, last), cfg)
    from costrank.models import predict_oos
    predictions, _ = predict_oos(full, world.calendar[world.calendar <= last], cfg,
                                 fold_list=forward_folds(world.calendar, cfg, days[0], last, 20))
    again = predictions[predictions.model == "ridge"][["date", "symbol", "score"]]
    assert differing_dates(stored, again[again.date.isin(stored.date)], ["date", "symbol"]) == []
    # The official record is the one-shot replay of the forward period with the stored scores.
    one_shot = simulate(full, stored, through(world, last), replace(cfg, initial_cash=protocol.initial_cash), "cost",
                        slippage_bps=10.0, views=execution_views(full, days[0], last), period=(days[0], last),
                        next_session=days[-1])
    record, _ = read_record(out / "ledger" / "daily")
    assert differing_dates(record, one_shot["daily"], ["date"]) == []
    alloc, _ = read_record(out / "ledger" / "allocations")
    assert alloc.set_index("date").loc[pd.Timestamp(failed), "status"] == "no_signal"
    trades = one_shot["orders"]
    assert not trades.decision_date.eq(pd.Timestamp(failed)).any()
    alerts = (out / "alerts.log").read_text(encoding="utf-8")
    assert f"{failed} data gate failed" in alerts
    assert benchmark_returns(full, days[0], last, through(world, last)).benchmark_return.notna().all()
    weeks = sorted((out / "reports").glob("week-*.md"))
    assert len(weeks) >= 4 and "| 平均秩 IC（已成熟标签） |" in weeks[-1].read_text(encoding="utf-8")
    assert sum("建仓周，不比较" in w.read_text(encoding="utf-8") for w in weeks) == 1
    assert "至" in weeks[-1].read_text(encoding="utf-8").split("| 双边换手 |")[1].split("\n")[0]  # a reference range
    daily_report = (out / "reports" / f"{last}.md").read_text(encoding="utf-8")
    assert "实施落差" in daily_report and "风险模型归因" in daily_report and "当日风格贡献最大的三项" in daily_report
    build = (out / "reports" / f"{days[1]}.md").read_text(encoding="utf-8")
    assert "跟踪误差从第二个成交日起计算" in build and "笔，成交额" in build
    assert not [p.name for p in (out / "reports").glob("*.md") if re.search(r"\bnan\b", p.read_text(encoding="utf-8"), re.I)]


def test_a_gate_that_fails_and_then_passes_records_one_consistent_session(world, tmp_path):
    root = tmp_path / "raw-cache"
    root.mkdir()
    (root / "collection-plan.json").write_text(json.dumps({"provider": "baostock", "schema": 3,
                                                           "start": "2018-01-02", "end": "2019-03-25"}))
    cfg = small(models=("ridge",), policies=("cost",))
    days = [d.strftime("%Y-%m-%d") for d in world.calendar[230:242]]
    protocol = Protocol(forward_start=days[0], model="ridge", policy="cost", refit_every=20)
    out = tmp_path / "paper"
    for i, day in enumerate(days[:-1]):
        if day == days[5]:  # a rebalance day whose first fetch was incomplete
            with pytest.raises(GateFailed):
                run_day(root, cfg, protocol, out, day, days[i + 1], bundle=through(world, day), today_gate=FAILED)
        result = run_day(root, cfg, protocol, out, day, days[i + 1], bundle=through(world, day), today_gate=PASSED)
        assert result["revisions"] == {}, (day, result["revisions"])
    alloc, _ = read_record(out / "ledger" / "allocations")
    assert (alloc.get("status", pd.Series(dtype=str)) != "no_signal").all()
    assert len(list((out / "gates").glob(f"{days[5]}@*.json"))) == 2  # both attempts kept


def test_a_rerun_of_the_same_session_changes_nothing(world, tmp_path):
    root = tmp_path / "raw-cache"
    root.mkdir()
    (root / "collection-plan.json").write_text(json.dumps({"provider": "baostock", "schema": 3,
                                                           "start": "2018-01-02", "end": "2019-03-25"}))
    cfg = small(models=("ridge",), policies=("cost",))
    days = [d.strftime("%Y-%m-%d") for d in world.calendar[230:236]]
    protocol = Protocol(forward_start=days[0], model="ridge", policy="cost", refit_every=20)
    out = tmp_path / "paper"
    for i, day in enumerate(days[:-1]):
        run_day(root, cfg, protocol, out, day, days[i + 1], bundle=through(world, day), today_gate=PASSED)
    files = {p: p.read_bytes() for p in (out / "ledger").rglob("*.csv")} | \
            {p: p.read_bytes() for p in (out / "scores").rglob("*.csv")}
    result = run_day(root, cfg, protocol, out, days[-2], days[-1], bundle=through(world, days[-2]), today_gate=PASSED)
    assert result["revisions"] == {}
    assert {p: p.read_bytes() for p in files} == files


DAILY = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "adjustflag", "turn",
         "tradestatus", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST"]


def live_cache(root, sessions, ahead, codes):
    """A raw cache as live.fetch leaves it: per-session files, segments, a calendar running ahead."""
    raw = root / "raw"

    def write(path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    days = list(sessions)
    calendar = pd.date_range(days[0], ahead[-1]).strftime("%Y-%m-%d")
    write(raw / f"calendar__{calendar[0]}__{calendar[-1]}.csv", ["calendar_date", "is_trading_day"],
          [[d, "1" if d in set(days) | set(ahead) else "0"] for d in calendar])
    write(raw / "master.csv", ["code", "code_name", "ipoDate", "outDate", "type", "status"],
          [[c, c, "2010-01-01", "", "1", "1"] for c in codes])
    rng = np.random.default_rng(0)
    close = {c: 10.0 + i for i, c in enumerate(codes)}
    for d in days:
        rows = []
        for c in codes:
            pre = close[c]
            close[c] = round(pre * (1 + rng.normal(0, 0.02)), 2)
            o = round(pre * (1 + rng.normal(0, 0.005)), 2)
            volume = int(rng.integers(2_000_000, 6_000_000))
            rows.append([d, c, f"{o:.2f}", f"{max(o, close[c]) + 0.01:.2f}", f"{min(o, close[c]) - 0.01:.2f}",
                         f"{close[c]:.2f}", f"{pre:.2f}", str(volume), f"{volume * close[c]:.2f}", "3",
                         f"{rng.uniform(0.5, 5):.4f}", "1", "0", f"{rng.normal(20, 5):.4f}", f"{rng.uniform(0.5, 5):.4f}",
                         "2.0", "8.0", "0"])
        write(raw / "daily" / f"{d}.csv", DAILY, rows)
        write(raw / "adjust" / f"{d}.csv", ["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor",
                                            "adjustFactor"], [])
    months = sorted(pd.Series(days).groupby(pd.to_datetime(pd.Series(days)).dt.to_period("M")).min())
    for d in months:
        write(raw / "industry" / f"{d}.csv", ["updateDate", "code", "code_name", "industry", "industryClassification"],
              [[d, c, c, ("C39计算机" if i % 2 else "J66金融"), "证监会行业分类"] for i, c in enumerate(codes)])
        for index in ("hs300", "zz500"):
            write(raw / index / f"{d}.csv", ["updateDate", "code", "code_name"], [[d, codes[0], codes[0]]])
    from costrank.acquire import INDICES
    span = f"__{days[0]}__{days[-1]}"
    for code in INDICES:
        write(raw / "index" / f"{code}{span}.csv", ["date", "code", "open", "high", "low", "close", "preclose",
                                                     "volume", "amount"], [[d, code] + ["1000"] * 7 for d in days])
    for kind, cols in (("forecast", ["code", "profitForcastExpPubDate", "profitForcastExpStatDate", "profitForcastType",
                                     "profitForcastAbstract", "profitForcastChgPctUp", "profitForcastChgPctDwn"]),
                       ("express", ["code", "performanceExpPubDate", "performanceExpStatDate", "performanceExpUpdateDate",
                                    "performanceExpressTotalAsset", "performanceExpressNetAsset",
                                    "performanceExpressEPSChgPct", "performanceExpressROEWa", "performanceExpressEPSDiluted",
                                    "performanceExpressGRYOY", "performanceExpressOPYOY"])):
        for c in codes:
            # Refreshed weekly in the forward period: report segments may end weeks before the session.
            write(raw / kind / f"{c}__{days[0]}__{days[min(150, len(days) - 1)]}.csv", cols, [])
    (root / "collection-plan.json").write_text(json.dumps({"provider": "baostock", "schema": 3, "start": days[0],
                                                           "end": days[-1]}))
    (root / "request-ledger.json").write_text(json.dumps({"2026-10-13": 1}))
    (root / "extended-from.json").write_text(json.dumps({"source_plan": {"end": days[0]}}))


def test_daily_runs_from_the_raw_cache_with_a_frozen_protocol(tmp_path, monkeypatch):
    from costrank.experiment import protocol_code_hash
    from costrank.live import daily
    sessions = pd.bdate_range("2018-01-02", periods=200).strftime("%Y-%m-%d")
    days, ahead = sessions[:175], sessions[175:]
    codes = [f"sh.{600000 + i}" for i in range(30)]
    root = tmp_path / "live"
    live_cache(root, days, ahead, codes)
    cfg = small(models=("ridge",), policies=("cost",), dividend_tax=False)
    freeze = tmp_path / "freeze.json"
    from costrank.experiment import config_hash
    freeze.write_text(json.dumps({"config": cfg.dict(), "protocol_code_hash": protocol_code_hash(),
                                  "config_hash": config_hash(cfg)}, default=list))
    protocol = tmp_path / "forward.json"
    protocol.write_text(json.dumps({"forward_start": days[160], "model": "ridge", "policy": "cost", "refit_every": 20,
                                    "resolutions": "none.csv"}))
    out = tmp_path / "paper"
    first = daily(root, freeze, protocol, out, day=days[160], do_fetch=False)
    assert first["gate"]["passed"] and (out / "orders" / f"{days[161]}.csv").exists()
    with pytest.raises(MissingRecord):  # sessions are recorded in order, none skipped
        daily(root, freeze, protocol, out, day=days[-1], do_fetch=False)
    # Unattended: every unrecorded session in order, through the latest cached one.
    caught_up = daily(root, freeze, protocol, out, do_fetch=False)
    assert caught_up["day"] == days[-1] and caught_up["sessions_run"] == list(days[161:])
    second = daily(root, freeze, protocol, out, do_fetch=False)  # all recorded: the latest again, a comparison
    assert second["sessions_run"] == [days[-1]] and second["revisions"] == {}
    pending = pd.read_csv(out / "orders" / f"{ahead[0]}.csv")
    assert set(pending.columns) >= {"decided", "execute", "symbol", "held", "target"}
    # An unattended run whose gate fails sets the session's raw files aside so the next fetch gets them again.
    raw = root / "raw"
    pd.read_csv(raw / "daily" / f"{days[-1]}.csv", dtype=str).iloc[1:].assign(date=ahead[0]).to_csv(
        raw / "daily" / f"{ahead[0]}.csv", index=False)  # a listed code missing
    (raw / "adjust" / f"{ahead[0]}.csv").write_bytes((raw / "adjust" / f"{days[-1]}.csv").read_bytes())
    plan = json.loads((root / "collection-plan.json").read_text())
    (root / "collection-plan.json").write_text(json.dumps(dict(plan, end=ahead[0])))
    with pytest.raises(GateFailed):
        daily(root, freeze, protocol, out, do_fetch=False)
    assert not (raw / "daily" / f"{ahead[0]}.csv").exists() and not (out / "ledger" / "daily" / f"{ahead[0]}.csv").exists()
    assert len(list((raw / "rejected" / "daily").glob(f"{ahead[0]}@*.csv"))) == 1
    assert f"{ahead[0]} raw files set aside" in (out / "alerts.log").read_text(encoding="utf-8")
    # The last session within the reviewed rules is recorded without a decision: its orders would trade beyond them.
    good = pd.read_csv(raw / "daily" / f"{days[-1]}.csv", dtype=str)
    good.assign(date=ahead[0], preclose=good.close, open=good.close, high=good.close, low=good.close).to_csv(
        raw / "daily" / f"{ahead[0]}.csv", index=False)
    (raw / "adjust" / f"{ahead[0]}.csv").write_bytes((raw / "adjust" / f"{days[-1]}.csv").read_bytes())
    import costrank.live
    monkeypatch.setattr(costrank.live, "RULES_REVIEWED_THROUGH", ahead[0])
    last = daily(root, freeze, protocol, out, do_fetch=False)
    assert last["sessions_run"] == [ahead[0]] and last["revisions"] == {}
    assert (out / "ledger" / "daily" / f"{ahead[0]}.csv").exists() and not (out / "orders" / f"{ahead[1]}.csv").exists()
    assert "前向期最后一个交易日" in (out / "reports" / f"{ahead[0]}.md").read_text(encoding="utf-8")
    monkeypatch.undo()
    original = freeze.read_text()
    tampered = json.loads(original)
    tampered["protocol_code_hash"] = "0" * 64
    freeze.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="Protocol code changed"):
        daily(root, freeze, protocol, out, do_fetch=False)
    edited = json.loads(original)
    edited["config"]["topk"] = 7
    freeze.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="does not match its hash"):
        daily(root, freeze, protocol, out, do_fetch=False)


def test_shortfall_is_measured_against_the_previous_close_and_the_session_vwap():
    day = pd.Timestamp("2020-01-03")
    panel = pd.DataFrame({"date": [day, day], "symbol": ["a", "b"], "preclose": [10.0, 20.0],
                          "amount": [1_010_000.0, 1_980_000.0], "volume": [100_000.0, 100_000.0]})
    orders = pd.DataFrame({"date": [day, day], "symbol": ["a", "b"], "side": ["buy", "sell"], "filled_qty": [100, 100],
                           "requested_qty": [100, 100], "open_price": [10.2, 19.8], "notional": [1020.0, 1980.0]})
    s = shortfall(orders, panel, "2020-01-03")
    assert s["fills"] == 2
    assert s["vs_previous_close_bps"] == pytest.approx((200 * 1020 + 100 * 1980) / 3000)  # buy 2% up, sell 1% down
    assert s["vs_session_vwap_bps"] == pytest.approx((10.2 / 10.1 - 1) * 1e4 * 1020 / 3000)  # the sale fills at VWAP


def test_monitor_flags_breaches_fallbacks_rejections_and_ic_drift():
    cfg = small()
    day = pd.Timestamp("2020-03-02")
    replayed = {"allocations": pd.DataFrame([{"date": day, "status": "hold_fallback", "reason": "x",
                                              "ex_ante_te": cfg.te_annual * 1.5, "max_style_active": cfg.style_band + 1,
                                              "max_industry_active": 0.0}]),
                "orders": pd.DataFrame({"date": [day] * 5, "filled_qty": [0, 0, 1, 1, 1], "requested_qty": [1] * 5})}
    ic = pd.DataFrame({"date": pd.bdate_range("2020-02-01", "2020-03-02"), "rank_ic": -0.01})
    levels = [level for level, _ in monitor(cfg, replayed, ic, "2020-03-02")]
    assert levels.count("ERROR") == 2 and levels.count("WARN") == 3
    calm = {"allocations": replayed["allocations"].assign(status="optimal", ex_ante_te=0.01, max_style_active=0.1),
            "orders": replayed["orders"].assign(filled_qty=1)}
    assert monitor(cfg, calm, ic.assign(rank_ic=0.02), "2020-03-02") == []
    # Planner records with excesses: solver-accuracy noise is quiet, a widened day warns, a broken bound errs.
    noisy = calm | {"allocations": calm["allocations"].assign(max_style_active=cfg.style_band + 1e-5, te_excess=0.0,
                                                              style_excess=1e-5, industry_excess=0.0, risk_limit_scale=1.0)}
    assert monitor(cfg, noisy, ic.assign(rank_ic=0.02), "2020-03-02") == []
    widened = calm | {"allocations": noisy["allocations"].assign(risk_limit_scale=1.4, max_style_active=0.4)}
    assert [lv for lv, _ in monitor(cfg, widened, ic.assign(rank_ic=0.02), "2020-03-02")] == ["WARN"]
    broken = calm | {"allocations": noisy["allocations"].assign(style_excess=0.01)}
    assert [lv for lv, _ in monitor(cfg, broken, ic.assign(rank_ic=0.02), "2020-03-02")] == ["ERROR"]


def test_a_recorded_receivable_payment_is_not_a_revision(tmp_path):
    """The payment has no symbol and the record reads it back as missing (the rehearsal's 2026-10-16
    replay of 10-15); only a changed value is a revision."""
    day = pd.Timestamp("2026-10-15")
    events = pd.DataFrame({"date": [day, day], "symbol": ["sh.600000", ""], "kind": ["explicit_action", "receivable_paid"],
                           "value": [345.0, 345.0]})
    replayed = {"daily": pd.DataFrame({"date": [day], "nav": [1e6]}),
                "orders": pd.DataFrame({"date": [day], "symbol": ["sh.600000"], "side": ["buy"]}),
                "holdings": pd.DataFrame({"date": [day], "symbol": ["sh.600000"]}),
                "events": events, "allocations": pd.DataFrame({"date": [day], "status": ["optimal"]})}
    assert record_session(tmp_path, replayed, "2026-10-15") == {}
    assert record_session(tmp_path, replayed, "2026-10-16") == {}
    changed = replayed | {"events": events.assign(value=[345.0, 300.0])}
    assert record_session(tmp_path, changed, "2026-10-19") == {"events": ["2026-10-15"]}


def test_a_weekly_report_before_any_label_matures_has_no_nan(tmp_path):
    """The forward period's first week: labels mature only horizon + 1 sessions after a score."""
    days = pd.bdate_range("2026-10-19", "2026-10-23")
    replayed = {"daily": pd.DataFrame({"date": days, "net_return": 0.001, "turnover_two_way": [0.0, 0.95, 0.0, 0.0, 0.0]}),
                "allocations": pd.DataFrame({"date": days, "status": "optimal"})}
    protocol = Protocol(forward_start="2026-10-19", model="lightgbm", policy="cost", refit_every=126)
    path, _, _ = weekly_report(tmp_path, "2026-10-23", protocol, replayed, pd.DataFrame(), pd.Series(0.0005, index=days),
                               0.95, build=days[1])
    text = path.read_text(encoding="utf-8")
    assert "已成熟标签的平均秩 IC 尚无" in text and not re.search(r"\bnan\b", text, re.I)


def test_gates_follow_the_assembly_repairs_and_require_every_listed_code(tmp_path):
    from costrank.forward import gates
    sessions = pd.bdate_range("2018-01-02", periods=40).strftime("%Y-%m-%d")
    codes = [f"sh.{600000 + i}" for i in range(30)]
    root = tmp_path / "live"
    live_cache(root, sessions[:30], sessions[30:], codes)
    day, before = sessions[29], sessions[28]
    assert gates(root, day, before)["passed"]
    path = root / "raw" / "daily" / f"{day}.csv"
    frame = pd.read_csv(path, dtype=str)
    # A suspended row with stale volume, no amount and a flat price: the assembly repairs it.
    frame.loc[0, ["tradestatus", "amount"]] = ["0", ""]
    frame.loc[0, ["open", "high", "low", "close"]] = frame.loc[0, "preclose"]
    frame.to_csv(path, index=False)
    assert gates(root, day, before)["passed"]
    frame.iloc[1:].to_csv(path, index=False)  # a listed code missing from the session
    result = gates(root, day, before)
    assert not result["passed"] and result["examples"]["listed_codes_present"] == [codes[0]]


def test_master_and_dividend_versions_are_read_as_of_the_session(tmp_path):
    from costrank.acquire import dividend_files, master_frame
    raw = tmp_path / "raw"
    (raw / "master-asof").mkdir(parents=True)
    pd.DataFrame([["sh.600000", "a", "2010-01-01", "", "1", "1"]],
                 columns=["code", "code_name", "ipoDate", "outDate", "type", "status"]).to_csv(raw / "master.csv", index=False)
    pd.DataFrame([["sh.600000", "a", "2010-01-01", "2026-05-06", "1", "0"]],
                 columns=["code", "code_name", "ipoDate", "outDate", "type", "status"]).to_csv(
        raw / "master-asof" / "2026-05-06.csv", index=False)
    assert master_frame(tmp_path, "2026-05-05").outDate.iloc[0] == ""
    assert master_frame(tmp_path, "2026-05-06").outDate.iloc[0] == "2026-05-06"
    (raw / "dividend").mkdir()
    for name in ("sh.600000-2026.csv", "sh.600000-2026@2026-06-01.csv", "sh.600000-2026@2026-07-01.csv"):
        (raw / "dividend" / name).write_text("code\n", encoding="utf-8")
    assert dividend_files(tmp_path, "2026-06-15")["sh.600000-2026"].name == "sh.600000-2026@2026-06-01.csv"
    assert dividend_files(tmp_path)["sh.600000-2026"].name == "sh.600000-2026@2026-07-01.csv"
    assert dividend_files(tmp_path, "2026-05-01")["sh.600000-2026"].name == "sh.600000-2026.csv"


def test_refetch_sets_raw_files_aside_and_a_failed_seal_leaves_evidence_not_a_snapshot(tmp_path):
    from costrank.data import seal
    from costrank.live import set_aside
    raw = tmp_path / "raw"
    for folder in ("daily", "adjust"):
        (raw / folder).mkdir(parents=True)
        (raw / folder / "2026-11-02.csv").write_text("x\n", encoding="utf-8")
    moved = set_aside(tmp_path, "2026-11-02")
    assert len(moved) == 2 and not (raw / "daily" / "2026-11-02.csv").exists()
    assert len(list((raw / "rejected" / "daily").glob("2026-11-02@*.csv"))) == 1
    from costrank.data import synthetic
    good = synthetic(tmp_path / "good", days=30, symbols=2)
    broken = tmp_path / "snap"
    broken.mkdir()
    for path in (tmp_path / "good").iterdir():
        if path.name != "metadata.json":
            (broken / path.name).write_bytes(path.read_bytes())
    bars = pd.read_csv(broken / "bars.csv.gz")
    pd.concat([bars, bars.head(1)]).to_csv(broken / "bars.csv.gz", index=False)  # a duplicate bar
    with pytest.raises(ValueError, match="duplicate"):
        seal(broken, dict(good.metadata))
    assert not (broken / "metadata.json").exists() and len(list(broken.glob("metadata.invalid-*.json"))) == 1


def test_forward_report_reads_the_records_and_the_development_run(reference, tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from forward_report import markdown, report
    from costrank.experiment import config_hash, protocol_code_hash
    from costrank.live import daily
    sessions = pd.bdate_range("2018-01-02", periods=200).strftime("%Y-%m-%d")
    days, ahead = sessions[:175], sessions[175:]
    root = tmp_path / "live"
    live_cache(root, days, ahead, [f"sh.{600000 + i}" for i in range(30)])
    cfg = small(models=("ridge",), policies=("cost",), dividend_tax=False)
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({"config": cfg.dict(), "protocol_code_hash": protocol_code_hash(),
                                  "config_hash": config_hash(cfg), "frozen_at": "test"}, default=list))
    protocol = tmp_path / "forward.json"
    protocol.write_text(json.dumps({"forward_start": days[150], "model": "ridge", "policy": "cost", "refit_every": 20,
                                    "resolutions": "none.csv", "reference_run": str(reference)}))
    out = tmp_path / "paper"
    for day in days[150:]:
        daily(root, freeze, protocol, out, day=day, do_fetch=False)
    r = report(out, protocol, freeze, samples=99)
    assert r["first_session"] == days[150] and r["last_session"] == days[-1]
    assert r["evaluation_start"] == r["build_session"]["date"] == days[151] and r["sessions"] == len(days) - 151
    assert r["build_session"]["two_way_turnover"] > 5 * r["forward"]["mean_two_way_turnover"] and r["build_session"]["costs"] > 0
    f, dev = r["forward"], r["development"]
    assert np.isfinite(f["information_ratio"]) and f["ir_interval_95"][0] <= f["ir_interval_95"][1]
    assert dev["window_sessions"] == r["sessions"] and dev["windows"] > 0
    assert 0 <= f["percentile_among_development_windows"] <= 1 and dev["window_ir_5_95"][0] <= dev["window_ir_5_95"][1]
    assert r["shortfall"]["fill_sessions"] > 0 and r["operations"]["gate_failures"] == 0
    assert r["minimum_detectable_ratio"] == pytest.approx(2.8016 * np.sqrt(252 / r["sessions"]), rel=1e-3)
    c = r["comparisons"]["series"]
    assert c["eligible-pool benchmark"]["information_ratio"] == pytest.approx(f["information_ratio"], rel=1e-12)
    assert c["wide equal weight"]["sessions"] == c["CSI 1000 price index"]["sessions"] == r["sessions"]
    # The cache's index closes are flat: the active return against the index is the net return.
    assert c["CSI 1000 price index"]["total_return"] == 0
    assert r["comparisons"]["net_total_return"] == pytest.approx(f["total_return"], abs=1e-9)
    text = markdown(r)
    assert "模拟盘前向报告" in text and "开发期窗口中的分位" in text and "中证 1000（价格指数）" in text


def test_the_wide_comparison_is_open_to_close_on_the_build_session(world):
    """Revision two of section 10: the wide equal weight holds the previous close's names listed
    `min_listing_days`, tradable and non-ST, without the ADV floor or rank cut; open to close on the
    build session, close to close after. A snapshot without index rows leaves those columns empty."""
    import sys
    sys.path.insert(0, "scripts")
    from forward_report import comparison_returns
    cfg = small(models=("ridge",), policies=("cost",))
    days = world.calendar[230:240]
    snapshot = through(world, days[-1])
    panel = prepare(snapshot, cfg)
    comp = comparison_returns(panel, snapshot, cfg, days[1], days[1:])
    listed = snapshot.instruments.set_index("symbol").listed

    def members(day):
        x = panel[panel.date.eq(day)].set_index("symbol")
        ok = ((day - listed.reindex(x.index)).dt.days >= cfg.min_listing_days) & x.tradable.eq(1) & x.is_st.eq(0)
        return set(x.index[ok])
    build = panel[panel.date.eq(days[1]) & panel.symbol.isin(members(days[0])) & panel.open.gt(0)]
    later = panel[panel.date.eq(days[5]) & panel.symbol.isin(members(days[4]))]
    assert comp.loc[days[1], "wide equal weight"] == pytest.approx((build.close / build.open - 1).mean(), abs=1e-12)
    assert comp.loc[days[5], "wide equal weight"] == pytest.approx((later.close / later.preclose - 1).mean(), abs=1e-12)
    assert len(members(days[4])) > int(panel[panel.date.eq(days[4])].eligible.sum())
    assert comp["CSI 1000 price index"].isna().all()


def test_a_new_listing_keeps_its_five_unlimited_sessions_across_a_long_holiday(tmp_path):
    from costrank.forward import gates
    days = pd.bdate_range("2018-01-02", periods=60)
    # A seven-weekday holiday right after the listing day: its fourth session falls 15 calendar days later.
    sessions = [d for i, d in enumerate(days) if not 31 <= i <= 37]
    sessions = [d.strftime("%Y-%m-%d") for d in sessions]
    codes = [f"sh.{600000 + i}" for i in range(30)]
    root = tmp_path / "live"
    live_cache(root, sessions[:50], sessions[50:], codes)
    listing = sessions[30]
    master = pd.read_csv(root / "raw" / "master.csv", dtype=str)
    master.loc[len(master)] = ["sz.001999", "new", listing, "", "1", "1"]
    master.to_csv(root / "raw" / "master.csv", index=False)
    close = 10.0
    for n, day in enumerate(sessions[30:50], start=1):
        path = root / "raw" / "daily" / f"{day}.csv"
        frame = pd.read_csv(path, dtype=str)
        row = frame.iloc[0].copy()
        pre, close = close, round(close * (1.25 if n in (4, 6) else 1.01), 2)
        row[["code", "open", "high", "low", "close", "preclose"]] = [
            "sz.001999", f"{pre:.2f}", f"{max(pre, close) + 0.01:.2f}", f"{min(pre, close) - 0.01:.2f}", f"{close:.2f}", f"{pre:.2f}"]
        pd.concat([frame, row.to_frame().T]).to_csv(path, index=False)
    fourth, sixth = sessions[33], sessions[35]
    assert (pd.Timestamp(fourth) - pd.Timestamp(listing)).days > 10
    assert gates(root, fourth, sessions[32])["passed"]
    result = gates(root, sixth, sessions[34])
    assert not result["passed"] and result["examples"]["price_band"] == ["sz.001999"]


def test_a_prefix_of_a_grown_cache_does_not_need_a_later_dividend_version(tmp_path):
    from costrank.snapshot import assemble
    sessions = pd.bdate_range("2018-01-02", periods=40).strftime("%Y-%m-%d")
    codes = [f"sh.{600000 + i}" for i in range(30)]
    root = tmp_path / "live"
    live_cache(root, sessions[:30], sessions[30:], codes)
    last = sessions[29]
    # An ex-date on the last cached session, with the dividend version fetched that day.
    pd.DataFrame([[codes[0], last, "1.0", "1.0", "1.0"]],
                 columns=["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor", "adjustFactor"]).to_csv(
        root / "raw" / "adjust" / f"{last}.csv", index=False)
    (root / "raw" / "dividend").mkdir(exist_ok=True)
    pd.DataFrame(columns=["code"]).to_csv(root / "raw" / "dividend" / f"{codes[0]}-2018@{last}.csv", index=False)
    bundle = assemble(root, sessions[0], sessions[28], target=tmp_path / "snap", forward=True)
    assert bundle.calendar[-1] == pd.Timestamp(sessions[28])


def test_the_evaluation_starts_at_the_first_execution_session(world):
    """The book is cash on the forward start, so that session has no benchmark; it is bought at the
    next open, so that session's benchmark is the pool's open-to-close return; close to close after."""
    cfg = small(models=("ridge",), policies=("cost",))
    days = world.calendar[230:240]
    snapshot = through(world, days[-1])
    panel = prepare(snapshot, cfg)
    protocol = Protocol(forward_start=days[0].strftime("%Y-%m-%d"), model="ridge", policy="cost", refit_every=20)
    bench = evaluation_benchmark(panel, snapshot, protocol, days[-1].strftime("%Y-%m-%d"))
    assert first_execution(snapshot.calendar, protocol) == days[1] and list(bench.index) == list(days[1:])
    close = benchmark_returns(panel, days[0], days[-1], snapshot).benchmark_return
    assert np.allclose(bench.loc[days[2]:], close.loc[days[2]:], rtol=0, atol=0)
    p = panel.sort_values(["symbol", "date"])
    pool = p[p.groupby("symbol").eligible.shift(fill_value=False).astype(bool) & p.date.eq(days[1])]
    assert len(pool) > 0 and bench.loc[days[1]] == pytest.approx((pool.close / pool.open - 1).mean(), abs=1e-15)
    assert abs(bench.loc[days[1]] - close.loc[days[1]]) > 1e-6  # the overnight move is left out


def test_evaluation_window_restates_a_run(reference):
    import sys
    sys.path.insert(0, "scripts")
    from evaluation_window import window
    r = window(reference, samples=99)
    invested = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))["config"]["invested"]
    m = r["models"]["ridge"]
    assert m["window"]["sessions"] == m["without_first"]["sessions"] == m["all_sessions"]["sessions"] - 1
    assert m["all_sessions"]["active_first"] == pytest.approx(-invested * r["first_session"]["benchmark_close_to_close"])
    build = r["build_session"]
    assert m["window"]["active_build"] - m["all_sessions"]["active_build"] == pytest.approx(
        invested * (build["benchmark_close_to_close"] - build["benchmark_open_to_close"]))


def test_a_failed_fetch_is_tried_again_until_six_and_waits_long_for_a_slow_login(tmp_path, monkeypatch):
    import socket
    from datetime import datetime, timedelta
    import costrank.acquire
    import costrank.live as live
    root, out = tmp_path / "live", tmp_path / "paper"
    root.mkdir()
    out.mkdir()
    (root / "collection-plan.json").write_text(json.dumps({"provider": "baostock", "schema": 3, "start": "2016-01-01",
                                                            "end": "2026-10-16"}))
    (root / "extended-from.json").write_text(json.dumps({"source_plan": {"end": "2026-10-14"}}))
    slow = RuntimeError("BaoStock login failed: 10002007: 网络接收错误。")
    outcomes, waits = [slow, slow, {"stopped": None}], []

    def acquire(*args, **kwargs):  # sets its own timeout first, as acquire() does
        before = socket.getdefaulttimeout()
        costrank.acquire.socket.setdefaulttimeout(60)
        waits.append(socket.getdefaulttimeout())
        costrank.acquire.socket.setdefaulttimeout(before)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    monkeypatch.setattr(costrank.acquire, "acquire", acquire)
    now, slept = [datetime(2026, 10, 19, 20, 30, tzinfo=live.BEIJING)], []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += timedelta(seconds=seconds)
    assert live.fetch_until(root, "2026-10-19", out, clock=lambda: now[0], sleep=sleep) == {"stopped": None}
    assert waits == [live.FETCH_TIMEOUT] * 3 and costrank.acquire.socket is socket
    assert slept == [live.FETCH_RETRY_EVERY] * 2
    alerts = (out / "alerts.log").read_text(encoding="utf-8").splitlines()
    assert len(alerts) == 1 and " WARN 2026-10-19 fetch failed, trying again" in alerts[0]
    # Shortly before six in the morning a failure is raised, not waited out.
    outcomes[:] = [slow]
    now[0] = datetime(2026, 10, 20, 5, 50, tzinfo=live.BEIJING)
    with pytest.raises(RuntimeError, match="login failed"):
        live.fetch_until(root, "2026-10-19", out, clock=lambda: now[0], sleep=sleep)
    assert len(slept) == 2 and costrank.acquire.socket is socket
