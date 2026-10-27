"""v0.4 schema-3 snapshots: offline assembly from the raw cache, hybrid corporate actions,
content-based prefix hashes and the side tables."""
import json
import numpy as np
import pandas as pd
import pytest
from costrank.acquire import INDICES
from costrank.data import frame_hash, read_bundle
from costrank.providers import repair_baostock_rows
from costrank.snapshot import assemble, repair_rows

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-02", periods=30)]
DAILY = ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount", "adjustflag", "turn",
         "tradestatus", "pctChg", "peTTM", "pbMRQ", "psTTM", "pcfNcfTTM", "isST"]
# Two names with an event on day 10: sh.600000 pays 0.5 and reconciles; sz.300001 is recorded as
# 0.5 cash but its preclose shows a 10-for-10 split, so it falls back to the proxy. sh.688001 has
# an unexplained preclose jump on day 20. sz.000002 is the old code of sz.001872 (same history),
# removed on day 25 with a suspended placeholder that day.
MASTER = [["sh.600000", "a", "2010-01-01", "", "1", "1"], ["sz.300001", "b", "2012-01-01", "", "1", "1"],
          ["sh.688001", "c", "2019-07-22", "", "1", "1"], ["sz.000002", "d", "1991-01-29", DAYS[25], "1", "0"],
          ["sz.001872", "d", "1991-01-29", "", "1", "1"], ["sh.689009", "e", "2020-10-29", "", "1", "1"]]


def price(code, i):
    base = {"sh.600000": 10.0, "sz.300001": 20.0, "sh.688001": 50.0, "sz.000002": 8.0, "sz.001872": 8.0}[code]
    return base + 0.01 * i


def rows_for(i, session):
    out = []
    for code in ("sh.600000", "sz.300001", "sh.688001", "sz.000002", "sz.001872", "sh.689009"):
        if code == "sh.689009":
            out.append([session, code] + ["9"] * 5 + ["100", "900", "3", "1", "1", "0", "1", "1", "1", "1", "0"])
            continue
        if code == "sz.000002" and i > 25:
            continue
        close = price(code, i)
        pre = price(code, i - 1) if i else close
        if code == "sh.600000" and i == 10:
            pre = round(price(code, 9) - 0.5, 2)
        if code == "sz.300001" and i == 10:
            pre = round(price(code, 9) / 2, 2)
        if code == "sh.688001" and i == 20:
            pre = round(price(code, 19) * 0.9, 2)
        if code == "sh.600000" and i == 6:  # day 5 did not trade and closed at day 4's price
            pre = price(code, 4)
        status, volume, amount = "1", "1000", f"{1000 * close:.2f}"
        if code == "sz.000002" and i == 25:
            status, volume, amount, close, pre = "0", "", "", price(code, 24), price(code, 24)
        if code == "sh.600000" and i == 5:  # provider artefact: active flag, no trades, flat price
            status, volume, amount, close, pre = "1", "0", "0", price(code, 4), price(code, 4)
        p = f"{close:.2f}"
        out.append([session, code, p, p, p, p, f"{pre:.2f}", volume, amount, "3", "1.5", status, "0",
                    "12.5", "1.2", "2.0", "8.0", "0"])
    return out


def write(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def raw_cache(root, *, drop_daily=None):
    raw = root / "raw"
    raw.mkdir(parents=True)
    (root / "collection-plan.json").write_text(json.dumps(
        {"provider": "baostock", "schema": 3, "start": DAYS[0], "end": DAYS[-1]}))
    (root / "request-ledger.json").write_text(json.dumps({"2026-10-13": 100}))
    write(raw / "calendar.csv", ["calendar_date", "is_trading_day"], [[d, "1"] for d in DAYS])
    write(raw / "master.csv", ["code", "code_name", "ipoDate", "outDate", "type", "status"], MASTER)
    for i, d in enumerate(DAYS):
        if d != drop_daily:
            write(raw / "daily" / f"{d}.csv", DAILY, rows_for(i, d))
        events = [[c, d, "1", "1", "1"] for c in ("sh.600000", "sz.300001")] if i == 10 else []
        write(raw / "adjust" / f"{d}.csv", ["code", "dividOperateDate", "foreAdjustFactor", "backAdjustFactor",
                                            "adjustFactor"], events)
    div = ["code", "dividPreNoticeDate", "dividAgmPumDate", "dividPlanAnnounceDate", "dividPlanDate", "dividRegistDate",
           "dividOperateDate", "dividPayDate", "dividStockMarketDate", "dividCashPsBeforeTax", "dividCashPsAfterTax",
           "dividStocksPs", "dividCashStock", "dividReserveToStockPs"]
    for code in ("sh.600000", "sz.300001"):
        write(raw / "dividend" / f"{code}-2020.csv", div,
              [[code, "", "", "", "", DAYS[9], DAYS[10], DAYS[12], "", "0.5", "0.45", "0", "", ""]])
    for month_first in ("2020-01-02", "2020-02-03"):
        write(raw / "industry" / f"{month_first}.csv", ["updateDate", "code", "code_name", "industry", "industryClassification"],
              [["2019-12-30", c, "x", "C39计算机", "证监会行业分类"] for c, *_ in MASTER])
        for index in ("hs300", "zz500"):
            write(raw / index / f"{month_first}.csv", ["updateDate", "code", "code_name"], [["2019-12-30", "sh.600000", "a"]])
    for code in INDICES:
        write(raw / "index" / f"{code}__{DAYS[0]}__{DAYS[-1]}.csv", ["date", "code", "open", "high", "low", "close", "preclose", "volume", "amount"],
              [[d, code, "1000", "1010", "990", "1005", "1000", "1", "1"] for d in DAYS])
    for code, *_ in MASTER[:5]:
        write(raw / "forecast" / f"{code}__{DAYS[0]}__{DAYS[-1]}.csv", ["code", "profitForcastExpPubDate", "profitForcastExpStatDate",
              "profitForcastType", "profitForcastAbstract", "profitForcastChgPctUp", "profitForcastChgPctDwn"],
              [[code, DAYS[3], "2019-12-31", "预增", "x", "50", "30"]])
        write(raw / "express" / f"{code}__{DAYS[0]}__{DAYS[-1]}.csv", ["code", "performanceExpPubDate", "performanceExpStatDate",
              "performanceExpUpdateDate", "performanceExpressTotalAsset", "performanceExpressNetAsset",
              "performanceExpressEPSChgPct", "performanceExpressROEWa", "performanceExpressEPSDiluted",
              "performanceExpressGRYOY", "performanceExpressOPYOY"],
              [[code, DAYS[4], "2019-12-31", "", "100", "50", "0.1", "10", "1", "0.2", "0.3"]])
        bars = [[d, f"{d.replace('-', '')}{t}00000", code, "10", "10.2", "9.9", f"{10 + k * 0.01:.2f}", "125", "1250"]
                for d in DAYS for k, t in enumerate(["1000", "1030", "1130", "1330", "1400", "1430", "1500", "1100"])
                if not (code == "sh.600000" and d == DAYS[5])]
        write(raw / "min30" / f"{code}-2020.csv", ["date", "time", "code", "open", "high", "low", "close", "volume",
                                                   "amount"], bars)
    return root


def test_assembly_types_tables_and_reconciles_events(tmp_path):
    bundle = assemble(raw_cache(tmp_path / "snap"), DAYS[0], DAYS[-1])
    meta = bundle.metadata
    assert meta["schema_version"] == 3 and meta["corporate_actions"] == "hybrid_explicit_proxy"
    assert sorted(bundle.instruments.symbol) == ["sh.600000", "sh.688001", "sz.001872", "sz.300001"]
    assert [(x["symbol"], x["alias_of"]) for x in meta["dropped_aliases"]] == [("sz.000002", "sz.001872")]
    assert [(x["rule"], x["symbol"]) for x in meta["provider_repairs"]] == [("active_flag_without_trades", "sh.600000")]
    a = bundle.actions
    assert a[["symbol", "cash_per_share", "split_ratio"]].values.tolist() == [["sh.600000", 0.5, 1.0]]
    assert a.pay_date.iloc[0] == pd.Timestamp(DAYS[12])
    assert a.share_listing_date.iloc[0] == a.date.iloc[0]  # a cash dividend adds no shares to list
    rec = bundle.tables["reconciliation"].set_index("symbol").status.to_dict()
    assert rec == {"sh.600000": "explicit", "sz.300001": "proxy_mismatch", "sh.688001": "proxy_no_event"}
    assert meta["corporate_action_reconciliation"] == {"explicit": 1, "proxy_mismatch": 1, "proxy_no_event": 1}
    attrs = bundle.tables["attributes"]
    assert len(attrs) == len(bundle.bars) and attrs.pe_ttm.eq(12.5).all()
    intraday = bundle.tables["intraday"]
    first = intraday[(intraday.symbol == "sh.600000") & (intraday.date == DAYS[0])].iloc[0]
    assert first.bars == 8 and first.first_close == 10.0 and first.last_close == 10.06 and first.day_volume == 1000
    # The 15:00 bar is the last one even when the provider lists bars out of order.
    assert meta["intraday"]["volume_mismatch_rows"] == 0
    assert set(bundle.tables["constituents"]["index"]) == {"hs300", "zz500"}
    assert bundle.tables["forecasts"].pub_date.eq(pd.Timestamp(DAYS[3])).all()
    assert bundle.tables["express"].update_date.isna().all()
    assert len(bundle.tables["indices"]) == 4 * len(DAYS)
    again = read_bundle(tmp_path / "snap")
    assert again.prefix_hash(DAYS[20]) == bundle.prefix_hash(DAYS[20])


def test_prefix_hash_binds_history_and_ignores_later_rows(tmp_path):
    base = assemble(raw_cache(tmp_path / "a"), DAYS[0], DAYS[-1])
    other = raw_cache(tmp_path / "b")
    late = pd.read_csv(other / "raw/daily" / f"{DAYS[-1]}.csv", dtype=str)
    late.loc[late.code == "sh.600000", "amount"] = "999999"
    late.to_csv(other / "raw/daily" / f"{DAYS[-1]}.csv", index=False)
    changed = assemble(other, DAYS[0], DAYS[-1])
    assert changed.prefix_hash(DAYS[-1]) == base.prefix_hash(DAYS[-1])
    assert changed.prefix_hash("2020-03-01") != base.prefix_hash("2020-03-01")


def test_prefix_hash_ignores_records_of_securities_listed_later(tmp_path):
    from dataclasses import replace as swap
    base = assemble(raw_cache(tmp_path / "a"), DAYS[0], DAYS[-1])
    cut = DAYS[10]
    # A security listed after the cutoff whose prospectus forecast is dated before it: an extended
    # snapshot carries the row, the development snapshot never had the code.
    ipo = base.instruments.iloc[[0]].assign(symbol="sz.301999", listed=pd.Timestamp(DAYS[-1]), delisted=pd.NaT)
    forecast = base.tables["forecasts"].iloc[[0]].assign(symbol="sz.301999", pub_date=pd.Timestamp(DAYS[2]))
    later = swap(base, instruments=pd.concat([base.instruments, ipo], ignore_index=True),
                 tables=dict(base.tables, forecasts=pd.concat([base.tables["forecasts"], forecast], ignore_index=True)))
    assert later.prefix_hash(cut) == base.prefix_hash(cut)
    # A listed security's record before the cutoff stays bound.
    revised = base.tables["forecasts"].copy()
    revised.loc[revised.index[0], "chg_up"] = 999.0
    assert revised.pub_date.iloc[0] < pd.Timestamp(cut)
    assert swap(base, tables=dict(base.tables, forecasts=revised)).prefix_hash(cut) != base.prefix_hash(cut)


def test_dated_subset_assembles_the_same_prefix(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from subset_raw import subset
    full = assemble(raw_cache(tmp_path / "full"), DAYS[0], DAYS[-1])
    end = DAYS[19]
    counts = subset(tmp_path / "full", tmp_path / "part", end)
    assert counts["daily"] == counts["adjust"] == 20 and counts["industry"] == 1 and counts["dividend"] == 2
    part = assemble(tmp_path / "part", DAYS[0], end)
    assert part.metadata["end"] == end and part.bars.date.max() == pd.Timestamp(end)
    head = lambda f: f[f.date <= end].reset_index(drop=True)
    pd.testing.assert_frame_equal(part.bars, head(full.bars))
    pd.testing.assert_frame_equal(part.actions, head(full.actions))
    with pytest.raises(FileExistsError):
        subset(tmp_path / "full", tmp_path / "part", end)
    with pytest.raises(ValueError, match="reaches the subset end"):
        subset(tmp_path / "full", tmp_path / "late", "2099-12-31")
    # The whole plan can be linked too (supplement 11: the snapshot without minute bars).
    assert subset(tmp_path / "full", tmp_path / "whole", DAYS[-1])["daily"] == len(DAYS)


def test_missing_market_file_blocks_assembly(tmp_path):
    with pytest.raises(ValueError, match="acquisition files missing"):
        assemble(raw_cache(tmp_path / "snap", drop_daily=DAYS[7]), DAYS[0], DAYS[-1])


def test_typed_repairs_match_the_main_board_collector():
    text = pd.DataFrame({"symbol": ["x", "y", "z", "w"], "date": ["2020-01-02"] * 4,
                         "open": ["10", "10", "10.5", "10"], "high": ["10", "10", "10.8", "10"],
                         "low": ["10", "10", "10.1", "10"], "close": ["10", "10", "10.6", "10"],
                         "preclose": ["10"] * 4, "volume": ["", "500", "300", "100"], "amount": ["", "0", "3150", "1000"],
                         "tradable": ["1", "0", "0", "1"], "is_st": ["0"] * 4})
    fixed, records = repair_baostock_rows(text)
    typed = pd.DataFrame({"code": text.symbol, "date": pd.to_datetime(text.date),
                          **{c: pd.to_numeric(text[c].replace("", np.nan)) for c in
                             ("open", "high", "low", "close", "preclose", "volume", "amount")},
                          "tradable": text.tradable.astype(float)})
    repaired, typed_records = repair_rows(typed)
    assert [r["rule"] for r in records] == [r["rule"] for r in typed_records]
    assert repaired.tradable.tolist() == pd.to_numeric(fixed.tradable).tolist()
    assert repaired.volume.fillna(0).tolist() == pd.to_numeric(fixed.volume.replace("", "0")).tolist()


def test_frame_hash_is_storage_independent(tmp_path):
    f = pd.DataFrame({"date": pd.to_datetime(["2020-01-02", "2020-01-03"]), "symbol": ["a", None],
                      "x": [1.5, np.nan], "n": [1, 2]})
    f.to_parquet(tmp_path / "f.parquet", index=False)
    assert frame_hash(pd.read_parquet(tmp_path / "f.parquet")) == frame_hash(f)
    assert frame_hash(f.assign(x=[1.5, 0.0])) != frame_hash(f)


def test_snapshot_audit_reports_schema3_sections(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from audit_snapshot import audit_snapshot
    from costrank.config import Config
    root = raw_cache(tmp_path / "snap")
    assemble(root, DAYS[0], DAYS[-1])
    report = audit_snapshot(root, Config(dev_start=DAYS[0], dev_end=DAYS[-1], holdout_start="2021-01-01",
                                         holdout_end="2021-12-31"), dividend_symbols=0)
    s3 = report["schema3"]
    assert s3["boards"]["star"]["instruments"] == 1 and s3["boards"]["chinext"]["instruments"] == 1
    assert s3["corporate_actions"]["reconciliation"] == {"explicit": 1, "proxy_mismatch": 1, "proxy_no_event": 1}
    assert s3["intraday"]["sessions_not_8_bars"] == 0 and s3["industries"]["snapshots"] == 2
    assert report["provider_universe"]["master_mainboard"] == 5  # the studied boards; the CDR is excluded


def test_report_segments_read_each_record_once_and_never_rewrite_history(tmp_path):
    import sys
    sys.path.insert(0, "scripts")
    from subset_raw import subset
    root = raw_cache(tmp_path / "full")
    express = root / "raw" / "express"
    cols = ["code", "performanceExpPubDate", "performanceExpStatDate", "performanceExpUpdateDate",
            "performanceExpressTotalAsset", "performanceExpressNetAsset", "performanceExpressEPSChgPct",
            "performanceExpressROEWa", "performanceExpressEPSDiluted", "performanceExpressGRYOY", "performanceExpressOPYOY"]
    (express / f"sh.600000__{DAYS[0]}__{DAYS[-1]}.csv").unlink()
    cut = DAYS[14]
    after = (pd.Timestamp(cut) + pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d")
    first = [["sh.600000", DAYS[3], "2019-12-31", "", "1", "1", "0.1", "1", "1", "1", "1"]]
    revised = ["sh.600000", DAYS[5], "2019-09-30", DAYS[20], "2", "2", "0.9", "2", "2", "2", "2"]
    # A fetch made after the revision returns the revised record in both ranges.
    write(express / f"sh.600000__{DAYS[0]}__{cut}.csv", cols, first + [revised])
    write(express / f"sh.600000__{after}__{DAYS[-1]}.csv", cols, [revised, ["sh.600000", DAYS[16], "2020-03-31", "", "3", "3",
                                                                          "0.3", "3", "3", "3", "3"]])
    index = root / "raw" / "index" / f"{INDICES[0]}__{DAYS[0]}__{DAYS[-1]}.csv"
    bars = pd.read_csv(index, dtype=str)
    index.unlink()
    bars[bars.date <= cut].to_csv(index.with_name(f"{INDICES[0]}__{DAYS[0]}__{cut}.csv"), index=False)
    bars[bars.date > cut].to_csv(index.with_name(f"{INDICES[0]}__{after}__{DAYS[-1]}.csv"), index=False)
    subset(root, tmp_path / "early", cut)
    full = assemble(root, DAYS[0], DAYS[-1])
    e = full.tables["express"]
    mine = e[e.symbol == "sh.600000"].sort_values("pub_date")
    assert mine.eps_chg_pct.tolist() == [0.1, 0.9, 0.3]
    assert mine.available_date.dt.strftime("%Y-%m-%d").tolist() == [DAYS[3], DAYS[20], DAYS[16]]
    assert (full.tables["indices"]["index"] == INDICES[0]).sum() == len(DAYS)
    early = assemble(tmp_path / "early", DAYS[0], cut)
    e = early.tables["express"]
    assert e[e.symbol == "sh.600000"].eps_chg_pct.tolist() == [0.1]  # the revision is not visible yet
    from costrank.snapshot import covering
    with pytest.raises(ValueError, match="Overlapping cache segments"):
        covering([("2020-01-01", "2020-01-20", "a"), ("2020-01-15", "2020-02-12", "b")], "2020-01-02", "2020-02-12")
    assert covering([("2020-01-01", "2020-01-20", "a")], "2020-01-02", "2020-02-12") is None  # not covered
    assert covering([("2020-01-01", "2020-01-20", "a"), ("2020-01-22", "2020-02-12", "b")], "2020-01-02", "2020-02-12") is None


def test_a_snapshot_without_minutes_serves_open_fills_and_refuses_vwap_runs(tmp_path):
    from costrank.config import Config
    from costrank.experiment import run
    root = raw_cache(tmp_path / "open")
    for path in (root / "raw" / "min30").glob("*.csv"):
        path.unlink()  # still being acquired: the open-fill decision path does not wait for them
    with pytest.raises(ValueError, match="per-code acquisition files missing"):
        assemble(root, DAYS[0], DAYS[-1], target=tmp_path / "strict")
    bundle = assemble(root, DAYS[0], DAYS[-1], target=tmp_path / "open-fills", minutes=False)
    assert bundle.metadata["intraday"]["omitted"] and bundle.tables["intraday"].empty
    assert bundle.metadata["partial_assembly"] is False and len(bundle.actions) == 1
    cfg = Config(dev_start=DAYS[5], dev_end=DAYS[-1], holdout_start="2021-01-01", holdout_end="2021-12-31",
                 universe_size=3, topk=2, min_adv=0, fill_price="vwap30", participation_basis="first30")
    with pytest.raises(RuntimeError, match="minute bars"):
        run(cfg, tmp_path / "open-fills", tmp_path / "runs", allow_proxy=True)


def test_split_shares_take_the_providers_listing_date(tmp_path):
    from costrank.snapshot import corporate_actions
    days = pd.to_datetime(["2019-06-14", "2019-06-17", "2019-06-18", "2019-06-19"])
    # sh.600338 10送4派0.4 (ex 06-18, new shares list 06-19); sz.000001 cash only; sz.000002 10转5 with
    # the listing date missing at the provider.
    bars = pd.DataFrame({"date": list(days) * 3, "symbol": ["sh.600338"] * 4 + ["sz.000001"] * 4 + ["sz.000002"] * 4,
                         "close": [14.0, 14.0, 10.0, 10.0] + [10.0, 10.0, 9.5, 9.5] + [9.0, 9.0, 6.0, 6.0],
                         "preclose": [14.0, 14.0, 9.97, 10.0] + [10.0, 10.0, 9.5, 9.5] + [9.0, 9.0, 6.0, 6.0]})
    div = ["code", "dividPreNoticeDate", "dividAgmPumDate", "dividPlanAnnounceDate", "dividPlanDate", "dividRegistDate",
           "dividOperateDate", "dividPayDate", "dividStockMarketDate", "dividCashPsBeforeTax", "dividCashPsAfterTax",
           "dividStocksPs", "dividCashStock", "dividReserveToStockPs"]
    for code, listing, cash, bonus, conversion in (("sh.600338", "2019-06-19", "0.04", "0.4", ""),
                                                   ("sz.000001", "", "0.5", "0", ""), ("sz.000002", "", "", "", "0.5")):
        write(tmp_path / "raw" / "dividend" / f"{code}-2019.csv", div,
              [[code, "", "", "", "", "2019-06-17", "2019-06-18", "2019-06-18", listing, cash, "", bonus, "", conversion]])
    actions, _ = corporate_actions(tmp_path, bars, "2019-06-14", "2019-06-19")
    a = actions.set_index("symbol")
    assert a.loc["sh.600338", ["split_ratio", "taxable_stock_per_share"]].tolist() == pytest.approx([1.4, 0.4])
    assert a.share_listing_date.dt.strftime("%Y-%m-%d").to_dict() == {
        "sh.600338": "2019-06-19", "sz.000001": "2019-06-18", "sz.000002": "2019-06-18"}


def test_latest_dividend_version_wins_and_assembly_can_target_new_directories(tmp_path):
    root = raw_cache(tmp_path / "live")
    div = ["code", "dividPreNoticeDate", "dividAgmPumDate", "dividPlanAnnounceDate", "dividPlanDate", "dividRegistDate",
           "dividOperateDate", "dividPayDate", "dividStockMarketDate", "dividCashPsBeforeTax", "dividCashPsAfterTax",
           "dividStocksPs", "dividCashStock", "dividReserveToStockPs"]
    # A refetch on the ex-date carries the same event (pay date now known): read once, from that version.
    write(root / "raw" / "dividend" / f"sh.600000-2020@{DAYS[10]}.csv", div,
          [["sh.600000", "", "", "", "", DAYS[9], DAYS[10], DAYS[13], "", "0.5", "0.45", "0", "", ""]])
    first = assemble(root, DAYS[0], DAYS[-1], target=tmp_path / "day1")
    again = assemble(root, DAYS[0], DAYS[-1], target=tmp_path / "day2")
    assert not (root / "metadata.json").exists()
    a = first.actions[first.actions.symbol == "sh.600000"]
    assert len(a) == 1 and a.cash_per_share.iloc[0] == 0.5 and a.pay_date.iloc[0] == pd.Timestamp(DAYS[13])
    assert first.prefix_hash(DAYS[-1]) == again.prefix_hash(DAYS[-1])


def test_partial_preview_assembles_without_dividend_details_and_is_refused_beyond_2019(tmp_path):
    from costrank.config import Config
    from costrank.experiment import run
    root = raw_cache(tmp_path / "preview")
    for path in (root / "raw" / "dividend").glob("*.csv"):
        path.unlink()
    with pytest.raises(ValueError, match="per-code acquisition files missing"):
        assemble(root, DAYS[0], DAYS[-1], target=tmp_path / "strict")
    bundle = assemble(root, DAYS[0], DAYS[-1], target=tmp_path / "partial", partial=True)
    assert bundle.metadata["partial_assembly"] is True
    assert bundle.actions.empty  # the event without details is priced by the proxy
    rec = bundle.tables["reconciliation"].set_index("symbol").status.to_dict()
    assert rec["sh.600000"] == "proxy_no_event"
    cfg = Config(dev_start=DAYS[5], dev_end=DAYS[-1], holdout_start="2021-01-01", holdout_end="2021-12-31",
                 universe_size=3, topk=2, min_adv=0)
    with pytest.raises(RuntimeError, match="partial preview"):
        run(cfg, tmp_path / "partial", tmp_path / "runs", allow_proxy=True)
