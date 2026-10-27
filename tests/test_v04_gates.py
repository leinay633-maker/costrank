"""Gate 1 (RESEARCH section 9): the per-session endpoint check passes only when it is complete."""
import json
import subprocess
import sys
import pandas as pd
import pytest

DAYS = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-01-02", periods=30)]
FIELDS = ["open", "high", "low", "close", "preclose", "volume", "amount", "tradestatus", "isST"]
MAIN = ["sh.600000", "sz.000001", "sz.000002"]  # sz.000002 is delisted on DAYS[25]


def rows(day, codes):
    return [[day, c, "10.0000", "10.5000", "9.8000", "10.2000", "10.0000", "1000", "10200.0000", "1", "0"] for c in codes]


def write(path, columns, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data, columns=columns).to_csv(path, index=False)


def caches(tmp_path, *, drop=None, late=False, mismatch=False):
    acq, dev4 = tmp_path / "acq", tmp_path / "dev4"
    acq.mkdir(parents=True)
    (acq / "collection-plan.json").write_text(json.dumps({"provider": "baostock", "schema": 3, "start": DAYS[0],
                                                          "end": DAYS[-1]}))
    write(acq / "raw" / "calendar.csv", ["calendar_date", "is_trading_day"], [[d, "1"] for d in DAYS])
    reference = []
    for i, d in enumerate(DAYS):
        main = [c for c in MAIN if not (c == "sz.000002" and i >= 25)]
        reference += rows(d, main)
        if d == drop:
            continue
        data = rows(d, main + ["sz.300001"])
        if mismatch and i == 3:
            data[0][5] = "10.2100"
        write(acq / "raw" / "daily" / f"{d}.csv", ["date", "code"] + FIELDS, data)
    write(dev4 / "raw" / "bars-0.csv", ["date", "code"] + FIELDS, [r for r in reference if r[0] != drop])
    delisted = [["sz.000002", DAYS[25]]] + ([["sh.600009", "2021-06-30"]] if late else [])
    write(tmp_path / "delisted.csv", ["code", "outDate"], delisted)
    write(tmp_path / "chinext.csv", ["date", "code"] + FIELDS, rows(DAYS[0], ["sz.300001"]) + rows(DAYS[1], ["sz.300001"]))
    out = tmp_path / "gate1.json"
    subprocess.run([sys.executable, "scripts/check_daily_endpoint.py", str(acq), str(dev4), "--delisted",
                    str(tmp_path / "delisted.csv"), "--chinext", "sz.300001", "--chinext-file", str(tmp_path / "chinext.csv"),
                    "--output", str(out)], check=True, capture_output=True)
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case,checks,complete", [
    ({}, True, True),
    ({"drop": DAYS[7]}, True, False),       # a planned session is not cached yet
    ({"late": True}, True, False),          # a delisting after the cached sessions cannot be checked
    ({"mismatch": True}, False, True),      # one close differs from the per-symbol cache
])
def test_gate1_passes_only_complete_and_equal(tmp_path, case, checks, complete):
    report = caches(tmp_path, **case)
    assert report["checks_passed"] is checks and report["complete"] is complete
    assert report["passed"] is (checks and complete)
    if case.get("late"):
        assert report["delisted_presence"]["unchecked"] == ["sh.600009 2021-06-30"]


def test_feasibility_report_refuses_a_directory_without_portfolios(tmp_path):
    sys.path.insert(0, "scripts")
    from feasibility_report import summarize
    with pytest.raises(ValueError, match="No mvo/mvo_rank portfolio"):
        summarize(tmp_path)  # e.g. the runs root instead of a run: no evidence is not a pass


def test_gate2_counts_only_unexplained_steps_before_publication(tmp_path):
    sys.path.insert(0, "scripts")
    from check_valuation_timing import evaluate
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-12-01", "2021-06-30")]
    stat, pub, step, ex_day, after = "2020-12-31", "2021-04-28", "2021-02-10", "2021-03-01", "2021-04-29"
    pb_step = {"sh.600001": step, "sh.600002": step, "sh.600003": ex_day, "sh.600004": after}
    for d in days:
        rows = []
        for code, when in pb_step.items():
            pre = "9.50" if (code == "sh.600003" and d == ex_day) else "10.00"
            rows.append([d, code, "10.00", pre, "1.1" if d >= when else "1.0", "12" if d >= after else "10"])
        write(tmp_path / "raw" / "daily" / f"{d}.csv", ["date", "code", "close", "preclose", "pbMRQ", "peTTM"], rows)
    write(tmp_path / "raw" / "express" / "sh.600002__2016-01-01__2024-12-31.csv",
          ["code", "performanceExpPubDate", "performanceExpStatDate"], [["sh.600002", "2021-02-05", stat]])
    pairs = [(code, 2020, pub, stat) for code in pb_step]
    report = evaluate(tmp_path, days, pairs, 4)
    leaks = {r["code"]: r["leak"] for r in report["details"]}
    assert leaks == {"sh.600001": True, "sh.600002": False, "sh.600003": False, "sh.600004": False}
    assert report["informative"] == {"bvps": 4, "eps_ttm": 4} and not report["passed"]
    clean = evaluate(tmp_path, days, pairs[1:], 3)
    assert clean["passed"] and clean["leaks"] == 0
