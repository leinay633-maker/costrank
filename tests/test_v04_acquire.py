"""v0.4 acquisition: board codes, request metering per page, daily budget and exact resume."""
import json
import sys
from types import SimpleNamespace
import pandas as pd
import pytest
from costrank import acquire as acq
from costrank.data import board, mainboard


@pytest.mark.parametrize("code,expected", [("sh.600000", "main"), ("sz.000001", "main"), ("sz.300750", "chinext"),
    ("sz.301001", "chinext"), ("sz.302132", "chinext"), ("sh.688981", "star"), ("sh.689009", None),
    ("sz.200002", None), ("sh.900901", None), ("bj.830799", None), ("sh.000300", None)])
def test_board_of_code(code, expected):
    assert board(code) == expected
    assert mainboard(code) == (expected == "main")


class Page:
    """BaoStock-like result: one message per page; a full page triggers a request for the next."""

    def __init__(self, sock, fields, rows, per=100):
        self.sock, self.fields, self.rows, self.per = sock, fields.split(","), rows, per
        self.error_code, self.error_msg, self.offset, self.i = "0", "ok", 0, 0
        self.page = self.fetch()

    def fetch(self):
        self.sock.send_msg("page")
        page = self.rows[self.offset:self.offset + self.per]
        self.offset += self.per
        return page

    def next(self):
        if self.i < len(self.page):
            return True
        if len(self.page) < self.per:
            return False
        self.page, self.i = self.fetch(), 0
        return bool(self.page)

    def get_row_data(self):
        self.i += 1
        return self.page[self.i - 1]


DAYS = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07", "2020-01-08", "2020-01-09"]
DAILY = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"


def install(monkeypatch, *, daily_rows=None, adjust_rows=None):
    sock = SimpleNamespace(send_msg=lambda msg: "")
    def login():
        sock.send_msg("login")
        return SimpleNamespace(error_code="0", error_msg="ok")
    def daily(date):
        rows = daily_rows(date) if daily_rows else [[date, c, "10", "10", "10", "10", "10", "100", str(1000 * (i + 1)),
                                                      "3", "1", "1", "0", "5", "1", "1", "1", "0"]
                                                     for i, c in enumerate(["sh.600000", "sz.300001", "sh.688001", "sh.689009"])]
        return Page(sock, DAILY, rows)
    bs = SimpleNamespace(login=login, logout=lambda: sock.send_msg("logout"),
        query_trade_dates=lambda **k: Page(sock, "calendar_date,is_trading_day", [[d, "1"] for d in DAYS]),
        query_stock_basic=lambda **k: Page(sock, "code,code_name,ipoDate,outDate,type,status",
                                           [["sh.600000", "a", "2000-01-01", "", "1", "1"]]),
        query_all_stock=lambda **k: Page(sock, "code,tradeStatus,code_name", [["sh.600000", "1", "a"]]),
        query_daily_history_k_AStock=daily,
        query_daily_adjust_factor=lambda date: Page(sock, "code,dividOperateDate,foreAdjustFactor,backAdjustFactor,adjustFactor",
                                                    (adjust_rows or {}).get(date, [])),
        query_dividend_data=lambda **k: Page(sock, "code,dividOperateDate,dividCashPsBeforeTax", [[k["code"], "x", "0.1"]]))
    monkeypatch.setitem(sys.modules, "baostock", bs)
    monkeypatch.setitem(sys.modules, "baostock.util", SimpleNamespace(socketutil=sock))
    monkeypatch.setitem(sys.modules, "baostock.util.socketutil", sock)
    monkeypatch.setattr(acq.time, "sleep", lambda _: None)
    return sock


def test_pages_are_metered_and_a_full_page_requests_the_next(tmp_path, monkeypatch):
    sock = install(monkeypatch)
    ledger = acq.Ledger(tmp_path / "ledger.json", 100)
    acq.meter(sock, ledger)
    for rows, sent in (([1, 2, 3], 2), ([1, 2, 3, 4], 3), ([], 1)):
        before = ledger.sent
        page = Page(sock, "a", [[x] for x in rows], per=2)
        while page.next():
            page.get_row_data()
        assert ledger.sent - before == sent
    assert json.loads((tmp_path / "ledger.json").read_text())[acq.beijing_day()] == ledger.sent


def test_budget_stops_cleanly_and_the_next_day_resumes_without_refetching(tmp_path, monkeypatch):
    install(monkeypatch)
    monkeypatch.setattr(acq, "beijing_day", lambda: "2026-10-13")
    # login + calendar + master + two universe snapshots = 5, leaving 3 daily requests.
    first = acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=("daily",), budget=8, pause=0)
    assert first["stopped"] and first["requests_this_run"] == 8 and first["completed"] == []
    assert len(list((tmp_path / "snap/raw/daily").glob("*.csv"))) == 3
    monkeypatch.setattr(acq, "beijing_day", lambda: "2026-10-14")
    second = acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=("daily",), budget=8, pause=0)
    assert second["stopped"] is None and second["completed"] == ["daily"]
    assert second["requests_this_run"] == 1 + 3 + 1  # login, the three missing days, logout
    ledger = json.loads((tmp_path / "snap/request-ledger.json").read_text())
    assert ledger == {"2026-10-13": 8, "2026-10-14": 5}
    with pytest.raises(ValueError, match="Resume plan mismatch"):
        acq.acquire(tmp_path / "snap", DAYS[0], "2020-01-08", stages=("daily",), budget=8, pause=0)


def test_empty_market_day_is_a_failure_not_a_cached_result(tmp_path, monkeypatch):
    install(monkeypatch, daily_rows=lambda date: [])
    with pytest.raises(RuntimeError, match="Acquisition failed after 3 attempts: daily/2020-01-02"):
        acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=("daily",), pause=0)
    assert not (tmp_path / "snap/raw/daily/2020-01-02.csv").exists()
    failures = (tmp_path / "snap/collection-failures.jsonl").read_text().splitlines()
    assert len(failures) == 3 and "empty response" in failures[0]


def test_dividends_only_for_studied_codes_with_an_adjustment_event(tmp_path, monkeypatch):
    adjust = {"2020-01-03": [["sh.600000", "2020-01-03", "1", "1.1", "1.1"], ["sh.689009", "2020-01-03", "1", "1", "1"]],
              "2020-01-08": [["sz.300001", "2020-01-08", "1", "1.2", "1.2"]]}
    install(monkeypatch, adjust_rows=adjust)
    with pytest.raises(ValueError, match="adjust stage"):
        acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=("daily", "dividend"), pause=0)
    summary = acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=("adjust", "dividend"), pause=0)
    assert summary["completed"] == ["adjust", "dividend"]
    assert sorted(p.name for p in (tmp_path / "snap/raw/dividend").glob("*.csv")) == \
        ["sh.600000-2020.csv", "sz.300001-2020.csv"]
    # Days without events are cached as empty files and reread as empty frames.
    assert acq.read_raw(tmp_path / "snap", "adjust").code.tolist() == ["sh.600000", "sh.689009", "sz.300001"]


def test_through_fetches_a_prefix_that_the_full_run_completes_without_refetching(tmp_path, monkeypatch):
    adjust = {DAYS[1]: [["sh.600000", DAYS[1], "1", "1.1", "1.1"]], DAYS[4]: [["sz.300001", DAYS[4], "1", "1.2", "1.2"]]}
    install(monkeypatch, adjust_rows=adjust)
    with pytest.raises(ValueError, match="whole calendar years"):
        acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=("daily", "min30"), through=DAYS[2], pause=0)
    stages = ("daily", "adjust", "dividend")
    first = acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=stages, through=DAYS[2], pause=0)
    assert first["completed"] == list(stages) and first["through"] == DAYS[2]
    assert sorted(p.stem for p in (tmp_path / "snap/raw/daily").glob("*.csv")) == DAYS[:3]
    assert [p.name for p in (tmp_path / "snap/raw/dividend").glob("*.csv")] == ["sh.600000-2020.csv"]
    second = acq.acquire(tmp_path / "snap", DAYS[0], DAYS[-1], stages=stages, pause=0)
    assert second["completed"] == list(stages)
    # login, the three later days of each per-session stage, the one new dividend year, logout
    assert second["requests_this_run"] == 1 + 3 + 3 + 1 + 1


def test_min30_symbols_rank_all_studied_codes_by_trailing_amount():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2019-12-02", periods=40)]
    amounts = {"sh.600000": 100, "sz.300001": 300, "sh.688001": 200, "sh.689009": 1000, "sz.000002": 50}
    daily = pd.DataFrame([[d, c, str(a)] for d in days for c, a in amounts.items()], columns=["date", "code", "amount"])
    assert acq.min30_symbols(daily, "2020-01-01", days[-1], 2) == ["sh.688001", "sz.300001"]
    assert acq.min30_symbols(daily, "2020-01-01", days[-1], 10) == ["sh.600000", "sh.688001", "sz.000002", "sz.300001"]


def test_gaps_are_the_uncovered_calendar_ranges():
    plan = ("2016-01-01", "2025-12-31")
    assert acq.gaps([], *plan) == [plan]
    assert acq.gaps([("2016-01-01", "2024-12-31", None)], *plan) == [("2025-01-01", "2025-12-31")]
    assert acq.gaps([("2016-01-01", "2019-12-31", None), ("2021-01-01", "2025-12-31", None)], *plan) == \
        [("2020-01-01", "2020-12-31")]
    assert acq.gaps([("2015-01-01", "2026-06-30", None)], *plan) == []


def test_extending_the_plan_fetches_only_new_sessions_and_uncovered_ranges(tmp_path, monkeypatch):
    import sys as system
    system.path.insert(0, "scripts")
    from extend_raw import extend
    sock = install(monkeypatch)
    bs = system.modules["baostock"]
    calls = []
    bs.query_trade_dates = lambda start_date, end_date: Page(sock, "calendar_date,is_trading_day",
                                                              [[d, "1"] for d in DAYS if start_date <= d <= end_date])
    def index(code, fields, start_date, end_date, frequency):
        calls.append(("index", code, start_date, end_date))
        return Page(sock, fields, [[d, code] + ["1"] * 7 for d in DAYS if start_date <= d <= end_date])
    def report(code, start_date, end_date):
        calls.append(("report", code, start_date, end_date))
        return Page(sock, "code,pubDate", [])
    bs.query_history_k_data_plus = index
    bs.query_forecast_report = bs.query_performance_express_report = report
    stages = ("daily", "index", "reports")
    acq.acquire(tmp_path / "dev", DAYS[0], DAYS[2], stages=stages, pause=0)
    before = {p.relative_to(tmp_path / "dev"): p.read_bytes() for p in (tmp_path / "dev" / "raw").rglob("*.csv")}
    extend(tmp_path / "dev", tmp_path / "final", DAYS[-1])
    calls.clear()
    acq.acquire(tmp_path / "final", DAYS[0], DAYS[-1], stages=stages, pause=0)
    nxt = (pd.Timestamp(DAYS[2]) + pd.Timedelta(1, unit="D")).strftime("%Y-%m-%d")
    assert {c[2:] for c in calls} == {(nxt, DAYS[-1])} and len(calls) == 4 + 2 * 3  # 4 indices, 2 kinds x 3 codes
    assert sorted(p.stem for p in (tmp_path / "final/raw/daily").glob("*.csv")) == DAYS
    for relative, data in before.items():
        assert (tmp_path / "final" / relative).read_bytes() == data
    calendars = sorted(p.name for p in (tmp_path / "final/raw").glob("calendar*.csv"))
    assert calendars == [f"calendar__{DAYS[0]}__{DAYS[2]}.csv", f"calendar__{nxt}__{DAYS[-1]}.csv"]
    assert sorted(p.name for p in (tmp_path / "final/raw/index").glob("sh.000300__*.csv")) == \
        [f"sh.000300__{DAYS[0]}__{DAYS[2]}.csv", f"sh.000300__{nxt}__{DAYS[-1]}.csv"]


def test_min30_years_never_grow_backwards_when_the_plan_is_extended():
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2020-11-02", "2022-02-28")]
    rows = []
    for i, d in enumerate(days):
        rows += [[d, "sh.600000", "300"], [d, "sz.000002", "100"],
                 [d, "sz.300001", "500" if d >= "2022-01-03" else "50"]]  # enters the top 2 only in 2022
    daily = pd.DataFrame(rows, columns=["date", "code", "amount"])
    short = acq.min30_items(daily[daily.date <= "2021-12-31"], "2020-01-01", "2021-12-31", 2)
    long = acq.min30_items(daily, "2020-01-01", "2022-12-31", 2)
    assert [x[:2] for x in short] == [("sh.600000", "2020"), ("sh.600000", "2021"), ("sz.000002", "2020"),
                                      ("sz.000002", "2021")]
    assert [x[:2] for x in long if x[1] <= "2021"] == [x[:2] for x in short]
    assert ("sz.300001", "2022") in [x[:2] for x in long] and ("sz.300001", "2021") not in [x[:2] for x in long]


def test_extend_moves_only_the_plan_end_forward(tmp_path, monkeypatch):
    install(monkeypatch)
    acq.acquire(tmp_path / "live", DAYS[0], DAYS[2], stages=("daily",), pause=0)
    with pytest.raises(ValueError, match="Resume plan mismatch"):
        acq.acquire(tmp_path / "live", DAYS[0], DAYS[4], stages=("daily",), pause=0)
    with pytest.raises(ValueError, match="Resume plan mismatch"):
        acq.acquire(tmp_path / "live", DAYS[0], DAYS[1], stages=("daily",), pause=0, extend=True)
    acq.acquire(tmp_path / "live", DAYS[0], DAYS[4], stages=("daily",), pause=0, extend=True)
    plan = json.loads((tmp_path / "live/collection-plan.json").read_text())
    assert plan["end"] == DAYS[4] and len(list((tmp_path / "live/raw/daily").glob("*.csv"))) == 5


def test_live_stages_fetch_a_calendar_ahead_and_a_dividend_version_per_ex_date(tmp_path, monkeypatch):
    sock = install(monkeypatch, adjust_rows={DAYS[3]: [["sh.600000", DAYS[3], "1", "1.1", "1.1"],
                                                       ["sh.689009", DAYS[3], "1", "1", "1"]]})
    bs = sys.modules["baostock"]
    every = pd.bdate_range(DAYS[0], periods=80).strftime("%Y-%m-%d")
    bs.query_trade_dates = lambda start_date, end_date: Page(sock, "calendar_date,is_trading_day",
                                                              [[d, "1"] for d in every if start_date <= d <= end_date])
    asked = []
    def dividends(**k):
        asked.append((k["code"], k["year"]))
        return Page(sock, "code,dividOperateDate,dividCashPsBeforeTax", [[k["code"], DAYS[3], "0.1"]])
    bs.query_dividend_data = dividends
    with pytest.raises(ValueError, match="live_since"):
        acq.acquire(tmp_path / "live", DAYS[0], DAYS[4], stages=("daily", "dividend_live"), pause=0)
    acq.acquire(tmp_path / "live", DAYS[0], DAYS[4], stages=("daily", "adjust", "dividend_live", "calendar_ahead"),
                pause=0, live_since=DAYS[2])
    assert asked == [("sh.600000", DAYS[3][:4])]  # studied codes only, from live_since on
    assert (tmp_path / f"live/raw/dividend/sh.600000-{DAYS[3][:4]}@{DAYS[3]}.csv").exists()
    ahead = acq.shift_day(DAYS[4], 40)
    assert acq.gaps(acq.calendar_parts(tmp_path / "live"), DAYS[0], ahead) == []
    assert acq.trading_days(tmp_path / "live", DAYS[0], ahead)[len(DAYS) - 1] == DAYS[-1]
