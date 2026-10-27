"""Free BaoStock acquisition for full A-share (schema v3) snapshots.

Network access lives only here. Every response is cached as one raw file, so an interrupted
run resumes exactly. A daily request budget (messages actually sent, one per result page,
login included, counted per Beijing calendar day) stops the run before BaoStock's per-IP limit;
rerunning the same command the next day continues where it stopped. Snapshot assembly from
the raw cache is offline (snapshot.py).
"""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import socket
import time
import pandas as pd
from .data import STUDIED, board
from .providers import guard_closed_socket

BEIJING = timezone(timedelta(hours=8))
STAGES = ("daily", "adjust", "master", "industry", "constituents", "index", "dividend", "reports", "min30")
# Forward paper trading only (live.fetch): a month of calendar ahead, and a dividend version per ex-date.
LIVE_ONLY = ("calendar_ahead", "dividend_live", "master_live")
INDICES = ("sh.000300", "sh.000905", "sh.000852", "sh.000906")
INDEX_FIELDS = "date,code,open,high,low,close,preclose,volume,amount"
MIN30_FIELDS = "date,time,code,open,high,low,close,volume,amount"
# BaoStock's 30-minute bars begin on 2020-01-02 (probed 2026-10-13: none in 2016 or 2019).
MIN30_FIRST_YEAR = 2020


class BudgetExhausted(RuntimeError):
    """Today's request budget is used up."""


def beijing_day():
    return datetime.now(BEIJING).strftime("%Y-%m-%d")


class Ledger:
    """Requests sent per Beijing calendar day, persisted after every request."""

    def __init__(self, path, limit):
        self.path, self.limit = Path(path), int(limit)
        self.counts = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.sent = 0

    def used(self, day=None):
        return self.counts.get(day or beijing_day(), 0)

    def charge(self):
        day = beijing_day()
        used = self.counts.get(day, 0)
        if used >= self.limit:
            raise BudgetExhausted(f"{used} requests sent on {day} (Beijing time); daily budget {self.limit}")
        self.counts[day] = used + 1
        self.sent += 1
        temp = self.path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.counts, indent=1, sort_keys=True), encoding="utf-8")
        temp.replace(self.path)


def meter(socketutil, ledger):
    """Charge every message BaoStock sends; each result page is a separate request."""
    original = getattr(socketutil, "costrank_unmetered_send", socketutil.send_msg)
    socketutil.costrank_unmetered_send = original

    def send(msg):
        ledger.charge()
        return original(msg)
    socketutil.send_msg = send


class Fetcher:
    def __init__(self, bs, root, *, pause, retries, backoff=8):
        self.bs, self.root, self.pause, self.retries, self.backoff = bs, Path(root), pause, retries, backoff
        self.fetched = 0

    def get(self, name, call, *, allow_empty=True):
        path = self.root / "raw" / (name + ".csv")
        if path.exists():
            try:
                return pd.read_csv(path, dtype=str, keep_default_na=False)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
        path.parent.mkdir(parents=True, exist_ok=True)
        last = None
        for attempt in range(self.retries):
            try:
                rs = call()
                if rs.error_code != "0":
                    raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":
                    raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
                if not rows and not allow_empty:
                    raise RuntimeError("empty response where data is required")
                frame = pd.DataFrame(rows, columns=rs.fields)
                temp = path.with_suffix(".tmp")
                frame.to_csv(temp, index=False, encoding="utf-8")
                temp.replace(path)
                self.fetched += 1
                time.sleep(self.pause)
                return frame
            except BudgetExhausted:
                raise
            except Exception as exc:
                last = exc
                with (self.root / "collection-failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "request": name,
                                        "attempt": attempt + 1, "error": str(exc)}, ensure_ascii=False) + "\n")
                if attempt + 1 < self.retries:
                    try:
                        self.bs.logout()
                    except BudgetExhausted:
                        raise
                    except Exception:
                        pass
                    time.sleep(min(2 ** attempt, self.backoff))
                    login = self.bs.login()
                    if login.error_code != "0":
                        last = RuntimeError(f"Reconnect failed: {login.error_code}")
        raise RuntimeError(f"Acquisition failed after {self.retries} attempts: {name}; {last}")


def segment(start, end):
    """Cache-name suffix of a query over a date range: a later range adds a file beside it
    instead of replacing it, and assembly reads each record from the segment it belongs to."""
    return f"__{start}__{end}"


def segment_index(root, folder):
    """{code: [(start, end, path)]} of a folder of segment files <code>__<start>__<end>.csv."""
    out = {}
    for path in (Path(root) / "raw" / folder).glob("*__*__*.csv"):
        code, first, last = path.stem.split("__")
        out.setdefault(code, []).append((first, last, path))
    return {k: sorted(v) for k, v in out.items()}


def shift_day(day, n):
    return (pd.Timestamp(day) + pd.Timedelta(n, unit="D")).strftime("%Y-%m-%d")


def gaps(parts, start, end):
    """Calendar sub-ranges of [start, end] that no existing segment covers, in order."""
    out, cursor = [], start
    for first, last, *_ in sorted(parts):
        if last < cursor:
            continue
        if first > end:
            break
        if first > cursor:
            out.append((cursor, shift_day(first, -1)))
        cursor = max(cursor, shift_day(last, 1))
    if cursor <= end:
        out.append((cursor, end))
    return out


def calendar_parts(root):
    """(first, last, frame) of every cached trading calendar; each covers its query's whole range."""
    out = []
    for path in sorted((Path(root) / "raw").glob("calendar*.csv")):
        f = pd.read_csv(path, dtype=str, keep_default_na=False)
        if len(f):
            out.append((f.calendar_date.min(), f.calendar_date.max(), f))
    return out


def trading_days(root, start, end):
    """Sessions in [start, end] from the cached calendars, which must cover the whole range."""
    parts = calendar_parts(root)
    if gaps(parts, start, end):
        raise ValueError(f"The cached trading calendar does not cover {start}..{end}")
    cal = pd.concat([f for *_, f in parts], ignore_index=True).drop_duplicates("calendar_date")
    return sorted(d for d in cal.loc[cal.is_trading_day.eq("1"), "calendar_date"] if start <= d <= end)


def first_sessions(days, freq):
    stamps = pd.to_datetime(pd.Series(days))
    return sorted(pd.Series(days).groupby(stamps.dt.to_period(freq).to_numpy()).first())


def read_raw(root, folder, usecols=None, through=None):
    """through: for folders of per-session files, skip files dated after that day."""
    frames = []
    for path in sorted((Path(root) / "raw" / folder).glob("*.csv")):
        if through is not None and path.stem > through:
            continue
        try:
            frames.append(pd.read_csv(path, dtype=str, keep_default_na=False, usecols=usecols))
        except pd.errors.EmptyDataError:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def studied_codes(daily):
    return sorted(c for c in daily.code.unique() if board(c))


def min30_entry(daily, start, end, top):
    """First session in [start, end] at which each studied code's 20-session mean amount ranks
    within `top` of all A shares: a superset of every eligible name, whose ranks exclude ST and
    new listings."""
    d = daily.loc[daily.code.str.fullmatch(STUDIED), ["date", "code", "amount"]].copy()
    d["amount"] = pd.to_numeric(d.amount, errors="coerce").fillna(0)
    d = d.sort_values(["code", "date"])
    d["adv"] = d.groupby("code").amount.transform(lambda x: x.rolling(20, min_periods=20).mean())
    d = d[(d.date >= start) & (d.date <= end)]
    ranks = d.groupby("date").adv.rank(method="first", ascending=False)
    return d.loc[ranks.le(top)].groupby("code").date.min()


def min30_symbols(daily, start, end, top):
    return sorted(min30_entry(daily, start, end, top).index)


def dividend_files(root, asof=None):
    """Latest version, as of `asof`, of each code-year dividend file: <code>-<year>.csv from the
    plan's fetch, or <code>-<year>@<day>.csv refetched on a later ex-date (forward period)."""
    latest = {}
    for path in (Path(root) / "raw" / "dividend").glob("*.csv"):
        base, _, version = path.stem.partition("@")
        if asof is not None and version > asof:
            continue
        if base not in latest or version > latest[base][0]:
            latest[base] = (version, path)
    return {base: path for base, (_, path) in latest.items()}


def master_frame(root, asof=None):
    """Security master as of `asof`: the latest full refresh (master-asof/<day>.csv, forward period)
    not after it, then the plan's master.csv, then per-code rows; the first listing of a code wins."""
    raw = Path(root) / "raw"
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    refreshes = sorted(p for p in (raw / "master-asof").glob("*.csv") if asof is None or p.stem <= asof)
    frames = ([read(refreshes[-1])] if refreshes else []) + [read(raw / "master.csv")]
    frames += [read(p) for p in sorted((raw / "master").glob("*.csv"))]
    return pd.concat(frames, ignore_index=True).drop_duplicates("code")


def dividend_items(root, through=None):
    """(code, year) pairs with an adjustment event: dividend details are queried per year."""
    events = read_raw(root, "adjust", usecols=["code", "dividOperateDate"], through=through)
    if events.empty:
        return []
    events = events[events.code.str.fullmatch(STUDIED)]
    return sorted(set(zip(events.code, events.dividOperateDate.str[:4])))


def report_items(daily):
    return [(kind, c) for kind in ("forecast", "express") for c in studied_codes(daily)]


def min30_items(daily, start, end, top):
    """(code, year, from, to) for each year, from the code's first top-`top` session onward, in
    which it traded. A year's set depends only on sessions up to that year's end, so a later plan
    end never adds earlier years (the frozen prefix stays the same)."""
    first = max(start, f"{MIN30_FIRST_YEAR}-01-01")
    entry = min30_entry(daily, first, end, top)
    recent = daily[daily.date >= first]
    seen = recent.date.str[:4].groupby(recent.code).unique()
    return [(c, y, max(first, f"{y}-01-01"), min(end, f"{y}-12-31"))
            for c, e in entry.items() for y in sorted(seen.get(c, [])) if y >= e[:4]]


def acquire(output, start, end, *, stages=STAGES, budget=45_000, pause=0.2, retries=3, min30_top=2500, through=None,
            extend=False, live_since=None, backoff=8):
    """Fill the raw cache stage by stage; returns a summary, stopping cleanly at the budget.

    through: fetch only items dated on or before that day (sessions, months, dividend years and
    codes seen by then) without changing the plan, so a prefix of the range can be completed and
    assembled first; every request is the one the full run would send, so nothing is fetched twice.

    extend: the plan's end may move later (same start), as the forward paper-trading cache grows.
    live_since: first session whose ex-dates get a fresh dividend version (stage dividend_live).
    """
    import baostock as bs
    from baostock.util import socketutil
    if set(stages) - set(STAGES) - set(LIVE_ONLY) or not stages:
        raise ValueError(f"Stages must be a nonempty subset of {STAGES + LIVE_ONLY}")
    if "dividend_live" in stages and live_since is None:
        raise ValueError("dividend_live needs live_since")
    if not 1 <= budget <= 50_000 or not 0 <= pause <= 60 or not 1 <= retries <= 10 or not 1 <= backoff <= 600:
        raise ValueError("Invalid acquisition limits")
    if through is not None:
        if not start <= through <= end:
            raise ValueError("through must lie within the plan range")
        if through < end and "min30" in stages:
            raise ValueError("The min30 stage caches whole calendar years; run it without through")
    root = Path(output)
    (root / "raw").mkdir(parents=True, exist_ok=True)
    if (root / "metadata.json").exists():
        raise FileExistsError("Sealed snapshot already exists; use a new directory")
    plan = {"provider": "baostock", "schema": 3, "start": start, "end": end}
    plan_path = root / "collection-plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        later = extend and dict(old, end=end) == plan and old["end"] <= end
        if old != plan and not later:
            raise ValueError("Resume plan mismatch; choose a new directory")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    ledger = Ledger(root / "request-ledger.json", budget)
    guard_closed_socket()
    meter(socketutil, ledger)
    summary = {"stages": list(stages), "through": through, "completed": [], "stopped": None}
    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(60)
    try:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code}: {login.error_msg}")
        f = Fetcher(bs, root, pause=pause, retries=retries, backoff=backoff)
        # The calendar too is cached in segments: an extended plan fetches only the dates not yet covered.
        for a, b in gaps(calendar_parts(root), start, end):
            f.get(f"calendar__{a}__{b}", lambda a=a, b=b: bs.query_trade_dates(start_date=a, end_date=b),
                  allow_empty=False)
        days = trading_days(root, start, end)
        if not days:
            raise ValueError("Provider returned an empty trading calendar")
        if "master_live" in stages:
            # Listings and delistings change in the forward period: a full refresh per session.
            f.get(f"master-asof/{end}", bs.query_stock_basic, allow_empty=False)
        if "calendar_ahead" in stages:
            # The next session must be known at today's close.
            for a, b in gaps(calendar_parts(root), shift_day(end, 1), shift_day(end, 40)):
                f.get(f"calendar__{a}__{b}", lambda a=a, b=b: bs.query_trade_dates(start_date=a, end_date=b),
                      allow_empty=False)
        f.get("master", bs.query_stock_basic, allow_empty=False)
        for day in sorted(set(first_sessions(days, "Y")) | {days[-1]}):
            f.get("universe/" + day, lambda d=day: bs.query_all_stock(day=d), allow_empty=False)
        if through is not None:
            days = [d for d in days if d <= through]  # a prefix, so month firsts are unchanged

        def progress(stage, done, total):
            if done % 100 == 0 or done == total:
                print(f"{stage} {done}/{total}, requests today {ledger.used()}", flush=True)

        def run(stage, items, fetch):
            for n, item in enumerate(items, 1):
                fetch(item)
                progress(stage, n, len(items))
            summary["completed"].append(stage)

        if "daily" in stages:
            run("daily", days, lambda d: f.get("daily/" + d, lambda: bs.query_daily_history_k_AStock(date=d),
                                               allow_empty=False))
        if "adjust" in stages:
            run("adjust", days, lambda d: f.get("adjust/" + d, lambda: bs.query_daily_adjust_factor(date=d)))
        if "industry" in stages:
            run("industry", first_sessions(days, "M"),
                lambda d: f.get("industry/" + d, lambda: bs.query_stock_industry(date=d), allow_empty=False))
        if "constituents" in stages:
            months = first_sessions(days, "M")
            run("constituents", [(i, d) for i in ("hs300", "zz500") for d in months],
                lambda x: f.get(f"{x[0]}/{x[1]}", lambda: getattr(bs, f"query_{x[0]}_stocks")(date=x[1]),
                                allow_empty=False))
        if "index" in stages:
            # Only the ranges no earlier fetch covers; a range without a session may be empty.
            parts = segment_index(root, "index")
            items = [(c, a, b) for c in INDICES for a, b in gaps(parts.get(c, []), start, end)]
            run("index", items, lambda x: f.get(
                f"index/{x[0]}{segment(x[1], x[2])}",
                lambda: bs.query_history_k_data_plus(x[0], INDEX_FIELDS, start_date=x[1], end_date=x[2], frequency="d"),
                allow_empty=not any(x[1] <= d <= x[2] for d in days)))
        if {"master", "dividend", "reports", "min30"} & set(stages):
            if any(not (root / "raw" / "daily" / (d + ".csv")).exists() for d in days):
                raise ValueError("Per-code stages need the daily stage completed first")
            daily = read_raw(root, "daily", usecols=["date", "code", "amount"], through=through)
        if "master" in stages:
            # Codes seen in the market files but missing from the current security master.
            known = set(f.get("master", bs.query_stock_basic).code)
            run("master", [c for c in studied_codes(daily) if c not in known],
                lambda c: f.get("master/" + c, lambda: bs.query_stock_basic(code=c), allow_empty=False))
        if "dividend" in stages:
            if any(not (root / "raw" / "adjust" / (d + ".csv")).exists() for d in days):
                raise ValueError("The dividend stage needs the adjust stage completed first")
            run("dividend", dividend_items(root, through), lambda x: f.get(f"dividend/{x[0]}-{x[1]}", lambda: bs.query_dividend_data(
                code=x[0], year=x[1], yearType="operate")))
        if "dividend_live" in stages:
            # On its ex-date an event is announced: fetch that code-year again so the latest version has it.
            items = []
            for d in (d for d in days if d >= live_since):
                path = root / "raw" / "adjust" / f"{d}.csv"
                if not path.exists():
                    continue
                try:
                    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
                except pd.errors.EmptyDataError:
                    continue
                codes = frame.loc[frame.code.str.fullmatch(STUDIED) & frame.dividOperateDate.eq(d), "code"]
                items += [(c, d) for c in sorted(set(codes))]
            run("dividend_live", items, lambda x: f.get(
                f"dividend/{x[0]}-{x[1][:4]}@{x[1]}",
                lambda: bs.query_dividend_data(code=x[0], year=x[1][:4], yearType="operate")))
        if "reports" in stages:
            parts = {k: segment_index(root, k) for k in ("forecast", "express")}
            items = [(k, c, a, b) for k, c in report_items(daily) for a, b in gaps(parts[k].get(c, []), start, end)]
            run("reports", items, lambda x: f.get(f"{x[0]}/{x[1]}{segment(x[2], x[3])}", lambda: (
                bs.query_forecast_report if x[0] == "forecast" else bs.query_performance_express_report)(
                x[1], start_date=x[2], end_date=x[3])))
        if "min30" in stages:
            items = min30_items(daily, start, end, min30_top)
            summary["min30_symbols"] = len({x[0] for x in items})
            run("min30", items, lambda x: f.get(f"min30/{x[0]}-{x[1]}", lambda: bs.query_history_k_data_plus(
                x[0], MIN30_FIELDS, start_date=x[2], end_date=x[3], frequency="30", adjustflag="3")))
    except BudgetExhausted as exc:
        summary["stopped"] = str(exc)
        print(f"stopped: {exc}; rerun the same command after midnight Beijing time to resume", flush=True)
    finally:
        try:
            bs.logout()
        except Exception:
            pass
        socket.setdefaulttimeout(old_timeout)
    summary["requests_today"] = ledger.used()
    summary["requests_this_run"] = ledger.sent
    with (root / "acquire-runs.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "budget": budget,
                              "min30_top": min30_top, **summary}, ensure_ascii=False) + "\n")
    return summary
