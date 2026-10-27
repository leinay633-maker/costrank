"""Free BaoStock adapter and strict canonical CSV handoff; no paid API calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import socket
import time
from types import SimpleNamespace
import pandas as pd
from .data import ACTIONS, BARS, DELISTINGS, MASTER, mainboard, seal, sha256


def guard_closed_socket():
    """baostock's send_msg spins forever when recv() returns b'' (peer closed, e.g. a
    gateway answering its raw protocol with HTTP 502), so socket timeouts and our
    retries never fire. Raising turns it into an ordinary BaoStock error code."""
    try:
        from baostock.util import socketutil
    except ImportError:
        return
    if getattr(socketutil, "costrank_guarded", False):
        return

    class GuardedSocket(socket.socket):
        def recv(self, *args):
            data = super().recv(*args)
            if not data:
                raise ConnectionError("BaoStock peer closed the connection")
            return data

    socketutil.socket = SimpleNamespace(socket=GuardedSocket, AF_INET=socket.AF_INET,
                                        SOCK_STREAM=socket.SOCK_STREAM)
    socketutil.costrank_guarded = True


def repair_baostock_rows(bars):
    """Repair provider artefacts only where the row's own fields show what happened (7 of
    6.37M rows in 2016-2024 main board); any other conflict is left for validate() to reject.
    Returns the repaired string frame and one record per changed row, original values kept."""
    num = {c: pd.to_numeric(bars[c], errors="coerce")
           for c in ("open", "high", "low", "close", "preclose", "volume", "amount")}
    flat = pd.concat([(num[c] - num["preclose"]).abs().le(1e-6)
                      for c in ("open", "high", "low", "close")], axis=1).all(axis=1)
    none = {c: bars[c].eq("") | num[c].eq(0) for c in ("volume", "amount")}
    active, halted = bars.tradable.eq("1"), bars.tradable.eq("0")
    vwap = num["amount"] / num["volume"]
    rules = {
        # Session after the last trade before delisting: no trade recorded, price pinned to preclose.
        "active_flag_without_trades": active & none["volume"] & none["amount"] & flat,
        # Suspended placeholder repeating the previous session's volume with zero amount.
        "stale_volume_on_suspended_row": halted & num["volume"].gt(0) & none["amount"] & flat,
        # Real trades under a suspended flag: the price moved and VWAP lies within the day's range.
        "suspended_flag_with_trades": halted & num["volume"].gt(0) & num["amount"].gt(0) & ~flat
            & vwap.ge(num["low"] - 0.01) & vwap.le(num["high"] + 0.01),
    }
    repairs = [dict(rule=rule, symbol=r.symbol, date=r.date, tradestatus=r.tradable,
                    volume=r.volume, amount=r.amount)
               for rule, mask in rules.items()
               for r in bars.loc[mask, ["symbol", "date", "tradable", "volume", "amount"]].itertuples()]
    bars = bars.copy()
    bars.loc[rules["active_flag_without_trades"], ["tradable", "volume", "amount"]] = "0"
    bars.loc[rules["stale_volume_on_suspended_row"], "volume"] = "0"
    bars.loc[rules["suspended_flag_with_trades"], "tradable"] = "1"
    return bars, sorted(repairs, key=lambda x: (x["date"], x["symbol"]))


def find_aliases(basic, frames, *, min_rows=20, agreement=0.99):
    """Old codes whose traded history reappears under another code with the same listing
    date: after a code change BaoStock re-files the whole history under the new code
    (sz.000022 -> sz.001872), so keeping both counts one company twice and books a false
    delisting. Requires the new code to cover every traded day of the old one."""
    prices = ["open", "high", "low", "close"]
    found = []
    for old in basic[basic.outDate.ne("")].itertuples():
        a = frames[old.code]
        a = a[a.tradestatus.eq("1") & (a.date < old.outDate)]
        peers = basic[basic.ipoDate.eq(old.ipoDate) & basic.code.ne(old.code) &
                      (basic.outDate.eq("") | (basic.outDate > old.outDate))]
        if len(a) < min_rows:
            continue
        for new in peers.code:
            j = a.merge(frames[new], on="date", suffixes=("", "_new"))
            if len(j) < len(a):
                continue
            num = lambda col: pd.to_numeric(j[col], errors="coerce")
            same = pd.concat([(num(c) - num(c + "_new")).abs().le(1e-6) for c in prices + ["volume"]],
                             axis=1).all(axis=1)
            if same.mean() >= agreement:
                found.append(dict(symbol=old.code, alias_of=new, listed=old.ipoDate, out_date=old.outDate,
                                  traded_rows=len(a), matching_rows=int(same.sum())))
                break
    return found


def curated_resolutions(path, basic, end):
    """Hand-curated share-exchange / cash outcomes (reference/, with public sources).
    A row must match the provider's removal date exactly; rows for securities outside
    this snapshot, or whose acquirer is outside it, are skipped and recorded."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    outdate = basic.set_index("code").outDate
    keep, skipped = [], []
    for r in table.itertuples():
        if r.symbol not in outdate.index or not outdate[r.symbol] or outdate[r.symbol] > end:
            skipped.append({"symbol": r.symbol, "reason": "not_removed_in_snapshot"})
            continue
        if outdate[r.symbol] != r.delisted:
            raise ValueError(f"Curated removal date {r.delisted} differs from provider {outdate[r.symbol]}: {r.symbol}")
        if r.kind == "share_exchange" and r.acquirer not in outdate.index:
            skipped.append({"symbol": r.symbol, "reason": "acquirer_outside_snapshot"})
            continue
        keep.append(dict(symbol=r.symbol, date=r.delisted, kind=r.kind, acquirer=r.acquirer or None,
                         ratio=float(r.ratio) if r.ratio else None,
                         cash_per_share=float(r.cash_per_share) if r.cash_per_share else None,
                         settle_date=r.settle_date))
    frame = pd.DataFrame(keep, columns=DELISTINGS).sort_values("symbol")
    return frame, {"file": Path(path).name, "sha256": sha256(path), "applied": len(frame), "skipped": skipped}


def collect_baostock(output, start, end, *, limit=0, pause=0.2, retries=3, resolutions=None):
    """Single session, serial requests: BaoStock uses process-global socket state.

    Resumes only the IDENTICAL acquisition plan. Complete snapshots are immutable.
    All-symbol master plus annual historical-list snapshots is not proof that the
    provider retained every delisted security; resulting evidence is exploratory.
    """
    import baostock as bs
    guard_closed_socket()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "metadata.json").exists():
        raise FileExistsError("Sealed snapshot already exists; use a new directory")
    if limit < 0 or not 0 <= pause <= 60 or not 1 <= retries <= 10:
        raise ValueError("Invalid collection limits")
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    plan = {"provider": "baostock", "start": start, "end": end, "limit": limit}
    plan_path = root / "collection-plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("Resume plan mismatch; choose a new directory")
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    def cached(name, call):
        path = raw / (name + ".csv")
        if path.exists():
            return pd.read_csv(path, dtype=str, keep_default_na=False)
        last = None
        for attempt in range(retries):
            try:
                rs = call()
                if rs.error_code != "0":
                    raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rs.error_code != "0":
                    raise RuntimeError(f"{rs.error_code}: {rs.error_msg}")
                result = pd.DataFrame(rows, columns=rs.fields)
                temp = path.with_suffix(".tmp")
                result.to_csv(temp, index=False, encoding="utf-8")
                temp.replace(path)
                time.sleep(pause)
                return result
            except Exception as exc:
                last = exc
                with (root / "collection-failures.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(),
                        "request": name, "attempt": attempt + 1, "error": str(exc)}) + "\n")
                if attempt + 1 < retries:
                    bs.logout()
                    time.sleep(min(2 ** attempt, 8))
                    login = bs.login()
                    if login.error_code != "0":
                        last = RuntimeError(f"Reconnect failed: {login.error_code}")
        raise RuntimeError(f"Acquisition failed after {retries} attempts: {name}; {last}")

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(30)
    try:
        login = bs.login()
        if login.error_code != "0":
            raise RuntimeError(f"BaoStock login failed: {login.error_code}: {login.error_msg}")
        cal = cached("calendar", lambda: bs.query_trade_dates(start_date=start, end_date=end))
        days = cal.loc[cal.is_trading_day.eq("1"), "calendar_date"].sort_values().tolist()
        if not days:
            raise ValueError("Provider returned an empty trading calendar")
        basic = cached("master", bs.query_stock_basic)
        basic = basic[basic.code.map(mainboard)].copy()
        discovered = set(basic.code)
        # Historical names, including those absent from the currently listed universe.
        annual = pd.Series(days).groupby(pd.to_datetime(days).year).first().tolist()
        for day in sorted(set(annual + [days[-1]])):
            snapshot = cached("universe-" + day, lambda d=day: bs.query_all_stock(day=d))
            discovered.update(snapshot.loc[snapshot.code.map(mainboard), "code"])
        absent = sorted(discovered - set(basic.code))
        for code in absent:
            item = cached("master-" + code, lambda c=code: bs.query_stock_basic(code=c))
            if item.empty:
                raise ValueError(f"Historical security missing from master: {code}")
            basic = pd.concat([basic, item], ignore_index=True)
        if basic.code.duplicated().any() or basic.ipoDate.eq("").any():
            raise ValueError("Duplicate master or missing listing date; cannot infer lifetime")
        basic = basic[(basic.ipoDate <= end) & ((basic.outDate == "") | (basic.outDate > start))]
        basic = basic.copy()
        basic["sample_key"] = basic.code.map(lambda x: hashlib.sha256(x.encode()).hexdigest())
        if limit:
            basic = basic.sort_values("sample_key").head(limit)
        basic = basic.sort_values("code")
        frames = {}
        fields = "date,code,open,high,low,close,preclose,volume,amount,tradestatus,isST"
        for number, code in enumerate(basic.code, 1):
            frame = cached("bars-" + code, lambda c=code: bs.query_history_k_data_plus(
                c, fields, start_date=start, end_date=end, frequency="d", adjustflag="3"))
            if frame.empty:
                raise ValueError(f"Empty history for {code}; refusing survivorship-by-download-success")
            frames[code] = frame
            if number % 25 == 0:
                print(f"cached {number}/{len(basic)} symbols", flush=True)
        if not frames:
            raise ValueError("No instruments overlap requested dates")
        aliases = find_aliases(basic, frames)
        if aliases:
            print(f"dropped {len(aliases)} alias codes; see metadata dropped_aliases", flush=True)
        dropped = {x["symbol"] for x in aliases}
        basic = basic[~basic.code.isin(dropped)]
        bars = pd.concat([frames[c] for c in basic.code], ignore_index=True).rename(
            columns={"code": "symbol", "tradestatus": "tradable", "isST": "is_st"})
        # outDate is the removal day (first day no longer listed), yet BaoStock still
        # returns a suspended placeholder row on it; an active row there is a real conflict.
        out = bars.symbol.map(basic.set_index("code").outDate)
        after = out.ne("") & (bars.date >= out)
        if bars.loc[after, "tradable"].ne("0").any():
            raise ValueError("Active bar on or after outDate; cannot infer the delisting day")
        bars, repairs = repair_baostock_rows(bars[~after])
        if repairs:
            print(f"repaired {len(repairs)} provider rows; see metadata provider_repairs", flush=True)
        # Explicit suspended rows occasionally encode numeric fields as empty strings.
        for col in ("volume", "amount"):
            bars.loc[bars.tradable.eq("0") & bars[col].eq(""), col] = "0"
        master = basic.rename(columns={"code": "symbol", "ipoDate": "listed", "outDate": "delisted"})[MASTER]
        bars[BARS].to_csv(root / "bars.csv.gz", index=False)
        master.to_csv(root / "instruments.csv", index=False)
        pd.DataFrame({"date": days}).to_csv(root / "calendar.csv", index=False)
        pd.DataFrame(columns=ACTIONS).to_csv(root / "actions.csv", index=False)
        delistings, curated = (curated_resolutions(resolutions, basic, end) if resolutions
                               else (pd.DataFrame(columns=DELISTINGS), None))
        delistings.to_csv(root / "delistings.csv", index=False)
        return seal(root, {"source": "BaoStock", "evidence": "free_data_exploratory",
            "corporate_actions": "reference_reinvestment_proxy", "coverage": "unverified",
            "start": start, "end": end, "sample_limit": limit,
            "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "historical_universe_snapshots": sorted(set(annual + [days[-1]])),
            "provider_repairs": repairs, "dropped_aliases": aliases, "delisting_resolutions": curated,
            "limitations": ["Delisted coverage has not been independently verified",
                "Reference-price reinvestment proxy is not a dividend/tax/rights ledger",
                "Daily ST/status fields lack independently verified publication timestamps",
                "Open-auction depth and fills are not observed",
                "Share-exchange and cash delisting outcomes are hand-curated; other removals assume zero recovery",
                "Deterministic hash subset, not a market universe" if limit else "Provider universe, not independently certified"]})
    finally:
        try:
            bs.logout()
        finally:
            socket.setdefaulttimeout(old_timeout)


def import_csv(output, bars, instruments, calendar, metadata, actions=None):
    """Adapter boundary for independently licensed data, with no token handling."""
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    pd.read_csv(bars, dtype={"symbol": str})[BARS].to_csv(root / "bars.csv.gz", index=False)
    shutil.copyfile(instruments, root / "instruments.csv")
    shutil.copyfile(calendar, root / "calendar.csv")
    if actions:
        shutil.copyfile(actions, root / "actions.csv")
    else:
        pd.DataFrame(columns=ACTIONS).to_csv(root / "actions.csv", index=False)
    return seal(root, json.loads(Path(metadata).read_text(encoding="utf-8")))
