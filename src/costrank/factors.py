"""Registered v0.4 factor library (RESEARCH section 9): causal, wide-matrix and vectorized.

A factor at close t uses only rows dated t or earlier. Time-series operations run on the causal
total-return index (open/high/low/close/vwap scaled by tr_close / close) so ex-dates leave no
jumps. Cross-sectional ranks of a single day use that day's raw prices and volumes. A term that
is a price level (degree 1) and is compared across stocks is converted back to day t's raw basis
with K = close / tr_close; volume compares in raw shares, W / K, where W = volume * K is the
split-continuous volume. Events published on day D are used from the next session onward.

The registry is fixed before any v0.4 result; selection happens inside each training window.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

POSITIVE_FORECASTS = {"预增", "略增", "扭亏", "续盈"}
NEGATIVE_FORECASTS = {"预减", "略减", "首亏", "续亏"}


# ---- operators on wide frames (rows: sessions, columns: symbols) ------------------------------

def mean(x, d):
    return x.rolling(d, min_periods=d).mean()


def total(x, d):
    return x.rolling(d, min_periods=d).sum()


def std(x, d):
    return x.rolling(d, min_periods=d).std()


def tsmin(x, d):
    return x.rolling(d, min_periods=d).min()


def tsmax(x, d):
    return x.rolling(d, min_periods=d).max()


def delay(x, d):
    return x.shift(d)


def delta(x, d):
    return x - x.shift(d)


def finite(x):
    return x.where(np.isfinite(x))


def corr(x, y, d):
    return finite(x.rolling(d, min_periods=d).corr(y))


def cov(x, y, d):
    return x.rolling(d, min_periods=d).cov(y)


def rank(x):
    return x.rank(axis=1, pct=True)


def sma(x, n, m):
    """GTJA SMA(A, n, m): Y = (m A + (n - m) Y_prev) / n."""
    return x.ewm(alpha=m / n, adjust=False, min_periods=n).mean()


def windows(x, d, fn, chunk=512):
    a = x.to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    for j in range(0, a.shape[1], chunk):
        w = sliding_window_view(a[:, j:j + chunk], d, axis=0)  # sessions-d+1, names, d
        bad = np.isnan(w).any(axis=-1)
        value = fn(np.where(np.isnan(w), 0.0, w))
        out[d - 1:, j:j + chunk] = np.where(bad, np.nan, value)
    return pd.DataFrame(out, index=x.index, columns=x.columns)


def tsrank(x, d):
    """Position of today's value within the trailing window, 0 (lowest) to 1 (highest)."""
    def fn(w):
        last = w[..., -1:]
        return ((w < last).sum(-1) + 0.5 * ((w == last).sum(-1) - 1)) / (d - 1)
    return windows(x, d, fn)


def argmax(x, d):
    return windows(x, d, lambda w: w.argmax(-1).astype(float))


def decay(x, d):
    weights = np.arange(1, d + 1, dtype=float)
    return windows(x, d, lambda w: w @ weights / weights.sum())


def where(condition, a, b):
    return a.where(condition, b) if isinstance(a, pd.DataFrame) else b.where(~condition, a)


# ---- market data ----------------------------------------------------------------------------

@dataclass
class Market:
    """Wide inputs. o/h/l/c/vwap are total-return-index prices, w is split-continuous volume;
    raw_* are the day's actual prices; k = raw close / index close."""
    o: pd.DataFrame
    h: pd.DataFrame
    l: pd.DataFrame
    c: pd.DataFrame
    vwap: pd.DataFrame
    w: pd.DataFrame
    k: pd.DataFrame
    raw_o: pd.DataFrame
    raw_h: pd.DataFrame
    raw_l: pd.DataFrame
    raw_c: pd.DataFrame
    raw_v: pd.DataFrame
    raw_vwap: pd.DataFrame
    amount: pd.DataFrame
    ret: pd.DataFrame
    turn: pd.DataFrame
    mcap: pd.DataFrame
    pe: pd.DataFrame
    pb: pd.DataFrame
    ps: pd.DataFrame
    pcf: pd.DataFrame
    mkt: pd.Series
    events: dict


def wide(frame, column, index, columns):
    return frame.pivot(index="date", columns="symbol", values=column).reindex(index=index, columns=columns)


def market(panel, bundle):
    """panel: features.prepare output (all bars, with tr_close and the eligible flag)."""
    dates = pd.DatetimeIndex(sorted(panel.date.unique()))
    symbols = sorted(panel.symbol.unique())
    get = lambda col, f=panel: wide(f, col, dates, symbols)
    raw_c, raw_o, raw_h, raw_l = get("close"), get("open"), get("high"), get("low")
    volume, amount = get("volume"), get("amount")
    traded = volume.gt(0)
    index_close = get("tr_close")
    scale = index_close / raw_c
    raw_vwap = (amount / volume).where(traded)
    k = raw_c / index_close
    attrs = bundle.tables["attributes"]
    attr = lambda col: wide(attrs, col, dates, symbols)
    turn = attr("turn").where(traded)
    # Float shares from turnover (percent of float traded) on days with at least 0.01% turnover,
    # carried in total-return-index units so a split or bonus on a day without trading rescales them.
    index_shares = (volume / (turn / 100) * (raw_c / index_close)).where(traded & turn.ge(0.01)).ffill()
    ret = (raw_c / get("preclose") - 1)
    eligible = get("eligible").fillna(False).astype(bool)
    # Equal weight over the previous session's eligible names, as the benchmark.
    mkt = ret.where(eligible.shift(1, fill_value=False)).mean(axis=1)
    return Market(o=raw_o * scale, h=raw_h * scale, l=raw_l * scale, c=index_close, vwap=raw_vwap * scale,
                  w=volume * k, k=k, raw_o=raw_o, raw_h=raw_h, raw_l=raw_l, raw_c=raw_c, raw_v=volume,
                  raw_vwap=raw_vwap, amount=amount.where(traded), ret=ret, turn=turn,
                  mcap=index_shares * index_close, pe=attr("pe_ttm"), pb=attr("pb_mrq"), ps=attr("ps_ttm"),
                  pcf=attr("pcf_ttm"), mkt=mkt, events=event_frames(bundle, dates, symbols))


def latest_event(table, value, dates, symbols, window):
    """Value of the most recent event published strictly before each session, if within `window`
    calendar days; NaN otherwise."""
    if table.empty:
        return pd.DataFrame(np.nan, index=dates, columns=symbols)
    t = table[table.symbol.isin(symbols)].sort_values("pub_date")
    usable = t.assign(session=dates[np.searchsorted(dates.values, t.pub_date.values, side="right")
                                    .clip(max=len(dates) - 1)])
    usable = usable[usable.pub_date < usable.session]
    last = usable.groupby(["session", "symbol"]).last()
    v = last[value].unstack().reindex(index=dates, columns=symbols).ffill()
    when = last["pub_date"].unstack().reindex(index=dates, columns=symbols).ffill()
    age = (dates.values[:, None] - when.to_numpy(dtype="datetime64[ns]")) / np.timedelta64(1, "D")
    return v.where(pd.DataFrame(age <= window, index=dates, columns=symbols))


def event_frames(bundle, dates, symbols):
    fc = bundle.tables.get("forecasts", pd.DataFrame())
    ex = bundle.tables.get("express", pd.DataFrame())
    out = {}
    if len(fc):
        fc = fc.assign(sentiment=np.where(fc.kind.isin(POSITIVE_FORECASTS), 1.0,
                                          np.where(fc.kind.isin(NEGATIVE_FORECASTS), -1.0, 0.0)),
                       midpoint=(fc.chg_up + fc.chg_down) / 2)
        for window in (20, 60):
            out[f"fc_sentiment_{window}"] = latest_event(fc, "sentiment", dates, symbols, window)
        out["fc_change_60"] = latest_event(fc, "midpoint", dates, symbols, 60)
    if len(ex):
        # The provider keeps only the latest version of a revised report: usable from its last update.
        ex = ex.assign(pub_date=ex.available_date)
        for col in ("eps_chg_pct", "gr_yoy", "op_yoy"):
            out[f"ex_{col}_60"] = latest_event(ex, col, dates, symbols, 60)
    parts = [x[["symbol", "pub_date"]] for x in (fc, ex) if len(x)]
    both = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if len(both):
        stamp = latest_event(both.assign(day=both.pub_date.astype("int64")), "day", dates, symbols, 3650)
        out["days_since_report"] = ((dates.asi8[:, None] - stamp) / 86_400e9).clip(upper=250)
    return out


# ---- the registry -----------------------------------------------------------------------------

REGISTRY = {}


def factor(name, family):
    def register(fn):
        if name in REGISTRY:
            raise ValueError(f"Duplicate factor {name}")
        REGISTRY[name] = (family, fn)
        return fn
    return register


def define(name, family, fn):
    factor(name, family)(fn)


def ratio(a, b):
    return finite(a / b)


def beta_parts(m, d):
    x = m.mkt
    mx = x.rolling(d, min_periods=d).mean()
    vx = x.rolling(d, min_periods=d).var()
    cxy = (m.ret.mul(x, axis=0)).rolling(d, min_periods=d).mean().sub(mean(m.ret, d).mul(mx, axis=0), axis=0) * d / (d - 1)
    beta = cxy.div(vx, axis=0)
    vy = m.ret.rolling(d, min_periods=d).var()
    return beta, vy, vx


# Momentum and reversal.
for n in (1, 3, 5, 6, 10, 20, 60, 120, 250):
    define(f"ret_{n}", "momentum", lambda m, n=n: m.c / delay(m.c, n) - 1)
for a, b in ((250, 20), (120, 20), (60, 5)):
    define(f"mom_{a}_{b}", "momentum", lambda m, a=a, b=b: delay(m.c, b) / delay(m.c, a) - 1)
for n in (5, 20):
    define(f"overnight_{n}", "momentum", lambda m, n=n: total(np.log(m.o / delay(m.c, 1)), n))
    define(f"intraday_{n}", "momentum", lambda m, n=n: total(np.log(m.c / m.o), n))
define("max_ret_20", "momentum", lambda m: tsmax(m.ret, 20))
define("min_ret_20", "momentum", lambda m: tsmin(m.ret, 20))
for n in (60, 250):
    define(f"high_dist_{n}", "momentum", lambda m, n=n: m.c / tsmax(m.h, n) - 1)
    define(f"low_dist_{n}", "momentum", lambda m, n=n: m.c / tsmin(m.l, n) - 1)
for n in (5, 10, 12, 20, 24, 60, 120):
    define(f"ma_ratio_{n}", "momentum", lambda m, n=n: m.c / mean(m.c, n) - 1)
define("ma_cross_5_20", "momentum", lambda m: mean(m.c, 5) / mean(m.c, 20) - 1)
define("ma_cross_20_60", "momentum", lambda m: mean(m.c, 20) / mean(m.c, 60) - 1)
for n in (12, 20, 60):
    define(f"up_ratio_{n}", "momentum", lambda m, n=n: mean(m.ret.gt(0).astype(float).where(m.ret.notna()), n))
define("mom_accel_20", "momentum", lambda m: (m.c / delay(m.c, 20)) - (delay(m.c, 20) / delay(m.c, 40)))
for n in (20, 60):
    define(f"ret_sharpe_{n}", "momentum", lambda m, n=n: ratio(total(m.ret, n), std(m.ret, n)))
define("path_efficiency_20", "momentum", lambda m: ratio((m.c - delay(m.c, 20)).abs(), total(delta(m.c, 1).abs(), 20)))
define("rsi_6", "momentum", lambda m: ratio(sma(delta(m.c, 1).clip(lower=0), 6, 1), sma(delta(m.c, 1).abs(), 6, 1)))
define("rsi_14", "momentum", lambda m: ratio(sma(delta(m.c, 1).clip(lower=0), 14, 1), sma(delta(m.c, 1).abs(), 14, 1)))
define("ma_mix_3_24", "momentum", lambda m: (mean(m.c, 3) + mean(m.c, 6) + mean(m.c, 12) + mean(m.c, 24)) / (4 * m.c))
define("stoch_k_9", "momentum", lambda m: sma(ratio(m.c - tsmin(m.l, 9), tsmax(m.h, 9) - tsmin(m.l, 9)), 3, 1))
define("stoch_d_9", "momentum", lambda m: sma(sma(ratio(m.c - tsmin(m.l, 9), tsmax(m.h, 9) - tsmin(m.l, 9)), 3, 1), 3, 1))
define("cci_12", "momentum", lambda m: ratio((m.h + m.l + m.c) / 3 - mean((m.h + m.l + m.c) / 3, 12),
                                             0.015 * mean((m.c - mean((m.h + m.l + m.c) / 3, 12)).abs(), 12)))

# Volatility and higher moments.
for n in (5, 20, 60, 120):
    define(f"vol_{n}", "volatility", lambda m, n=n: std(m.ret, n))
for n in (20, 60):
    define(f"downside_vol_{n}", "volatility", lambda m, n=n: np.sqrt(mean(m.ret.clip(upper=0) ** 2, n)))
    define(f"skew_{n}", "volatility", lambda m, n=n: m.ret.rolling(n, min_periods=n).skew())
    define(f"kurt_{n}", "volatility", lambda m, n=n: m.ret.rolling(n, min_periods=n).kurt())
    define(f"range_{n}", "volatility", lambda m, n=n: mean((m.h - m.l) / m.c, n))
define("range_5", "volatility", lambda m: mean((m.h - m.l) / m.c, 5))
define("upside_vol_20", "volatility", lambda m: np.sqrt(mean(m.ret.clip(lower=0) ** 2, 20)))
define("vol_ratio_5_60", "volatility", lambda m: ratio(std(m.ret, 5), std(m.ret, 60)))
define("parkinson_20", "volatility", lambda m: np.sqrt(mean(np.log(m.h / m.l) ** 2, 20) / (4 * np.log(2))))
define("garman_klass_20", "volatility", lambda m: np.sqrt(mean(
    0.5 * np.log(m.h / m.l) ** 2 - (2 * np.log(2) - 1) * np.log(m.c / m.o) ** 2, 20).clip(lower=0)))
for n in (60, 250):
    define(f"beta_{n}", "volatility", lambda m, n=n: beta_parts(m, n)[0])
def idio_var(m, d):
    beta, vy, vx = beta_parts(m, d)
    return beta, vy, (vy - (beta ** 2).mul(vx, axis=0)).clip(lower=0)


for n in (20, 60):
    define(f"idio_vol_{n}", "volatility", lambda m, n=n: np.sqrt(idio_var(m, n)[2]))
define("r2_60", "volatility", lambda m: (lambda b, vy, ve: finite(1 - ve / vy))(*idio_var(m, 60)))
define("coskew_60", "volatility", lambda m: ratio(
    mean((m.ret - mean(m.ret, 60)).mul((m.mkt - m.mkt.rolling(60, min_periods=60).mean()) ** 2, axis=0), 60),
    std(m.ret, 60).mul(m.mkt.rolling(60, min_periods=60).var(), axis=0)))
define("vol_of_vol_60", "volatility", lambda m: ratio(std(std(m.ret, 5), 60), mean(std(m.ret, 5), 60)))
define("max_drawdown_60", "volatility", lambda m: (m.c / m.c.rolling(60, min_periods=60).max() - 1).rolling(60, min_periods=60).min())
define("abs_ret_20", "volatility", lambda m: mean(m.ret.abs(), 20))
define("gap_vol_20", "volatility", lambda m: std(np.log(m.o / delay(m.c, 1)), 20))
def true_range(m):
    return np.maximum(np.maximum(m.h - m.l, (delay(m.c, 1) - m.h).abs()), (delay(m.c, 1) - m.l).abs())


define("atr_ratio_14", "volatility", lambda m: ratio(mean(true_range(m), 14), m.c))
define("range_trend_10", "volatility", lambda m: ratio(sma(m.h - m.l, 10, 2), sma(sma(m.h - m.l, 10, 2), 10, 2)))

# Liquidity and turnover.
for n in (5, 20, 60, 120):
    define(f"turn_{n}", "liquidity", lambda m, n=n: mean(m.turn.fillna(0), n))
define("turn_ratio_5_60", "liquidity", lambda m: ratio(mean(m.turn.fillna(0), 5), mean(m.turn.fillna(0), 60)))
define("turn_ratio_20_120", "liquidity", lambda m: ratio(mean(m.turn.fillna(0), 20), mean(m.turn.fillna(0), 120)))
define("turn_std_20", "liquidity", lambda m: std(m.turn.fillna(0), 20))
define("turn_cv_20", "liquidity", lambda m: ratio(std(m.turn.fillna(0), 20), mean(m.turn.fillna(0), 20)))
for n in (20, 60):
    define(f"amihud_{n}", "liquidity", lambda m, n=n: m.ret.abs().div(m.amount / 1e8).rolling(n, min_periods=n // 2).mean())
define("log_amount_20", "liquidity", lambda m: np.log(m.amount.fillna(0).rolling(20, min_periods=20).mean().where(lambda x: x > 0)))
define("amount_ratio_5_60", "liquidity", lambda m: ratio(mean(m.amount.fillna(0), 5), mean(m.amount.fillna(0), 60)))
define("amount_std_6", "liquidity", lambda m: std(m.amount.fillna(0), 6))
define("amount_std_20", "liquidity", lambda m: std(m.amount.fillna(0), 20))
define("halt_days_60", "liquidity", lambda m: mean(m.amount.isna().astype(float), 60))
for n in (10, 20):
    define(f"price_volume_corr_{n}", "liquidity", lambda m, n=n: corr(m.c, m.w, n))
define("return_volume_corr_20", "liquidity", lambda m: corr(m.ret, delta(np.log(m.w.where(m.w > 0)), 1), 20))
define("turn_weighted_ret_20", "liquidity", lambda m: ratio(total(m.ret * m.turn.fillna(0), 20), total(m.turn.fillna(0), 20)))
define("vwap_gap_5", "liquidity", lambda m: mean(m.c / m.vwap - 1, 5))
define("vwap_gap_20", "liquidity", lambda m: mean(m.c / m.vwap - 1, 20))
define("volume_trend_20", "liquidity", lambda m: corr(np.log(m.w.where(m.w > 0)),
    pd.DataFrame(np.arange(len(m.w))[:, None].repeat(m.w.shape[1], 1).astype(float), index=m.w.index, columns=m.w.columns), 20))
define("volume_change_5", "liquidity", lambda m: ratio(m.w, delay(m.w, 5)) - 1)
define("volume_std_20", "liquidity", lambda m: std(m.w, 20) / m.k)
define("up_down_volume_26", "liquidity", lambda m: ratio(total(m.w.where(m.ret > 0, 0), 26), total(m.w.where(m.ret <= 0, 0), 26)))
define("obv_20", "liquidity", lambda m: total(np.sign(delta(m.c, 1)) * m.w, 20) / m.k)
for n in (6, 20):
    define(f"clv_volume_{n}", "liquidity", lambda m, n=n: total(ratio((m.c - m.l) - (m.h - m.c), m.h - m.l) * m.w, n) / m.k)

# Value and quality from provider valuation ratios. Yields of nonpositive ratios are missing, not
# negative; per-share values are taken on the index basis (index price / ratio) to avoid split jumps.
def positive_yield(x):
    return 1 / x.where(x > 0)


define("ep", "value", lambda m: positive_yield(m.pe))
define("bp", "value", lambda m: positive_yield(m.pb))
define("sp", "value", lambda m: positive_yield(m.ps))
define("cfp", "value", lambda m: positive_yield(m.pcf))
define("roe_proxy", "value", lambda m: m.pb.where(m.pb > 0) * positive_yield(m.pe))
define("log_bp", "value", lambda m: np.log(positive_yield(m.pb)))
for n in (63, 126, 252):
    define(f"eps_growth_{n}", "value", lambda m, n=n: ratio(m.c / m.pe - delay(m.c / m.pe, n), delay(m.c / m.pe, n).abs()))
for n in (126, 252):
    define(f"sales_growth_{n}", "value", lambda m, n=n: ratio(m.c / m.ps.where(m.ps > 0), delay(m.c / m.ps.where(m.ps > 0), n)) - 1)
define("bp_change_63", "value", lambda m: positive_yield(m.pb) - delay(positive_yield(m.pb), 63))
define("ep_zscore_250", "value", lambda m: ratio(positive_yield(m.pe) - mean(positive_yield(m.pe), 250),
                                                std(positive_yield(m.pe), 250)))
define("loss_flag", "value", lambda m: m.pe.lt(0).astype(float).where(m.pe.notna()))
define("cash_earnings_gap", "value", lambda m: positive_yield(m.pcf) - positive_yield(m.pe))
define("price_level", "value", lambda m: np.log(m.raw_c))

# Earnings forecasts and flash reports.
for name in ("fc_sentiment_20", "fc_sentiment_60", "fc_change_60", "ex_eps_chg_pct_60", "ex_gr_yoy_60",
             "ex_op_yoy_60", "days_since_report"):
    define(name, "events", lambda m, name=name: m.events.get(name))


# Formulaic alphas (Kakushadze 2016, "101 Formulaic Alphas"): daily-bar formulas only, without
# industry neutralization terms; K converts degree-1 price terms to the day's raw basis.
def k1(m, x):
    return x * m.k


define("alpha002", "alpha101", lambda m: -corr(rank(delta(np.log(m.w.where(m.w > 0)), 2)), rank((m.c - m.o) / m.o), 6))
define("alpha003", "alpha101", lambda m: -corr(rank(m.raw_o), rank(m.raw_v), 10))
define("alpha004", "alpha101", lambda m: -tsrank(rank(m.raw_l), 9))
define("alpha006", "alpha101", lambda m: -corr(m.o, m.w, 10))
define("alpha008", "alpha101", lambda m: -rank(k1(m, delta(total(m.o, 5) * total(m.ret, 5), 10))))
define("alpha012", "alpha101", lambda m: np.sign(delta(m.w, 1)) * -k1(m, delta(m.c, 1)))
define("alpha013", "alpha101", lambda m: -rank(cov(rank(m.raw_c), rank(m.raw_v), 5)))
define("alpha014", "alpha101", lambda m: -rank(delta(m.ret, 3)) * corr(m.o, m.w, 10))
define("alpha015", "alpha101", lambda m: -total(rank(corr(rank(m.raw_h), rank(m.raw_v), 3)), 3))
define("alpha016", "alpha101", lambda m: -rank(cov(rank(m.raw_h), rank(m.raw_v), 5)))
define("alpha018", "alpha101", lambda m: -rank(k1(m, std((m.c - m.o).abs(), 5) + (m.c - m.o)) + corr(m.c, m.o, 10)))
define("alpha019", "alpha101", lambda m: -np.sign(m.c / delay(m.c, 7) - 1) * (1 + rank(1 + total(m.ret, 250))))
define("alpha020", "alpha101", lambda m: -rank(k1(m, m.o - delay(m.h, 1))) * rank(k1(m, m.o - delay(m.c, 1)))
       * rank(k1(m, m.o - delay(m.l, 1))))
define("alpha022", "alpha101", lambda m: -delta(corr(m.h, m.w, 5), 5) * rank(k1(m, std(m.c, 20))))
define("alpha023", "alpha101", lambda m: k1(m, (-delta(m.h, 2)).where(mean(m.h, 20) < m.h, 0.0)))
define("alpha025", "alpha101", lambda m: rank(-m.ret * mean(m.amount.fillna(0), 20) * k1(m, m.vwap) * k1(m, m.h - m.c)))
define("alpha026", "alpha101", lambda m: -tsmax(corr(tsrank(m.w, 5), tsrank(m.h, 5), 5), 3))
define("alpha030", "alpha101", lambda m: (1 - rank(np.sign(delta(m.c, 1)) + np.sign(delay(delta(m.c, 1), 1))
                                                   + np.sign(delay(delta(m.c, 1), 2)))) * ratio(total(m.w, 5), total(m.w, 20)))
define("alpha033", "alpha101", lambda m: rank(m.o / m.c - 1))
define("alpha034", "alpha101", lambda m: rank((1 - rank(ratio(std(m.ret, 2), std(m.ret, 5)))) + (1 - rank(k1(m, delta(m.c, 1))))))
define("alpha035", "alpha101", lambda m: tsrank(m.w, 32) * (1 - tsrank(m.c + m.h - m.l, 16)) * (1 - tsrank(m.ret, 32)))
define("alpha037", "alpha101", lambda m: rank(corr(delay(m.o - m.c, 1), m.c, 200)) + rank(k1(m, m.o - m.c)))
define("alpha038", "alpha101", lambda m: -rank(tsrank(m.c, 10)) * rank(m.c / m.o))
define("alpha040", "alpha101", lambda m: -rank(k1(m, std(m.h, 10))) * corr(m.h, m.w, 10))
define("alpha041", "alpha101", lambda m: k1(m, np.sqrt(m.h * m.l) - m.vwap))
define("alpha042", "alpha101", lambda m: rank(m.raw_vwap - m.raw_c) / rank(m.raw_vwap + m.raw_c))
# adv20 is the 20-day mean traded amount (Kakushadze A.2), as in alpha025, not mean volume.
define("alpha043", "alpha101", lambda m: tsrank(ratio(m.w, mean(m.amount.fillna(0), 20)), 20) * tsrank(-delta(m.c, 7), 8))
define("alpha044", "alpha101", lambda m: -corr(m.h, rank(m.raw_v), 5))
define("alpha045", "alpha101", lambda m: -(rank(k1(m, mean(delay(m.c, 5), 20))) * corr(m.c, m.w, 2)
                                             * rank(corr(total(m.c, 5), total(m.c, 20), 2))))
define("alpha050", "alpha101", lambda m: -tsmax(rank(corr(rank(m.raw_v), rank(m.raw_vwap), 5)), 5))
define("alpha052", "alpha101", lambda m: k1(m, -tsmin(m.l, 5) + delay(tsmin(m.l, 5), 5))
       * rank((total(m.ret, 240) - total(m.ret, 20)) / 220) * tsrank(m.w, 5))
define("alpha053", "alpha101", lambda m: -delta(ratio((m.c - m.l) - (m.h - m.c), m.c - m.l), 9))
define("alpha054", "alpha101", lambda m: ratio(-(m.l - m.c) * m.o ** 5, (m.l - m.h) * m.c ** 5))
define("alpha055", "alpha101", lambda m: -corr(rank(ratio(m.c - tsmin(m.l, 12), tsmax(m.h, 12) - tsmin(m.l, 12))),
                                               rank(m.raw_v), 6))
define("alpha060", "alpha101", lambda m: -(2 * rank(ratio((m.c - m.l) - (m.h - m.c), m.h - m.l) * m.raw_v)
                                              - rank(argmax(m.c, 10))))
define("alpha101", "alpha101", lambda m: (m.raw_c - m.raw_o) / ((m.raw_h - m.raw_l) + 0.001))


# Guotai Junan short-horizon alphas (GTJA 191), daily-bar formulas not already listed above.
define("gtja001", "gtja191", lambda m: -corr(rank(delta(np.log(m.w.where(m.w > 0)), 1)), rank((m.c - m.o) / m.o), 6))
define("gtja002", "gtja191", lambda m: -delta(ratio((m.c - m.l) - (m.h - m.c), m.h - m.l), 1))
define("gtja003", "gtja191", lambda m: k1(m, total(where(m.c == delay(m.c, 1), m.c * 0, m.c - where(
    m.c > delay(m.c, 1), np.minimum(m.l, delay(m.c, 1)), np.maximum(m.h, delay(m.c, 1)))), 6)))
define("gtja006", "gtja191", lambda m: -rank(np.sign(delta(m.o * 0.85 + m.h * 0.15, 4))))
define("gtja007", "gtja191", lambda m: (rank(k1(m, tsmax(m.vwap - m.c, 3))) + rank(k1(m, tsmin(m.vwap - m.c, 3))))
       * rank(delta(m.w, 3) / m.k))
define("gtja008", "gtja191", lambda m: -rank(k1(m, delta((m.h + m.l) / 2 * 0.2 + m.vwap * 0.8, 4))))
define("gtja014", "gtja191", lambda m: k1(m, m.c - delay(m.c, 5)))
define("gtja015", "gtja191", lambda m: m.o / delay(m.c, 1) - 1)
define("gtja024", "gtja191", lambda m: k1(m, sma(m.c - delay(m.c, 5), 5, 1)))
define("gtja029", "gtja191", lambda m: (m.c / delay(m.c, 6) - 1) * m.raw_v)
define("gtja065", "gtja191", lambda m: mean(m.c, 6) / m.c)
define("gtja080", "gtja191", lambda m: ratio(m.w - delay(m.w, 5), delay(m.w, 5)))
define("gtja106", "gtja191", lambda m: k1(m, m.c - delay(m.c, 20)))
define("gtja126", "gtja191", lambda m: (m.raw_c + m.raw_h + m.raw_l) / 3)
define("gtja160", "gtja191", lambda m: k1(m, sma(std(m.c, 20).where(m.c <= delay(m.c, 1), 0.0), 20, 1)))
define("gtja161", "gtja191", lambda m: k1(m, mean(np.maximum(np.maximum(m.h - m.l, (delay(m.c, 1) - m.h).abs()),
                                                            (delay(m.c, 1) - m.l).abs()), 12)))
define("gtja167", "gtja191", lambda m: k1(m, total(delta(m.c, 1).clip(lower=0), 12)))
define("gtja175", "gtja191", lambda m: k1(m, mean(np.maximum(np.maximum(m.h - m.l, (delay(m.c, 1) - m.h).abs()),
                                                            (delay(m.c, 1) - m.l).abs()), 6)))
define("gtja189", "gtja191", lambda m: k1(m, mean((m.c - mean(m.c, 6)).abs(), 6)))
define("gtja100", "gtja191", lambda m: std(m.w, 20) / m.k)
define("gtja109", "gtja191", lambda m: ratio(sma(m.h - m.l, 10, 2), sma(sma(m.h - m.l, 10, 2), 10, 2)))
define("gtja040", "gtja191", lambda m: ratio(total(m.w.where(m.c > delay(m.c, 1), 0), 26),
                                            total(m.w.where(m.c <= delay(m.c, 1), 0), 26)))
define("gtja046", "gtja191", lambda m: (mean(m.c, 3) + mean(m.c, 6) + mean(m.c, 12) + mean(m.c, 24)) / (4 * m.c))


def raw_factors(m, names=None):
    """Yield (name, family, wide frame) for every registered factor, computed lazily."""
    for name in (names or REGISTRY):
        family, fn = REGISTRY[name]
        value = fn(m)
        if value is None:
            continue
        yield name, family, finite(value.astype(float))
