"""Features at close t; labels open(t+1) -> open(t+h+1), exchange sessions."""
import numpy as np
import pandas as pd

FEATURES = ["ret1", "ret5", "ret20", "ret60", "vol20", "range", "intraday",
            "gap", "volume_ratio", "price_ma20"]


def prepare(bundle, config):
    b = bundle.bars.copy()
    g = b.groupby("symbol", sort=False)
    # Causal total-return proxy. No back-adjusted price levels or future anchor.
    b["tr_close"] = (b.close / b.preclose).groupby(b.symbol).cumprod() * 100
    b["tr_open"] = b.tr_close * b.open / b.close
    for n in (1, 5, 20, 60):
        b[f"ret{n}"] = b.tr_close / b.groupby("symbol").tr_close.shift(n) - 1
    b["vol20"] = b.groupby("symbol").ret1.transform(lambda x: x.rolling(20, min_periods=20).std())
    b["adv20"] = g.amount.transform(lambda x: x.rolling(20, min_periods=20).mean())
    b["range"] = (b.high - b.low) / b.close
    b["intraday"] = b.close / b.open - 1
    b["gap"] = b.open / b.preclose - 1
    v20 = g.volume.transform(lambda x: x.rolling(20, min_periods=20).mean())
    b["volume_ratio"] = b.volume / v20.replace(0, np.nan)
    ma = b.groupby("symbol").tr_close.transform(lambda x: x.rolling(20, min_periods=20).mean())
    b["price_ma20"] = b.tr_close / ma - 1
    listed = b.symbol.map(bundle.instruments.set_index("symbol").listed)
    base = ((b.date - listed).dt.days >= config.min_listing_days) & b.tradable.eq(1) & b.is_st.eq(0)
    base &= b.adv20.ge(config.min_adv) & np.isfinite(b[FEATURES]).all(axis=1)
    # Date-t liquidity selection, not today's surviving constituents or full-period ADV.
    rank_adv = b.adv20.where(base).groupby(b.date).rank(method="first", ascending=False)
    b["eligible"] = base & rank_adv.le(config.universe_size)
    g = b.groupby("symbol", sort=False)
    # Marked returns do not condition training inclusion on FUTURE tradability.
    # Suspended endpoints use the explicit carried mark; execution may still fail.
    entry = g.tr_open.shift(-1)
    exit_ = g.tr_open.shift(-(config.horizon + 1))
    maturity = pd.Series(bundle.calendar, index=bundle.calendar).shift(-(config.horizon + 1))
    b["label_end"] = b.date.map(maturity)
    removal = b.symbol.map(bundle.instruments.set_index("symbol").delisted)
    # Include terminal losses in labels; do not drop delisting tails as NaNs.
    terminal = removal.notna() & b.label_end.notna() & b.label_end.ge(removal)
    res = bundle.delistings.set_index("symbol")
    resolved = terminal & b.symbol.isin(res.index)
    exit_ = exit_.mask(terminal & ~resolved, 0.0)
    b["label"] = exit_ / entry - 1
    if resolved.any():
        # Holders received acquirer shares or cash instead: value that in raw prices at label end.
        # Before the acquirer trades, the exchange carries the last close, as in the ledger.
        r = res.reindex(b.symbol[resolved])
        acquirer_open = b.set_index(["date", "symbol"]).open.reindex(
            pd.MultiIndex.from_arrays([b.label_end[resolved], r.acquirer])).to_numpy()
        carried = g.close.transform("last")[resolved].to_numpy()
        exchanged = np.where(np.isfinite(acquirer_open), r.ratio.to_numpy() * acquirer_open, carried)
        received = np.where(r.kind.eq("share_exchange"), exchanged, r.cash_per_share.to_numpy())
        b.loc[resolved, "label"] = received / g.open.shift(-1)[resolved].to_numpy() - 1
    b["target"] = b.label.where(b.eligible).groupby(b.date).rank(pct=True) - 0.5
    # Cross-sectional transforms use only the eligible names visible at this close.
    for name in FEATURES:
        b["x_" + name] = b[name].where(b.eligible).groupby(b.date).rank(pct=True) - 0.5
    return b


def qlib_frame(panel):
    """Native Qlib DataHandlerLP.from_df input; no Qlib exchange assumptions."""
    frame = panel.loc[panel.eligible].set_index(["date", "symbol"]).sort_index()
    frame.index.names = ["datetime", "instrument"]
    x = frame[["x_" + x for x in FEATURES]].copy()
    x.columns = pd.MultiIndex.from_product([["feature"], FEATURES])
    x[("label", "LABEL0")] = frame.target
    return x
