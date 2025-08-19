# utils.py
# Backtest runner using your DoL detector on MEXC 5m data
# Safe timezone handling; returns structured results (does not crash on tz-aware indices).

from datetime import datetime, timedelta
import pandas as pd

from mexc_client import get_klines as get_klines_mexc
from strategy import DoLDetector, TradeManager
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan

# Initialize lightweight models for backtest (separate from live singletons)
_bandit_bt = PolicyArms()
_knn_bt = PatternKNN()
_bayes_bt = BayesModel()
_bans_bt = CFPBan()
_detector_bt = DoLDetector(_bandit_bt)

def _normalize_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df.index is tz-aware UTC.
    Never tz_localize on an already-aware index.
    """
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def backtest_strategy(days: int = 7):
    """
    Simulate last `days` on 5m using MEXC data.
    Returns a list of trade dicts.
    """
    end_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    start_utc = end_utc - pd.Timedelta(days=days)

    # Pull a large chunk (MEXC limit ~1000), then slice the window
    df5 = get_klines_mexc("5", limit=1000, include_partial=False)
    if df5 is None or df5.empty:
        return []

    df5 = _normalize_index_utc(df5)
    df5 = df5.loc[(df5.index >= start_utc) & (df5.index <= end_utc)]
    if len(df5) < 250:
        return []

    tm_bt = TradeManager()
    results = []

    # Walk forward candle-by-candle
    for i in range(250, len(df5)):
        view = df5.iloc[: i + 1]
        sig = _detector_bt.find(view, _bayes_bt, _knn_bt, _bandit_bt, _bans_bt)
        if not sig:
            continue

        # open pseudo-trade
        st = tm_bt.open_from_signal(sig)
        if not st:
            continue

        # Minimal execution sim: check TP1/SL within next N bars (you can expand to TP2 if you like)
        outcome = "MISS"
        entry = sig.entry_ref
        sl = sig.stop_px
        tp = sig.tp1_px

        look_ahead = df5.iloc[i : min(i + 48, len(df5))]  # next ~4h on 5m
        for _, row in look_ahead.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if sig.side == "BUY":
                if low <= sl:
                    outcome = "SL"
                    break
                if high >= tp:
                    outcome = "TP"
                    break
            else:  # SELL
                if high >= sl:
                    outcome = "SL"
                    break
                if low <= tp:
                    outcome = "TP"
                    break

        results.append({
            "time": str(view.index[-1]),
            "symbol": "BTCUSDT",
            "side": sig.side,
            "entry": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "result": outcome
        })

    return results
