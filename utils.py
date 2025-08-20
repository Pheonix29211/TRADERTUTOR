# utils.py
# Backtest runner using your DoL detector on MEXC 5m data
# - Uses paged range fetcher to cover full window
# - Safe timezone (no tz_localize on tz-aware)
# - Persists to /data/trades.json so /logs shows results

from __future__ import annotations

import os
import json
import time
from typing import List, Dict, Any

import pandas as pd

from mexc_client import get_klines_between
from strategy import DoLDetector, TradeManager
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan

# Paths (match bot.py)
/* keep consistent with bot.py */
DATA_DIR = os.getenv("DATA_DIR", "/data")
os.makedirs(DATA_DIR, exist_ok=True)
TRADES_PATH = os.path.join(DATA_DIR, "trades.json")
if not os.path.exists(TRADES_PATH):
    with open(TRADES_PATH, "w", encoding="utf-8") as f:
        json.dump([], f)

# Backtest-local models (separate from live)
_bandit_bt = PolicyArms()
_knn_bt = PatternKNN()
_bayes_bt = BayesModel()
_bans_bt = CFPBan()
_detector_bt = DoLDetector(_bandit_bt)

def _normalize_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _load_logs() -> List[Dict[str, Any]]:
    try:
        with open(TRADES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_logs(rows: List[Dict[str, Any]]):
    try:
        with open(TRADES_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_log(row: Dict[str, Any]):
    rows = _load_logs()
    rows.append(row)
    _save_logs(rows)

def backtest_strategy(days: int = 7) -> List[Dict[str, Any]]:
    """
    Simulate last `days` of BTCUSDT on 5m using DoLDetector.
    Returns results and appends them to /data/trades.json (source="backtest").
    """
    end_utc = pd.Timestamp.utcnow().tz_localize("UTC")
    start_utc = end_utc - pd.Timedelta(days=days)

    # Range fetch to truly cover the whole window (more than 1000 bars)
    df5 = get_klines_between("5", start_utc, end_utc, include_partial=False)
    if df5 is None or df5.empty:
        return []

    df5 = _normalize_index_utc(df5)
    # require history
    if len(df5) < 300:
        # not enough bars to build context
        return []

    tm_bt = TradeManager()
    results: List[Dict[str, Any]] = []

    # Walk forward candle-by-candle
    for i in range(250, len(df5)):
        window = df5.iloc[: i + 1]
        sig = _detector_bt.find(window, _bayes_bt, _knn_bt, _bandit_bt, _bans_bt)
        if not sig:
            continue

        st = tm_bt.open_from_signal(sig)
        if not st:
            continue

        entry = float(getattr(sig, "entry_ref", window["close"].iloc[-1]))
        sl    = float(getattr(sig, "stop_px",  entry - 50.0))
        tp1   = float(getattr(sig, "tp1_px",   entry + 600.0))
        tp2   = float(getattr(sig, "tp2_px",   entry + 1500.0))
        side  = getattr(sig, "side", "BUY")

        look_ahead = df5.iloc[i : min(i + 48, len(df5))]  # next ~4h
        outcome = "MISS"

        if side == "BUY":
            for _, row in look_ahead.iterrows():
                high = float(row["high"]); low = float(row["low"])
                if low <= sl:
                    outcome = "SL"; break
                if high >= tp2:
                    outcome = "TP2"; break
                if high >= tp1:
                    outcome = "TP1"; break
        else:
            for _, row in look_ahead.iterrows():
                high = float(row["high"]); low = float(row["low"])
                if high >= sl:
                    outcome = "SL"; break
                if low <= tp2:
                    outcome = "TP2"; break
                if low <= tp1:
                    outcome = "TP1"; break

        row = {
            "time": str(window.index[-1]),
            "symbol": "BTCUSDT",
            "side": side,
            "entry": entry,
            "sl": sl,
            "tp": tp2 if outcome == "TP2" else tp1,
            "result": outcome,
            "source": "backtest"
        }
        results.append(row)
        _append_log(row)

    return results
