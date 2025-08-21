# utils.py
# Backtest utilities:
# - Uses mexc_client.get_klines_between with robust paging
# - Lock-free simulation windows
# - BASELINE tuning matches live (looser for more entries)
# - SL from P&L dollars (qty); TP1/TP2 from price points
# - Results appended to /data/trades.json for /logs

from __future__ import annotations

import os, json
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd

from mexc_client import get_klines_between
from strategy import DoLDetector
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan

def _resolve_data_dir():
    candidates = [
        os.getenv("DATA_DIR"),
        "/data",
        "/tmp/data",
        os.path.join(os.path.dirname(__file__), "data"),
    ]
    for p in candidates:
        if not p: continue
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            t = Path(p) / ".write_test"
            t.write_text("ok", encoding="utf-8")
            t.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    raise RuntimeError("No writable data directory found")

DATA_DIR = _resolve_data_dir()
TRADES_PATH = os.path.join(DATA_DIR, "trades.json")
if not Path(TRADES_PATH).exists():
    Path(TRADES_PATH).write_text("[]", encoding="utf-8")

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

# ---------- detector context ----------
_bandit_bt = PolicyArms()
_knn_bt    = PatternKNN()
_bayes_bt  = BayesModel()
_bans_bt   = CFPBan()
_detector_bt = DoLDetector(_bandit_bt)

# ---------- risk / sizing ----------
QTY_BTC      = float(os.getenv("QTY_BTC",      "0.101"))
RISK_DOLLARS = float(os.getenv("RISK_DOLLARS", "40"))
TP1_POINTS   = float(os.getenv("TP1_POINTS",   "600"))
TP2_POINTS   = float(os.getenv("TP2_POINTS",   "1500"))

def _normalize_index_utc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty: return df
    if df.index.tz is None: df.index = df.index.tz_localize("UTC")
    else: df.index = df.index.tz_convert("UTC")
    return df

def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")

def _baseline_tuning():
    # Match bot.py BASELINE (looser for more entries)
    return {"disp_mult": 0.80, "fvg_mult": 0.85, "return_mult": 1.15, "conf_adj": -0.09}

def backtest_strategy(days: int = 7, tf: str = "5") -> List[Dict[str, Any]]:
    end_utc = _utc_now()
    start_utc = end_utc - pd.Timedelta(days=days)

    df = get_klines_between(tf, start_utc, end_utc, include_partial=False)
    if df is None or df.empty: return []

    df = _normalize_index_utc(df)
    if len(df) < 300: return []

    results: List[Dict[str, Any]] = []

    tf_minutes = {"1":"1","5":"5","15":"15","30":"30","60":"60"}.get(tf, "5")
    tfm = int(tf_minutes)
    look_bars = max(12, int(8 * 60 / tfm))  # 8h window

    risk_dist = RISK_DOLLARS / max(1e-9, QTY_BTC)
    tp1_dist  = TP1_POINTS
    tp2_dist  = TP2_POINTS

    tuning = _baseline_tuning()

    for i in range(250, len(df)):
        window = df.iloc[: i + 1]
        try:
            sig = _detector_bt.find(window, _bayes_bt, _knn_bt, _bandit_bt, _bans_bt, tuning=tuning)
        except TypeError:
            sig = _detector_bt.find(window, _bayes_bt, _knn_bt, _bandit_bt, _bans_bt)
        if not sig: continue

        last_close = float(window["close"].iloc[-1])
        side  = getattr(sig, "side", "BUY")
        entry = float(getattr(sig, "entry_ref", last_close))

        stop_px = getattr(sig, "stop_px", None)
        tp1_px  = getattr(sig, "tp1_px",  None)
        tp2_px  = getattr(sig, "tp2_px",  None)

        if stop_px is None:
            stop_px = entry - risk_dist if side == "BUY" else entry + risk_dist
        if tp1_px is None:
            tp1_px = entry + tp1_dist  if side == "BUY" else entry - tp1_dist
        if tp2_px is None:
            tp2_px = entry + tp2_dist  if side == "BUY" else entry - tp2_dist

        # direction sanity
        if side == "BUY":
            if stop_px >= entry: stop_px = entry - abs(risk_dist)
            if tp1_px  <= entry: tp1_px  = entry + abs(tp1_dist)
            if tp2_px  <= entry: tp2_px  = entry + abs(tp2_dist)
        else:
            if stop_px <= entry: stop_px = entry + abs(risk_dist)
            if tp1_px  >= entry: tp1_px  = entry - abs(tp1_dist)
            if tp2_px  >= entry: tp2_px  = entry - abs(tp2_dist)

        stop_px = float(stop_px); tp1_px = float(tp1_px); tp2_px = float(tp2_px)

        look_ahead = df.iloc[i : min(i + look_bars, len(df))]
        outcome = "MISS"
        if side == "BUY":
            for _, row in look_ahead.iterrows():
                high = float(row["high"]); low = float(row["low"])
                if low <= stop_px: outcome = "SL"; break
                if high >= tp2_px: outcome = "TP2"; break
                if high >= tp1_px: outcome = "TP1"; break
        else:
            for _, row in look_ahead.iterrows():
                high = float(row["high"]); low = float(row["low"])
                if high >= stop_px: outcome = "SL"; break
                if low <= tp2_px: outcome = "TP2"; break
                if low <= tp1_px: outcome = "TP1"; break

        row = {
            "time": str(window.index[-1]),
            "symbol": "BTCUSDT",
            "side": side,
            "entry": entry,
            "sl": stop_px,
            "tp": tp2_px if outcome == "TP2" else tp1_px,
            "result": outcome,
            "tf": f"{tfm}m",
            "source": "backtest"
        }
        results.append(row)
        _append_log(row)

    return results

def backtest_strategy_debug(days: int = 7, tf: str = "5") -> Dict[str, Any]:
    end_utc = _utc_now()
    start_utc = end_utc - pd.Timedelta(days=days)

    df = get_klines_between(tf, start_utc, end_utc, include_partial=False)
    if df is None or df.empty:
        return {'bars': 0, 'windows_tested': 0, 'signals_found': 0, 'first_ts': None, 'last_ts': None}

    df = _normalize_index_utc(df)
    bars = len(df)
    if bars < 300:
        return {'bars': bars, 'windows_tested': 0, 'signals_found': 0,
                'first_ts': str(df.index[0]) if bars else None,
                'last_ts': str(df.index[-1]) if bars else None}

    tuning = _baseline_tuning()
    tested = 0; signals = 0
    for i in range(250, len(df)):
        tested += 1
        window = df.iloc[: i + 1]
        try:
            sig = _detector_bt.find(window, _bayes_bt, _knn_bt, _bandit_bt, _bans_bt, tuning=tuning)
        except TypeError:
            sig = _detector_bt.find(window, _bayes_bt, _knn_bt, _bandit_bt, _bans_bt)
        if sig: signals += 1

    return {
        'bars': bars,
        'windows_tested': tested,
        'signals_found': signals,
        'first_ts': str(df.index[0]),
        'last_ts': str(df.index[-1])
    }
