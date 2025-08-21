# mexc_client.py
# MEXC v3 spot klines utilities for SpiralBot
# - get_klines: latest N klines (1/5/15/30/60 minutes)
# - get_klines_between: robust range pager (startTime-only) for backtests
# Returns pandas.DataFrame with UTC index and columns: open, high, low, close, volume

from __future__ import annotations

import os
import math
from typing import Optional

import pandas as pd
import requests

# ----------------- Config / constants -----------------
try:
    from config import SYMBOL, TZ_NAME  # TZ_NAME not required here but kept for completeness
except Exception:
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    TZ_NAME = os.getenv("TZ_NAME", "Asia/Kolkata")

MEXC_V3_URL = "https://api.mexc.com/api/v3/klines"
TIMEOUT_SECS = 12

# Map our tf strings to MEXC intervals
_MEXC_TF_MAP = {
    "1": "1m", "1m": "1m",
    "5": "5m", "5m": "5m",
    "15": "15m", "15m": "15m",
    "30": "30m", "30m": "30m",
    "60": "1h", "1h": "1h"
}

# ----------------- Helpers -----------------
def _ensure_iv(tf: str) -> str:
    """Normalize timeframe input to MEXC interval."""
    if tf is None:
        return "5m"
    tf = str(tf).strip().lower()
    return _MEXC_TF_MAP.get(tf, tf if tf.endswith(("m", "h")) else f"{tf}m")

def _bar_minutes(iv: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(iv, 5)

def _normalize_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    # open_time is already UTC; ensure tz-aware
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _df_from_raw(raw, include_partial: bool, start_ms: int = None, end_ms: int = None) -> Optional[pd.DataFrame]:
    """Build dataframe from raw list rows; handle types, UTC index, and partial-bar trimming."""
    if not isinstance(raw, list) or len(raw) == 0:
        return None

    cols_full = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trade_count",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    n = len(raw[0])
    df = pd.DataFrame(raw, columns=cols_full[:n])

    # numeric
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # time columns -> UTC
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # optional trimming for range bounds
    if start_ms is not None:
        df = df[df["open_time"].astype("int64") // 10**6 >= start_ms]
    if end_ms is not None:
        df = df[df["open_time"].astype("int64") // 10**6 <  end_ms]

    # partial last bar removal (if requested and we have close_time)
    if not include_partial and "close_time" in df.columns and len(df):
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        last_close_ms = int(df["close_time"].iloc[-1].timestamp() * 1000)
        if last_close_ms > now_ms:
            df = df.iloc[:-1]

    df = df.set_index("open_time").sort_index()
    df = df[["open","high","low","close","volume"]].dropna()
    return _normalize_utc_index(df)

# ----------------- Public API -----------------
def get_klines(tf: str, limit: int = 500, include_partial: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch the latest klines window (most recent N bars).
    Returns DataFrame indexed by UTC open_time with: open, high, low, close, volume.
    """
    try:
        iv = _ensure_iv(tf)
        limit = int(max(1, min(int(limit), 1000)))  # MEXC max=1000

        params = {"symbol": SYMBOL, "interval": iv, "limit": limit}
        r = requests.get(MEXC_V3_URL, params=params, timeout=TIMEOUT_SECS)
        if r.status_code != 200:
            return None

        raw = r.json()
        df = _df_from_raw(raw, include_partial=include_partial)
        return df
    except Exception:
        return None

def get_klines_between(tf: str, start_utc, end_utc, include_partial: bool = False) -> Optional[pd.DataFrame]:
    """
    Robust range fetcher using startTime-only paging (avoids under-returns).
    Trims to [start_utc, end_utc) and optionally drops partial last kline.
    """
    try:
        iv = _ensure_iv(tf)
        mins = _bar_minutes(iv)
        bar_ms = mins * 60 * 1000

        # Normalize bounds to UTC ms
        s = pd.Timestamp(start_utc)
        if s.tz is None: s = s.tz_localize("UTC")
        else: s = s.tz_convert("UTC")
        e = pd.Timestamp(end_utc)
        if e.tz is None: e = e.tz_localize("UTC")
        else: e = e.tz_convert("UTC")

        start_ms = int(s.timestamp() * 1000)
        end_ms   = int(e.timestamp() * 1000)

        def _page(limit: int) -> Optional[pd.DataFrame]:
            cursor = start_ms
            last_ot = None
            out = []
            while cursor < end_ms:
                params = {"symbol": SYMBOL, "interval": iv, "limit": int(limit), "startTime": int(cursor)}
                try:
                    r = requests.get(MEXC_V3_URL, params=params, timeout=TIMEOUT_SECS)
                except Exception:
                    break
                if r.status_code != 200:
                    break
                data = r.json()
                if not isinstance(data, list) or len(data) == 0:
                    break

                advanced = False
                for row in data:
                    ot = int(row[0])  # open_time ms
                    if last_ot is not None and ot <= last_ot:
                        continue
                    if ot >= end_ms:
                        advanced = True
                        break
                    out.append(row)
                    last_ot = ot
                    advanced = True

                # advance cursor (nudge at least one bar)
                if last_ot is not None:
                    cursor = last_ot + bar_ms
                else:
                    cursor += bar_ms

                # if fewer than limit returned, probably near end; avoid tight loop
                if len(data) < int(limit) and not advanced:
                    break

            if not out:
                return None
            # Build DF with bound trimming & partial handling
            return _df_from_raw(out, include_partial=include_partial, start_ms=start_ms, end_ms=end_ms)

        # First pass with big pages; if under-returning, fallback to smaller pages
        df = _page(1000)
        if df is None:
            return None

        expected = int(math.floor((end_ms - start_ms) / bar_ms))
        if expected > 0 and len(df) < 0.8 * expected:
            df2 = _page(500)
            if df2 is not None and len(df2) > len(df):
                df = df2

        return df
    except Exception:
        return None
