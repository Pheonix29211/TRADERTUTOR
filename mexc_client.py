# mexc_client.py
# MEXC v3 spot klines utilities
# - get_klines: latest N klines (1/5/15/30/60 minutes)
# - get_klines_between: robust range pager (startTime+endTime) for backtests
# Returns pandas.DataFrame with UTC index and columns: open, high, low, close, volume

from __future__ import annotations

import os
import math
from typing import Optional, List

import pandas as pd
import requests

__all__ = ["get_klines", "get_klines_between"]

# ----------------- Config / constants -----------------
try:
    from config import SYMBOL, TZ_NAME  # TZ_NAME not used here; kept for completeness
except Exception:
    SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
    TZ_NAME = os.getenv("TZ_NAME", "Asia/Kolkata")

MEXC_V3_URL  = "https://api.mexc.com/api/v3/klines"
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
    """Normalize timeframe input to a MEXC interval string."""
    if tf is None:
        return "5m"
    tf = str(tf).strip().lower()
    return _MEXC_TF_MAP.get(tf, tf if tf.endswith(("m", "h")) else f"{tf}m")

def _bar_minutes(iv: str) -> int:
    return {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}.get(iv, 5)

def _normalize_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure tz-aware UTC index (avoids tz-localize errors elsewhere)."""
    if df is None or df.empty:
        return df
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    return df

def _df_from_raw(raw: List[list], include_partial: bool, start_ms: int | None = None, end_ms: int | None = None) -> Optional[pd.DataFrame]:
    """
    Build dataframe from raw list rows; handle numeric types, UTC index, and partial-bar trimming.
    If start_ms/end_ms are provided, trims to [start_ms, end_ms).
    If include_partial=False, drops any row whose close_time exceeds the limit (end_ms or now).
    """
    if not isinstance(raw, list) or len(raw) == 0:
        return None

    cols_full = [
        "open_time","open","high","low","close","volume",
        "close_time","quote_volume","trade_count",
        "taker_buy_base","taker_buy_quote","ignore"
    ]
    n = len(raw[0])
    df = pd.DataFrame(raw, columns=cols_full[:n])

    # numeric consistency
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # time columns → UTC
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    if "close_time" in df.columns:
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    # trim to bounds (by open_time)
    if start_ms is not None:
        df = df[df["open_time"].astype("int64") // 10**6 >= start_ms]
    if end_ms is not None:
        df = df[df["open_time"].astype("int64") // 10**6 <  end_ms]

    # partial last bar removal:
    # - if end_ms given: drop rows whose close_time > end_ms
    # - else: drop rows whose close_time > now
    if not include_partial and "close_time" in df.columns and len(df):
        limit_ts = pd.to_datetime(end_ms, unit="ms", utc=True) if end_ms is not None else pd.Timestamp.now(tz="UTC")
        df = df[df["close_time"] <= limit_ts]

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
        return _df_from_raw(raw, include_partial=include_partial)
    except Exception:
        return None

def get_klines_between(tf: str, start_utc, end_utc, include_partial: bool = False) -> Optional[pd.DataFrame]:
    """
    Robust range fetcher using startTime+endTime paging.
    Walks forward window-by-window, retries blanks, guarantees progress.
    Trims to [start_utc, end_utc) and optionally drops the last partial kline.
    """
    try:
        iv = _ensure_iv(tf)
        mins   = _bar_minutes(iv)
        bar_ms = mins * 60 * 1000

        # normalize bounds to UTC ms
        s = pd.Timestamp(start_utc)
        if s.tz is None: s = s.tz_localize("UTC")
        else: s = s.tz_convert("UTC")
        e = pd.Timestamp(end_utc)
        if e.tz is None: e = e.tz_localize("UTC")
        else: e = e.tz_convert("UTC")

        start_ms = int(s.timestamp() * 1000)
        end_ms   = int(e.timestamp() * 1000)

        # page sizing — default 800 bars; fallback to 400 if under-returning
        def _run_pager(page_bars: int) -> Optional[pd.DataFrame]:
            page_bars  = max(200, min(950, int(page_bars)))
            page_limit = min(1000, page_bars)

            cursor  = start_ms
            last_ot = None
            out: List[list] = []
            empty_stalls = 0

            while cursor < end_ms:
                page_end = min(end_ms, cursor + page_bars * bar_ms)
                params = {
                    "symbol": SYMBOL,
                    "interval": iv,
                    "startTime": int(cursor),
                    "endTime": int(page_end - 1),
                    "limit": int(page_limit),
                }

                # up to 3 quick retries for transient issues
                ok = False
                raw = None
                for _ in range(3):
                    try:
                        r = requests.get(MEXC_V3_URL, params=params, timeout=TIMEOUT_SECS)
                        if r.status_code == 200:
                            raw = r.json()
                            ok = True
                            break
                    except Exception:
                        pass
                if not ok:
                    # nudge ahead one bar to avoid infinite loops
                    cursor += bar_ms
                    empty_stalls += 1
                    if empty_stalls > 5:
                        break
                    continue

                if not isinstance(raw, list) or len(raw) == 0:
                    # API sometimes returns empty for narrow slices—nudge ahead
                    cursor += bar_ms
                    empty_stalls += 1
                    if empty_stalls > 20:
                        break
                    continue

                advanced = False
                for row in raw:
                    ot = int(row[0])  # open_time ms
                    if ot < start_ms or ot >= end_ms:
                        continue
                    if last_ot is not None and ot <= last_ot:
                        continue
                    out.append(row)
                    last_ot = ot
                    advanced = True

                # advance cursor
                if advanced and last_ot is not None:
                    cursor = last_ot + bar_ms
                    empty_stalls = 0
                else:
                    # no progress in this window — jump one full window to force movement
                    cursor = min(end_ms, cursor + page_bars * bar_ms)
                    empty_stalls += 1
                    if empty_stalls > 10:
                        break

            if not out:
                return None

            return _df_from_raw(out, include_partial=include_partial, start_ms=start_ms, end_ms=end_ms)

        # first pass: larger pages
        page_bars_env = int(os.getenv("MEXC_PAGE_BARS", "800"))
        df = _run_pager(page_bars_env)
        if df is None:
            return None

        # sanity: expected bars
        expected = int(math.floor((end_ms - start_ms) / bar_ms))
        if expected > 0 and len(df) < int(0.9 * expected):
            # fallback to smaller pages (some MEXC edges truncate bigger pages)
            df2 = _run_pager(400)
            if df2 is not None and len(df2) > len(df):
                df = df2

        return df
    except Exception:
        return None
