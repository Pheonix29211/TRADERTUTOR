# mexc_client.py
# Robust MEXC V3 spot klines fetcher with strict TZ handling
import os
import requests
import pandas as pd
from typing import Optional

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TZ_NAME = os.getenv("TZ", "Asia/Kolkata")  # your desired local tz, e.g. Asia/Kolkata

MEXC_V3_URL = "https://api.mexc.com/api/v3/klines"
_MEXC_TF_MAP = {"1": "1m", "5": "5m", "15": "15m", "30": "30m", "60": "1h", "1h": "1h", "5m": "5m", "1m": "1m"}

# ---------- TZ helpers (bulletproof against tz-aware/naive) ----------
def _to_utc_index(series_ms) -> pd.DatetimeIndex:
    """
    Convert milliseconds to a tz-aware UTC DatetimeIndex safely.
    Never tz_localize on an already-aware index.
    """
    # pd.to_datetime(..., utc=True) ALWAYS returns tz-aware UTC
    idx = pd.to_datetime(series_ms, unit="ms", utc=True)
    # idx is tz-aware UTC here; do NOT tz_localize again anywhere else
    return idx

def _convert_index(idx: pd.DatetimeIndex, tz_name: str) -> pd.DatetimeIndex:
    """
    Convert an already tz-aware index to the requested timezone.
    If idx is naive (shouldn't be), localize; else convert.
    """
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert(tz_name)
    return idx.tz_convert(tz_name)

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Keep only the essentials in correct dtypes
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

# ---------- Public API ----------
def get_klines(tf: str, limit: int = 200, include_partial: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch klines from MEXC V3.
    Returns DataFrame:
      index = open_time localized to TZ_NAME,
      cols  = open, high, low, close, volume (floats)
    """
    try:
        iv = _MEXC_TF_MAP.get(tf, tf)
        params = {"symbol": SYMBOL, "interval": iv, "limit": int(limit)}
        r = requests.get(MEXC_V3_URL, params=params, timeout=12)
        if r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None

        # MEXC may return 8–12 fields; map flexibly
        cols_full = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        n = len(data[0])
        df = pd.DataFrame(data, columns=cols_full[:n])

        # Build a tz-aware UTC index from open_time ms (always tz-aware)
        if "open_time" not in df.columns:
            return None
        idx_utc = _to_utc_index(df["open_time"])

        # Optionally drop the most-recent partial candle
        # (leave as-is; MEXC usually provides closed candles, but keep the flag for parity)
        if not include_partial and len(idx_utc) > 0:
            # Heuristic: if the last candle's close_time is in the future vs. now_utc, drop it
            # If close_time not provided, we keep all.
            if "close_time" in df.columns and pd.notna(df["close_time"].iloc[-1]):
                now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
                last_close = pd.to_datetime(df["close_time"].iloc[-1], unit="ms", utc=True)
                if last_close > now_utc:
                    df = df.iloc[:-1, :]
                    idx_utc = idx_utc[:-1]

        # Assign index and standardize columns
        df.index = idx_utc
        df = _normalize_cols(df)

        # Convert index to requested local timezone safely (never tz_localize here)
        try:
            df.index = _convert_index(df.index, TZ_NAME)
        except Exception:
            # If TZ_NAME invalid, keep UTC; avoids crashes
            pass

        return df
    except Exception:
        return None


# Optional range fetch (if you added it elsewhere). For now, alias:
def get_klines_between(*args, **kwargs):
    return get_klines(*args, **kwargs)
