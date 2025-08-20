# mexc_client.py
# Robust MEXC V3 klines fetcher (spot) with:
# - safe timezone handling (never tz_localize an aware index)
# - resilient HTTPS retries (fixes intermittent SSL BAD_RECORD_MAC)
# - paginated range fetcher (startTime) for multi-day backtests

import os
from typing import Optional, List, Tuple
import time

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TZ_NAME = os.getenv("TZ", "Asia/Kolkata")

MEXC_V3_URL = "https://api.mexc.com/api/v3/klines"
_MEXC_TF_MAP = {
    "1": "1m", "5": "5m", "15": "15m", "30": "30m", "60": "1h",
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1h"
}

# ---------------- HTTP session with retries ----------------
_SESSION = None

def _reset_session():
    global _SESSION
    try:
        if _SESSION is not None:
            _SESSION.close()
    except Exception:
        pass
    _SESSION = None

def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is not None:
        return _SESSION
    retry = Retry(
        total=4, connect=4, read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"])
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    s.headers.update({"User-Agent": "TraderTutor/1.0", "Connection": "close"})
    _SESSION = s
    return _SESSION

def _http_get(url: str, params: dict, timeout: int = 12):
    s = _get_session()
    try:
        return s.get(url, params=params, timeout=timeout)
    except requests.exceptions.SSLError:
        _reset_session()
        s = _get_session()
        return s.get(url, params=params, timeout=timeout)
    except requests.exceptions.RequestException:
        return None

# ---------------- Timezone helpers ----------------
def _to_utc_index(series_ms) -> pd.DatetimeIndex:
    # ALWAYS tz-aware UTC
    return pd.to_datetime(series_ms, unit="ms", utc=True)

def _convert_index(idx: pd.DatetimeIndex, tz_name: str) -> pd.DatetimeIndex:
    if idx.tz is None:
        return idx.tz_localize("UTC").tz_convert(tz_name)
    return idx.tz_convert(tz_name)

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
    df = df[keep].copy()
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna()

# ---------------- Single page fetch ----------------
def get_klines(tf: str, limit: int = 200, include_partial: bool = True) -> Optional[pd.DataFrame]:
    """
    Fetch latest klines (single page).
    Returns tz-aware index in TZ_NAME with columns: open, high, low, close, volume.
    """
    try:
        iv = _MEXC_TF_MAP.get(tf, tf)
        params = {"symbol": SYMBOL, "interval": iv, "limit": int(limit)}
        r = _http_get(MEXC_V3_URL, params=params, timeout=12)
        if r is None or r.status_code != 200:
            return None
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return None

        cols_full = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trade_count",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ]
        n = len(data[0])
        df = pd.DataFrame(data, columns=cols_full[:n])

        idx_utc = _to_utc_index(df["open_time"])
        if not include_partial and "close_time" in df.columns and pd.notna(df["close_time"].iloc[-1]):
            now_utc = pd.Timestamp.utcnow().tz_localize("UTC")
            last_close = pd.to_datetime(df["close_time"].iloc[-1], unit="ms", utc=True)
            if last_close > now_utc:
                df = df.iloc[:-1, :]
                idx_utc = idx_utc[:-1]

        df.index = idx_utc
        df = _normalize_cols(df)
        try:
            df.index = _convert_index(df.index, TZ_NAME)
        except Exception:
            pass
        return df
    except Exception:
        return None

# ---------------- Paged range fetch (for backtests) ----------------
def get_klines_between(tf: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, include_partial: bool = False) -> Optional[pd.DataFrame]:
    """
    Fetch klines covering [start_utc, end_utc] using paging via startTime.
    Returns tz-aware index in TZ_NAME.
    """
    try:
        iv = _MEXC_TF_MAP.get(tf, tf)
        # Ensure tz-aware UTC inputs
        if start_utc.tz is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tz is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")

        frames: List[pd.DataFrame] = []
        cursor_ms = int(start_utc.value // 10**6)  # ms
        end_ms = int(end_utc.value // 10**6)

        # MEXC page size cap ~1000; we’ll loop until we pass end_ms or no progress
        safety = 0
        while cursor_ms < end_ms and safety < 200:  # hard cap pages
            safety += 1
            params = {"symbol": SYMBOL, "interval": iv, "limit": 1000, "startTime": cursor_ms}
            r = _http_get(MEXC_V3_URL, params=params, timeout=15)
            if r is None or r.status_code != 200:
                break
            data = r.json()
            if not isinstance(data, list) or len(data) == 0:
                break

            cols_full = [
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_volume", "trade_count",
                "taker_buy_base", "taker_buy_quote", "ignore"
            ]
            n = len(data[0])
            df = pd.DataFrame(data, columns=cols_full[:n])
            idx_utc = _to_utc_index(df["open_time"])
            df.index = idx_utc
            df = _normalize_cols(df)
            frames.append(df)

            # advance cursor to AFTER last open_time to avoid duplicates
            last_ot = int(data[-1][0])  # ms
            if last_ot <= cursor_ms:
                break
            cursor_ms = last_ot + 1
            time.sleep(0.15)  # be nice to the API

        if not frames:
            return None

        big = pd.concat(frames).sort_index()
        # trim to end_utc
        big = big.loc[(big.index.tz_convert("UTC") >= start_utc) & (big.index.tz_convert("UTC") <= end_utc)]
        # convert index to TZ_NAME
        try:
            big.index = _convert_index(big.index, TZ_NAME)
        except Exception:
            pass

        if not include_partial and "close" in big.columns and big.index.size > 1:
            # best-effort: keep all; MEXC range fetch is already mostly closed bars
            pass

        return big
    except Exception:
        return None
