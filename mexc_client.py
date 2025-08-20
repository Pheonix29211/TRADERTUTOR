# mexc_client.py
# Robust MEXC V3 klines fetcher (spot) with:
# - safe timezone handling (never tz_localize an aware index)
# - resilient HTTPS retries (fixes intermittent SSL BAD_RECORD_MAC)
# - paginated range fetcher (startTime) for multi-day windows
# - single-page fetch that auto-falls back to range if empty

import os
import time
from typing import Optional, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TZ_NAME = os.getenv("TZ", "Asia/Kolkata")

MEXC_V3_URL = "https://api.mexc.com/api/v3/klines"
_MEXC_TF_MAP = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
}
_MINUTES_PER_IV = {"1m": 1, "5m": 5, "15m": 15, "30m": 30, "1h": 60}

# ------------- HTTP session with retries -------------
_SESSION: Optional[requests.Session] = None


def _reset_session() -> None:
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
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
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


# ------------- TZ helpers -------------
def _to_utc_index(series_ms) -> pd.DatetimeIndex:
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


# ------------- Range fetch (paged) -------------
def get_klines_between(
    tf: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, include_partial: bool = False
) -> Optional[pd.DataFrame]:
    """
    Fetch klines covering [start_utc, end_utc] using paging via startTime.
    Returns tz-aware index in TZ_NAME.
    """
    try:
        iv = _MEXC_TF_MAP.get(tf, tf)
        if start_utc.tz is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tz is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")

        frames: List[pd.DataFrame] = []
        cursor_ms = int(start_utc.value // 10**6)
        end_ms = int(end_utc.value // 10**6)

        safety = 0
        while cursor_ms < end_ms and safety < 200:
            safety += 1
            params = {"symbol": SYMBOL, "interval": iv, "limit": 1000, "startTime": cursor_ms}
            r = _http_get(MEXC_V3_URL, params=params, timeout=15)
            if r is None or r.status_code != 200:
                break
            data = r.json()
            # Some gateways send dicts; normalize to list if possible
            if isinstance(data, dict):
                data = data.get("data", [])
            if not isinstance(data, list) or len(data) == 0:
                break

            cols_full = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_volume",
                "trade_count",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ]
            n = len(data[0])
            df = pd.DataFrame(data, columns=cols_full[:n])
            idx_utc = _to_utc_index(df["open_time"])
            df.index = idx_utc
            df = _normalize_cols(df)
            frames.append(df)

            last_ot = int(data[-1][0])
            if last_ot <= cursor_ms:
                break
            cursor_ms = last_ot + 1
            time.sleep(0.15)

        if not frames:
            return None
        big = pd.concat(frames).sort_index()
        # Trim to requested end
        big = big.loc[
            (big.index.tz_convert("UTC") >= start_utc)
            & (big.index.tz_convert("UTC") <= end_utc)
        ]
        try:
            big.index = _convert_index(big.index, TZ_NAME)
        except Exception:
            pass
        return big
    except Exception:
        return None


# ------------- Single page fetch w/ fallback -------------
def get_klines(tf: str, limit: int = 200, include_partial: bool = True) -> Optional[pd.DataFrame]:
    """
    Latest klines (single page). If empty/failed, falls back to a short range fetch.
    Index returned in TZ_NAME (tz-aware). Columns: open, high, low, close, volume.
    """
    try:
        iv = _MEXC_TF_MAP.get(tf, tf)
        params = {"symbol": SYMBOL, "interval": iv, "limit": int(limit)}
        r = _http_get(MEXC_V3_URL, params=params, timeout=12)
        if r is None or r.status_code != 200:
            return _fallback_range(iv, limit)

        data = r.json()
        if isinstance(data, dict):
            data = data.get("data", data.get("kline", []))
        if not isinstance(data, list) or len(data) == 0:
            return _fallback_range(iv, limit)

        cols_full = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trade_count",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ]
        n = len(data[0])
        df = pd.DataFrame(data, columns=cols_full[:n])

        if "open_time" not in df.columns:
            return _fallback_range(iv, limit)
        idx_utc = _to_utc_index(df["open_time"])

        # Safer partial-bar handling: only drop if last close is in the future
        try:
            if (
                not include_partial
                and "close_time" in df.columns
                and pd.notna(df["close_time"].iloc[-1])
            ):
                now_utc = pd.Timestamp.now(tz="UTC")
                last_close = pd.to_datetime(df["close_time"].iloc[-1], unit="ms", utc=True)
                if last_close > now_utc:
                    df = df.iloc[:-1, :]
                    idx_utc = idx_utc[:-1]
        except Exception:
            pass

        if len(df) == 0:
            return _fallback_range(iv, limit)

        df.index = idx_utc
        df = _normalize_cols(df)
        if df.empty:
            return _fallback_range(iv, limit)

        try:
            df.index = _convert_index(df.index, TZ_NAME)
        except Exception:
            pass

        return df
    except Exception:
        try:
            return _fallback_range(_MEXC_TF_MAP.get(tf, tf), limit)
        except Exception:
            return None


def _fallback_range(iv: str, limit: int) -> Optional[pd.DataFrame]:
    """Fetch a small range (~limit bars) ending now as a safety net."""
    mins = _MINUTES_PER_IV.get(iv, 5)
    end_utc = pd.Timestamp.now(tz="UTC")
    start_utc = end_utc - pd.Timedelta(minutes=mins * max(10, min(limit, 1000)))
    df = get_klines_between(iv, start_utc, end_utc, include_partial=False)
    return df
