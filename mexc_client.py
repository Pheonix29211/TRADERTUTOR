import os, time
from typing import Optional
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from config import DATA_DIR, SYMBOL

MEXC_V3_URL = "https://api.mexc.com/api/v3/klines"
_TF_MAP = {"1":"1m","3":"3m","5":"5m","15":"15m","30":"30m","60":"1h"}

CACHE_ROOT = os.path.join(DATA_DIR, "cache", "mexc", SYMBOL)

def _ensure_dirs():
    os.makedirs(CACHE_ROOT, exist_ok=True)

def _cols(n: int):
    return ["open_time","open","high","low","close","volume",
            "close_time","quote_volume","trade_count",
            "taker_buy_base","taker_buy_quote","ignore"][:n]

def _parse_df(raw) -> Optional[pd.DataFrame]:
    if not isinstance(raw, list) or len(raw) == 0:
        return None
    df = pd.DataFrame(raw, columns=_cols(len(raw[0])))
    for c in ("open","high","low","close","volume"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    idx = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df.index = idx
    df = df[["open","high","low","close","volume"]].sort_index().dropna().drop_duplicates()
    return df

def _request(params, retries=3, backoff=0.2):
    for i in range(retries):
        r = requests.get(MEXC_V3_URL, params=params, timeout=12)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429,500,502,503,504):
            time.sleep(backoff * (2**i))
        else:
            break
    return None

def get_klines(interval: str, limit: int = 500, include_partial: bool = False) -> Optional[pd.DataFrame]:
    _ensure_dirs()
    iv = _TF_MAP.get(interval, interval)
    params = {"symbol": SYMBOL, "interval": iv, "limit": min(int(limit), 1000)}
    raw = _request(params)
    if raw is None: return None
    df = _parse_df(raw)
    if df is None or df.empty: return None
    if not include_partial and len(df) >= 2:
        dt = (df.index[-1] - df.index[-2]).total_seconds()
        if (pd.Timestamp.utcnow().tz_localize("UTC") - df.index[-1]).total_seconds() < dt:
            df = df.iloc[:-1]
    return df

def _cache_path(iv: str, day: pd.Timestamp) -> str:
    yyyy = day.strftime("%Y%m%d")
    folder = os.path.join(CACHE_ROOT, iv)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{yyyy}.parquet")

def get_klines_between(interval: str, start_ms: int, end_ms: int, cache: bool = True) -> Optional[pd.DataFrame]:
    _ensure_dirs()
    iv = _TF_MAP.get(interval, interval)
    start = pd.to_datetime(start_ms, unit="ms", utc=True)
    end = pd.to_datetime(end_ms, unit="ms", utc=True)

    parts = []
    # load cached days
    day = start.normalize()
    while day <= end.normalize():
        p = _cache_path(iv, day)
        if cache and os.path.exists(p):
            try: parts.append(pd.read_parquet(p))
            except: pass
        day += pd.Timedelta(days=1)

    fetched = pd.DataFrame()
    bar_sec = {"1m":60,"3m":180,"5m":300,"15m":900,"30m":1800,"1h":3600}[iv]
    cur = start_ms
    while cur < end_ms:
        raw = _request({"symbol": SYMBOL, "interval": iv, "startTime": cur, "endTime": end_ms, "limit": 1000})
        if raw is None or len(raw) == 0: break
        df = _parse_df(raw)
        if df is None or df.empty: break
        fetched = pd.concat([fetched, df])
        last_ts = int(df.index[-1].timestamp() * 1000)
        cur = max(cur + bar_sec*1000, last_ts + bar_sec*1000)
        if len(df) < 1000: break

    if parts or not fetched.empty:
        out = pd.concat(parts + ([fetched] if not fetched.empty else []))
        out = out.sort_index().drop_duplicates()
        out = out[(out.index >= start) & (out.index <= end)]
        if cache and not out.empty:
            for day, d in out.groupby(out.index.normalize()):
                p = _cache_path(iv, day)
                try: d.to_parquet(p)
                except: pass
        return out
    return None
