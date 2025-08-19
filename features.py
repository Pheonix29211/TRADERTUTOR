from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
from indicators import ema, rsi, atr, body_size, wick_sizes
from config import (EQUAL_TOL_PCT, SWING_LOOKBACK, PSY_STEP, DISPLACEMENT_MULT, FVG_MIN_PCT)

@dataclass
class LiquidityLevel:
    price: float
    kind: str
    ts_ms: int

def within_tol(a: float, b: float, tol_pct: float) -> bool:
    return abs(a - b) <= tol_pct * ((a + b) / 2.0)

def find_swings(df: pd.DataFrame, lookback: int = SWING_LOOKBACK) -> Tuple[List[LiquidityLevel], List[LiquidityLevel]]:
    highs, lows = [], []
    idx = df.index
    H = df['high'].values
    L = df['low'].values
    n = len(df)
    for i in range(lookback, n - lookback):
        if H[i] == H[i-lookback:i+lookback+1].max():
            highs.append(LiquidityLevel(float(H[i]), 'swing_high', int(idx[i].timestamp()*1000)))
        if L[i] == L[i-lookback:i+lookback+1].min():
            lows.append(LiquidityLevel(float(L[i]), 'swing_low', int(idx[i].timestamp()*1000)))
    return highs, lows

def group_equals(levels: List[LiquidityLevel], tol_pct: float = EQUAL_TOL_PCT, kind_hi=True) -> List[LiquidityLevel]:
    if not levels: return []
    levels_sorted = sorted(levels, key=lambda x: x.price)
    groups, cur = [], [levels_sorted[0]]
    for lvl in levels_sorted[1:]:
        if within_tol(lvl.price, cur[-1].price, tol_pct): cur.append(lvl)
        else:
            if len(cur) >= 2:
                avg = sum(x.price for x in cur)/len(cur)
                groups.append(LiquidityLevel(avg, 'equal_highs' if kind_hi else 'equal_lows', cur[-1].ts_ms))
            cur = [lvl]
    if len(cur) >= 2:
        avg = sum(x.price for x in cur)/len(cur)
        groups.append(LiquidityLevel(avg, 'equal_highs' if kind_hi else 'equal_lows', cur[-1].ts_ms))
    return groups

def psych_levels(df: pd.DataFrame, step: int = PSY_STEP) -> List[LiquidityLevel]:
    pmin, pmax = df['low'].min(), df['high'].max()
    start = int(np.floor(pmin / step) * step)
    end = int(np.ceil(pmax / step) * step)
    out = []
    for p in range(start, end + step, step):
        out.append(LiquidityLevel(float(p), 'psych', int(df.index[-1].timestamp()*1000)))
    return out

def displacement_and_fvg(df: pd.DataFrame, i: int) -> Tuple[bool, Optional[Tuple[float,float]], str]:
    if i < 2: return False, None, ""
    bodies = body_size(df)
    avgb = bodies.iloc[max(0,i-20):i].mean() if i >= 20 else bodies.iloc[:i].mean()
    body = bodies.iloc[i]
    if avgb <= 0 or body < DISPLACEMENT_MULT * avgb:
        return False, None, ""
    hi_i2, lo_i2 = df['high'].iloc[i-2], df['low'].iloc[i-2]
    hi_i, lo_i = df['high'].iloc[i], df['low'].iloc[i]
    op_i, cl_i = df['open'].iloc[i], df['close'].iloc[i]
    if lo_i > hi_i2 and cl_i > op_i:
        gap = lo_i - hi_i2
        if gap / ((lo_i + hi_i2)/2.0 + 1e-12) >= FVG_MIN_PCT:
            return True, (hi_i2, lo_i), "bull"
    if hi_i < lo_i2 and cl_i < op_i:
        gap = lo_i2 - hi_i
        if gap / ((hi_i + lo_i2)/2.0 + 1e-12) >= FVG_MIN_PCT:
            return True, (hi_i, lo_i2), "bear"
    return False, None, ""
