import numpy as np
import pandas as pd

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.clip(lower=0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))

def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.ewm(span=n, adjust=False).mean()

def body_size(df: pd.DataFrame) -> pd.Series:
    return (df['close'] - df['open']).abs()

def wick_sizes(df: pd.DataFrame):
    upper = df['high'] - df[['open', 'close']].max(axis=1)
    lower = df[['open', 'close']].min(axis=1) - df['low']
    return upper.abs(), lower.abs()
