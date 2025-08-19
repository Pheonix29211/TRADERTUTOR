import os, json, time, math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, List
import numpy as np
import pandas as pd

from config import *
from mexc_client import get_klines, get_klines_between
from indicators import ema, rsi, atr, body_size, wick_sizes
from features import LiquidityLevel, find_swings, group_equals, psych_levels, displacement_and_fvg
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan, combined_conf, hazard

# ---------- Utilities ----------
def now_ms(): return int(time.time()*1000)

def fmt_price(x: float) -> str: return f"${x:,.2f}"

# ---------- Signal / Trade ----------
@dataclass
class Signal:
    side: str                  # BUY / SELL
    entry_zone: Tuple[float,float]
    entry_ref: float
    stop_px: float             # structural stop price (pre-risk sizing)
    tp1_px: float
    tp2_px: float
    fvg_zone: Tuple[float,float]
    swept_level: float
    ts_ms: int
    reason: str
    arm_id: str
    conf: float
    feats: Dict

@dataclass
class TradeState:
    id: str
    side: str
    entry_zone: Tuple[float,float]
    entry_ref: float
    stop_px: float
    tp1_px: float
    tp2_px: float
    qty: float
    filled: bool=False
    entry_price: Optional[float]=None
    tp1_done: bool=False
    closed: bool=False
    exit_price: Optional[float]=None
    exit_reason: Optional[str]=None
    arm_id: str=""
    opened_ms: int=0
    closed_ms: Optional[int]=None
    stretched: bool=False
    be_buffer: float=BE_MIN_BUFFER

# ---------- DoL Detector ----------
class DoLDetector:
    def __init__(self, arms: PolicyArms):
        self.arms = arms

    def _bias(self, df5: pd.DataFrame) -> str:
        e50 = ema(df5['close'], 50); e200 = ema(df5['close'], 200)
        if e50.iloc[-1] > e200.iloc[-1]: return "long"
        if e50.iloc[-1] < e200.iloc[-1]: return "short"
        return "neutral"

    def _session(self, ts: pd.Timestamp) -> str:
        hm = ts.time()
        if ASIA_UTC[0] <= hm <= ASIA_UTC[1]: return "ASIA"
        if LONDON_UTC[0] <= hm <= LONDON_UTC[1]: return "LONDON"
        if NY_UTC[0] <= hm <= NY_UTC[1]: return "NY"
        return "OFF"

    def _buckets(self, df5: pd.DataFrame, i: int) -> Dict:
        a = atr(df5, 14).iloc[:i+1]; atrp = float((a.iloc[-1] / df5['close'].iloc[-1]) * 100.0)
        disp_ok, fvg, direction = displacement_and_fvg(df5, i)
        disp_bucket = "lo"
        bodies = body_size(df5)
        avgb = bodies.iloc[max(0,i-20):i].mean() if i >= 20 else bodies.iloc[:i].mean()
        body = bodies.iloc[i]
        mult = body / max(1e-9, avgb)
        if mult >= 2.0: disp_bucket = "hi"
        elif mult >= 1.8: disp_bucket = "mid"
        fvg_bucket = "small"
        if fvg is not None:
            span = abs(fvg[1]-fvg[0]); fvg_bucket = "big" if span/df5['close'].iloc[i] > 0.006 else ("mid" if span/df5['close'].iloc[i]>0.003 else "small")
        atr_bucket = "lo" if atrp < 0.5 else ("mid" if atrp < 1.5 else "hi")
        return {"atrp":atrp, "atr_bucket":atr_bucket, "disp_bucket":disp_bucket, "fvg_bucket":fvg_bucket}

    def find(self, df5: pd.DataFrame, p_model, knn: PatternKNN, bandit: PolicyArms, bans: CFPBan) -> Optional[Signal]:
        if len(df5) < max(POOL_LOOKBACK, 210): return None
        i = len(df5)-1
        price = float(df5['close'].iloc[i])
        swings_h, swings_l = find_swings(df5.iloc[-POOL_LOOKBACK:])
        eq_highs = group_equals([x for x in swings_h], EQUAL_TOL_PCT, True)
        eq_lows  = group_equals([x for x in swings_l], EQUAL_TOL_PCT, False)
        psy = psych_levels(df5.iloc[-POOL_LOOKBACK:])

        last = df5.iloc[i]; prev = df5.iloc[i-1]
        high_pools = eq_highs + [x for x in swings_h] + [x for x in psy if x.price > price]
        low_pools  = eq_lows  + [x for x in swings_l] + [x for x in psy if x.price < price]

        swept_high = None; swept_low = None
        for lvl in sorted(high_pools, key=lambda x: abs(x.price - price))[:10]:
            if prev['close'] < lvl.price and last['high'] > lvl.price and last['close'] < lvl.price:
                swept_high = lvl.price; break
        for lvl in sorted(low_pools, key=lambda x: abs(x.price - price))[:10]:
            if prev['close'] > lvl.price and last['low'] < lvl.price and last['close'] > lvl.price:
                swept_low = lvl.price; break

        disp_ok, fvg, direction = displacement_and_fvg(df5, i)
        if not disp_ok or fvg is None: return None
        bias = self._bias(df5)
        session = self._session(df5.index[-1])

        # features for AI
        feats = self._buckets(df5, i)
        feats.update({
            "session": session,
            "bias": bias,
            "sweep": "up" if swept_high else ("down" if swept_low else "none")
        })

        # probability & knn EV
        p = p_model.predict(feats)
        ev, wr = knn.query(self._to_vec(df5, i, feats), k=64)
        band_prior = 0.5  # neutral prior; bandit mean is updated on close

        conf = combined_conf(p, ev, band_prior)

        # ban check
        if bans.banned(feats): return None

        ts_ms = int(df5.index[-1].timestamp()*1000)
        arm_id = bandit.pick()

        if direction == "bear" and swept_high and bias in ("short","neutral"):
            entry_top, entry_bot = fvg
            entry_ref = (entry_top + entry_bot)/2.0
            stop_px = entry_top  # structural stop beyond gap
            tp1 = entry_ref - TP1_DOLLARS
            tp2 = entry_ref - TP2_DOLLARS
            reason = f"sweep↑ {round(swept_high,2)} + displacement + FVG ret short"
            return Signal("SELL",(entry_bot, entry_top), entry_ref, stop_px, tp1, tp2, fvg, swept_high, ts_ms, reason, arm_id, conf, feats)

        if direction == "bull" and swept_low and bias in ("long","neutral"):
            entry_top, entry_bot = fvg
            entry_ref = (entry_top + entry_bot)/2.0
            stop_px = entry_bot
            tp1 = entry_ref + TP1_DOLLARS
            tp2 = entry_ref + TP2_DOLLARS
            reason = f"sweep↓ {round(swept_low,2)} + displacement + FVG ret long"
            return Signal("BUY",(entry_bot, entry_top), entry_ref, stop_px, tp1, tp2, fvg, swept_low, ts_ms, reason, arm_id, conf, feats)

        return None

    def _to_vec(self, df5: pd.DataFrame, i: int, feats: Dict) -> np.ndarray:
        # compact 32-dim vector from recent structure
        seg = df5.iloc[max(0,i-19):i+1].copy()
        seg["body"] = (seg["close"]-seg["open"]).abs()
        seg["rng"] = (seg["high"]-seg["low"])
        v = np.array([
            seg["body"].mean(), seg["rng"].mean(),
            seg["body"].std(), seg["rng"].std(),
            float(feats["atrp"]),
            1.0 if feats["atr_bucket"]=="hi" else (0.5 if feats["atr_bucket"]=="mid" else 0.0),
            1.0 if feats["disp_bucket"]=="hi" else (0.5 if feats["disp_bucket"]=="mid" else 0.0),
            1.0 if feats["fvg_bucket"]=="big" else (0.5 if feats["fvg_bucket"]=="mid" else 0.0),
            1.0 if feats.get("sweep")=="up" else (0.0 if feats.get("sweep")=="down" else 0.5),
        ], dtype=np.float32)
        # pad to 32
        if v.shape[0] < 32:
            v = np.pad(v, (0,32-v.shape[0]))
        return v

# ---------- Risk Engine ----------
def qty_for_risk(entry: float, stop: float, risk_dollars: float) -> float:
    dist = abs(entry - stop)
    if dist <= 0: return 0.0
    return float(risk_dollars / dist)

# ---------- Trade Manager ----------
class TradeManager:
    def __init__(self):
        self.active: Optional[TradeState] = None
        self.cooldown_until_ms = 0
        self.last_signal_fingerprint = None
        self.sl1_recent_ts = 0
        self.sl_in_last_2h = []

    def _fingerprint(self, sig: Signal) -> str:
        return f"{sig.side}|{round(sig.entry_ref/5.0)*5}|{sig.ts_ms//60_000}"

    def can_open(self) -> bool:
        if self.active and not self.active.closed: return False
        if int(time.time()*1000) < self.cooldown_until_ms: return False
        return True

    def open_from_signal(self, sig: Signal) -> TradeState:
        if not self.can_open(): return self.active
        # qty for $40 risk
        qty = max(0.0001, qty_for_risk(sig.entry_ref, sig.stop_px, RISK_DOLLARS))
        tid = f"{int(time.time())}-{SYMBOL}-{sig.side}"
        st = TradeState(
            id=tid, side=sig.side, entry_zone=sig.entry_zone, entry_ref=sig.entry_ref,
            stop_px=sig.stop_px, tp1_px=sig.tp1_px, tp2_px=sig.tp2_px,
            qty=qty, arm_id=sig.arm_id, opened_ms=now_ms(),
            be_buffer=BE_MIN_BUFFER
        )
        self.active = st
        self.cooldown_until_ms = int(time.time()*1000) + SIGNAL_MIN_COOLDOWN_MS
        self.last_signal_fingerprint = self._fingerprint(sig)
        return st

    # ---- AI-gated stretch to $50 (once) ----
    def maybe_stretch(self, st: TradeState, ai_conf: float):
        if st.stretched: return
        if ai_conf < (0.80 + 0.04*QUALITY_BIAS): return
        # compute new stop allowing up to $50 risk on current qty
        max_dist = STRETCH_MAX_DOLLARS / max(st.qty, 1e-9)
        cur_dist = abs(st.entry_ref - st.stop_px)
        new_dist = min(cur_dist * 1.25, max_dist)
        if new_dist > cur_dist:
            if st.side == "BUY": st.stop_px = st.entry_ref - new_dist
            else: st.stop_px = st.entry_ref + new_dist
            st.stretched = True

    # ---- Early rejection detection ----
    def _rejection(self, df1: pd.DataFrame, side: str) -> bool:
        if len(df1) < 8: return False
        sub = df1.iloc[-5:].copy()
        up, low = wick_sizes(sub); bodies = body_size(sub) + 1e-9
        wick_ratio = np.maximum(up, low) / bodies
        wick_flag = (wick_ratio.iloc[-1] >= WICK_REJ_RATIO)
        e9, e21 = ema(df1["close"], 9), ema(df1["close"], 21)
        if side == "BUY":
            ema_flip = (e9.iloc[-1] < e21.iloc[-1]) and (e9.iloc[-2] > e21.iloc[-2])
        else:
            ema_flip = (e9.iloc[-1] > e21.iloc[-1]) and (e9.iloc[-2] < e21.iloc[-2])
        rs = rsi(df1["close"], 14)
        if side == "BUY":
            r_hook = (rs.iloc[-2] > 60) and (rs.iloc[-1] < 55)
        else:
            r_hook = (rs.iloc[-2] < 40) and (rs.iloc[-1] > 45)
        return sum([wick_flag, ema_flip, r_hook]) >= 2

    def maybe_fill(self, st: TradeState, price: float):
        if st.filled: return
        lo, hi = st.entry_zone
        if lo <= price <= hi:
            st.filled = True
            st.entry_price = float(np.clip(st.entry_ref, lo, hi))

    def manage(self, st: TradeState, df1: pd.DataFrame, df5: pd.DataFrame) -> Optional[Dict]:
        last_price = float(df1["close"].iloc[-1])
        self.maybe_fill(st, last_price)
        if not st.filled:
            # cancel pre-entry if heavy rejection before fill
            if self._rejection(df1, st.side):
                return {"event":"CANCELLED", "price":last_price}
            return None

        # TP/SL logic
        if st.side == "BUY":
            if not st.tp1_done and self._rejection(df1, "BUY"):
                st.tp1_done = True; return {"event":"EARLY_TP1", "price":last_price}
            if last_price >= st.tp1_px and not st.tp1_done:
                st.tp1_done = True; return {"event":"TP1", "price": st.tp1_px}
            # trail after TP1
            if st.tp1_done:
                be = st.entry_price + st.be_buffer
                a5 = float(atr(df5, 14).iloc[-1]); trail = last_price - max(a5*ATR_TRAIL_PRE, 1.0)
                st.stop_px = max(st.stop_px, be, trail)
                if (st.tp2_px - last_price) <= TP2_TIGHTEN_DIST:
                    st.stop_px = max(st.stop_px, last_price - max(a5*ATR_TRAIL_NEAR, 1.0))
                if self._rejection(df1, "BUY"): return {"event":"EARLY_TP2", "price": last_price}
            if last_price >= st.tp2_px: return {"event":"TP2", "price": st.tp2_px}
            if last_price <= st.stop_px: return {"event":"SL", "price": st.stop_px}
        else:
            if not st.tp1_done and self._rejection(df1, "SELL"):
                st.tp1_done = True; return {"event":"EARLY_TP1", "price":last_price}
            if last_price <= st.tp1_px and not st.tp1_done:
                st.tp1_done = True; return {"event":"TP1", "price": st.tp1_px}
            if st.tp1_done:
                be = st.entry_price - st.be_buffer
                a5 = float(atr(df5, 14).iloc[-1]); trail = last_price + max(a5*ATR_TRAIL_PRE, 1.0)
                st.stop_px = min(st.stop_px, be, trail)
                if (last_price - st.tp2_px) <= TP2_TIGHTEN_DIST:
                    st.stop_px = min(st.stop_px, last_price + max(a5*ATR_TRAIL_NEAR, 1.0))
                if self._rejection(df1, "SELL"): return {"event":"EARLY_TP2", "price": last_price}
            if last_price <= st.tp2_px: return {"event":"TP2", "price": st.tp2_px}
            if last_price >= st.stop_px: return {"event":"SL", "price": st.stop_px}
        return None

    def close(self, st: TradeState, event: str, px: float):
        st.closed = True; st.exit_price = px; st.exit_reason = event; st.closed_ms = now_ms()
        self.active = None

# ---------- Reward & Learning helpers ----------
def trade_reward(st: TradeState, event: str) -> float:
    if not st.entry_price or not st.exit_price: return 0.0
    pnl = (st.exit_price - st.entry_price) if st.side=="BUY" else (st.entry_price - st.exit_price)
    pnl_$ = pnl * st.qty
    rr = pnl_$ / RISK_DOLLARS
    base = 1.0 if event in ("TP1","TP2","EARLY_TP1","EARLY_TP2") and pnl_$ > 0 else -1.0
    eff = 0.2*max(0.0, rr) if event in ("TP2","EARLY_TP2") else 0.0
    R = base + rr + eff
    # punish stretch loss a bit more
    if st.stretched and event=="SL": R -= 0.2
    return float(np.clip(R, -2.0, 4.0))

def norm_reward(R: float) -> float:
    return float((R + 2.0) / 6.0)
