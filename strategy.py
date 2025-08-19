import os, json, time, math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

from config import (
    SYMBOL,
    TP1_DOLLARS, TP2_DOLLARS,
    RISK_DOLLARS, STRETCH_MAX_DOLLARS,
    BE_MIN_BUFFER, ATR_TRAIL_PRE, ATR_TRAIL_NEAR, TP2_TIGHTEN_DIST,
    POOL_LOOKBACK, EQUAL_TOL_PCT, SWING_LOOKBACK,
    QUALITY_BIAS,
    ASIA_UTC, LONDON_UTC, NY_UTC,
)
from mexc_client import get_klines, get_klines_between
from indicators import ema, rsi, atr, body_size, wick_sizes
from features import LiquidityLevel, find_swings, group_equals, psych_levels, displacement_and_fvg
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan, combined_conf, hazard

# ------------------ Utilities ------------------ #
def now_ms() -> int:
    return int(time.time() * 1000)

def fmt_price(x: float) -> str:
    return f"${x:,.2f}"

# ------------------ Data Models ------------------ #
@dataclass
class Signal:
    side: str                  # "BUY" or "SELL"
    entry_zone: Tuple[float, float]
    entry_ref: float
    stop_px: float
    tp1_px: float
    tp2_px: float
    fvg_zone: Tuple[float, float]
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
    entry_zone: Tuple[float, float]
    entry_ref: float
    stop_px: float
    tp1_px: float
    tp2_px: float
    qty: float
    filled: bool = False
    entry_price: Optional[float] = None
    tp1_done: bool = False
    closed: bool = False
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    arm_id: str = ""
    opened_ms: int = 0
    closed_ms: Optional[int] = None
    stretched: bool = False
    be_buffer: float = BE_MIN_BUFFER

# ------------------ Detector (ICT DoL) ------------------ #
class DoLDetector:
    """
    Detects ICT DoL: sweep -> displacement -> FVG return, with bias and session context.
    """
    def __init__(self, arms: PolicyArms):
        self.arms = arms

    def _bias(self, df5: pd.DataFrame) -> str:
        e50 = ema(df5["close"], 50)
        e200 = ema(df5["close"], 200)
        if e50.iloc[-1] > e200.iloc[-1]:
            return "long"
        if e50.iloc[-1] < e200.iloc[-1]:
            return "short"
        return "neutral"

    def _session(self, ts: pd.Timestamp) -> str:
        hm = ts.time()
        if ASIA_UTC[0] <= hm <= ASIA_UTC[1]:
            return "ASIA"
        if LONDON_UTC[0] <= hm <= LONDON_UTC[1]:
            return "LONDON"
        if NY_UTC[0] <= hm <= NY_UTC[1]:
            return "NY"
        return "OFF"

    def _buckets(self, df5: pd.DataFrame, i: int) -> Dict:
        a = atr(df5, 14).iloc[: i + 1]
        atrp = float((a.iloc[-1] / df5["close"].iloc[-1]) * 100.0)

        disp_ok, fvg, direction = displacement_and_fvg(df5, i)
        # Body mult bucket
        bodies = body_size(df5)
        avgb = bodies.iloc[max(0, i - 20) : i].mean() if i >= 20 else bodies.iloc[:i].mean()
        body = bodies.iloc[i]
        mult = body / max(1e-9, avgb)
        if mult >= 2.0:
            disp_bucket = "hi"
        elif mult >= 1.8:
            disp_bucket = "mid"
        else:
            disp_bucket = "lo"

        # FVG bucket by relative span
        fvg_bucket = "none"
        if fvg is not None:
            span = abs(fvg[1] - fvg[0])
            rel = span / max(1e-9, df5["close"].iloc[i])
            fvg_bucket = "big" if rel > 0.006 else ("mid" if rel > 0.003 else "small")

        atr_bucket = "lo" if atrp < 0.5 else ("mid" if atrp < 1.5 else "hi")
        return {"atrp": atrp, "atr_bucket": atr_bucket, "disp_bucket": disp_bucket, "fvg_bucket": fvg_bucket}

    def _to_vec(self, df5: pd.DataFrame, i: int, feats: Dict) -> np.ndarray:
        """
        Build a compact 32-dim vector from recent structure for KNN memory.
        """
        seg = df5.iloc[max(0, i - 19) : i + 1].copy()
        seg["body"] = (seg["close"] - seg["open"]).abs()
        seg["rng"] = (seg["high"] - seg["low"])
        v = np.array(
            [
                seg["body"].mean(),
                seg["rng"].mean(),
                seg["body"].std(),
                seg["rng"].std(),
                float(feats["atrp"]),
                1.0 if feats["atr_bucket"] == "hi" else (0.5 if feats["atr_bucket"] == "mid" else 0.0),
                1.0 if feats["disp_bucket"] == "hi" else (0.5 if feats["disp_bucket"] == "mid" else 0.0),
                1.0 if feats["fvg_bucket"] == "big" else (0.5 if feats["fvg_bucket"] == "mid" else (0.25 if feats["fvg_bucket"] == "small" else 0.0)),
                1.0 if feats.get("sweep") == "up" else (0.0 if feats.get("sweep") == "down" else 0.5),
            ],
            dtype=np.float32,
        )
        if v.shape[0] < 32:
            v = np.pad(v, (0, 32 - v.shape[0]))
        return v

    def find(
        self,
        df5: pd.DataFrame,
        p_model: BayesModel,
        knn: PatternKNN,
        bandit: PolicyArms,
        bans: CFPBan,
    ) -> Optional[Signal]:
        """
        Returns a complete Signal if conditions align.
        """
        if len(df5) < max(POOL_LOOKBACK, 210):
            return None

        i = len(df5) - 1
        price = float(df5["close"].iloc[i])
        session = self._session(df5.index[-1])
        bias = self._bias(df5)

        # Pools
        swings_h, swings_l = find_swings(df5.iloc[-POOL_LOOKBACK:], lookback=SWING_LOOKBACK)
        eq_highs = group_equals(swings_h, EQUAL_TOL_PCT, True)
        eq_lows = group_equals(swings_l, EQUAL_TOL_PCT, False)
        psy = psych_levels(df5.iloc[-POOL_LOOKBACK:])

        # Sweep detection (last 2 bars behavior)
        last = df5.iloc[i]
        prev = df5.iloc[i - 1]
        high_pools = eq_highs + swings_h + [x for x in psy if x.price > price]
        low_pools = eq_lows + swings_l + [x for x in psy if x.price < price]

        swept_high = None
        for lvl in sorted(high_pools, key=lambda x: abs(x.price - price))[:10]:
            if prev["close"] < lvl.price and last["high"] > lvl.price and last["close"] < lvl.price:
                swept_high = lvl.price
                break

        swept_low = None
        for lvl in sorted(low_pools, key=lambda x: abs(x.price - price))[:10]:
            if prev["close"] > lvl.price and last["low"] < lvl.price and last["close"] > lvl.price:
                swept_low = lvl.price
                break

        # Displacement + FVG on bar i
        disp_ok, fvg, direction = displacement_and_fvg(df5, i)
        if not disp_ok or fvg is None:
            return None

        # Features for AI
        feats = self._buckets(df5, i)
        feats.update(
            {
                "session": session,
                "bias": bias,
                "sweep": "up" if swept_high else ("down" if swept_low else "none"),
            }
        )

        # Probabilities / EV
        p = p_model.predict(feats)  # Beta-smoothed class prob
        ev_norm, winrate = knn.query(self._to_vec(df5, i, feats), k=64)
        band_prior = 0.5  # neutral; bandit mean is updated post-trade
        conf = combined_conf(p, ev_norm, band_prior)

        # Banlist check — if we recently lost in an almost-identical context, skip
        if bans.banned(feats):
            return None

        ts_ms = int(df5.index[-1].timestamp() * 1000)
        arm_id = bandit.pick()

        # Build Signal
        if direction == "bear" and swept_high and bias in ("short", "neutral"):
            entry_top, entry_bot = fvg           # NOTE: for bear, fvg tuple is (hi_i, lo_i2)
            entry_ref = (entry_top + entry_bot) / 2.0
            stop_px = entry_top                  # beyond the gap
            tp1 = entry_ref - TP1_DOLLARS
            tp2 = entry_ref - TP2_DOLLARS
            reason = f"sweep↑ {round(swept_high,2)} + displacement + FVG ret short"
            return Signal(
                side="SELL",
                entry_zone=(entry_bot, entry_top),
                entry_ref=entry_ref,
                stop_px=stop_px,
                tp1_px=tp1,
                tp2_px=tp2,
                fvg_zone=(entry_bot, entry_top),
                swept_level=swept_high,
                ts_ms=ts_ms,
                reason=reason,
                arm_id=arm_id,
                conf=conf,
                feats=feats,
            )

        if direction == "bull" and swept_low and bias in ("long", "neutral"):
            entry_top, entry_bot = fvg           # NOTE: for bull, fvg tuple is (hi_i2, lo_i)
            entry_ref = (entry_top + entry_bot) / 2.0
            stop_px = entry_bot
            tp1 = entry_ref + TP1_DOLLARS
            tp2 = entry_ref + TP2_DOLLARS
            reason = f"sweep↓ {round(swept_low,2)} + displacement + FVG ret long"
            return Signal(
                side="BUY",
                entry_zone=(entry_bot, entry_top),
                entry_ref=entry_ref,
                stop_px=stop_px,
                tp1_px=tp1,
                tp2_px=tp2,
                fvg_zone=(entry_bot, entry_top),
                swept_level=swept_low,
                ts_ms=ts_ms,
                reason=reason,
                arm_id=arm_id,
                conf=conf,
                feats=feats,
            )

        return None

# ------------------ Risk/Qty ------------------ #
def qty_for_risk(entry: float, stop: float, risk_dollars: float) -> float:
    dist = abs(entry - stop)
    if dist <= 0:
        return 0.0
    return float(risk_dollars / dist)

# ------------------ Trade Manager ------------------ #
class TradeManager:
    """
    One-trade-at-a-time management:
    - pre-fill cancel on rejection
    - TP1/TP2, early TP logic
    - trailing & BE buffer
    - optional stretch to $50 when AI is confident
    """
    def __init__(self):
        self.active: Optional[TradeState] = None
        self.cooldown_until_ms = 0
        self.last_signal_fingerprint = None
        self.sl1_recent_ts = 0
        self.sl_in_last_2h: List[int] = []

    def _fingerprint(self, sig: Signal) -> str:
        return f"{sig.side}|{round(sig.entry_ref/5.0)*5}|{sig.ts_ms//60_000}"

    def can_open(self) -> bool:
        if self.active and not self.active.closed:
            return False
        if int(time.time() * 1000) < self.cooldown_until_ms:
            return False
        return True

    def open_from_signal(self, sig: Signal) -> TradeState:
        if not self.can_open():
            return self.active
        qty = max(0.0001, qty_for_risk(sig.entry_ref, sig.stop_px, RISK_DOLLARS))
        tid = f"{int(time.time())}-{SYMBOL}-{sig.side}"
        st = TradeState(
            id=tid,
            side=sig.side,
            entry_zone=sig.entry_zone,
            entry_ref=sig.entry_ref,
            stop_px=sig.stop_px,
            tp1_px=sig.tp1_px,
            tp2_px=sig.tp2_px,
            qty=qty,
            arm_id=sig.arm_id,
            opened_ms=now_ms(),
            be_buffer=BE_MIN_BUFFER,
        )
        self.active = st
        # simple cooldown between signals to avoid spam
        self.cooldown_until_ms = int(time.time() * 1000) + 60_000 * 15  # mirrored from config default
        self.last_signal_fingerprint = self._fingerprint(sig)
        return st

    # ---- AI-gated stretch to $50 (once) ----
    def maybe_stretch(self, st: TradeState, ai_conf: float):
        if st.stretched:
            return
        # Require higher confidence for stretch (personality-aware via QUALITY_BIAS)
        if ai_conf < (0.80 + 0.04 * QUALITY_BIAS):
            return
        # compute new stop allowing up to $50 risk on current qty
        max_dist = STRETCH_MAX_DOLLARS / max(st.qty, 1e-9)
        cur_dist = abs(st.entry_ref - st.stop_px)
        new_dist = min(cur_dist * 1.25, max_dist)
        if new_dist > cur_dist:
            if st.side == "BUY":
                st.stop_px = st.entry_ref - new_dist
            else:
                st.stop_px = st.entry_ref + new_dist
            st.stretched = True

    # ---- Rejection detection for early cuts ----
    def _rejection(self, df1: pd.DataFrame, side: str) -> bool:
        if len(df1) < 8:
            return False
        sub = df1.iloc[-5:].copy()
        up, low = wick_sizes(sub)
        bodies = body_size(sub) + 1e-9
        wick_ratio = np.maximum(up, low) / bodies
        wick_flag = bool(wick_ratio.iloc[-1] >= 1.5)  # WICK_REJ_RATIO from config

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

        # Early rejection if 2+ signals agree
        return sum([wick_flag, ema_flip, r_hook]) >= 2

    def maybe_fill(self, st: TradeState, price: float):
        if st.filled:
            return
        lo, hi = st.entry_zone
        if lo <= price <= hi:
            st.filled = True
            st.entry_price = float(np.clip(st.entry_ref, lo, hi))

    def manage(self, st: TradeState, df1: pd.DataFrame, df5: pd.DataFrame) -> Optional[Dict]:
        """
        Called every minute with fresh 1m/5m. Returns an event dict when something happens:
        {"event": "TP1|TP2|EARLY_TP1|EARLY_TP2|SL|CANCELLED", "price": float}
        """
        last_price = float(df1["close"].iloc[-1])
        self.maybe_fill(st, last_price)

        # Cancel if pre-entry rejection happens
        if not st.filled:
            if self._rejection(df1, st.side):
                return {"event": "CANCELLED", "price": last_price}
            return None

        # TP/SL/early-cut logic
        if st.side == "BUY":
            # Early cut to TP1 if rejection pre-TP1
            if not st.tp1_done and self._rejection(df1, "BUY"):
                st.tp1_done = True
                return {"event": "EARLY_TP1", "price": last_price}

            # TP1 hit
            if last_price >= st.tp1_px and not st.tp1_done:
                st.tp1_done = True
                return {"event": "TP1", "price": st.tp1_px}

            # After TP1: trail + BE protection, and early TP2 if rejection
            if st.tp1_done:
                be = st.entry_price + st.be_buffer
                a5 = float(atr(df5, 14).iloc[-1])
                trail = last_price - max(a5 * ATR_TRAIL_PRE, 1.0)
                st.stop_px = max(st.stop_px, be, trail)
                # tighten near TP2
                if (st.tp2_px - last_price) <= TP2_TIGHTEN_DIST:
                    st.stop_px = max(st.stop_px, last_price - max(a5 * ATR_TRAIL_NEAR, 1.0))
                if self._rejection(df1, "BUY"):
                    return {"event": "EARLY_TP2", "price": last_price}

            # Final TP2 / SL
            if last_price >= st.tp2_px:
                return {"event": "TP2", "price": st.tp2_px}
            if last_price <= st.stop_px:
                return {"event": "SL", "price": st.stop_px}

        else:  # SELL
            if not st.tp1_done and self._rejection(df1, "SELL"):
                st.tp1_done = True
                return {"event": "EARLY_TP1", "price": last_price}

            if last_price <= st.tp1_px and not st.tp1_done:
                st.tp1_done = True
                return {"event": "TP1", "price": st.tp1_px}

            if st.tp1_done:
                be = st.entry_price - st.be_buffer
                a5 = float(atr(df5, 14).iloc[-1])
                trail = last_price + max(a5 * ATR_TRAIL_PRE, 1.0)
                st.stop_px = min(st.stop_px, be, trail)
                if (last_price - st.tp2_px) <= TP2_TIGHTEN_DIST:
                    st.stop_px = min(st.stop_px, last_price + max(a5 * ATR_TRAIL_NEAR, 1.0))
                if self._rejection(df1, "SELL"):
                    return {"event": "EARLY_TP2", "price": last_price}

            if last_price <= st.tp2_px:
                return {"event": "TP2", "price": st.tp2_px}
            if last_price >= st.stop_px:
                return {"event": "SL", "price": st.stop_px}

        return None

    def close(self, st: TradeState, event: str, px: float):
        st.closed = True
        st.exit_price = px
        st.exit_reason = event
        st.closed_ms = now_ms()
        self.active = None

# ------------------ Reward & Shaping ------------------ #
def trade_reward(st: TradeState, event: str) -> float:
    """
    Compute reward (R) using PnL in $ normalized by risk, plus quality shaping.
    Caps to [-2, +4] to stabilize bandit/KNN updates.
    """
    if not st.entry_price or not st.exit_price:
        return 0.0

    pnl = (st.exit_price - st.entry_price) if st.side == "BUY" else (st.entry_price - st.exit_price)
    pnl_usd = pnl * st.qty  # *** FIXED: legal variable name ***
    rr = pnl_usd / RISK_DOLLARS

    # Base outcome
    base = 1.0 if (event in ("TP1", "TP2", "EARLY_TP1", "EARLY_TP2") and pnl_usd > 0) else -1.0

    # Extra credit for reaching TP2 (or early TP2) — encourages letting runners run cleanly
    eff = 0.2 * max(0.0, rr) if event in ("TP2", "EARLY_TP2") else 0.0

    R = base + rr + eff

    # Slight extra penalty if we stretched the stop and still lost (we gave it room)
    if st.stretched and event == "SL":
        R -= 0.2

    # Clamp
    return float(np.clip(R, -2.0, 4.0))

def norm_reward(R: float) -> float:
    """
    Normalize reward R in [-2, +4] to [0,1] for bandit updates.
    """
    return float((R + 2.0) / 6.0)

