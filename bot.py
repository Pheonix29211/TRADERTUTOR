# bot.py
# Webhook-ready Telegram bot (PTB v13 + Flask 3.x)
# Features:
# - MEXC-only data feed (mexc_client.py)
# - AI-mode (Q/M/R) with hysteresis; mode-based cooldown after SL
# - Looser BASELINE & AGGRESSIVE selection gates (more entries; risk unchanged)
# - Scans 5m (primary) and 15m (fallback); larger context windows
# - Scan cadence 30s (manage cadence 30s)
# - One-trade lock, SL=$40 P&L (stretch to $50 on high conf), TP1=±$600, TP2=±$1500
# - Single, full entry message; live logging to /data/trades.json
# - Session commentary + daily/weekly recaps
# - Safe Telegram sends (TLS retry) + /menu command fixed
# - Flask webhook; no polling

import os, sys, traceback, time, json, math
from datetime import datetime
from datetime import time as dtime
from typing import Optional, List, Dict
from pathlib import Path

sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from flask import Flask, request
from pytz import UTC
import pandas as pd

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext
from telegram.utils.request import Request  # TLS-hardened small pool

from config import (
    TOKEN, ADMIN_CHAT_ID, TZ_NAME, SYMBOL,
    ASSISTANT_MODE, ASSISTANT_WARN, ASSISTANT_CUT, ASSISTANT_STRONG,
    RISK_DOLLARS, STRETCH_MAX_DOLLARS, TP1_DOLLARS, TP2_DOLLARS
)
try:
    from config import QTY_BTC as CFG_QTY_BTC
except Exception:
    CFG_QTY_BTC = float(os.getenv("QTY_BTC", "0.101"))

from mexc_client import get_klines
from strategy import DoLDetector, TradeManager, trade_reward, norm_reward, Signal, TradeState
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan

# ----------------- init -----------------
load_dotenv()

def _resolve_data_dir():
    candidates = [
        os.getenv("DATA_DIR"),
        "/data",
        "/tmp/data",
        os.path.join(os.path.dirname(__file__), "data"),
    ]
    for p in candidates:
        if not p: continue
        try:
            Path(p).mkdir(parents=True, exist_ok=True)
            t = Path(p) / ".write_test"
            t.write_text("ok", encoding="utf-8")
            t.unlink(missing_ok=True)
            return p
        except Exception:
            continue
    raise RuntimeError("No writable data directory found")

DATA_DIR = _resolve_data_dir()
TRADES_PATH = os.path.join(DATA_DIR, "trades.json")
if not Path(TRADES_PATH).exists():
    Path(TRADES_PATH).write_text("[]", encoding="utf-8")

bandit = PolicyArms()
knn    = PatternKNN()
bayes  = BayesModel()
bans   = CFPBan()
detector = DoLDetector(bandit)
tm       = TradeManager()

# ----------------- helpers -----------------
def safe_utc(ts):
    try:
        from pandas import Timestamp
        if isinstance(ts, Timestamp):
            if ts.tz is None:
                return ts.tz_localize("UTC").to_pydatetime()
            else:
                return ts.tz_convert("UTC").to_pydatetime()
        else:
            import pytz
            if getattr(ts, "tzinfo", None) is None:
                return pytz.UTC.localize(ts)
            return ts.astimezone(UTC)
    except Exception:
        return ts

def _is_tls_hiccup(err: Exception) -> bool:
    s = str(err).lower()
    return ("decryption_failed" in s) or ("bad record mac" in s)

def safe_send(context: CallbackContext, **kw):
    for i in range(3):
        try:
            return context.bot.send_message(**kw)
        except Exception as e:
            if _is_tls_hiccup(e):
                time.sleep(0.6 * (i + 1))
                continue
            raise

def safe_reply(update: Update, **kw):
    for i in range(3):
        try:
            return update.message.reply_text(**kw)
        except Exception as e:
            if _is_tls_hiccup(e):
                time.sleep(0.6 * (i + 1))
                continue
            raise

def _is_admin(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    if not ADMIN_CHAT_ID or str(ADMIN_CHAT_ID).strip() in ("", "0", "None"):
        return True
    return str(uid) == str(ADMIN_CHAT_ID)

def _require_admin(func):
    def wrapper(update: Update, context: CallbackContext, *a, **kw):
        if not _is_admin(update):
            try: safe_reply(update, text="Access denied.")
            except Exception: pass
            return
        return func(update, context, *a, **kw)
    return wrapper

def _read_trades_safe() -> List[dict]:
    try:
        if not os.path.exists(TRADES_PATH): return []
        with open(TRADES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_trades(rows: List[dict]):
    try:
        with open(TRADES_PATH, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def _append_trade(row: dict):
    rows = _read_trades_safe()
    rows.append(row)
    _save_trades(rows)

def _now_iso_utc():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def _p(v, nd=2):
    try: return f"{float(v):.{nd}f}"
    except Exception: return str(v)

# ----------------- AI mode scoring -----------------
COOLDOWN_CAUTIOUS_MIN  = int(os.getenv("COOLDOWN_CAUTIOUS_MIN",  "8"))
COOLDOWN_BASELINE_MIN  = int(os.getenv("COOLDOWN_BASELINE_MIN",  "7"))
COOLDOWN_AGGR_MIN      = int(os.getenv("COOLDOWN_AGGR_MIN",      "7"))
AI_MODE_HOLD_MIN       = int(os.getenv("AI_MODE_HOLD_MIN",       "60"))

AI_STATE = {"mode": "BASELINE", "A": 0.5, "last_change_ts": 0.0, "Q": 0.5, "M": 0.5, "R": 0.5}

def _sigmoid(x: float) -> float:
    try: return 1.0 / (1.0 + math.exp(-x))
    except OverflowError: return 0.0 if x < 0 else 1.0

def _score_Q_from_logs(n: int = 30) -> float:
    rows = _read_trades_safe()
    if not rows: return 0.5
    recent = rows[-n:]
    s = 0.0; cnt = 0
    for r in recent:
        res = str(r.get("result","")).upper()
        if res in ("TP1","TP2","EARLY_TP1","EARLY_TP2"): s += 1.0; cnt += 1
        elif res in ("SL",): s -= 1.0; cnt += 1
        elif res in ("MISS", "OPEN"): cnt += 1
    if cnt == 0: return 0.5
    x = s / cnt
    return 0.5 + 0.5 * x

def _score_R_from_logs() -> float:
    rows = _read_trades_safe()
    if not rows: return 0.3
    last8 = rows[-8:]; last4 = rows[-4:]
    sl8 = sum(1 for r in last8 if str(r.get("result","")).upper()=="SL")
    sl4 = sum(1 for r in last4 if str(r.get("result","")).upper()=="SL")
    base = 0.15 * sl8
    cluster = 0.4 if sl4 >= 2 else 0.0
    return float(min(1.0, base + cluster))

def _score_M_from_market() -> float:
    df = get_klines("5", limit=300, include_partial=True)
    if df is None or len(df) < 80: return 0.5
    close = df["close"].astype(float).copy()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()
    hl = (df["high"] - df["low"]).abs().astype(float)
    atr14 = hl.rolling(14).mean()
    with pd.option_context('mode.use_inf_as_na', True):
        atr_ratio = (atr14 / close).fillna(0)
    spread = ema20 - ema50
    slope = (spread.iloc[-1] - spread.iloc[-20]) / 20.0
    slope_norm = float(max(0.0, min(1.0, abs(slope) / (0.002 * max(1.0, close.iloc[-1])))))
    ar = float((atr14.iloc[-1] / max(1.0, close.iloc[-1])))
    if ar <= 0: vol_score = 0.3
    else:
        vol_score = math.exp(-((ar - 0.007) ** 2) / (2 * (0.006 ** 2)))
        vol_score = float(max(0.0, min(1.0, vol_score)))
    body = (df["close"] - df["open"]).abs().astype(float)
    tr = (df["high"] - df["low"]).astype(float)
    with pd.option_context('mode.use_inf_as_na', True):
        br = (body.rolling(20).mean() / tr.rolling(20).mean()).fillna(0.0)
    noise_penalty = float(max(0.0, min(0.4, 0.4 * (1.0 - float(br.iloc[-1])))))
    raw = 0.6 * slope_norm + 0.6 * vol_score - noise_penalty
    return float(max(0.0, min(1.0, raw)))

def _decide_mode(Q: float, M: float, R: float):
    A = _sigmoid(1.1*Q + 0.9*M - 0.7*R - 0.1)
    prev = AI_STATE["mode"]
    now  = time.time()
    hold_ok = (now - AI_STATE.get("last_change_ts",0)) >= AI_MODE_HOLD_MIN*60
    to_caut = A < (0.35 if prev != "CAUTIOUS" else 0.38)
    to_aggr = A > (0.65 if prev != "AGGRESSIVE" else 0.62)
    new_mode = prev
    if hold_ok:
        if to_caut: new_mode = "CAUTIOUS"
        elif to_aggr: new_mode = "AGGRESSIVE"
        else: new_mode = "BASELINE"
    changed = (new_mode != prev) and hold_ok
    if changed: AI_STATE["last_change_ts"] = now
    AI_STATE.update({"mode": new_mode, "A": float(A), "Q": float(Q), "M": float(M), "R": float(R)})
    return changed

def refresh_ai_mode(context: CallbackContext = None, announce: bool = True):
    try:
        Q = _score_Q_from_logs(); M = _score_M_from_market(); R = _score_R_from_logs()
        changed = _decide_mode(Q,M,R)
        if changed and announce and ADMIN_CHAT_ID:
            mode = AI_STATE["mode"]; A = AI_STATE["A"]
            if mode == "CAUTIOUS":
                spec = f"disp +15%, fvg +10%, return −5%, gate +0.07; cooldown={COOLDOWN_CAUTIOUS_MIN}m"
            elif mode == "AGGRESSIVE":
                spec = f"disp −20%, fvg −18%, return +20%, gate −0.10; cooldown={COOLDOWN_AGGR_MIN}m"
            else:
                spec = f"baseline: disp −20%, fvg −15%, return +15%, gate −0.09; cooldown={COOLDOWN_BASELINE_MIN}m"
            safe_send(context, chat_id=ADMIN_CHAT_ID,
                      text=f"🧠 AI Mode: {mode} (A={A:.2f}) — {spec}",
                      parse_mode=ParseMode.MARKDOWN)
    except Exception:
        pass

def _current_tuning() -> Dict[str, float]:
    """
    Tuning multipliers for detector thresholds.
    """
    m = AI_STATE["mode"]
    if m == "CAUTIOUS":
        return {"disp_mult": 1.15, "fvg_mult": 1.10, "return_mult": 0.95, "conf_adj": +0.07}
    if m == "AGGRESSIVE":
        # a touch looser so Aggressive fires more often
        return {"disp_mult": 0.80, "fvg_mult": 0.82, "return_mult": 1.20, "conf_adj": -0.10}
    # BASELINE — one notch looser for more entries
    return {"disp_mult": 0.80, "fvg_mult": 0.85, "return_mult": 1.15, "conf_adj": -0.09}

# ----------------- P&L → price distances -----------------
def _dist_from_pnl_usd(pnl_usd: float, qty_btc: float) -> float:
    if qty_btc <= 0: qty_btc = 0.101
    return float(abs(pnl_usd) / qty_btc)

def _enforce_levels_from_dollars(st: TradeState, risk_usd: float, tp1_pts: float, tp2_pts: float, qty_btc: float):
    entry = float(st.entry_price or st.entry_ref)
    risk_dist = _dist_from_pnl_usd(risk_usd, qty_btc)
    tp1_dist  = float(tp1_pts); tp2_dist = float(tp2_pts)
    if st.side == "BUY":
        stop_px = entry - risk_dist
        tp1_px  = entry + tp1_dist
        tp2_px  = entry + tp2_dist
        if getattr(st,"stop_px",None) is not None: stop_px = min(stop_px,float(st.stop_px))
        if getattr(st,"tp1_px",None)  is not None: tp1_px  = max(tp1_px, float(st.tp1_px))
        if getattr(st,"tp2_px",None)  is not None: tp2_px  = max(tp2_px, float(st.tp2_px))
    else:
        stop_px = entry + risk_dist
        tp1_px  = entry - tp1_dist
        tp2_px  = entry - tp2_dist
        if getattr(st,"stop_px",None) is not None: stop_px = max(stop_px,float(st.stop_px))
        if getattr(st,"tp1_px",None)  is not None: tp1_px  = min(tp1_px, float(st.tp1_px))
        if getattr(st,"tp2_px",None)  is not None: tp2_px  = min(tp2_px, float(st.tp2_px))
    st.stop_px = float(stop_px); st.tp1_px = float(tp1_px); st.tp2_px = float(tp2_px)

# ----------------- commentary builders -----------------
def _commentary_trade_event(st: TradeState, event: str, px: float) -> str:
    base = f"`{st.id}` {event} @ ${px:,.2f}"
    if event == "TP1":       return f"✅ **TP1 HIT** — +$600 bag secured 🔥\n{base}"
    if event == "TP2":       return f"💰 **TP2 SNIPED** — +$1500 locked 🚀\n{base}"
    if event == "EARLY_TP1": return f"✅ **Early TP1** — Protecting edge 🎯\n{base}"
    if event == "EARLY_TP2": return f"✅ **Early TP2** — Reversal risk spiking; banking win ⚡️\n{base}"
    if event == "SL":        return f"❌ **SL -$40** — pain logged, filters tightening ⚔️\n{base}"
    if event == "CANCELLED": return f"⛔ **Cancelled** — Pre-entry rejection saved us.\n{base}"
    return f"ℹ️ {base}"

# ----------------- core loops -----------------
def _full_setup_text(sig: Signal) -> str:
    try:
        meta = getattr(sig, "meta", {}) or {}
        sweep = meta.get("swept_level") or getattr(sig, "swept_level", None)
        fvg_lo = meta.get("fvg_low") or getattr(sig, "fvg_low", None)
        fvg_hi = meta.get("fvg_high") or getattr(sig, "fvg_high", None)
        disp   = meta.get("disp_size") or getattr(sig, "disp_size", None)
        ote_lo = meta.get("ote_low") or getattr(sig, "ote_low", None)
        ote_hi = meta.get("ote_high") or getattr(sig, "ote_high", None)
        parts = []
        if sweep is not None: parts.append(f"Swept: <b>{_p(sweep)}</b>")
        if fvg_lo is not None and fvg_hi is not None: parts.append(f"FVG: <b>{_p(fvg_lo)}</b>–<b>{_p(fvg_hi)}</b>")
        if disp is not None: parts.append(f"Displacement: <b>{_p(disp)}</b>")
        if ote_lo is not None and ote_hi is not None: parts.append(f"OTE: <b>{_p(ote_lo)}</b>–<b>{_p(ote_hi)}</b>")
        if parts:
            return " • ".join(parts)
        return "Pattern: sweep → displacement → FVG return"
    except Exception:
        return "Pattern: sweep → displacement → FVG return"

def scan_once(context: CallbackContext):
    refresh_ai_mode(context, announce=False)
    if tm.active and not tm.active.closed:
        return

    # Larger windows, still within API limits
    df5  = get_klines("5",  limit=800, include_partial=False)
    if df5 is None or len(df5) < 210:
        return

    tuning = _current_tuning()
    try:
        sig = detector.find(df5, bayes, knn, bandit, bans, tuning=tuning)
    except TypeError:
        sig = detector.find(df5, bayes, knn, bandit, bans)

    # If no 5m signal, try 15m
    if not sig:
        df15 = get_klines("15", limit=600, include_partial=False)
        if df15 is not None and len(df15) >= 210:
            try:
                sig = detector.find(df15, bayes, knn, bandit, bans, tuning=tuning)
            except TypeError:
                sig = detector.find(df15, bayes, knn, bandit, bans)

    if not sig or not tm.can_open():
        return

    st = tm.open_from_signal(sig)
    if not st:
        return

    # Enforce levels per your rules
    try:
        qty = float(getattr(st, "qty", None) or CFG_QTY_BTC or 0.101)
        _enforce_levels_from_dollars(st, RISK_DOLLARS, 600.0, 1500.0, qty)
    except Exception:
        pass

    # Persist OPEN
    _append_trade({
        "time": _now_iso_utc(),
        "symbol": SYMBOL,
        "side": st.side,
        "entry": float(st.entry_ref),
        "sl": float(st.stop_px),
        "tp1": float(st.tp1_px),
        "tp2": float(st.tp2_px),
        "result": "OPEN",
        "tf": "5m",
        "source": "live"
    })

    # Single, full entry message (HTML)
    if ADMIN_CHAT_ID:
        setup_lines = _full_setup_text(sig)
        txt = (
            f"🎯 <b>ENTRY LIVE</b> — <code>{st.side}</code> on <code>{SYMBOL}</code>\n"
            f"Entry <b>{_p(st.entry_ref)}</b> | SL <b>{_p(st.stop_px)}</b> | "
            f"TP1 <b>{_p(st.tp1_px)}</b> | TP2 <b>{_p(st.tp2_px)}</b>\n"
            f"Risk: ${RISK_DOLLARS:.0f} (stretch ≤ ${STRETCH_MAX_DOLLARS:.0f})\n"
            f"{setup_lines}\n"
            f"Confidence: {_p(getattr(sig,'conf',0.0),2)}"
        )
        safe_send(context, chat_id=ADMIN_CHAT_ID, text=txt, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

def manage_once(context: CallbackContext):
    if not (tm.active and not tm.active.closed):
        return
    st = tm.active
    df1 = get_klines("1", limit=120, include_partial=True)
    df5 = get_klines("5", limit=210, include_partial=True)
    if df1 is None or df5 is None or len(df1) < 30 or len(df5) < 50:
        return

    ai_conf = 0.78
    tm.maybe_stretch(st, ai_conf)

    ev = tm.manage(st, df1, df5)
    if not ev:
        if not ASSISTANT_MODE or not ADMIN_CHAT_ID:
            return
        try:
            cur = float(df1["close"].iloc[-1])
            mae = abs(cur - (st.entry_price or st.entry_ref))
            delta = abs(st.entry_ref - st.stop_px) or 1.0
            mae_ratio = float(min(1.0, mae / delta))
            prob_sl = 0.35; ev_knn = 0.55; rej_score = 0.1; struct_loss = 0.3; vol_shift = 0.1
            H = 0.32*prob_sl + 0.18*(1-ev_knn) + 0.20*rej_score + 0.12*mae_ratio + 0.12*struct_loss + 0.06*vol_shift
            if H >= ASSISTANT_STRONG:
                safe_send(context, chat_id=ADMIN_CHAT_ID, text="🛑 <b>Cut Recommended</b> — risk outweighs hold. Bank discipline now.", parse_mode=ParseMode.HTML)
            elif H >= ASSISTANT_CUT:
                safe_send(context, chat_id=ADMIN_CHAT_ID, text="⚠️ <b>Weakening</b> — EV_exit > EV_hold. Consider partial or exit.", parse_mode=ParseMode.HTML)
            elif H >= ASSISTANT_WARN:
                safe_send(context, chat_id=ADMIN_CHAT_ID, text="ℹ️ <b>Heads-up</b> — momentum softening; tighten BE buffer.", parse_mode=ParseMode.HTML)
        except Exception:
            pass
        return

    if ADMIN_CHAT_ID:
        safe_send(context, chat_id=ADMIN_CHAT_ID, text=_commentary_trade_event(st, ev["event"], ev["price"]), parse_mode=ParseMode.MARKDOWN)

    if ev["event"] in ("TP2", "SL", "EARLY_TP2", "TP1", "EARLY_TP1", "CANCELLED"):
        if ev["event"] in ("TP2", "SL", "EARLY_TP2"): tm.close(st, ev["event"], ev["price"])
        R = trade_reward(st, ev["event"])
        bandit.update(st.arm_id, norm_reward(R), weight=1.5 if "TP2" in ev["event"] else 1.0)

        _append_trade({
            "time": _now_iso_utc(),
            "symbol": SYMBOL,
            "side": st.side,
            "entry": float(st.entry_price or st.entry_ref),
            "exit": float(ev["price"]),
            "sl": float(st.stop_px),
            "tp1": float(st.tp1_px),
            "tp2": float(st.tp2_px),
            "result": ev["event"],
            "tf": "5m",
            "source": "live"
        })

        # mode-based cooldown after SL
        if ev["event"] == "SL":
            refresh_ai_mode(context, announce=False)
            mode = AI_STATE["mode"]
            if mode == "CAUTIOUS": cd_min = COOLDOWN_CAUTIOUS_MIN
            elif mode == "AGGRESSIVE": cd_min = COOLDOWN_AGGR_MIN
            else: cd_min = COOLDOWN_BASELINE_MIN
            now_ms = int(time.time()*1000)
            tm.cooldown_until_ms = max(tm.cooldown_until_ms, now_ms + cd_min*60*1000)
            if ADMIN_CHAT_ID:
                safe_send(context, chat_id=ADMIN_CHAT_ID, text=f"⏳ Cooldown set to {cd_min} min (AI mode: {mode}; chop filters still ON).")

# ----------------- session & recaps -----------------
_last_session_sent = {"LONDON": None, "NY": None}
def session_commentary(context: CallbackContext):
    import pytz
    ist = pytz.timezone(TZ_NAME if TZ_NAME else "Asia/Kolkata")
    now = datetime.now(ist); h = now.hour; day = now.strftime("%Y-%m-%d")
    global _last_session_sent
    if 12 <= h <= 17:
        if _last_session_sent.get("LONDON") != day and ADMIN_CHAT_ID:
            safe_send(context, chat_id=ADMIN_CHAT_ID, text="📈 London in play: likely liquidity engineering. Watch for sweep → displacement. Sniper mode ON 🔥")
            _last_session_sent["LONDON"] = day
    if 19 <= h <= 23:
        if _last_session_sent.get("NY") != day and ADMIN_CHAT_ID:
            safe_send(context, chat_id=ADMIN_CHAT_ID, text="⚡️ NY session: breakout probability elevated. Track displacement legs and FVG returns 🚀")
            _last_session_sent["NY"] = day

def daily_recap(context: CallbackContext):
    if not ADMIN_CHAT_ID: return
    today = datetime.now().strftime("%Y-%m-%d")
    safe_send(context, chat_id=ADMIN_CHAT_ID,
        text=(f"📅 <b>Daily Recap — {today}</b>\nAsia: context logged • London/NY: commentary posted\nTrades: check logs.\n🔥 One clean kill > many messy shots."),
        parse_mode=ParseMode.HTML)

def weekly_recap(context: CallbackContext):
    if not ADMIN_CHAT_ID: return
    safe_send(context, chat_id=ADMIN_CHAT_ID,
        text="📆 <b>Weekly Recap</b>\nWins/Losses, EV improvements, chop avoidance, and session chains summarised. Keep sniping ⚔️",
        parse_mode=ParseMode.HTML)

# ----------------- commands -----------------
@_require_admin
def cmd_start(update: Update, context: CallbackContext):
    safe_reply(update, text=f"Online. Symbol `{SYMBOL}`. 1-trade lock; risk ${RISK_DOLLARS:.0f}.", parse_mode=ParseMode.MARKDOWN)

@_require_admin
def cmd_menu(update: Update, context: CallbackContext):
    text = (
        "<b>Menu</b>\n"
        "/start – Bot online & risk\n"
        "/aistats – AI mode, Q/M/R, cooldown map\n"
        "/diag – Data freshness & lock/cooldown\n"
        "/status – Active trade & levels\n"
        "/scan – Force a 5m/15m scan now\n"
        "/backtest <days> [tf] – e.g. /backtest 7 5 or /backtest 30 15\n"
        "/btdebug <days> [tf] – bars & signals count\n"
        "/logs – Last logged trades\n"
        "/check_data – MEXC health\n"
        "/force_unlock – Clear 1-trade lock & cooldown\n"
    )
    safe_reply(update, text=text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@_require_admin
def cmd_scan(update: Update, context: CallbackContext):
    try:
        scan_once(context)
        safe_reply(update, text="Scan done.")
    except Exception as e:
        safe_reply(update, text=f"Scan error: {e}")

@_require_admin
def cmd_status(update: Update, context: CallbackContext):
    if tm.active and not tm.active.closed:
        st = tm.active
        msg = (f"*Active*: `{st.id}` {st.side} qty={st.qty:.6f}\n"
               f"Entry {st.entry_ref:.2f} • Stop {st.stop_px:.2f} • TP1 {st.tp1_px:.2f} • TP2 {st.tp2_px:.2f}\n"
               f"Filled={st.filled} • Stretched={st.stretched}")
    else:
        left_ms = max(0, tm.cooldown_until_ms - int(time.time()*1000))
        msg = f"No active trade. Cooldown {left_ms//1000}s."
    safe_reply(update, text=msg, parse_mode=ParseMode.MARKDOWN)

@_require_admin
def cmd_check_data(update: Update, context: CallbackContext):
    try:
        df5 = get_klines("5", limit=10, include_partial=True)
        if df5 is None or df5.empty:
            safe_reply(update, text="❌ No data from MEXC.")
            return
        ts = safe_utc(df5.index[-1])
        safe_reply(update, text=f"✅ MEXC OK. 5m candles: {len(df5)}. Last bar @ {ts.isoformat()}")
    except Exception as e:
        safe_reply(update, text=f"❌ MEXC error: {e}")

@_require_admin
def cmd_backtest(update: Update, context: CallbackContext):
    try:
        days = 7; tf = "5"
        if context.args:
            if len(context.args) >= 1:
                try: days = max(1, min(30, int(context.args[0])))
                except Exception: pass
            if len(context.args) >= 2 and context.args[1] in ("1","5","15","30","60","1m","5m","15m","30m","1h"):
                tf = context.args[1].replace("m","");  tf = "60" if tf=="1h" else tf
        safe_reply(update, text=f"⏳ Running backtest for last {days} days on {tf}m...")
        from utils import backtest_strategy
        results = backtest_strategy(days=days, tf=tf)
        if not results:
            safe_reply(update, text="✅ Backtest complete. No valid setups in the window.")
            return
        lines = [f"📊 Backtest Results (last {days} days, {tf}m):"]
        for r in results[-10:]:
            lines.append(f"{r['time']} | {r['symbol']} | {r['side']} @ {r['entry']} SL {r['sl']} TP {r['tp']} → {r['result']}")
        safe_reply(update, text="\n".join(lines))
    except Exception as e:
        safe_reply(update, text=f"⚠️ Backtest failed: {e}")

@_require_admin
def cmd_logs(update: Update, context: CallbackContext):
    trades = _read_trades_safe()
    if not trades:
        safe_reply(update, text="No saved trades yet.")
        return
    lines = ["🧾 Last trades:"]
    for t in trades[-10:]:
        when = t.get("time","?")
        sym  = t.get("symbol", SYMBOL)
        side = t.get("side","?")
        ent  = t.get("entry","?")
        res  = t.get("result", t.get("exit_reason","?"))
        src  = t.get("source","live")
        tf   = t.get("tf","")
        tf_s = f" [{tf}]" if tf else ""
        lines.append(f"{when} | {sym}{tf_s} | {side} @ {ent} → {res} ({src})")
    safe_reply(update, text="\n".join(lines))

@_require_admin
def cmd_btdebug(update: Update, context: CallbackContext):
    try:
        days = 7; tf = "5"
        if context.args:
            if len(context.args) >= 1:
                try: days = max(1, min(30, int(context.args[0])))
                except Exception: pass
            if len(context.args) >= 2 and context.args[1] in ("1","5","15","30","60","1m","5m","15m","30m","1h"):
                tf = context.args[1].replace("m",""); tf = "60" if tf=="1h" else tf
        from utils import backtest_strategy_debug
        d = backtest_strategy_debug(days=days, tf=tf)
        msg = (
            f"🔎 Backtest Debug ({days}d, {tf}m)\n"
            f"Bars: {d.get('bars',0)}\n"
            f"Windows tested: {d.get('windows_tested',0)}\n"
            f"Signals found: {d.get('signals_found',0)}\n"
            f"First bar: {d.get('first_ts')}\n"
            f"Last bar: {d.get('last_ts')}\n"
        )
        safe_reply(update, text=msg)
    except Exception as e:
        safe_reply(update, text=f"btdebug error: {e}")

@_require_admin
def cmd_diag(update: Update, context: CallbackContext):
    try:
        df5 = get_klines("5", limit=800, include_partial=False)
        df1 = get_klines("1", limit=120, include_partial=True)
        parts = []
        if df5 is None or df5.empty:
            parts.append("5m: ❌ no data")
        else:
            ts5 = safe_utc(df5.index[-1]); parts.append(f"5m: {len(df5)} bars (last {ts5.isoformat()})")
        if df1 is None or df1.empty:
            parts.append("1m: ❌ no data")
        else:
            ts1 = safe_utc(df1.index[-1]); parts.append(f"1m: {len(df1)} bars (last {ts1.isoformat()})")
        if tm.active and not tm.active.closed:
            st = tm.active
            parts.append(f"Active: {st.id} {st.side} @ {st.entry_ref:.2f} SL {st.stop_px:.2f} TP1 {st.tp1_px:.2f} TP2 {st.tp2_px:.2f}")
        else:
            left_ms = max(0, tm.cooldown_until_ms - int(time.time()*1000))
            parts.append(f"No active trade. Cooldown: {left_ms//1000}s")
        safe_reply(update, text=" • ".join(parts))
    except Exception as e:
        safe_reply(update, text=f"diag error: {e}")

@_require_admin
def cmd_force_unlock(update: Update, context: CallbackContext):
    try:
        tm.cooldown_until_ms = 0
        if tm.active and tm.active.closed:
            tm.active = None
        safe_reply(update, text="🔓 Trade lock & cooldown cleared.")
    except Exception as e:
        safe_reply(update, text=f"force_unlock error: {e}")

@_require_admin
def cmd_aistats(update: Update, context: CallbackContext):
    try:
        refresh_ai_mode(context, announce=False)
        mins_left = max(0, int((AI_MODE_HOLD_MIN*60 - (time.time() - AI_STATE.get("last_change_ts",0)))//60))
        s = AI_STATE
        msg = (f"🤖 AI Stats\n"
               f"Mode: {s['mode']} (A={s['A']:.2f}), hold ≥{AI_MODE_HOLD_MIN}m (≈{mins_left}m left)\n"
               f"Q={s['Q']:.2f} (quality) • M={s['M']:.2f} (regime) • R={s['R']:.2f} (risk pressure)\n"
               f"Cooldown map → Caut:{COOLDOWN_CAUTIOUS_MIN}m | Base:{COOLDOWN_BASELINE_MIN}m | Aggr:{COOLDOWN_AGGR_MIN}m")
        safe_reply(update, text=msg)
    except Exception as e:
        safe_reply(update, text=f"aistats error: {e}")

def error_handler(update: object, context: CallbackContext):
    try:
        msg = f"Error: {context.error}\n{traceback.format_exc()[:900]}"
        if ADMIN_CHAT_ID:
            safe_send(context, chat_id=ADMIN_CHAT_ID, text=msg)
    except Exception:
        pass

# ----------------- PTB bootstrap (no polling) -----------------
req = Request(con_pool_size=1, connect_timeout=20, read_timeout=20)
updater = Updater(TOKEN, use_context=True, request=req)
dp = updater.dispatcher
jq = updater.job_queue

dp.add_handler(CommandHandler("start",        cmd_start))
dp.add_handler(CommandHandler("menu",         cmd_menu))
dp.add_handler(CommandHandler("scan",         cmd_scan))
dp.add_handler(CommandHandler("status",       cmd_status))
dp.add_handler(CommandHandler("check_data",   cmd_check_data))
dp.add_handler(CommandHandler("backtest",     cmd_backtest))
dp.add_handler(CommandHandler("logs",         cmd_logs))
dp.add_handler(CommandHandler("btdebug",      cmd_btdebug))
dp.add_handler(CommandHandler("diag",         cmd_diag))
dp.add_handler(CommandHandler("force_unlock", cmd_force_unlock))
dp.add_handler(CommandHandler("aistats",      cmd_aistats))
dp.add_error_handler(error_handler)

# jobs (30s cadence for more responsiveness)
jq.run_repeating(manage_once,         interval=30,  first=10)
jq.run_repeating(scan_once,           interval=30,  first=15)
jq.run_repeating(session_commentary,  interval=60*30, first=30)
jq.run_repeating(lambda ctx: refresh_ai_mode(ctx, announce=True), interval=300, first=10)
jq.run_daily(daily_recap,  time=dtime(17,  0, 0, tzinfo=UTC))
jq.run_daily(weekly_recap, time=dtime(17, 30, 0, tzinfo=UTC), days=(6,))
jq.start()

# ----------------- Flask webhook -----------------
app = Flask(__name__)

def setup_webhook_once():
    host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    token = os.environ.get("TOKEN")
    if not host or not token:
        return
    url = f"https://{host}/{token}"
    try: updater.bot.delete_webhook()
    except Exception: pass
    updater.bot.set_webhook(url=url)

setup_webhook_once()

@app.route(f"/{os.environ['TOKEN']}", methods=["POST"])
def telegram_webhook():
    update = Update.de_json(request.get_json(force=True), updater.bot)
    updater.dispatcher.process_update(update)
    return "ok", 200

@app.get("/")
def index():
    return "OK", 200
