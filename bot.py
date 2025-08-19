import os, time, math, traceback
from datetime import datetime, timedelta, timezone
import os
from flask import Flask, request
from telegram import Update
# you already import Updater, CommandHandler, etc. earlier in your file
from typing import Optional
import pandas as pd
from dotenv import load_dotenv
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext, MessageHandler, Filters

from config import (
    TOKEN, ADMIN_CHAT_ID, TZ_NAME, SYMBOL,
    ASSISTANT_MODE, ASSISTANT_WARN, ASSISTANT_CUT, ASSISTANT_STRONG,
    RISK_DOLLARS, STRETCH_MAX_DOLLARS, TP1_DOLLARS, TP2_DOLLARS,
    COOLDOWN_MIN, FOCUS_COOLDOWN_MIN, POST_SL2_LOCK_MIN,
)
from storage import ensure_dirs, daily_maintenance
from mexc_client import get_klines
from strategy import DoLDetector, TradeManager, trade_reward, norm_reward
from strategy import Signal, TradeState
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan, combined_conf, hazard
from indicators import atr

load_dotenv()
ensure_dirs()

# --- Global singletons (kept lightweight for 2GB/1CPU) ---
bandit = PolicyArms()
knn = PatternKNN()
bayes = BayesModel()
bans = CFPBan()
detector = DoLDetector(bandit)
tm = TradeManager()

# --- Helpers ---
def is_admin(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return str(uid) == str(ADMIN_CHAT_ID)

def require_admin(func):
    def wrapper(update: Update, context: CallbackContext, *a, **kw):
        if not is_admin(update):
            update.message.reply_text("Access denied.")
            return
        return func(update, context, *a, **kw)
    return wrapper

def send(context: CallbackContext, text: str, pm: bool = True):
    try:
        context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception:
        pass

def fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts/1000, tz=timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S")

def commentary_entry(sig: Signal) -> str:
    side = "Short" if sig.side=="SELL" else "Long"
    return (
        f"🎯 **Entry lined up** — *{side}* on `{SYMBOL}`\n"
        f"• Risk: ${RISK_DOLLARS:.0f} (stretch ≤ ${STRETCH_MAX_DOLLARS:.0f})\n"
        f"• TP1: +${TP1_DOLLARS:.0f} • TP2: +${TP2_DOLLARS:.0f}\n"
        f"• Reason: {sig.reason}\n"
        f"• Confidence: {sig.conf:.2f}"
    )

def commentary_trade_event(st: TradeState, event: str, px: float) -> str:
    base = f"`{st.id}` {event} @ ${px:,.2f}"
    if event == "TP1": return f"✅ **TP1 HIT** — +$600 bag secured 🔥\n{base}"
    if event == "TP2": return f"💰 **TP2 SNIPED** — +$1500 locked 🚀\n{base}"
    if event == "EARLY_TP1": return f"✅ **Early TP1** — Protecting edge 🎯\n{base}"
    if event == "EARLY_TP2": return f"✅ **Early TP2** — Reversal risk spiking; banking win ⚡️\n{base}"
    if event == "SL": return f"❌ **SL -$40** — pain logged, filters tightening ⚔️\n{base}"
    if event == "CANCELLED": return f"⛔ **Cancelled** — Pre-entry rejection saved us.\n{base}"
    return f"ℹ️ {base}"

# --- Core scan/open loop ---
def scan_once(context: CallbackContext):
    # If active trade exists, we only manage it from job_manage.
    if tm.active and not tm.active.closed:
        return

    # 5m data for setup
    df5 = get_klines("5", limit=500, include_partial=False)
    if df5 is None or len(df5) < 210:
        return

    sig = detector.find(df5, bayes, knn, bandit, bans)
    if not sig:
        return

    # Open if gates okay and cooldown allows
    if not tm.can_open():
        return

    st = tm.open_from_signal(sig)
    if st:
        send(context, commentary_entry(sig))
        # On entry we post expert note too (concise)
        send(context, f"Expert: Liquidity swept, displacement confirmed, FVG return in play. "
                      f"Risk sized to ${RISK_DOLLARS:.0f}. Trade `{st.id}` tracking.")

def manage_once(context: CallbackContext):
    # manage active trade every minute
    if not (tm.active and not tm.active.closed):
        return

    st = tm.active
    # Live 1m/5m
    df1 = get_klines("1", limit=120, include_partial=True)
    df5 = get_klines("5", limit=210, include_partial=True)
    if df1 is None or df5 is None or len(df1) < 30 or len(df5) < 50:
        return

    # AI stretch gate (once)
    # Estimate confidence using current features around last 5m bar (cheap proxy)
    a5 = float(atr(df5, 14).iloc[-1])
    # simple proxy for model prob/ev using last signal's feats not stored; we just keep conservative conf bump
    ai_conf = 0.75  # base; full calc is in detector; here we keep conservative default
    tm.maybe_stretch(st, ai_conf)

    ev = tm.manage(st, df1, df5)
    if not ev:
        # Assistant Advisor: hazard-based warnings (lightweight calc)
        if not ASSISTANT_MODE: return
        # rough hazard inputs
        prob_sl = 0.35  # baseline; replaced by model when integrated with full features
        ev_knn = 0.55
        rej_score = 1.0 if "EARLY" in (st.exit_reason or "") else 0.0
        # MAE/Δ proxy:
        cur = float(df1["close"].iloc[-1])
        mae = abs(cur - (st.entry_price or st.entry_ref))
        delta = abs(st.entry_ref - st.stop_px)
        mae_ratio = float(np.clip(mae / max(delta, 1e-9), 0, 1))
        # structure loss & vol shift (cheap proxies)
        struct_loss = 0.3
        vol_shift = 0.1

        H = hazard(prob_sl, ev_knn, rej_score, mae_ratio, struct_loss, vol_shift)
        if H >= ASSISTANT_STRONG:
            send(context, "🛑 *Cut Recommended* — risk outweighs hold. Bank discipline now.",)
        elif H >= ASSISTANT_CUT:
            send(context, "⚠️ *Weakening* — EV_exit > EV_hold. Consider partial or exit.")
        elif H >= ASSISTANT_WARN:
            send(context, "ℹ️ *Heads-up* — momentum softening; tighten BE buffer.")
        return

    # Trade event fired
    send(context, commentary_trade_event(st, ev["event"], ev["price"]))
    # Close if terminal
    if ev["event"] in ("TP2", "SL", "EARLY_TP2"):
        tm.close(st, ev["event"], ev["price"])
        # Reward + learning
        R = trade_reward(st, ev["event"])
        bandit.update(st.arm_id, norm_reward(R), weight=1.5 if "TP2" in ev["event"] else 1.0)

def session_commentary(context: CallbackContext):
    # Lightweight market scout commentary
    now = datetime.now(timezone.utc).astimezone()
    hour = now.hour
    # Just-in-time note; you can enrich with detector telemetry if desired
    if 13 <= hour <= 17:  # London-ish (IST display anyway)
        send(context, "📈 London open: liquidity engineering likely. Watch for sweep → displacement. Sniper mode ON 🔥")
    if 19 <= hour <= 22:  # NY window (IST)
        send(context, "⚡️ NY session: breakout probabilities elevated. Track VWAP decision and displacement legs 🚀")

def daily_recap(context: CallbackContext):
    # Very light placeholder; reporter.py provides full report if you want file output
    now = datetime.now().strftime("%Y-%m-%d")
    send(context, f"📅 *Daily Recap — {now}*\n"
                  f"Asia: context logged • London/NY: commentary posted\n"
                  f"Trades: see logs.\n"
                  f"🔥 Clean execution focus tomorrow.")

def weekly_recap(context: CallbackContext):
    send(context, "📆 *Weekly Recap*\nWins/Losses, EV improvements, chop avoidance, and session chains summarised. Keep sniping ⚔️")

# --- Telegram commands ---
@require_admin
def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text(f"Online. Symbol `{SYMBOL}`. 1-trade lock; risk ${RISK_DOLLARS:.0f}.",
                              parse_mode=ParseMode.MARKDOWN)

@require_admin
def cmd_scan(update: Update, context: CallbackContext):
    try:
        scan_once(context)
        update.message.reply_text("Scan done.", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        update.message.reply_text(f"Scan error: {e}")

@require_admin
def cmd_status(update: Update, context: CallbackContext):
    if tm.active and not tm.active.closed:
        st = tm.active
        msg = (f"*Active*: `{st.id}` {st.side} qty={st.qty:.6f}\n"
               f"EntryRef {st.entry_ref:.2f} • Stop {st.stop_px:.2f} • TP1 {st.tp1_px:.2f} • TP2 {st.tp2_px:.2f}\n"
               f"Filled={st.filled} • Stretched={st.stretched}")
    else:
        left_ms = max(0, tm.cooldown_until_ms - int(time.time()*1000))
        msg = f"No active trade. Cooldown {left_ms//1000}s."
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

@require_admin
def cmd_help(update: Update, context: CallbackContext):
    update.message.reply_text(
        "/start • /scan • /status\n"
        "/daily_now • /weekly_now\n"
        "Bot runs auto-jobs: scan (2m), manage (1m), session notes, daily & weekly recaps.",
        parse_mode=ParseMode.MARKDOWN,
    )

@require_admin
def cmd_daily_now(update: Update, context: CallbackContext):
    daily_recap(context)
    update.message.reply_text("Daily recap sent.")

@require_admin
def cmd_weekly_now(update: Update, context: CallbackContext):
    weekly_recap(context)
    update.message.reply_text("Weekly recap sent.")

def error_handler(update: object, context: CallbackContext):
    try:
        msg = f"Error: {context.error}\n{traceback.format_exc()[:800]}"
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
    except Exception:
        pass

def main():
    updater = Updater(TOKEN, use_context=True)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler("start", cmd_start))
    dp.add_handler(CommandHandler("scan", cmd_scan))
    dp.add_handler(CommandHandler("status", cmd_status))
    dp.add_handler(CommandHandler("help", cmd_help))
    dp.add_handler(CommandHandler("daily_now", cmd_daily_now))
    dp.add_handler(CommandHandler("weekly_now", cmd_weekly_now))
    dp.add_error_handler(error_handler)

    jq = updater.job_queue
    # Jobs: manage active trade every 60s, scan every 120s (only opens if none active)
    jq.run_repeating(manage_once, interval=60, first=10)
    jq.run_repeating(scan_once, interval=120, first=15)
    # Session commentary (rough cadence)
    jq.run_repeating(session_commentary, interval=60*30, first=30)    # every 30m
    # Daily recap ~ 22:30 IST (17:00 UTC approx)
    now_utc = datetime.now(timezone.utc)
    daily_first = (now_utc + timedelta(minutes=5))  # start soon; adjust to your schedule
    jq.run_daily(daily_recap, time=time(17,0,0, tzinfo=timezone.utc))
    # Weekly recap: Sunday 17:30 UTC (Mon 23:00 IST approx)
    jq.run_daily(weekly_recap, time=time(17,30,0, tzinfo=timezone.utc), days=(6,))  # Sunday

# ---- Flask webhook server for Render/Gunicorn ----
app = Flask(__name__)

@app.before_first_request
def _setup_webhook():
    # Set Telegram webhook when the server starts
    host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    token = os.environ["TOKEN"]
    if host and token:
        url = f"https://{host}/{token}"
        try:
            updater.bot.delete_webhook()
        except Exception:
            pass
        updater.bot.set_webhook(url=url)

@app.route(f"/{os.environ['TOKEN']}", methods=["POST"])
def telegram_webhook():
    update = Update.de_json(request.get_json(force=True), updater.bot)
    updater.dispatcher.process_update(update)
    return "ok", 200

# Optional health check
@app.route("/", methods=["GET"])
def index():
    return "OK", 200
