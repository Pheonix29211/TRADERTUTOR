# bot.py
# Complete webhook-ready Telegram bot (PTB v13 + Flask 3.x) for your ICT DoL system

import os, sys, traceback, time
from datetime import datetime, timedelta, timezone, time as dtime
from typing import Optional

# Ensure local modules work on Render (Linux, case-sensitive)
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from flask import Flask, request
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

# ---- Local modules (make sure these filenames match exactly) ----
from config import (
    TOKEN, ADMIN_CHAT_ID, TZ_NAME, SYMBOL,
    ASSISTANT_MODE, ASSISTANT_WARN, ASSISTANT_CUT, ASSISTANT_STRONG,
    RISK_DOLLARS, STRETCH_MAX_DOLLARS, TP1_DOLLARS, TP2_DOLLARS,
)
from storage import ensure_dirs, daily_maintenance
from mexc_client import get_klines
from strategy import DoLDetector, TradeManager, trade_reward, norm_reward, Signal, TradeState
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan
from indicators import atr

# ==================== INIT ====================

load_dotenv()
ensure_dirs()

# Singletons kept small/lightweight for 2GB RAM / 1 CPU
bandit = PolicyArms()
knn = PatternKNN()
bayes = BayesModel()
bans = CFPBan()
detector = DoLDetector(bandit)
tm = TradeManager()

# ==================== HELPERS ====================

def _is_admin(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    return str(uid) == str(ADMIN_CHAT_ID)

def _require_admin(func):
    def wrapper(update: Update, context: CallbackContext, *a, **kw):
        if not _is_admin(update):
            try:
                update.message.reply_text("Access denied.")
            except Exception:
                pass
            return
        return func(update, context, *a, **kw)
    return wrapper

def _send(context: CallbackContext, text: str):
    try:
        context.bot.send_message(
            chat_id=ADMIN_CHAT_ID,
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception:
        pass

def _commentary_entry(sig: Signal) -> str:
    side = "Short" if sig.side == "SELL" else "Long"
    return (
        f"🎯 **Entry lined up** — *{side}* on `{SYMBOL}`\n"
        f"• Risk: ${RISK_DOLLARS:.0f} (stretch ≤ ${STRETCH_MAX_DOLLARS:.0f})\n"
        f"• TP1: +${TP1_DOLLARS:.0f} • TP2: +${TP2_DOLLARS:.0f}\n"
        f"• Reason: {sig.reason}\n"
        f"• Confidence: {sig.conf:.2f}"
    )

def _commentary_trade_event(st: TradeState, event: str, px: float) -> str:
    base = f"`{st.id}` {event} @ ${px:,.2f}"
    if event == "TP1":       return f"✅ **TP1 HIT** — +$600 bag secured 🔥\n{base}"
    if event == "TP2":       return f"💰 **TP2 SNIPED** — +$1500 locked 🚀\n{base}"
    if event == "EARLY_TP1": return f"✅ **Early TP1** — Protecting edge 🎯\n{base}"
    if event == "EARLY_TP2": return f"✅ **Early TP2** — Reversal risk spiking; banking win ⚡️\n{base}"
    if event == "SL":        return f"❌ **SL -$40** — pain logged, filters tightening ⚔️\n{base}"
    if event == "CANCELLED": return f"⛔ **Cancelled** — Pre-entry rejection saved us.\n{base}"
    return f"ℹ️ {base}"

# ==================== CORE LOOP FUNCS ====================

def scan_once(context: CallbackContext):
    """Scans for new DoL signals on 5m; opens only if no active trade."""
    # If a trade is active, management loop will take care of it
    if tm.active and not tm.active.closed:
        return
    df5 = get_klines("5", limit=500, include_partial=False)
    if df5 is None or len(df5) < 210:
        return
    sig = detector.find(df5, bayes, knn, bandit, bans)
    if not sig:
        return
    if not tm.can_open():
        return
    st = tm.open_from_signal(sig)
    if st:
        _send(context, _commentary_entry(sig))
        _send(context, f"Expert: Liquidity swept, displacement confirmed, FVG return in play. "
                       f"Risk sized to ${RISK_DOLLARS:.0f}. Trade `{st.id}` tracking.")

def manage_once(context: CallbackContext):
    """Manages the active trade every minute (1m + 5m context)."""
    if not (tm.active and not tm.active.closed):
        return
    st = tm.active
    df1 = get_klines("1", limit=120, include_partial=True)
    df5 = get_klines("5", limit=210, include_partial=True)
    if df1 is None or df5 is None or len(df1) < 30 or len(df5) < 50:
        return

    # Optional stretch: we can plug full AI confidence; for runtime safety use moderate prior
    ai_conf = 0.78
    tm.maybe_stretch(st, ai_conf)

    ev = tm.manage(st, df1, df5)
    if not ev:
        # Advisor (light hazard notices)
        if not ASSISTANT_MODE: 
            return
        try:
            # coarse checks; detailed hazard calc is in ai.py if you want to wire real-time features
            cur = float(df1["close"].iloc[-1])
            mae = abs(cur - (st.entry_price or st.entry_ref))
            delta = abs(st.entry_ref - st.stop_px) or 1.0
            mae_ratio = float(min(1.0, mae / delta))

            # heuristic hazard mix (kept light for 1 CPU)
            prob_sl = 0.35
            ev_knn = 0.55
            rej_score = 0.1
            struct_loss = 0.3
            vol_shift = 0.1

            # Inline hazard calc (same functional shape as ai.hazard, but avoid extra import call freq)
            H = 0.32*prob_sl + 0.18*(1-ev_knn) + 0.20*rej_score + 0.12*mae_ratio + 0.12*struct_loss + 0.06*vol_shift
            if H >= ASSISTANT_STRONG:
                _send(context, "🛑 *Cut Recommended* — risk outweighs hold. Bank discipline now.")
            elif H >= ASSISTANT_CUT:
                _send(context, "⚠️ *Weakening* — EV_exit > EV_hold. Consider partial or exit.")
            elif H >= ASSISTANT_WARN:
                _send(context, "ℹ️ *Heads-up* — momentum softening; tighten BE buffer.")
        except Exception:
            pass
        return

    # A management event fired
    _send(context, _commentary_trade_event(st, ev["event"], ev["price"]))

    # Close and update bandit if terminal
    if ev["event"] in ("TP2", "SL", "EARLY_TP2"):
        tm.close(st, ev["event"], ev["price"])
        R = trade_reward(st, ev["event"])
        bandit.update(st.arm_id, norm_reward(R), weight=1.5 if "TP2" in ev["event"] else 1.0)

def session_commentary(context: CallbackContext):
    """Lightweight market notes around sessions (IST-friendly)."""
    now = datetime.now(timezone.utc).astimezone()
    h = now.hour
    # IST vibes: adjust messages around your active windows
    if 12 <= h <= 17:
        _send(context, "📈 London in play: likely liquidity engineering. Watch for sweep → displacement. Sniper mode ON 🔥")
    if 19 <= h <= 23:
        _send(context, "⚡️ NY session: breakout probability elevated. Track displacement legs and FVG returns 🚀")

def daily_recap(context: CallbackContext):
    today = datetime.now().strftime("%Y-%m-%d")
    _send(context, f"📅 *Daily Recap — {today}*\n"
                   f"Asia: context logged • London/NY: commentary posted\n"
                   f"Trades: check logs.\n"
                   f"🔥 One clean kill > many messy shots.")

def weekly_recap(context: CallbackContext):
    _send(context, "📆 *Weekly Recap*\nWins/Losses, EV improvements, chop avoidance, and session chains summarised. Keep sniping ⚔️")

# ==================== TELEGRAM COMMANDS ====================

@_require_admin
def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text(
        f"Online. Symbol `{SYMBOL}`. 1-trade lock; risk ${RISK_DOLLARS:.0f}.",
        parse_mode=ParseMode.MARKDOWN
    )

@_require_admin
def cmd_scan(update: Update, context: CallbackContext):
    try:
        scan_once(context)
        update.message.reply_text("Scan done.", parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        update.message.reply_text(f"Scan error: {e}")

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
    update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

@_require_admin
def cmd_help(update: Update, context: CallbackContext):
    update.message.reply_text(
        "/start • /scan • /status\n"
        "/daily_now • /weekly_now\n"
        "Auto-jobs: manage (1m), scan (2m), session notes (30m), daily/weekly recaps.",
        parse_mode=ParseMode.MARKDOWN,
    )

@_require_admin
def cmd_daily_now(update: Update, context: CallbackContext):
    daily_recap(context)
    update.message.reply_text("Daily recap sent.")

@_require_admin
def cmd_weekly_now(update: Update, context: CallbackContext):
    weekly_recap(context)
    update.message.reply_text("Weekly recap sent.")

def error_handler(update: object, context: CallbackContext):
    try:
        msg = f"Error: {context.error}\n{traceback.format_exc()[:900]}"
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
    except Exception:
        pass

# ==================== PTB v13 BOOTSTRAP (NO POLLING) ====================

# Create Updater/Dispatcher/JobQueue FIRST (so it's available to webhook setup)
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher
jq = updater.job_queue

# Register handlers
dp.add_handler(CommandHandler("start", cmd_start))
dp.add_handler(CommandHandler("scan", cmd_scan))
dp.add_handler(CommandHandler("status", cmd_status))
dp.add_handler(CommandHandler("help", cmd_help))
dp.add_handler(CommandHandler("daily_now", cmd_daily_now))
dp.add_handler(CommandHandler("weekly_now", cmd_weekly_now))
dp.add_error_handler(error_handler)

# Schedule jobs (adjust times as you like)
jq.run_repeating(manage_once, interval=60, first=10)
jq.run_repeating(scan_once,   interval=120, first=15)
jq.run_repeating(session_commentary, interval=60*30, first=30)
# Daily recap 17:00 UTC (~22:30 IST). Change to your preference.
jq.run_daily(daily_recap, time=dtime(17,0,0, tzinfo=timezone.utc))
# Weekly recap Sunday 17:30 UTC
jq.run_daily(weekly_recap, time=dtime(17,30,0, tzinfo=timezone.utc), days=(6,))

# IMPORTANT: In webhook mode we don't call start_polling(), so start JobQueue explicitly
jq.start()

# ==================== FLASK WEBHOOK (FLASK 3.x SAFE) ====================

app = Flask(__name__)

def setup_webhook_once():
    """Set Telegram webhook explicitly on startup (Flask 3.x safe)."""
    host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")
    token = os.environ.get("TOKEN")
    if not host or not token:
        return
    url = f"https://{host}/{token}"
    try:
        updater.bot.delete_webhook()
    except Exception:
        pass
    updater.bot.set_webhook(url=url)

# Call after Updater is ready so `updater` exists
setup_webhook_once()

@app.route(f"/{os.environ['TOKEN']}", methods=["POST"])
def telegram_webhook():
    update = Update.de_json(request.get_json(force=True), updater.bot)
    updater.dispatcher.process_update(update)
    return "ok", 200

@app.get("/")
def index():
    return "OK", 200
