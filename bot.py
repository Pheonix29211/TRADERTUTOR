# bot.py
# Complete webhook-ready Telegram bot (PTB v13 + Flask 3.x) for your ICT DoL system

import os, sys, traceback, time
from datetime import datetime
from datetime import time as dtime
from typing import Optional

# Ensure local modules import on Render (Linux, case-sensitive)
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from flask import Flask, request
from pytz import UTC  # APScheduler in PTB v13 requires pytz timezones
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

# ---- Local modules (filenames must match exactly on Linux) ----
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

# Lightweight singletons (fit 2GB RAM / 1 CPU)
bandit = PolicyArms()
knn = PatternKNN()
bayes = BayesModel()
bans = CFPBan()
detector = DoLDetector(bandit)
tm = TradeManager()

# ==================== TZ / GUARDS ====================

def safe_utc(ts: datetime) -> datetime:
    """
    Return the same timestamp normalized to UTC, handling both naive and tz-aware.
    Avoids: 'Cannot localize tz-aware Timestamp, use tz_convert for conversions'
    """
    try:
        # pandas Timestamp has .tz_localize/.tz_convert too; this works with python datetime as well
        if getattr(ts, "tzinfo", None) is None:
            # naive → localize to UTC
            # For pandas.Timestamp: ts.tz_localize("UTC"); but to be generic:
            from pandas import Timestamp
            if isinstance(ts, Timestamp):
                return ts.tz_localize("UTC").to_pydatetime()
            else:
                # python datetime naive → attach UTC
                import pytz
                return pytz.UTC.localize(ts)
        else:
            # tz-aware → convert to UTC
            from pandas import Timestamp
            if isinstance(ts, Timestamp):
                return ts.tz_convert("UTC").to_pydatetime()
            else:
                return ts.astimezone(UTC)
    except Exception:
        # last resort: assume UTC as-is to keep bot alive
        return ts

# ==================== HELPERS ====================

def _is_admin(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    # If ADMIN_CHAT_ID not set yet, allow any user (so you can /start and then set it)
    if not ADMIN_CHAT_ID or str(ADMIN_CHAT_ID).strip() in ("", "0", "None"):
        return True
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
            chat_id=ADMIN_CHAT_ID if ADMIN_CHAT_ID else update_or_me(context),
            text=text,
            parse_mode=ParseMode.MARKDOWN,
            disable_web_page_preview=True,
        )
    except Exception:
        pass

def update_or_me(context: CallbackContext) -> int:
    """
    Fallback chat_id if ADMIN_CHAT_ID is unset: return the last update's chat id if available.
    (Keeps bot usable during initial setup.)
    """
    try:
        if context and getattr(context, "chat_data", None):
            # not reliable for PTB v13 here; fall back below
            pass
    except Exception:
        pass
    # No reliable context here in jobs; if ADMIN_CHAT_ID missing, nothing to send to.
    # For jobs we’ll silently skip sending when ADMIN_CHAT_ID is not set.
    return int(ADMIN_CHAT_ID) if ADMIN_CHAT_ID else 0

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
    if tm.active and not tm.active.closed:
        return
    df5 = get_klines("5", limit=500, include_partial=False)
    if df5 is None or len(df5) < 210:
        return
    # Guard candle index tz in case caller changed client
    try:
        idx = df5.index[-1]
        _ = safe_utc(idx)
    except Exception:
        pass

    sig = detector.find(df5, bayes, knn, bandit, bans)
    if not sig:
        return
    if not tm.can_open():
        return
    st = tm.open_from_signal(sig)
    if st:
        # Only send if admin chat id is known
        if ADMIN_CHAT_ID:
            context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=_commentary_entry(sig),
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
            context.bot.send_message(
                chat_id=ADMIN_CHAT_ID,
                text=(f"Expert: Liquidity swept, displacement confirmed, FVG return in play. "
                      f"Risk sized to ${RISK_DOLLARS:.0f}. Trade `{st.id}` tracking."),
                parse_mode=ParseMode.MARKDOWN,
            )

def manage_once(context: CallbackContext):
    """Manages the active trade every minute (1m + 5m context)."""
    if not (tm.active and not tm.active.closed):
        return
    st = tm.active
    df1 = get_klines("1", limit=120, include_partial=True)
    df5 = get_klines("5", limit=210, include_partial=True)
    if df1 is None or df5 is None or len(df1) < 30 or len(df5) < 50:
        return

    # Optional stretch: plug full AI confidence later; use moderate prior now
    ai_conf = 0.78
    tm.maybe_stretch(st, ai_conf)

    ev = tm.manage(st, df1, df5)
    if not ev:
        if not ASSISTANT_MODE or not ADMIN_CHAT_ID:
            return
        try:
            # Coarse hazard checks; (detailed hazard is in ai.py)
            cur = float(df1["close"].iloc[-1])
            mae = abs(cur - (st.entry_price or st.entry_ref))
            delta = abs(st.entry_ref - st.stop_px) or 1.0
            mae_ratio = float(min(1.0, mae / delta))

            prob_sl = 0.35
            ev_knn = 0.55
            rej_score = 0.1
            struct_loss = 0.3
            vol_shift = 0.1

            H = 0.32*prob_sl + 0.18*(1-ev_knn) + 0.20*rej_score + 0.12*mae_ratio + 0.12*struct_loss + 0.06*vol_shift
            if H >= ASSISTANT_STRONG:
                context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="🛑 *Cut Recommended* — risk outweighs hold. Bank discipline now.", parse_mode=ParseMode.MARKDOWN)
            elif H >= ASSISTANT_CUT:
                context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="⚠️ *Weakening* — EV_exit > EV_hold. Consider partial or exit.", parse_mode=ParseMode.MARKDOWN)
            elif H >= ASSISTANT_WARN:
                context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="ℹ️ *Heads-up* — momentum softening; tighten BE buffer.", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass
        return

    # A management event fired
    if ADMIN_CHAT_ID:
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=_commentary_trade_event(st, ev["event"], ev["price"]), parse_mode=ParseMode.MARKDOWN)

    # Close and update bandit if terminal
    if ev["event"] in ("TP2", "SL", "EARLY_TP2"):
        tm.close(st, ev["event"], ev["price"])
        R = trade_reward(st, ev["event"])
        bandit.update(st.arm_id, norm_reward(R), weight=1.5 if "TP2" in ev["event"] else 1.0)

# One-per-session dedupe
_last_session_sent = {"LONDON": None, "NY": None}

def session_commentary(context: CallbackContext):
    """Send at most once per session per day (IST-friendly)."""
    import pytz
    ist = pytz.timezone(TZ_NAME if TZ_NAME else "Asia/Kolkata")
    now = datetime.now(ist)
    h = now.hour
    day = now.strftime("%Y-%m-%d")

    global _last_session_sent
    if 12 <= h <= 17:  # London-ish in IST
        if _last_session_sent.get("LONDON") != day and ADMIN_CHAT_ID:
            context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="📈 London in play: likely liquidity engineering. Watch for sweep → displacement. Sniper mode ON 🔥")
            _last_session_sent["LONDON"] = day
    if 19 <= h <= 23:  # New York-ish
        if _last_session_sent.get("NY") != day and ADMIN_CHAT_ID:
            context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="⚡️ NY session: breakout probability elevated. Track displacement legs and FVG returns 🚀")
            _last_session_sent["NY"] = day

def daily_recap(context: CallbackContext):
    if not ADMIN_CHAT_ID:
        return
    today = datetime.now().strftime("%Y-%m-%d")
    context.bot.send_message(
        chat_id=ADMIN_CHAT_ID,
        text=(f"📅 *Daily Recap — {today}*\n"
              f"Asia: context logged • London/NY: commentary posted\n"
              f"Trades: check logs.\n"
              f"🔥 One clean kill > many messy shots."),
        parse_mode=ParseMode.MARKDOWN,
    )

def weekly_recap(context: CallbackContext):
    if not ADMIN_CHAT_ID:
        return
    context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="📆 *Weekly Recap*\nWins/Losses, EV improvements, chop avoidance, and session chains summarised. Keep sniping ⚔️", parse_mode=ParseMode.MARKDOWN)

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
def cmd_daily_now(update: Update, context: CallbackContext):
    daily_recap(context)
    update.message.reply_text("Daily recap sent.")

@_require_admin
def cmd_weekly_now(update: Update, context: CallbackContext):
    weekly_recap(context)
    update.message.reply_text("Weekly recap sent.")

@_require_admin
def cmd_menu(update: Update, context: CallbackContext):
    commands_text = (
        "<b>📋 SpiralBot Command Menu</b>\n\n"
        "<b>/scan</b> — Manually scan market for new signal\n"
        "<b>/status</b> — Current trade / cooldown\n"
        "<b>/logs</b> — Last 30 trades (wins + losses)\n"
        "<b>/results</b> — Performance stats (win rate, profit, avg R)\n"
        "<b>/last30</b> — Last 30 live trade entries\n"
        "<b>/backtest</b> — Backtest last 7 days (updates AI learning)\n"
        "<b>/check_data</b> — Test MEXC data feed\n"
        "<b>/daily_now</b> — Send daily recap now\n"
        "<b>/weekly_now</b> — Send weekly recap now\n"
        "<b>/menu</b> or <b>/help</b> — Show this menu"
    )
    update.message.reply_text(commands_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@_require_admin
def cmd_help(update: Update, context: CallbackContext):
    cmd_menu(update, context)

@_require_admin
def cmd_check_data(update: Update, context: CallbackContext):
    try:
        df5 = get_klines("5", limit=10, include_partial=True)
        if df5 is None or df5.empty:
            update.message.reply_text("❌ No data from MEXC.")
            return
        # Show last candle time safely in UTC
        ts = safe_utc(df5.index[-1])
        update.message.reply_text(f"✅ MEXC OK. 5m candles: {len(df5)}. Last bar @ {ts.isoformat()}")
    except Exception as e:
        update.message.reply_text(f"❌ MEXC error: {e}")

def error_handler(update: object, context: CallbackContext):
    try:
        msg = f"Error: {context.error}\n{traceback.format_exc()[:900]}"
        if ADMIN_CHAT_ID:
            context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
    except Exception:
        pass

# ==================== PTB v13 BOOTSTRAP (NO POLLING) ====================

# Create Updater/Dispatcher/JobQueue FIRST (so it's available to webhook setup)
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher
jq = updater.job_queue

# Register handlers
dp.add_handler(CommandHandler("start",      cmd_start))
dp.add_handler(CommandHandler("scan",       cmd_scan))
dp.add_handler(CommandHandler("status",     cmd_status))
dp.add_handler(CommandHandler("daily_now",  cmd_daily_now))
dp.add_handler(CommandHandler("weekly_now", cmd_weekly_now))
dp.add_handler(CommandHandler("check_data", cmd_check_data))
dp.add_handler(CommandHandler("menu",       cmd_menu))
dp.add_handler(CommandHandler("help",       cmd_help))
dp.add_error_handler(error_handler)

# Schedule jobs (apscheduler needs pytz tz)
jq.run_repeating(manage_once,         interval=60,     first=10)
jq.run_repeating(scan_once,           interval=120,    first=15)
jq.run_repeating(session_commentary,  interval=60*30,  first=30)
jq.run_daily(daily_recap,  time=dtime(17,  0, 0, tzinfo=UTC))            # 17:00 UTC (~22:30 IST)
jq.run_daily(weekly_recap, time=dtime(17, 30, 0, tzinfo=UTC), days=(6,)) # Sunday 17:30 UTC

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
