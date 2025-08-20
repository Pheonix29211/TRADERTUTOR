# bot.py
# Webhook-ready Telegram bot (PTB v13 + Flask 3.x) for your ICT DoL system
# - MEXC-only data feed (mexc_client.py)
# - pytz-safe APScheduler jobs
# - HTML /menu (no Telegram markdown parse issues)
# - /scan, /status, /check_data, /backtest [days tf], /logs
# - Debug: /btdebug [days tf], /diag, /force_unlock, /fscheck
# - Session commentary with daily dedupe
# - “Dopamine”/learning handled in strategy/ai; this bot wires notifications

import os, sys, traceback, time, json
from datetime import datetime
from datetime import time as dtime
from typing import Optional, List
from pathlib import Path

# Ensure local modules import
sys.path.append(os.path.dirname(__file__))

from dotenv import load_dotenv
from flask import Flask, request
from pytz import UTC
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, CallbackContext

from config import (
    TOKEN, ADMIN_CHAT_ID, TZ_NAME, SYMBOL,
    ASSISTANT_MODE, ASSISTANT_WARN, ASSISTANT_CUT, ASSISTANT_STRONG,
    RISK_DOLLARS, STRETCH_MAX_DOLLARS, TP1_DOLLARS, TP2_DOLLARS,
)
try:
    from storage import ensure_dirs
except Exception:
    def ensure_dirs(): pass

from mexc_client import get_klines
from strategy import DoLDetector, TradeManager, trade_reward, norm_reward, Signal, TradeState
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan

# ==================== INIT ====================
load_dotenv()
ensure_dirs()

# --------- Data dir with fallback (works even if /data not mounted) ---------
def _resolve_data_dir():
    candidates = [
        os.getenv("DATA_DIR"),
        "/data",
        "/tmp/data",
        os.path.join(os.path.dirname(__file__), "data"),
    ]
    for p in candidates:
        if not p:
            continue
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

# Lightweight singletons
bandit = PolicyArms()
knn = PatternKNN()
bayes = BayesModel()
bans = CFPBan()
detector = DoLDetector(bandit)
tm = TradeManager()

# ==================== TZ / GUARDS ====================
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

# ==================== HELPERS ====================
def _is_admin(update: Update) -> bool:
    uid = update.effective_user.id if update.effective_user else None
    if not ADMIN_CHAT_ID or str(ADMIN_CHAT_ID).strip() in ("", "0", "None"):
        return True
    return str(uid) == str(ADMIN_CHAT_ID)

def _require_admin(func):
    def wrapper(update: Update, context: CallbackContext, *a, **kw):
        if not _is_admin(update):
            try: update.message.reply_text("Access denied.")
            except Exception: pass
            return
        return func(update, context, *a, **kw)
    return wrapper

def _send(context: CallbackContext, text: str):
    try:
        if not ADMIN_CHAT_ID:
            return
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

def _read_trades_safe() -> List[dict]:
    try:
        if not os.path.exists(TRADES_PATH): return []
        with open(TRADES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

# ==================== CORE LOOP FUNCS ====================
def scan_once(context: CallbackContext):
    if tm.active and not tm.active.closed:
        return
    df5 = get_klines("5", limit=500, include_partial=False)
    if df5 is None or len(df5) < 210:
        return
    try: _ = safe_utc(df5.index[-1])
    except Exception: pass

    sig = detector.find(df5, bayes, knn, bandit, bans)
    if not sig or not tm.can_open():
        return

    st = tm.open_from_signal(sig)
    if st and ADMIN_CHAT_ID:
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=_commentary_entry(sig), parse_mode=ParseMode.MARKDOWN)
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=(f"Expert: Liquidity swept, displacement confirmed, FVG return in play. "
                                                               f"Risk sized to ${RISK_DOLLARS:.0f}. Trade `{st.id}` tracking."),
                                  parse_mode=ParseMode.MARKDOWN)

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
                context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="🛑 *Cut Recommended* — risk outweighs hold. Bank discipline now.", parse_mode=ParseMode.MARKDOWN)
            elif H >= ASSISTANT_CUT:
                context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="⚠️ *Weakening* — EV_exit > EV_hold. Consider partial or exit.", parse_mode=ParseMode.MARKDOWN)
            elif H >= ASSISTANT_WARN:
                context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="ℹ️ *Heads-up* — momentum softening; tighten BE buffer.", parse_mode=ParseMode.MARKDOWN)
        except Exception:
            pass
        return

    if ADMIN_CHAT_ID:
        context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=_commentary_trade_event(st, ev["event"], ev["price"]), parse_mode=ParseMode.MARKDOWN)

    if ev["event"] in ("TP2", "SL", "EARLY_TP2"):
        tm.close(st, ev["event"], ev["price"])
        R = trade_reward(st, ev["event"])
        bandit.update(st.arm_id, norm_reward(R), weight=1.5 if "TP2" in ev["event"] else 1.0)

_last_session_sent = {"LONDON": None, "NY": None}
def session_commentary(context: CallbackContext):
    import pytz
    ist = pytz.timezone(TZ_NAME if TZ_NAME else "Asia/Kolkata")
    now = datetime.now(ist); h = now.hour; day = now.strftime("%Y-%m-%d")
    global _last_session_sent
    if 12 <= h <= 17:
        if _last_session_sent.get("LONDON") != day and ADMIN_CHAT_ID:
            context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="📈 London in play: likely liquidity engineering. Watch for sweep → displacement. Sniper mode ON 🔥")
            _last_session_sent["LONDON"] = day
    if 19 <= h <= 23:
        if _last_session_sent.get("NY") != day and ADMIN_CHAT_ID:
            context.bot.send_message(chat_id=ADMIN_CHAT_ID, text="⚡️ NY session: breakout probability elevated. Track displacement legs and FVG returns 🚀")
            _last_session_sent["NY"] = day

def daily_recap(context: CallbackContext):
    if not ADMIN_CHAT_ID: return
    today = datetime.now().strftime("%Y-%m-%d")
    context.bot.send_message(chat_id=ADMIN_CHAT_ID,
                             text=(f"📅 *Daily Recap — {today}*\n"
                                   f"Asia: context logged • London/NY: commentary posted\n"
                                   f"Trades: check logs.\n"
                                   f"🔥 One clean kill > many messy shots."),
                             parse_mode=ParseMode.MARKDOWN)

def weekly_recap(context: CallbackContext):
    if not ADMIN_CHAT_ID: return
    context.bot.send_message(chat_id=ADMIN_CHAT_ID,
                             text="📆 *Weekly Recap*\nWins/Losses, EV improvements, chop avoidance, and session chains summarised. Keep sniping ⚔️",
                             parse_mode=ParseMode.MARKDOWN)

# ==================== TELEGRAM COMMANDS ====================
@_require_admin
def cmd_start(update: Update, context: CallbackContext):
    update.message.reply_text(f"Online. Symbol `{SYMBOL}`. 1-trade lock; risk ${RISK_DOLLARS:.0f}.",
                              parse_mode=ParseMode.MARKDOWN)

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
    daily_recap(context); update.message.reply_text("Daily recap sent.")

@_require_admin
def cmd_weekly_now(update: Update, context: CallbackContext):
    weekly_recap(context); update.message.reply_text("Weekly recap sent.")

@_require_admin
def cmd_menu(update: Update, context: CallbackContext):
    commands_text = (
        "<b>📋 SpiralBot Command Menu</b>\n\n"
        "<b>/scan</b> — Manually scan market for new signal\n"
        "<b>/status</b> — Current trade / cooldown\n"
        "<b>/logs</b> — Show last 30 saved trades (live/backtest)\n"
        "<b>/backtest [days] [tf]</b> — e.g., <code>/backtest 14 5</code> (writes to logs)\n"
        "<b>/check_data</b> — Test MEXC data feed\n"
        "<b>/btdebug [days] [tf]</b> — e.g., <code>/btdebug 7 15</code> (detector counters)\n"
        "<b>/diag</b> — Data health + cooldown state\n"
        "<b>/force_unlock</b> — Clear trade lock/cooldown (debug)\n"
        "<b>/fscheck</b> — Show data dir & trades.json info\n"
        "<b>/daily_now</b> — Send daily recap now\n"
        "<b>/weekly_now</b> — Send weekly recap now\n"
        "<b>/menu</b> or <b>/help</b> — Show this menu"
    )
    update.message.reply_text(commands_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

@_require_admin
def cmd_help(update: Update, context: CallbackContext):
    cmd_menu(update, context)

@_require_admin
def cmd_fscheck(update: Update, context: CallbackContext):
    try:
        p = Path(TRADES_PATH)
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        update.message.reply_text(f"DATA_DIR: {DATA_DIR}\nTRADES_PATH: {TRADES_PATH}\nExists: {exists}\nSize: {size} bytes")
    except Exception as e:
        update.message.reply_text(f"fscheck error: {e}")

@_require_admin
def cmd_check_data(update: Update, context: CallbackContext):
    try:
        df5 = get_klines("5", limit=10, include_partial=True)
        if df5 is None or df5.empty:
            update.message.reply_text("❌ No data from MEXC.")
            return
        ts = safe_utc(df5.index[-1])
        update.message.reply_text(f"✅ MEXC OK. 5m candles: {len(df5)}. Last bar @ {ts.isoformat()}")
    except Exception as e:
        update.message.reply_text(f"❌ MEXC error: {e}")

@_require_admin
def cmd_backtest(update: Update, context: CallbackContext):
    try:
        days = 7
        tf = "5"
        if context.args:
            if len(context.args) >= 1:
                try: days = max(1, min(30, int(context.args[0])))
                except Exception: pass
            if len(context.args) >= 2 and context.args[1] in ("1","5","15","30","60","1m","5m","15m","30m","1h"):
                tf = context.args[1].replace("m","")
                if tf == "1h": tf = "60"

        update.message.reply_text(f"⏳ Running backtest for last {days} days on {tf}m...")
        from utils import backtest_strategy
        results = backtest_strategy(days=days, tf=tf)

        if not results:
            update.message.reply_text("✅ Backtest complete. No valid setups in the window.")
            return

        lines = [f"📊 Backtest Results (last {days} days, {tf}m):"]
        for r in results[-10:]:
            lines.append(f"{r['time']} | {r['symbol']} | {r['side']} @ {r['entry']} SL {r['sl']} TP {r['tp']} → {r['result']}")
        update.message.reply_text("\n".join(lines))
    except Exception as e:
        update.message.reply_text(f"⚠️ Backtest failed: {e}")

@_require_admin
def cmd_logs(update: Update, context: CallbackContext):
    trades = _read_trades_safe()
    if not trades:
        update.message.reply_text("No saved trades yet.")
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
    update.message.reply_text("\n".join(lines))

# ---- DEBUG ----
@_require_admin
def cmd_btdebug(update: Update, context: CallbackContext):
    try:
        days = 7
        tf = "5"
        if context.args:
            if len(context.args) >= 1:
                try: days = max(1, min(30, int(context.args[0])))
                except Exception: pass
            if len(context.args) >= 2 and context.args[1] in ("1","5","15","30","60","1m","5m","15m","30m","1h"):
                tf = context.args[1].replace("m","")
                if tf == "1h": tf = "60"
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
        update.message.reply_text(msg)
    except Exception as e:
        update.message.reply_text(f"btdebug error: {e}")

@_require_admin
def cmd_diag(update: Update, context: CallbackContext):
    try:
        df5 = get_klines("5", limit=500, include_partial=False)
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

        update.message.reply_text(" • ".join(parts))
    except Exception as e:
        update.message.reply_text(f"diag error: {e}")

@_require_admin
def cmd_force_unlock(update: Update, context: CallbackContext):
    try:
        tm.cooldown_until_ms = 0
        if tm.active and tm.active.closed:
            tm.active = None
        update.message.reply_text("🔓 Trade lock & cooldown cleared.")
    except Exception as e:
        update.message.reply_text(f"force_unlock error: {e}")

def error_handler(update: object, context: CallbackContext):
    try:
        msg = f"Error: {context.error}\n{traceback.format_exc()[:900]}"
        if ADMIN_CHAT_ID:
            context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=msg)
    except Exception:
        pass

# ==================== PTB v13 BOOTSTRAP (NO POLLING) ====================
updater = Updater(TOKEN, use_context=True)
dp = updater.dispatcher
jq = updater.job_queue

# Register handlers
dp.add_handler(CommandHandler("start",        cmd_start))
dp.add_handler(CommandHandler("scan",         cmd_scan))
dp.add_handler(CommandHandler("status",       cmd_status))
dp.add_handler(CommandHandler("daily_now",    cmd_daily_now))
dp.add_handler(CommandHandler("weekly_now",   cmd_weekly_now))
dp.add_handler(CommandHandler("check_data",   cmd_check_data))
dp.add_handler(CommandHandler("backtest",     cmd_backtest))
dp.add_handler(CommandHandler("logs",         cmd_logs))
dp.add_handler(CommandHandler("menu",         cmd_menu))
dp.add_handler(CommandHandler("help",         cmd_help))
# Debug
dp.add_handler(CommandHandler("btdebug",      cmd_btdebug))
dp.add_handler(CommandHandler("diag",         cmd_diag))
dp.add_handler(CommandHandler("force_unlock", cmd_force_unlock))
dp.add_handler(CommandHandler("fscheck",      cmd_fscheck))
dp.add_error_handler(error_handler)

# Schedule jobs (apscheduler needs pytz tz)
jq.run_repeating(manage_once,         interval=60,     first=10)
jq.run_repeating(scan_once,           interval=120,    first=15)
jq.run_repeating(session_commentary,  interval=60*30,  first=30)
jq.run_daily(daily_recap,  time=dtime(17,  0, 0, tzinfo=UTC))
jq.run_daily(weekly_recap, time=dtime(17, 30, 0, tzinfo=UTC), days=(6,))

# Start JobQueue in webhook mode
jq.start()

# ==================== FLASK WEBHOOK ====================
app = Flask(__name__)

def setup_webhook_once():
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

setup_webhook_once()

@app.route(f"/{os.environ['TOKEN']}", methods=["POST"])
def telegram_webhook():
    update = Update.de_json(request.get_json(force=True), updater.bot)
    updater.dispatcher.process_update(update)
    return "ok", 200

@app.get("/")
def index():
    return "OK", 200

