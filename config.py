import os
from datetime import time

# ENV
DATA_DIR = os.getenv("DATA_DIR", "data")
TZ_NAME = os.getenv("TZ", "Asia/Kolkata")
SYMBOL = os.getenv("SYMBOL", "BTCUSDT").upper()

# Telegram
TOKEN = os.getenv("TOKEN", "")
ADMIN_CHAT_ID = int(os.getenv("ADMIN_CHAT_ID", "0"))
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
PORT = int(os.getenv("PORT", "10000"))

# Risk model
RISK_DOLLARS = 40.0                 # baseline risk per trade
STRETCH_MAX_DOLLARS = 50.0          # single stretch cap
TP1_DOLLARS = 600.0
TP2_DOLLARS = 1500.0
TP1_PARTIAL = 0.5

# Mechanics
BE_MIN_BUFFER = 5.0                 # minimum BE buffer in $
ATR_TRAIL_PRE = 1.0
ATR_TRAIL_NEAR = 0.6
TP2_TIGHTEN_DIST = 40.0             # tighten trail when within this of TP2

# Strategy base thresholds
DISPLACEMENT_MULT = 1.6
FVG_MIN_PCT = 0.04 / 100
WICK_REJ_RATIO = 1.5
EQUAL_TOL_PCT = 0.05 / 100
SWING_LOOKBACK = 3
POOL_LOOKBACK = 288                 # ~24h of 5m
PSY_STEP = 500

# Anti-spam & locks
MAX_CONCURRENT_TRADES = 1
COOLDOWN_MIN = int(os.getenv("COOLDOWN_MIN", "15"))
FOCUS_COOLDOWN_MIN = int(os.getenv("FOCUS_COOLDOWN_MIN", "8"))
POST_SL2_LOCK_MIN = int(os.getenv("POST_SL2_LOCK_MIN", "90"))
SIGNAL_MIN_COOLDOWN_MS = COOLDOWN_MIN * 60_000

# Assistant Advisor thresholds
ASSISTANT_MODE = os.getenv("ASSISTANT_MODE", "on").lower() == "on"
ASSISTANT_WARN = float(os.getenv("ASSISTANT_WARN", "0.65"))
ASSISTANT_CUT = float(os.getenv("ASSISTANT_CUT", "0.80"))
ASSISTANT_STRONG = float(os.getenv("ASSISTANT_STRONG", "0.90"))

# Personality (0..1)
WIN_DRIVE = float(os.getenv("WIN_DRIVE", "0.78"))
LOSS_AVERSION = float(os.getenv("LOSS_AVERSION", "0.82"))
QUALITY_BIAS = float(os.getenv("QUALITY_BIAS", "0.75"))
REVENGE_GUARD = float(os.getenv("REVENGE_GUARD", "0.85"))

# Sessions (UTC windows; only for tagging/context)
ASIA_UTC = (time(0, 0), time(7, 59))
LONDON_UTC = (time(8, 0), time(11, 59))
NY_UTC = (time(13, 30), time(21, 30))

# Retention
CACHE_RETENTION_DAYS = int(os.getenv("CACHE_RETENTION_DAYS", "90"))
BT_MAX_RUNS = int(os.getenv("BT_MAX_RUNS", "20"))
LOG_RETENTION_DAYS = int(os.getenv("LOG_RETENTION_DAYS", "7"))
LOW_SPACE_MB = int(os.getenv("LOW_SPACE_MB", "150"))
MODEL_KEEP = int(os.getenv("MODEL_KEEP", "5"))
