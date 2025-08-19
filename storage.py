import os, json, gzip, shutil, glob
from datetime import datetime, timezone, timedelta
import pandas as pd
from config import DATA_DIR, CACHE_RETENTION_DAYS, BT_MAX_RUNS, LOG_RETENTION_DAYS, LOW_SPACE_MB, MODEL_KEEP

PATHS = {
    "root": DATA_DIR,
    "cache": os.path.join(DATA_DIR, "cache"),
    "logs": os.path.join(DATA_DIR, "logs"),
    "daily": os.path.join(DATA_DIR, "logs", "daily"),
    "weekly": os.path.join(DATA_DIR, "logs", "weekly"),
    "models": os.path.join(DATA_DIR, "models"),
    "dataset": os.path.join(DATA_DIR, "dataset"),
    "trade_logs": os.path.join(DATA_DIR, "trade_logs"),
    "ai": os.path.join(DATA_DIR, "ai"),
}

FILES = {
    "arms": os.path.join(PATHS["ai"], "arms.json"),
    "dopamine": os.path.join(PATHS["ai"], "dopamine.json"),
    "cfp": os.path.join(PATHS["ai"], "cfp_memory.json"),
    "feature_stats": os.path.join(PATHS["ai"], "feature_stats.json"),
    "knn": os.path.join(PATHS["ai"], "knn_index.npz"),
    "dataset_parquet": os.path.join(PATHS["dataset"], "dataset.parquet"),
}

def ensure_dirs():
    for p in PATHS.values():
        os.makedirs(p, exist_ok=True)

def disk_free_mb(path: str) -> int:
    total, used, free = shutil.disk_usage(path)
    return int(free / (1024*1024))

def rotate_backtests():
    bt_files = sorted(glob.glob(os.path.join(DATA_DIR, "backtest_*.json")), key=os.path.getmtime)
    if len(bt_files) > BT_MAX_RUNS:
        for f in bt_files[:-BT_MAX_RUNS]:
            try: os.remove(f)
            except: pass

def compress_old_logs():
    # compress logs older than retention days
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOG_RETENTION_DAYS)
    for root in (PATHS["logs"], PATHS["daily"], PATHS["weekly"]):
        for f in glob.glob(os.path.join(root, "*.log")):
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(f), tz=timezone.utc)
                if mtime < cutoff and not f.endswith(".gz"):
                    with open(f, "rb") as src, gzip.open(f + ".gz", "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    os.remove(f)
            except: pass

def prune_cache():
    cutoff = (datetime.utcnow() - timedelta(days=CACHE_RETENTION_DAYS)).date()
    for f in glob.glob(os.path.join(PATHS["cache"], "**", "*.parquet"), recursive=True):
        try:
            # infer date from filename …/YYYYMMDD.parquet
            base = os.path.basename(f).split(".")[0]
            dt = datetime.strptime(base, "%Y%m%d").date()
            if dt < cutoff:
                os.remove(f)
        except: pass

def prune_backtests_over():
    rotate_backtests()

def low_space_cleanup():
    if disk_free_mb(DATA_DIR) >= LOW_SPACE_MB:
        return
    # prefer deleting cache first, then old backtests, then gz logs > retention
    prune_cache()
    prune_backtests_over()
    compress_old_logs()

def daily_maintenance():
    ensure_dirs()
    compress_old_logs()
    prune_backtests_over()
    low_space_cleanup()
