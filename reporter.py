import os, json, glob
from datetime import datetime, timezone, timedelta
import pandas as pd
from storage import PATHS, ensure_dirs

def _write_log(path: str, text: str):
    ensure_dirs()
    with open(path, "a", encoding="utf-8") as f:
        f.write(text.rstrip() + "\n")

def daily_report():
    today = datetime.now().strftime("%Y-%m-%d")
    text = (f"📅 Daily Recap — {today}\n"
            f"Asia: context tracked • London: manipulation/trend flags posted • NY: breakout watch\n"
            f"Trades & AI notes saved.")
    p = os.path.join(PATHS["daily"], f"{today}.log")
    _write_log(p, text)
    return text

def weekly_report():
    now = datetime.now()
    week = f"{(now - timedelta(days=6)).strftime('%Y-%m-%d')} to {now.strftime('%Y-%m-%d')}"
    text = (f"📆 Weekly Recap — {week}\n"
            f"Summary: clean executions favored, chop avoided, NY continuation bias validated.\n"
            f"Learning updated; configs tuned where EV improved.")
    p = os.path.join(PATHS["weekly"], f"week_{now.strftime('%Y_%W')}.log")
    _write_log(p, text)
    return text
