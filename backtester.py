import os, json, math, time
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

from config import SYMBOL, RISK_DOLLARS
from mexc_client import get_klines_between
from strategy import DoLDetector, TradeManager, qty_for_risk, trade_reward, norm_reward
from ai import PolicyArms, PatternKNN, BayesModel, CFPBan

def backtest(start: str, end: str, interval5="5", interval1="1") -> dict:
    """
    Simulates signals on 5m and manages with 1m between [start,end] ISO8601 (UTC).
    Returns summary dict. Stores a JSON results file under /data/backtest_*.json
    """
    t0 = int(pd.Timestamp(start, tz='UTC').timestamp()*1000)
    t1 = int(pd.Timestamp(end, tz='UTC').timestamp()*1000)

    df5 = get_klines_between(interval5, t0, t1)
    df1 = get_klines_between(interval1, t0, t1)
    if df5 is None or df1 is None or df5.empty or df1.empty:
        return {"ok": False, "error": "insufficient data"}

    bandit = PolicyArms()
    knn = PatternKNN()
    bayes = BayesModel()
    bans = CFPBan()
    det = DoLDetector(bandit)
    tm = TradeManager()

    results = []
    # Iterate across 5m bars, build signals, then check 1m path forward until terminal event or next signal area
    for i in range(210, len(df5)-1):
        window5 = df5.iloc[:i+1]
        sig = det.find(window5, bayes, knn, bandit, bans)
        if not sig:
            continue

        # open
        st = tm.open_from_signal(sig)
        # simulate 1m from next minute
        start_ts = window5.index[-1] + pd.Timedelta(minutes=1)
        path = df1[(df1.index >= start_ts)]
        if path.empty:
            continue

        # Walk the path until trade closes by strategy.manage()
        # We reuse manage logic lightly adapted by copying parts
        st.entry_price = sig.entry_ref  # assume mid fill for BT
        st.filled = True
        for j in range(len(path)):
            # Terminal checks simplified for speed:
            px = float(path["close"].iloc[j])
            # Stops / TPs based on side
            if st.side == "BUY":
                if not st.tp1_done and px >= st.tp1_px:
                    st.tp1_done = True
                if px >= st.tp2_px:
                    st.exit_price = st.tp2_px; st.exit_reason="TP2"; break
                if px <= st.stop_px:
                    st.exit_price = st.stop_px; st.exit_reason="SL"; break
            else:
                if not st.tp1_done and px <= st.tp1_px:
                    st.tp1_done = True
                if px <= st.tp2_px:
                    st.exit_price = st.tp2_px; st.exit_reason="TP2"; break
                if px >= st.stop_px:
                    st.exit_price = st.stop_px; st.exit_reason="SL"; break
        if not st.exit_price:
            # time stop if not closed by end
            st.exit_price = float(path["close"].iloc[-1])
            st.exit_reason = "TIMEOUT"

        R = trade_reward(st, st.exit_reason)
        results.append({
            "ts": int(window5.index[-1].timestamp()*1000),
            "side": st.side, "entry": st.entry_ref, "stop": st.stop_px,
            "tp1": st.tp1_px, "tp2": st.tp2_px, "exit": st.exit_price,
            "reason": st.exit_reason, "R": R
        })
        bandit.update(st.arm_id, norm_reward(R))

    if not results:
        return {"ok": False, "error": "no signals"}

    df = pd.DataFrame(results)
    winrate = float((df["R"] > 0).mean())
    pnlR = float(df["R"].sum())
    out = {"ok": True, "trades": len(df), "winrate": winrate, "sumR": pnlR}
    # Save
    fname = f"backtest_{SYMBOL}_{int(time.time())}.json"
    with open(os.path.join("data", fname), "w") as f:
        json.dump({"summary": out, "rows": results}, f)
    return out

if __name__ == "__main__":
    # Example: last 7 days UTC
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=7)
    print(backtest(start.isoformat(), end.isoformat()))
