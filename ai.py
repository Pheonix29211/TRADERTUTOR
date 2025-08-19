import os, json, math
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from indicators import ema, rsi, atr, body_size, wick_sizes
from config import (DATA_DIR, WIN_DRIVE, LOSS_AVERSION, QUALITY_BIAS, REVENGE_GUARD,
                    ASSISTANT_WARN, ASSISTANT_CUT, ASSISTANT_STRONG)

AI_DIR = os.path.join(DATA_DIR, "ai")
os.makedirs(AI_DIR, exist_ok=True)

def _safe_load_json(p, default):
    try:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f: return json.load(f)
    except: pass
    return default

def _safe_save_json(p, obj):
    try:
        with open(p, "w", encoding="utf-8") as f: json.dump(obj, f)
    except: pass

# ---------------- UCB Bandit ----------------
class PolicyArms:
    def __init__(self, path=os.path.join(AI_DIR, "arms.json")):
        self.path = path
        self.arms = {
            "A1": {"disp": 1.6, "fvg": 0.04/100, "eq": 0.05/100},
            "A2": {"disp": 1.8, "fvg": 0.05/100, "eq": 0.05/100},
            "A3": {"disp": 2.0, "fvg": 0.06/100, "eq": 0.06/100},
        }
        self.stats = _safe_load_json(self.path, {"total":0,"stats":{k:{"n":0,"mean":0.0} for k in self.arms}})
        self.total = self.stats.get("total", 0)

    def pick(self) -> str:
        self.total += 1
        for k,st in self.stats["stats"].items():
            if st["n"] < 3: return k
        logn = math.log(self.total + 1)
        def ucb(k):
            s = self.stats["stats"][k]
            return s["mean"] + math.sqrt(2*logn/(s["n"]))
        return max(self.stats["stats"].keys(), key=ucb)

    def update(self, arm_id: str, reward_norm: float, weight: float = 1.0):
        s = self.stats["stats"][arm_id]
        n, m = s["n"], s["mean"]
        new_mean = (m*n + reward_norm*weight) / (n + weight)
        self.stats["stats"][arm_id] = {"n": n + weight, "mean": new_mean}
        self.stats["total"] = self.total
        _safe_save_json(self.path, self.stats)

# --------------- Pattern Memory (KNN) ---------------
class PatternKNN:
    def __init__(self, path=os.path.join(AI_DIR, "knn_index.npz")):
        self.path = path
        self.X = None  # features (n,d)
        self.y = None  # R in [-2..+4], or outcome score
        self.meta = None  # dict rows
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            try:
                d = np.load(self.path, allow_pickle=True)
                self.X = d["X"]; self.y = d["y"]; self.meta = d["meta"].tolist()
            except: pass
        if self.X is None:
            self.X = np.zeros((0, 32), dtype=np.float32)
            self.y = np.zeros((0,), dtype=np.float32)
            self.meta = []

    def save(self):
        try:
            np.savez_compressed(self.path, X=self.X, y=self.y, meta=np.array(self.meta, dtype=object))
        except: pass

    def add(self, x: np.ndarray, y: float, meta: dict):
        x = x.astype(np.float32).reshape(1, -1)
        self.X = np.vstack([self.X, x]) if self.X.size else x
        self.y = np.append(self.y, np.float32(y))
        self.meta.append(meta)
        # Cap memory for 2GB RAM: keep last 50k patterns
        if self.X.shape[0] > 50000:
            self.X = self.X[-50000:]; self.y = self.y[-50000:]; self.meta = self.meta[-50000:]

    def query(self, x: np.ndarray, k: int = 64) -> Tuple[float, float]:
        if self.X.shape[0] == 0: return 0.0, 0.5
        x = x.astype(np.float32).reshape(1,-1)
        # cosine similarity
        Xn = self.X / (np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-9)
        xn = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
        sims = (Xn @ xn.T).flatten()
        idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
        w = np.clip(sims[idx], 0, 1) + 1e-6
        ev = float(np.sum(self.y[idx] * w) / np.sum(w))  # expected R
        winrate = float(np.mean(self.y[idx] > 0.0))
        # Normalize EV to 0..1 (assuming R in [-2..+4])
        ev_norm = (ev + 2.0) / 6.0
        return ev_norm, winrate

# --------------- CFP Banlist ---------------
class CFPBan:
    def __init__(self, path=os.path.join(AI_DIR, "cfp_memory.json")):
        self.path = path
        self.data: Dict[str, float] = _safe_load_json(self.path, {})

    def _key(self, d: dict) -> str:
        # Coarse buckets to generalize contexts
        return f"{d.get('session')}|{d.get('atr_bucket')}|{d.get('bias')}|{d.get('sweep')}|{d.get('fvg_bucket')}"

    def ban(self, d: dict, minutes: int):
        self.data[self._key(d)] = pd.Timestamp.utcnow().timestamp() + minutes*60
        _safe_save_json(self.path, self.data)

    def banned(self, d: dict) -> bool:
        t = self.data.get(self._key(d), 0)
        return pd.Timestamp.utcnow().timestamp() < t

# --------------- Confidence Combiner ---------------
def combined_conf(p_model: float, ev_knn: float, bandit_prior: float) -> float:
    # weights shaped by personality
    w1 = 0.45 + 0.15*QUALITY_BIAS
    w2 = 0.35 + 0.10*WIN_DRIVE
    w3 = 1.0 - (w1 + w2)
    return float(np.clip(w1*p_model + w2*ev_knn + w3*bandit_prior, 0.0, 1.0))

# --------------- Simple probability model ---------------
class BayesModel:
    """Light calibrated predictor over coarse buckets; robust on 1 CPU."""
    def __init__(self, path=os.path.join(AI_DIR, "feature_stats.json")):
        self.path = path
        self.table = {}  # key -> (wins, total)
        self._load()

    def _load(self):
        self.table = _safe_load_json(self.path, {})

    def save(self):
        _safe_save_json(self.path, self.table)

    def _key(self, feats: dict) -> str:
        # hash a few coarse buckets
        return f"{feats.get('session')}|{feats.get('atr_bucket')}|{feats.get('bias')}|{feats.get('sweep')}|{feats.get('disp_bucket')}|{feats.get('fvg_bucket')}"

    def update(self, feats: dict, win: bool):
        k = self._key(feats)
        w,t = self.table.get(k, [0,0])
        self.table[k] = [w + (1 if win else 0), t + 1]

    def predict(self, feats: dict) -> float:
        k = self._key(feats)
        w,t = self.table.get(k, [0,0])
        # Beta(1,1) prior smoothing
        return (w + 1) / (t + 2) if t >= 0 else 0.5

# --------------- Assistant Advisor (Hazard) ---------------
def hazard(prob_sl: float, ev_knn: float, rej_score: float, mae_ratio: float, struct_loss: float, vol_shift: float) -> float:
    # Higher LossAversion → earlier warnings/cuts (we'll lower thresholds outside)
    w1,w2,w3,w4,w5,w6 = 0.32,0.18,0.20,0.12,0.12,0.06
    H = w1*prob_sl + w2*(1-ev_knn) + w3*rej_score + w4*mae_ratio + w5*struct_loss + w6*vol_shift
    return float(np.clip(H, 0.0, 1.0))
