"""Strategy planner: searches strategy x timeframe x coin combinations for a target, honestly.

Search space: 5 strategy styles x 4 entry timeframes (1m, 3m, 5m, 15m) x the most liquid coins.
For every combination the strategy is replayed on recent real candles (fees + slippage included).

Honesty rules:
  * Combinations are SELECTED using only the older ~70% of the data ("selection period").
  * The headline chance is computed from the newest ~30% ("unseen period"), which played no part in
    the selection. Searching many combinations and scoring them on the same data they were picked on
    would make almost any target look achievable.
  * The chance of reaching a target rises with bet size, and so does the chance of a big loss; the
    plan shows both for every risk level.
"""
from __future__ import annotations

import json
import time
from dataclasses import replace

import numpy as np

from .agents import MeanReversion, MomentumBurst, MomentumChaser, Pullback, TrendBreakout, regime_on, simulate_agent
from .analyse import monte_carlo
from .client import RESOLUTION_SECONDS
from .config import Settings
from .data import candles_df
from .scanner import liquid_universe
from .state import STATE_DIR

STYLES = {
    "breakout": (TrendBreakout, "Breakout"),
    "pullback": (Pullback, "Pullback"),
    "meanrev": (MeanReversion, "Mean reversion"),
    "momentum": (MomentumBurst, "Volume burst"),
    "chaser": (MomentumChaser, "Momentum chase"),
}
REGIME_FOR = {"1m": "15m", "3m": "15m", "5m": "1h", "15m": "1h"}
DAYS_FOR = {"1m": 5, "3m": 6, "5m": 14, "15m": 21}
RISK_LEVELS = [0.01, 0.02, 0.03, 0.05, 0.08]
BASE_RISK = 0.01
SELECT_FRAC = 0.7
MAX_ACTIVE, MAX_BACKUPS = 4, 8
PLAN_PATH = STATE_DIR / "plan.json"


def make_agent(spec: dict):
    cls, label = STYLES[spec["style"]]
    return cls(spec["name"], f"{label} {spec['tf']} · {spec['sym']}", spec["desc"], REGIME_FOR[spec["tf"]],
               spec["tf"], max_positions=1, max_hold_bars=48, symbols=[spec["sym"]])


def _per_hour(n: int, hours: float) -> float:
    return n / hours if hours > 0 else 0.0


def plan(client, products, s: Settings, start_inr: float, target_inr: float, minutes: float,
         coins: int = 10, progress=lambda m, p: None) -> dict:
    t0 = time.time()
    base = replace(s, risk_per_trade=BASE_RISK)
    syms = [x for x, _ in liquid_universe(client, products, s)[:coins]]
    cache: dict = {}

    def get(sym, tf, days):
        k = (sym, tf)
        if k not in cache:
            cache[k] = candles_df(client, sym, tf, int(days * 86400 / RESOLUTION_SECONDS[tf]) + 120)
        return cache[k]

    combos = []
    total = len(STYLES) * len(DAYS_FOR) * len(syms)
    k = 0
    for tf, days in DAYS_FOR.items():
        rtf = REGIME_FOR[tf]
        for sym in syms:
            try:
                e = get(sym, tf, days)
                r = get(sym, rtf, days + 10)
            except Exception:
                k += len(STYLES)
                continue
            if len(e) < 300 or len(r) < s.regime_ema_slow + 20:
                k += len(STYLES)
                continue
            reg = regime_on(e, r, s, rtf, tf)
            split_t = e.index[0] + (e.index[-1] - e.index[0]) * SELECT_FRAC
            sel_h = (split_t - e.index[0]).total_seconds() / 3600
            oos_h = (e.index[-1] - split_t).total_seconds() / 3600
            for style, (cls, label) in STYLES.items():
                k += 1
                progress(f"Testing {label} {tf} on {sym}", 3 + 77 * k / total)
                ag = cls("x", "", "", rtf, tf, max_positions=1, max_hold_bars=48)
                trades, _ = simulate_agent(e, reg, ag, products[sym].contract_value, base, 1000.0)
                sel = [t for t in trades if t["time"] < split_t]
                oos = [t for t in trades if t["time"] >= split_t]
                rs = np.array([t["pnl"] / 1000 for t in sel])
                ro = np.array([t["pnl"] / 1000 for t in oos])
                hold_h = np.mean([t["bars"] for t in trades]) * RESOLUTION_SECONDS[tf] / 3600 if trades else 1
                cap = 1 / max(hold_h, RESOLUTION_SECONDS[tf] / 3600)        # one position per combo
                combos.append({
                    "style": style, "tf": tf, "sym": sym, "label": f"{label} {tf} · {sym}",
                    "sel_n": int(len(rs)), "sel_avg": float(rs.mean()) if len(rs) else 0.0,
                    "sel_win": float((rs > 0).mean()) if len(rs) else 0.0,
                    "sel_score": float(rs.mean() / (rs.std() + 1e-9) * np.sqrt(len(rs))) if len(rs) > 1 else -9,
                    "oos_n": int(len(ro)), "oos_avg": float(ro.mean()) if len(ro) else 0.0,
                    "oos_win": float((ro > 0).mean()) if len(ro) else 0.0,
                    "rate_sel": min(_per_hour(len(rs), sel_h), cap), "rate_oos": min(_per_hour(len(ro), oos_h), cap),
                    "rets_sel": rs.round(6).tolist(), "rets_oos": ro.round(6).tolist(),
                })

    # ---- select on the selection period only, aimed at THIS target and session length
    min_trades = {"1m": 15, "3m": 10, "5m": 8, "15m": 5}
    eligible = [c for c in combos if c["sel_n"] >= min_trades[c["tf"]] and c["sel_avg"] > 0]
    eligible.sort(key=lambda c: c["sel_score"], reverse=True)

    def pools(cs, key_rate, key_rets, risk):
        return [{"rate": c[key_rate], "rets": np.array(c[key_rets]) * (risk / BASE_RISK)}
                for c in cs if c[key_rate] > 0 and c[key_rets]]

    def chance(cs, risk, which="sel", sims=800, target=target_inr):
        kr, kx = ("rate_sel", "rets_sel") if which == "sel" else ("rate_oos", "rets_oos")
        return monte_carlo(pools(cs, kr, kx, risk), minutes, start_inr, target, 0.5, sims=sims)

    progress("Building the best team for this target", 82)
    teams = {}
    for risk in RISK_LEVELS:
        chosen, used, best = [], set(), chance([], risk)["p_target"]
        for _ in range(MAX_ACTIVE):
            pick, pick_p = None, best
            for c in eligible[:30]:
                if c["sym"] in used:
                    continue
                pr = chance(chosen + [c], risk)["p_target"]
                if pr > pick_p + 1e-9:
                    pick, pick_p = c, pr
            if pick is None:
                break
            chosen.append(pick); used.add(pick["sym"]); best = pick_p
        if not chosen:                                    # nothing helps: fall back to the strongest edges
            for c in eligible:
                if c["sym"] not in used and len(chosen) < MAX_ACTIVE:
                    chosen.append(c); used.add(c["sym"])
        teams[risk] = chosen

    progress("Testing each team on the unseen days", 90)
    risk_table = []
    for risk in RISK_LEVELS:
        cs = teams[risk]
        sel = chance(cs, risk, "sel", sims=3000)
        oos = chance(cs, risk, "oos", sims=6000)
        risk_table.append({"risk": risk, "team": [c["label"] for c in cs],
                           "p_target_selection": sel["p_target"], "big_loss_selection": sel["p_big_loss"],
                           "p_target": oos["p_target"], "p_profit": oos["p_profit"], "p_loss": oos["p_loss"],
                           "median": oos["median"], "p5": oos["p5"], "p95": oos["p95"],
                           "p_big_loss": oos["p_big_loss"], "expected_trades": oos["expected_trades"]})
    # risk is chosen with the SELECTION data only (no peeking at the unseen days)
    ok = [r for r in risk_table if r["big_loss_selection"] <= 0.20]
    rec = max(ok or risk_table[:1], key=lambda r: (r["p_target_selection"], -r["risk"]))
    active = teams[rec["risk"]]
    used = {c["sym"] for c in active}
    backups = [c for c in eligible if c not in active][:MAX_BACKUPS]
    for i, c in enumerate(active + backups):
        c["name"] = f"p{i}_{c['style']}_{c['tf']}_{c['sym']}".lower()
        c["desc"] = (f"Older data: {c['sel_n']} trades, {c['sel_avg'] * 100:+.3f}% avg. "
                     f"Unseen days: {c['oos_n']} trades, {c['oos_avg'] * 100:+.3f}% avg")

    # what IS realistic with this team (unseen-day results)
    achievable = []
    for gain in (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        tgt = start_inr * (1 + gain)
        achievable.append({"gain": gain, "target_inr": tgt,
                           "p": chance(active, rec["risk"], "oos", sims=3000, target=tgt)["p_target"]})
    progress("Done", 100)
    return {
        "time": time.time(), "seconds": round(time.time() - t0, 1),
        "start_inr": start_inr, "target_inr": target_inr, "minutes": minutes,
        "required_pct": target_inr / start_inr - 1,
        "tested": len(combos), "eligible": len(eligible),
        "active": active, "backups": backups,
        "risk_table": risk_table, "recommended_risk": rec["risk"],
        "p_target": rec["p_target"], "p_target_optimistic": rec["p_target_selection"],
        "achievable": achievable,
        "oos_all": {"n": int(sum(c["oos_n"] for c in active)),
                    "avg": float(np.mean(sum((c["rets_oos"] for c in active), [])) if active and any(c["rets_oos"] for c in active) else 0.0)},
    }


def save_plan(p: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    PLAN_PATH.write_text(json.dumps(p))


def load_plan() -> dict | None:
    return json.loads(PLAN_PATH.read_text()) if PLAN_PATH.exists() else None


def live_chance(p: dict, equity_inr: float, target_inr: float, minutes_left: float, risk: float) -> float:
    """Re-estimate the chance from the CURRENT balance and time left (unseen-period results)."""
    if equity_inr >= target_inr:
        return 1.0
    if minutes_left <= 0:
        return 0.0
    pools = [{"rate": c["rate_oos"], "rets": np.array(c["rets_oos"]) * (risk / BASE_RISK)}
             for c in p.get("active", []) if c["rate_oos"] > 0 and c["rets_oos"]]
    return monte_carlo(pools, minutes_left, equity_inr, target_inr, 0.5, sims=1500)["p_target"]
