"""How achievable is a target? Estimated BEFORE the bot starts.

1. Replay the chosen mode's rule bots on recent real Delta candles (fees + slippage included) to measure
   how often they trade and what each trade returns as a fraction of the account.
2. Monte Carlo: simulate many sessions of the chosen length by drawing that many trades (Poisson) from
   the measured outcomes. A session "reaches the target" if the balance touches it (the bot stops there);
   the account kill switch ends a session early.

Limits (shown on the dashboard): Claude has no trading history so it is not included; per-bot exposure
caps and simultaneous-position limits are approximated; the past 1-2 weeks may not resemble the next session.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

from .agents import default_team, regime_on, simulate_agent, target_team
from .client import RESOLUTION_SECONDS, DeltaClient
from .config import Settings
from .data import candles_df
from .scanner import liquid_universe

FUNDING_YIELD = 0.065          # cash-and-carry backtest, per year on capital (for comparison only)


def measure(client: DeltaClient, products, s: Settings, target_mode: bool, coins: int = 10, progress=lambda m, p: None):
    team = target_team() if target_mode else default_team()
    syms = [x for x, _ in liquid_universe(client, products, s)[:coins]]
    per_agent = []
    steps = len(team) * len(syms)
    k = 0
    for ag in team:
        days = 5 if ag.entry_tf in ("1m", "3m") else 14
        eb = int(days * 86400 / RESOLUTION_SECONDS[ag.entry_tf]) + 80
        rb = int(days * 86400 / RESOLUTION_SECONDS[ag.regime_tf]) + s.regime_ema_slow + 60
        rets, holds, n_trades = [], [], 0
        for sym in syms:
            k += 1
            progress(f"Replaying {ag.label} on {sym}", 5 + 80 * k / steps)
            try:
                e = candles_df(client, sym, ag.entry_tf, eb)
                r = candles_df(client, sym, ag.regime_tf, rb)
            except Exception:
                continue
            if len(e) < 200 or len(r) < s.regime_ema_slow + 20:
                continue
            start = 1000.0
            trades, _ = simulate_agent(e, regime_on(e, r, s, ag.regime_tf, ag.entry_tf), ag,
                                       products[sym].contract_value, s, start)
            for t in trades:
                rets.append(t["pnl"] / start)
                holds.append(t["bars"] * RESOLUTION_SECONDS[ag.entry_tf] / 3600)
            n_trades += len(trades)
        if not rets:
            continue
        raw_rate = n_trades / (days * 24)                                  # trades per hour, all coins
        avg_hold = max(np.mean(holds), RESOLUTION_SECONDS[ag.entry_tf] / 3600)
        rate = min(raw_rate, ag.max_positions / avg_hold)                  # can't hold more than max_positions
        per_agent.append({"label": ag.label, "rate": rate, "rets": np.array(rets), "days": days,
                          "win": float(np.mean(np.array(rets) > 0)), "avg": float(np.mean(rets))})
    return per_agent


def monte_carlo(per_agent, minutes: float, start_inr: float, target_inr: float, kill_dd: float,
                sims: int = 20000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    hours = minutes / 60
    goal = target_inr / start_inr
    if not per_agent:
        return {"p_target": 0.0, "p_profit": 0.0, "p_loss": 0.0, "median": start_inr, "p5": start_inr,
                "p95": start_inr, "expected_trades": 0.0, "p_big_loss": 0.0}
    rates = np.array([a["rate"] for a in per_agent])
    total_rate = rates.sum()
    pools = [a["rets"] for a in per_agent]
    probs = rates / total_rate
    finals, hits = np.empty(sims), 0
    n_trades = rng.poisson(total_rate * hours, sims)
    for i in range(sims):
        eq, peak, hit = 1.0, 1.0, False
        if n_trades[i]:
            which = rng.choice(len(pools), n_trades[i], p=probs)
            for w in which:
                eq *= 1 + pools[w][rng.integers(len(pools[w]))]
                peak = max(peak, eq)
                if eq >= goal:
                    hit = True
                    break
                if eq <= peak * (1 - kill_dd):
                    break
        hits += hit
        finals[i] = eq
    finals_inr = finals * start_inr
    return {
        "p_target": hits / sims,
        "p_profit": float(np.mean(finals > 1.0005)),
        "p_loss": float(np.mean(finals < 0.9995)),
        "median": float(np.median(finals_inr)),
        "p5": float(np.percentile(finals_inr, 5)),
        "p95": float(np.percentile(finals_inr, 95)),
        "expected_trades": float(total_rate * hours),
        "p_big_loss": float(np.mean(finals <= 0.70)),
    }


def analyse(client, products, s: Settings, target_mode: bool, start_inr: float, target_inr: float,
            minutes: float, progress=lambda m, p: None) -> dict:
    t0 = time.time()
    agents = measure(client, products, s, target_mode, progress=progress)
    progress("Simulating 20,000 sessions", 90)
    mc = monte_carlo(agents, minutes, start_inr, target_inr, s.max_drawdown)
    need = target_inr / start_inr - 1
    days_funding = math.log(target_inr / start_inr) / math.log(1 + FUNDING_YIELD) * 365 if target_inr > start_inr else 0
    progress("Done", 100)
    return {
        **mc,
        "mode": "target" if target_mode else "normal",
        "required_pct": need,
        "minutes": minutes,
        "start_inr": start_inr,
        "target_inr": target_inr,
        "agents": [{k: v for k, v in a.items() if k != "rets"} | {"trades_measured": int(len(a["rets"]))} for a in agents],
        "funding_days": days_funding,
        "seconds": round(time.time() - t0, 1),
        "time": time.time(),
    }
