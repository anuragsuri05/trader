"""All-weather portfolio - live PAPER engine on Delta Exchange data.

* Same position logic as the backtest (allweather.trend_weights / carry_states), fed with Delta daily
  candles and Delta funding history.
* Rebalances once a day shortly after the daily close (00:05 UTC) and once at start.
* Funding is charged / earned every hour at Delta's live rate; fees on every rebalance.
* AI risk committee (once a day, before the rebalance): Claude Fable searches the web for danger
  (hacks, insolvencies, bans, depegs); ChatGPT reviews the numbers and Claude's headlines. They can
  only REDUCE risk: final trend scale = the lower of the two; carry stays on only if both agree.
* Model-portfolio ledger with fractional sizes: real trading needs enough capital for Delta's minimum
  contract sizes - the panel shows the minimum.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import allweather as AW
from .ai_traders import find_claude_cli
from .client import DeltaClient
from .data import candles_df, load_products
from .notify import Notifier
from .state import State

log = logging.getLogger("deltabot.allweather")
MODE = "paper_allweather"
PROFILES = {"balanced": {"carry": 0.5, "trend": 0.5}, "conservative": {"carry": 0.7, "trend": 0.3}}

COMMITTEE_PROMPT = """You are one member of a two-person RISK COMMITTEE for an automated crypto portfolio on Delta Exchange.
The portfolio has two parts: (1) a market-neutral carry trade (long spot + short perpetual on BTC/ETH/SOL/XRP,
earning funding) and (2) a diversified trend-following book (long/short ~16 coins, daily rebalanced, volatility
targeted). You do NOT predict prices. Your only job is to protect capital from abnormal danger.

Return:
- risk_scale: multiplier for the trend book, 1.0 = normal. Lower it ONLY for concrete, current danger: exchange
  hacks or insolvency (especially Delta Exchange or a major venue), stablecoin depegs, sudden bans, a market-wide
  crash in progress, broken data. 0 = close the trend book.
- carry_ok: false only if the carry trade itself is endangered (exchange solvency, funding turning sharply
  negative, spot/perp disconnect).
- headlines: the specific facts you relied on (short).
- reason: one or two sentences.
Normal volatility, ordinary dips and generic macro worries are NOT reasons to cut risk - the strategy is built to
handle them, and cutting risk without real cause costs returns."""

SCHEMA = {"type": "object", "properties": {"risk_scale": {"type": "number"}, "carry_ok": {"type": "boolean"},
          "headlines": {"type": "array", "items": {"type": "string"}}, "reason": {"type": "string"}},
          "required": ["risk_scale", "carry_ok", "headlines", "reason"], "additionalProperties": False}


class Shim:
    """Minimal broker-like object so the dashboard can read equity / quotes."""

    def __init__(self, bot):
        self.bot = bot
        self.marks: dict = {}

    def equity(self) -> float:
        return self.bot.equity()

    def quote(self, symbol: str) -> dict:
        m = self.marks.get(symbol) or 0.0
        return {"mark": m, "bid": m, "ask": m, "spot": m}


class AllWeatherBot:
    def __init__(self, s, profile: str = "balanced", backtest: dict | None = None, mode: str = MODE,
                 committee: bool = True):
        if s.is_live:
            raise RuntimeError("All-weather engine is paper-only in this build")
        self.s = s
        self.profile = profile if profile in PROFILES else "balanced"
        self.client = DeltaClient(s.base_url)
        self.notify = Notifier(s)
        self.products = load_products(self.client)
        self.state = State.load(mode)
        self.state.mode = mode
        self.use_committee = committee
        self.broker = Shim(self)
        self.candidates = []
        self.backtest = backtest
        self.lock = threading.RLock()
        self.coins = [c for c in sorted(set(AW.UNIVERSE_2020) | set(AW.CARRY_COINS)) if f"{c}USD" in self.products]
        self.cli = find_claude_cli()
        self._committee_busy = False
        self._data_cache = None
        aw = self.state.extra.get("aw")
        if not aw:
            self._fresh(s.paper_start_equity)

    # ------------------------------------------------------------ ledger
    def _fresh(self, equity_usd: float) -> None:
        self.state.extra["aw"] = {"cash": equity_usd, "start": equity_usd, "pos": {}, "carry": {}, "last_rebalance": 0.0,
                                  "last_accrual": time.time(), "peak": equity_usd, "braked": False,
                                  "committee": {}, "fees": 0.0, "fund_trend": 0.0, "fund_carry": 0.0,
                                  "trades": [], "history": [], "targets": {}, "carry_info": {}}
        self.state.trades.clear()
        self.state.halted_reason = ""

    @property
    def aw(self) -> dict:
        return self.state.extra["aw"]

    def equity(self) -> float:
        aw, marks = self.aw, self.broker.marks
        upnl = sum(p["units"] * ((marks.get(f"{c}USD") or p["avg"]) - p["avg"]) for c, p in aw["pos"].items())
        return aw["cash"] + upnl

    def new_session(self) -> None:
        """Dashboard 'Start' = start (or restart) the model portfolio from the configured capital."""
        self._fresh(self.s.paper_start_equity)
        self.state.save()
        self.notify(f"All-weather portfolio started ({self.profile}): {self.s.paper_start_equity:.2f} USD")

    # ------------------------------------------------------------ data
    def market_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        if self._data_cache and time.time() - self._data_cache[0] < 3600:
            return self._data_cache[1], self._data_cache[2]
        prices, funding = {}, {}
        for c in self.coins:
            try:
                d = candles_df(self.client, f"{c}USD", "1d", 420)
                f = candles_df(self.client, f"FUNDING:{c}USD", "1h", 24 * 30, closed_only=False)
            except Exception as e:
                log.warning("data %s: %s", c, e)
                continue
            if len(d) < 200:
                continue
            prices[c] = d["close"]
            funding[c] = (f["close"] / 100).resample("1D").mean() * 3 if len(f) else pd.Series(dtype=float)
        P = pd.DataFrame(prices).sort_index()
        F = pd.DataFrame(funding).reindex(P.index).fillna(0.0)
        self._data_cache = (time.time(), P, F)
        return P, F

    def refresh_marks(self) -> dict:
        tk = self.client.tickers("perpetual_futures")
        marks = {t["symbol"]: float(t.get("mark_price") or t.get("close") or 0) for t in tk}
        self.broker.marks = marks
        self.funding_now = {t["symbol"]: float(t.get("funding_rate") or 0) / 100 for t in tk}   # per 8h
        return marks

    # ------------------------------------------------------------ committee
    def _ask_claude(self, text: str) -> dict:
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        p = subprocess.run([self.cli, "-p", "--output-format", "json", "--no-session-persistence", "--model", "claude-fable-5-1",
                            "--tools", "WebSearch", "--allowedTools", "WebSearch", "--system-prompt", COMMITTEE_PROMPT,
                            "--json-schema", json.dumps(SCHEMA)], input=text, capture_output=True, text=True, timeout=600,
                           env=env, cwd=Path(__file__).resolve().parent.parent / "state")
        out = json.loads(p.stdout)
        if out.get("is_error"):
            raise RuntimeError(str(out.get("result"))[:200])
        return out.get("structured_output") or json.loads(out.get("result") or "{}")

    def _ask_gpt(self, text: str) -> dict:
        from openai import OpenAI
        r = OpenAI(timeout=180).chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
            messages=[{"role": "system", "content": COMMITTEE_PROMPT}, {"role": "user", "content": text}],
            response_format={"type": "json_schema", "json_schema": {"name": "risk", "strict": True, "schema": SCHEMA}})
        return json.loads(r.choices[0].message.content)

    def committee_snapshot(self) -> dict:
        P, F = self.market_data()
        snap = {"date_utc": time.strftime("%Y-%m-%d %H:%M"), "portfolio_drawdown_pct": round((self.equity() / self.aw["peak"] - 1) * 100, 2)}
        for c in ("BTC", "ETH", "SOL"):
            if c in P:
                p = P[c].dropna()
                snap[c] = {"chg_1d_pct": round((p.iloc[-1] / p.iloc[-2] - 1) * 100, 2), "chg_7d_pct": round((p.iloc[-1] / p.iloc[-8] - 1) * 100, 2),
                           "chg_30d_pct": round((p.iloc[-1] / p.iloc[-31] - 1) * 100, 2),
                           "vol_30d_ann_pct": round(p.pct_change().iloc[-30:].std() * np.sqrt(365) * 100, 1),
                           "funding_7d_avg_pct_per_8h": round(F[c].iloc[-7:].mean() / 3 * 100, 4)}
        return snap

    def run_committee(self) -> None:
        if self._committee_busy or not self.use_committee:
            return
        self._committee_busy = True

        def work():
            rec = {"time": time.time(), "claude": None, "gpt": None, "errors": []}
            try:
                snap = self.committee_snapshot()
                text = "Current numbers:\n" + json.dumps(snap, indent=1)
                if self.cli:
                    try:
                        rec["claude"] = self._ask_claude(text + "\n\nSearch the web for today's crypto risk news before deciding.")
                    except Exception as e:
                        rec["errors"].append(f"Claude: {str(e)[:150]}")
                if os.environ.get("OPENAI_API_KEY"):
                    heads = (rec["claude"] or {}).get("headlines", [])
                    try:
                        rec["gpt"] = self._ask_gpt(text + "\n\nNews found by the other committee member:\n" + json.dumps(heads))
                    except Exception as e:
                        rec["errors"].append(f"ChatGPT: {str(e)[:150]}")
                votes = [v for v in (rec["claude"], rec["gpt"]) if v]
                rec["risk_scale"] = float(min([1.0] + [max(0.0, min(1.0, v["risk_scale"])) for v in votes]))
                rec["carry_ok"] = all(v["carry_ok"] for v in votes) if votes else True
                with self.lock:
                    self.aw["committee"] = rec
                    self.state.save()
                self.notify(f"Risk committee: trend scale {rec['risk_scale']:.2f}, carry {'on' if rec['carry_ok'] else 'OFF'}"
                            + (f" ({'; '.join(rec['errors'])})" if rec["errors"] else ""))
            except Exception as e:
                log.exception("committee failed: %s", e)
            finally:
                self._committee_busy = False
        threading.Thread(target=work, daemon=True, name="committee").start()

    # ------------------------------------------------------------ rebalance + accrual
    def rebalance(self) -> None:
        P, F = self.market_data()
        marks = self.refresh_marks()
        with self.lock:
            aw = self.aw
            eq = self.equity()
            # settle open P&L at the current mark (daily futures settlement), then trade to targets
            for c, p in aw["pos"].items():
                m = marks.get(f"{c}USD") or p["avg"]
                aw["cash"] += p["units"] * (m - p["avg"])
                p["avg"] = m
            dd = eq / aw["peak"] - 1
            aw["braked"] = dd < -AW.DD_BRAKE or (aw["braked"] and dd < -AW.DD_RELEASE)
            com = aw.get("committee") or {}
            weights = PROFILES[self.profile]
            trend_cap = eq * weights["trend"] * (0.5 if aw["braked"] else 1.0) * float(com.get("risk_scale", 1.0))
            w = AW.trend_weights(P).iloc[-1]
            targets = {}
            for c, wt in w.items():
                m = marks.get(f"{c}USD")
                if not m:
                    continue
                units = wt * trend_cap / m
                old = aw["pos"].get(c, {"units": 0.0, "avg": m})
                delta = units - old["units"]
                if abs(delta) * m > 1e-9:
                    fee = abs(delta) * m * AW.PERP_COST
                    aw["cash"] -= fee
                    aw["fees"] += fee
                    if abs(delta) * m > 0.01 * eq:
                        aw["trades"].append({"time": time.time(), "coin": c, "from": old["units"] * m, "to": units * m, "fee": fee})
                aw["pos"][c] = {"units": units, "avg": m}
                targets[c] = {"weight": float(wt), "value": units * m}
            aw["pos"] = {c: p for c, p in aw["pos"].items() if abs(p["units"]) > 0}
            # carry: on/off by the same rule as the backtest, vetoed by the committee
            on = AW.carry_states(P, F).iloc[-1]
            if not com.get("carry_ok", True):
                on[:] = False
            n_on = int(on.sum())
            carry_cap = eq * weights["carry"] * AW.CARRY_HEDGE_FRACTION
            info = {}
            for c in on.index:
                new = carry_cap / n_on if on[c] and n_on else 0.0
                old = aw["carry"].get(c, 0.0)
                if abs(new - old) > 0.05 * max(new, old, 1e-9):
                    cost = abs(new - old) * AW.CARRY_SWITCH_COST
                    aw["cash"] -= cost
                    aw["fees"] += cost
                    aw["carry"][c] = new
                info[c] = {"on": bool(on[c]), "notional": aw["carry"].get(c, 0.0),
                           "funding_7d_pct": float(F[c].iloc[-7:].mean() / 3 * 100) if c in F else 0.0}
            aw["targets"], aw["carry_info"] = targets, info
            aw["trades"] = aw["trades"][-40:]
            aw["last_rebalance"] = time.time()
            self.state.save()
        self.notify(f"Rebalanced: trend book {sum(abs(v['value']) for v in targets.values()):.2f} USD gross across "
                    f"{len(targets)} coins, carry on {n_on} coins, equity {self.equity():.2f} USD")

    def accrue(self) -> None:
        """Charge / earn funding for the time since the last accrual at Delta's live rates."""
        with self.lock:
            aw = self.aw
            hours = (time.time() - aw["last_accrual"]) / 3600
            if hours < 0.25:
                return
            fr = getattr(self, "funding_now", {})
            for c, p in aw["pos"].items():
                m = self.broker.marks.get(f"{c}USD") or p["avg"]
                pay = p["units"] * m * fr.get(f"{c}USD", 0.0) * hours / 8       # longs pay positive funding
                aw["cash"] -= pay
                aw["fund_trend"] -= pay
            for c, notional in aw["carry"].items():
                get = notional * fr.get(f"{c}USD", 0.0) * hours / 8             # short perp receives
                aw["cash"] += get
                aw["fund_carry"] += get
            aw["last_accrual"] = time.time()
            eq = self.equity()
            aw["peak"] = max(aw["peak"], eq)
            aw["history"].append([time.time(), eq])
            aw["history"] = aw["history"][-5000:]
            self.state.save()

    # ------------------------------------------------------------ loop
    def step(self) -> None:
        self.refresh_marks()
        self.accrue()

    def run(self, stop=None) -> None:
        if not self.aw.get("committee"):
            self.run_committee()
            t0 = time.time()
            while self._committee_busy and time.time() - t0 < 600 and not (stop and stop.is_set()):
                (stop.wait if stop else time.sleep)(5)
        while not (stop and stop.is_set()):
            try:
                now = time.time()
                last = self.aw["last_rebalance"]
                next_reb = (last // 86400 + 1) * 86400 + 300 if last else 0
                if now >= next_reb:
                    if last and now - self.aw.get("committee", {}).get("time", 0) > 20 * 3600:
                        self.run_committee()                       # fresh committee for the new day
                        t0 = time.time()
                        while self._committee_busy and time.time() - t0 < 600:
                            time.sleep(5)
                    self.rebalance()
                self.step()
            except Exception as e:
                log.exception("all-weather loop: %s", e)
            (stop.wait if stop else time.sleep)(60)

    # ------------------------------------------------------------ view
    def min_capital_usd(self) -> float:
        need = 0.0
        for c, t in self.aw.get("targets", {}).items():
            p = self.products.get(f"{c}USD")
            if p and abs(t["value"]) > 0:
                one = p.contract_value * (self.broker.marks.get(f"{c}USD") or 0)
                need = max(need, one / (abs(t["value"]) / max(self.equity(), 1e-9)))
        return need

    def aw_view(self, rate: float) -> dict:
        aw = self.aw
        last = aw["last_rebalance"]
        return {"profile": self.profile, "equity": self.equity(), "start": aw["start"], "cash": aw["cash"],
                "fees": aw["fees"], "fund_trend": aw["fund_trend"], "fund_carry": aw["fund_carry"],
                "braked": aw["braked"], "committee": aw.get("committee", {}), "committee_busy": self._committee_busy,
                "targets": aw.get("targets", {}), "carry": aw.get("carry_info", {}), "trades": aw["trades"][-12:],
                "last_rebalance": last, "next_rebalance": (last // 86400 + 1) * 86400 + 300 if last else None,
                "min_capital_usd": self.min_capital_usd(), "history": aw["history"][-1500:],
                "backtest": self.backtest}
