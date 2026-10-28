"""Local web dashboard: paper-trade the bot with a pretend INR balance and watch it work.

    python3 -m deltabot dashboard        ->  http://127.0.0.1:8050

Everything here is PAPER trading. Your Delta API key (optional) is only used to read your
real balance; no real orders are ever sent from the dashboard.
"""
from __future__ import annotations

import collections
import csv
import json
import logging
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from .client import RESOLUTION_SECONDS, DeltaClient
from .config import BASE_URLS, ROOT, load_settings
from .data import candles_df, unix_seconds
from .state import STATE_DIR

log = logging.getLogger("deltabot.dashboard")
WEB = Path(__file__).resolve().parent / "web"
CONF_PATH = STATE_DIR / "dashboard.json"
EQUITY_PATH = STATE_DIR / "equity_paper.json"
DEFAULT_CONF = {"start_inr": 5000.0, "target_inr": 7000.0, "inr_rate": 95.0,
                "instruments": "perpetual", "risk_per_trade": 0.01,
                "timeframe": "auto", "session_minutes": 30.0, "engine": "arena", "aggression": "normal",
                "aw_profile": "balanced", "rent_month": 200.0}


class MemoryLog(logging.Handler):
    def __init__(self, n=300):
        super().__init__(logging.INFO)
        self.lines = collections.deque(maxlen=n)
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%d %b %H:%M:%S"))

    def snapshot(self, n: int = 150) -> list[str]:
        self.acquire()                      # same lock logging holds while emitting
        try:
            return list(self.lines)[-n:]
        finally:
            self.release()

    def emit(self, record):
        if record.name.startswith(("urllib3", "deltabot.dashboard")):
            return
        self.lines.append(self.format(record))


def _clean(x):
    """JSON-safe numbers."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


class App:
    def __init__(self):
        self.memlog = MemoryLog()
        logging.getLogger().addHandler(self.memlog)
        logging.getLogger().setLevel(logging.INFO)
        STATE_DIR.mkdir(exist_ok=True)
        self.conf = {**DEFAULT_CONF, **(json.loads(CONF_PATH.read_text()) if CONF_PATH.exists() else {})}
        self.lock = threading.RLock()
        self.bot = None
        self.thread = None
        self.stop = threading.Event()
        self.bot_error = ""
        self.connected = {"ok": False}
        self.equity_hist = json.loads(EQUITY_PATH.read_text()) if EQUITY_PATH.exists() else []
        self.replay = {"status": "idle", "progress": 0, "msg": "", "result": None}
        self.analysis = {"status": "idle", "progress": 0, "msg": "", "result": None}
        self.planning = {"status": "idle", "progress": 0, "msg": ""}
        self._replay_obj = None
        self._build_bot()
        s = self.settings()
        if s.api_key and s.api_secret:
            threading.Thread(target=self._check_account, args=(s.api_key, s.api_secret, s.region), daemon=True).start()

    # ---------------------------------------------------------------- setup
    def current_plan(self):
        from .planner import load_plan
        return load_plan()

    def settings(self):
        s = load_settings()
        s.mode = "paper"                                       # dashboard never trades for real
        s.paper_start_equity = self.conf["start_inr"] / self.conf["inr_rate"]
        s.instruments = [x for x in self.conf["instruments"].split(",") if x]
        s.risk_per_trade = float(self.conf["risk_per_trade"])
        s.timeframe_mode = self.conf["timeframe"]
        s.apply_timeframe_mode()
        if self.conf.get("engine") in ("arena", "plan", "ai"):
            s.max_positions = 6                               # bots share the account
            if self.conf.get("aggression") == "target" or self.conf.get("engine") in ("plan", "ai"):
                # TARGET MODE (paper only): bigger bets, more positions, no daily stop
                s.risk_per_trade = max(s.risk_per_trade, 0.03)
                s.max_positions = 10
                s.max_total_notional_x = 10
                s.leverage = 20
                s.daily_loss_limit = 0.99
                s.max_drawdown = 0.50
            if self.conf.get("engine") == "plan":
                p = self.current_plan()
                if p:
                    s.risk_per_trade = p["recommended_risk"]
        return s

    AW_PATH = STATE_DIR / "allweather_backtest.json"

    def aw_backtest(self, force: bool = False):
        """Backtest summary (2020-today) for the dashboard, cached for 12 hours."""
        if not force and self.AW_PATH.exists() and time.time() - self.AW_PATH.stat().st_mtime < 12 * 3600:
            return json.loads(self.AW_PATH.read_text())
        from . import allweather as AW
        saved = AW.WEIGHTS
        out = {}
        for prof, w in (("balanced", {"carry": 0.5, "trend": 0.5}), ("conservative", {"carry": 0.7, "trend": 0.3})):
            AW.WEIGHTS = w
            try:
                r = AW.run(log=lambda *_: None)
            finally:
                AW.WEIGHTS = saved
            port = r["series"]["portfolio"]
            eq = (1 + port).cumprod().resample("W").last()
            prices, _ = AW.load_all(log=lambda *_: None)
            btc = prices["BTC"].reindex(eq.index, method="ffill")
            btc = btc / btc.iloc[0]
            out[prof] = {k: r[k] for k in ("period", "portfolio", "carry", "trend", "btc_hold", "regimes", "years")}
            out[prof]["curve"] = [[int(t.timestamp()), round(float(a), 4), round(float(b), 4)] for t, a, b in zip(eq.index, eq.values, btc.values)]
        self.AW_PATH.write_text(json.dumps(out))
        return out

    MOON_PATH = STATE_DIR / "moonshot_analysis.json"

    def moon_analysis(self):
        if not self.MOON_PATH.exists():
            return None
        a = json.loads(self.MOON_PATH.read_text())
        same = (a["start"], a["target"], a["minutes"]) == (self.conf["start_inr"], self.conf["target_inr"],
                                                          float(self.conf.get("session_minutes") or 60))
        return {**a, "stale": not same}

    def run_moon_analysis(self):
        if self.planning["status"] == "running":
            return
        from .moonshot import analyse
        self.planning = {"status": "running", "progress": 0, "msg": "Starting"}

        def prog(msg, pct):
            self.planning.update(msg=msg, progress=pct)

        def target():
            try:
                a = analyse(self.bot.client, self.bot.products, getattr(self.bot, "symbols", None) or
                            [x for x, _ in __import__("deltabot.scanner", fromlist=["x"]).liquid_universe(
                                self.bot.client, self.bot.products, self.settings())[:8]],
                            self.conf["start_inr"], self.conf["target_inr"],
                            float(self.conf.get("session_minutes") or 60), progress=prog)
                self.MOON_PATH.write_text(json.dumps(a))
                self.planning.update(status="done", progress=100, msg="Done")
                if self.conf.get("engine") == "moonshot" and not self.running:
                    self._build_bot()
                b = a.get("best") or {}
                log.info("Moonshot odds: %.2f%% to reach target, %.1f%% to bust (best: %sx, +%s per win)",
                         b.get("p_target", 0) * 100, b.get("p_bust", 0) * 100, b.get("leverage"), b.get("step"))
            except Exception as e:
                log.exception("moonshot analysis failed")
                self.planning.update(status="error", msg=str(e))
        threading.Thread(target=target, daemon=True, name="moon").start()

    def _build_bot(self):
        if self.conf.get("engine") == "daytrader":
            from .daytrader import DayTraderBot
            self.bot = DayTraderBot(self.settings())
            self.bot.state.save()
            return
        if self.conf.get("engine") == "survival":
            from .survival import SurvivalBot
            self.bot = SurvivalBot(self.settings(), rent_per_day=float(self.conf.get("rent_month", 200.0)) / 30)
            self.bot.state.save()
            return
        if self.conf.get("engine") == "allweather":
            from .allweather_live import AllWeatherBot
            try:
                bt = self.aw_backtest()
            except Exception as e:
                log.warning("all-weather backtest unavailable: %s", e)
                bt = None
            prof = self.conf.get("aw_profile", "balanced")
            self.bot = AllWeatherBot(self.settings(), prof, backtest=(bt or {}).get(prof))
            self.bot.state.save()
            return
        if self.conf.get("engine") == "moonshot":
            from .moonshot import MoonshotBot
            a = self.moon_analysis()
            self.bot = MoonshotBot(self.settings(), analysis=a if a and not a["stale"] else None)
            self.bot.state.save()
            return
        if self.conf.get("engine") == "ai":
            from .arena import ArenaBot
            self.bot = ArenaBot(self.settings(), ai_duel=True)
        elif self.conf.get("engine") == "plan" and self.current_plan():
            from .arena import ArenaBot
            p = self.current_plan()
            p["inr_rate"] = self.conf["inr_rate"]
            self.bot = ArenaBot(self.settings(), plan=p)
        elif self.conf.get("engine") in ("arena", "plan"):
            from .arena import ArenaBot
            self.bot = ArenaBot(self.settings(), target_mode=self.conf.get("aggression") == "target")
        else:
            from .bot import Bot
            self.bot = Bot(self.settings())
        self.bot.state.save()

    @property
    def journal_path(self) -> Path:
        return STATE_DIR / f"trades_{self.bot.state.mode}.csv"

    def save_conf(self):
        CONF_PATH.write_text(json.dumps(self.conf, indent=2))

    @property
    def running(self):
        return bool(self.thread and self.thread.is_alive())

    # ---------------------------------------------------------------- actions
    def start_bot(self, resume: bool = False):
        """resume=True continues the existing paper ledger instead of starting a new session."""
        with self.lock:
            if self.running:
                return
            if self.conf.get("engine") == "plan":
                pv = self.plan_view()
                if not pv or pv["stale"]:
                    self.bot_error = ("Your target, balance or session length changed since the plan was made. "
                                      "Press Plan & analyse, then Use this plan, before starting.")
                    return
            self.stop.clear()
            self.bot_error = ""
            st, s = self.bot.state, self.bot.s
            if st.halted_reason.startswith(("target:", "session:")):
                st.halted_reason = ""                       # new session
            s.target_equity = self.conf["target_inr"] / self.conf["inr_rate"]
            mins = float(self.conf.get("session_minutes") or 0)
            s.session_end = time.time() + mins * 60 if mins > 0 else 0.0
            if hasattr(self.bot, "new_session") and not resume:
                self.bot.new_session()                      # re-hire every bot
            st.save()
            log.info("Session started: target ₹%.0f, %s", self.conf["target_inr"],
                     f"{mins:g} minutes" if mins > 0 else "no time limit")

            def target():
                try:
                    self.bot.run(self.stop)
                except Exception as e:                      # surfaced in the UI
                    self.bot_error = str(e)
                    log.exception("bot crashed")
            self.thread = threading.Thread(target=target, daemon=True, name="bot")
            self.thread.start()

    def stop_bot(self):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=60)

    def reset(self, new_conf: dict | None = None):
        self.stop_bot()
        with self.lock:
            if new_conf:
                for k in DEFAULT_CONF:
                    if k in new_conf and new_conf[k] not in (None, ""):
                        self.conf[k] = type(DEFAULT_CONF[k])(new_conf[k])
                self.save_conf()
            for f in ("state_paper.json", "trades_paper.csv", "equity_paper.json",
                      "state_paper_moonshot.json", "trades_paper_moonshot.csv", "state_paper_allweather.json",
                      "state_paper_survival.json", "trades_paper_survival.csv", "state_paper_survival_core.json",
                      "state_paper_daytrader.json", "trades_paper_daytrader.csv",
                      "state_paper_arena.json", "trades_paper_arena.csv"):
                (STATE_DIR / f).unlink(missing_ok=True)
            self.equity_hist = []
            self.analysis = {"status": "idle", "progress": 0, "msg": "", "result": None}
            self._build_bot()
            log.info("Paper account reset: start ₹%.0f, target ₹%.0f", self.conf["start_inr"], self.conf["target_inr"])

    def _check_account(self, key, secret, region):
        try:
            c = DeltaClient(BASE_URLS[(region, False)], key, secret)
            data = c.wallet_balances()
            meta = data.get("meta") or {}
            usd = float(meta.get("net_equity") or 0)
            if not usd:
                usd = sum(float(w.get("balance") or 0) for w in data["result"] if w.get("asset_symbol") in ("USD", "USDT"))
            self.connected = {"ok": True, "balance_usd": usd, "region": region, "checked": time.time()}
        except Exception as e:
            self.connected = {"ok": False, "error": str(e)[:200]}
        return self.connected

    def connect(self, body):
        key, secret = body.get("key", "").strip(), body.get("secret", "").strip()
        region = body.get("region", "india")
        res = self._check_account(key, secret, region)
        if res["ok"] and body.get("save"):
            env = ROOT / ".env"
            lines = env.read_text().splitlines() if env.exists() else []
            vals = {"DELTA_API_KEY": key, "DELTA_API_SECRET": secret, "DELTA_REGION": region}
            lines = [ln for ln in lines if ln.split("=", 1)[0].strip() not in vals]
            lines += [f"{k}={v}" for k, v in vals.items()]
            env.write_text("\n".join(lines) + "\n")
            env.chmod(0o600)
        return res

    def set_anthropic_key(self, body):
        import os
        import anthropic
        key = body.get("key", "").strip()
        if not key:
            return {"ok": False, "error": "empty key"}
        try:
            anthropic.Anthropic(api_key=key).models.retrieve("claude-opus-5-5")
        except anthropic.AuthenticationError:
            return {"ok": False, "error": "Anthropic rejected this API key"}
        except anthropic.APIStatusError as e:
            return {"ok": False, "error": f"Anthropic API error {e.status_code}"}
        except anthropic.APIConnectionError:
            return {"ok": False, "error": "could not reach the Anthropic API"}
        os.environ["ANTHROPIC_API_KEY"] = key
        if hasattr(self.bot, "claude"):
            self.bot.claude._client = None
        if body.get("save"):
            env = ROOT / ".env"
            lines = env.read_text().splitlines() if env.exists() else []
            lines = [ln for ln in lines if ln.split("=", 1)[0].strip() != "ANTHROPIC_API_KEY"] + [f"ANTHROPIC_API_KEY={key}"]
            env.write_text("\n".join(lines) + "\n")
            env.chmod(0o600)
        log.info("Anthropic API key accepted - Claude AI can trade")
        return {"ok": True}

    def run_plan(self):
        if self.planning["status"] == "running":
            return
        from .planner import plan, save_plan
        self.planning = {"status": "running", "progress": 0, "msg": "Starting"}

        def prog(msg, pct):
            self.planning.update(msg=msg, progress=pct)

        def target():
            try:
                p = plan(self.bot.client, self.bot.products, self.settings(), self.conf["start_inr"],
                         self.conf["target_inr"], float(self.conf.get("session_minutes") or 60), progress=prog)
                p["inr_rate"] = self.conf["inr_rate"]
                save_plan(p)
                self.planning.update(status="done", progress=100, msg="Done")
                log.info("Plan ready: %.1f%% chance (unseen days) at %.0f%% risk per trade, %d strategies, %d backups",
                         p["p_target"] * 100, p["recommended_risk"] * 100, len(p["active"]), len(p["backups"]))
            except Exception as e:
                log.exception("planning failed")
                self.planning.update(status="error", msg=str(e))
        threading.Thread(target=target, daemon=True, name="planner").start()

    def use_plan(self):
        if not self.current_plan():
            return {"ok": False, "error": "make a plan first"}
        self.reset({"engine": "plan"})
        return {"ok": True}

    def plan_view(self):
        p = self.current_plan()
        if not p:
            return None
        strip = lambda c: {k: v for k, v in c.items() if k not in ("rets_sel", "rets_oos")}
        return {**{k: v for k, v in p.items() if k not in ("active", "backups", "live_active")},
                "active": [strip(c) for c in p["active"]], "backups": [strip(c) for c in p["backups"]],
                "stale": (p["start_inr"], p["target_inr"], p["minutes"]) !=
                         (self.conf["start_inr"], self.conf["target_inr"], float(self.conf.get("session_minutes") or 60)),
                "in_use": self.conf.get("engine") == "plan"}

    def run_analysis(self):
        if self.analysis["status"] == "running":
            return
        from .analyse import analyse
        self.analysis = {"status": "running", "progress": 0, "msg": "Starting", "result": None}

        def prog(msg, pct):
            self.analysis.update(msg=msg, progress=pct)

        def target():
            try:
                s = self.settings()
                res = analyse(self.bot.client, self.bot.products, s, self.conf.get("aggression") == "target",
                              self.conf["start_inr"], self.conf["target_inr"],
                              float(self.conf.get("session_minutes") or 60), prog)
                self.analysis.update(status="done", progress=100, msg="Done", result=res)
                log.info("Target analysis: %.1f%% chance of reaching ₹%.0f in %g min",
                         res["p_target"] * 100, res["target_inr"], res["minutes"])
            except Exception as e:
                log.exception("analysis failed")
                self.analysis.update(status="error", msg=str(e))
        threading.Thread(target=target, daemon=True, name="analysis").start()

    def run_replay(self, days: int):
        if self.replay["status"] == "running":
            return
        from .replay import run_replay
        self.replay = {"status": "running", "progress": 0, "msg": "Starting", "result": None}

        def prog(msg, pct):
            self.replay.update(msg=msg, progress=pct)

        def target():
            try:
                s = self.settings()
                rate = self.conf["inr_rate"]
                start = self.conf["start_inr"] / rate
                r = run_replay(self.bot.client, self.bot.products, s, days, start, prog)
                self._replay_obj = r
                curve = [(t, e * rate) for t, e in r.curve]
                eqs = [e for _, e in curve] or [self.conf["start_inr"]]
                peak, dd = eqs[0], 0.0
                for e in eqs:
                    peak = max(peak, e)
                    dd = min(dd, e / peak - 1)
                target_inr = self.conf["target_inr"]
                hit = next((t for t, e in curve if e >= target_inr), None)
                trades = [{**t, "pnl_inr": t["pnl"] * rate} for t in r.trades]
                wins = [t for t in trades if t["pnl"] > 0]
                self.replay.update(status="done", progress=100, msg="Done", result={
                    "days": days, "start_inr": self.conf["start_inr"], "target_inr": target_inr,
                    "end_inr": eqs[-1], "best_inr": max(eqs), "worst_inr": min(eqs), "max_dd": dd,
                    "trades": trades, "curve": curve[:: max(1, len(curve) // 1500)] + curve[-1:],
                    "win_rate": len(wins) / len(trades) if trades else 0, "target_hit": hit,
                    "skipped_small": r.skipped_small, "symbols": sorted({t["symbol"] for t in trades}),
                })
            except Exception as e:
                log.exception("replay failed")
                self.replay.update(status="error", msg=str(e))
        threading.Thread(target=target, daemon=True, name="replay").start()

    # ---------------------------------------------------------------- views
    def state_view(self):
        rate = self.conf["inr_rate"]
        bot = self.bot
        st = bot.state
        try:
            equity = bot.broker.equity()
        except Exception as e:
            equity = st.paper_cash
            log.warning("equity calc failed: %s", e)
        now = time.time()
        if not self.equity_hist or now - self.equity_hist[-1][0] >= 60:
            self.equity_hist.append([now, equity * rate])
            self.equity_hist = self.equity_hist[-5000:]
            EQUITY_PATH.write_text(json.dumps(self.equity_hist))
        open_rows = []
        for sym, t in list(st.trades.items()):
            p = bot.products.get(sym)
            if not p:
                continue
            try:
                mark = bot.broker.quote(sym)["bid" if t.kind == "option" else "mark"]
            except Exception:
                mark = t.entry_price
            sign = 1 if t.side == "buy" else -1
            pnl = sign * (mark - t.entry_price) * t.size * p.contract_value
            open_rows.append({**asdict(t), "mark": mark, "pnl_inr": pnl * rate,
                              "chart_symbol": t.signal_symbol or sym})
        closed = []
        jp = self.journal_path
        if jp.exists():
            with jp.open() as f:
                for r in csv.DictReader(f):
                    try:
                        r["pnl_inr"] = float(r["pnl"]) * rate
                    except (TypeError, ValueError):
                        r["pnl_inr"] = None
                    closed.append(r)
        wins = [r for r in closed if (r["pnl_inr"] or 0) > 0]
        start, target = self.conf["start_inr"], self.conf["target_inr"]
        eq_inr = equity * rate
        return _clean({
            "conf": self.conf,
            "mode": "paper",
            "connected": {**self.connected, "balance_inr": self.connected.get("balance_usd", 0) * rate}
            if self.connected.get("ok") else self.connected,
            "session": {"end": bot.s.session_end, "now": time.time(), "target_hit": st.halted_reason.startswith("target:"),
                        "ended": st.halted_reason.startswith("session:")},
            "bot": {"running": self.running, "error": self.bot_error, "halted": st.halted_reason,
                    "killed": st.killed, "last_scan": st.last_scan,
                    "entry_tf": bot.s.entry_tf, "regime_tf": bot.s.regime_tf, "tf_mode": bot.s.timeframe_mode},
            "equity_inr": eq_inr, "pnl_inr": eq_inr - start, "pnl_pct": eq_inr / start - 1,
            "progress": (eq_inr - start) / (target - start) if target != start else 0,
            "open": open_rows, "closed": closed[::-1][:200],
            "stats": {"trades": len(closed), "win_rate": len(wins) / len(closed) if closed else None},
            "candidates": [asdict(c) for c in bot.candidates[:15]],
            "arena": bot.arena_view(rate) if hasattr(bot, "arena_view") else None,
            "analysis": self.analysis,
            "planning": self.planning,
            "plan": self.plan_view(),
            "moonshot": bot.moon_view(rate) if hasattr(bot, "moon_view") else None,
            "allweather": bot.aw_view(rate) if hasattr(bot, "aw_view") else None,
            "survival": bot.sv_view(rate) if hasattr(bot, "sv_view") else None,
            "daytrader": bot.dt_view(rate) if hasattr(bot, "dt_view") else None,
            "moon_analysis": self.moon_analysis(),
            "equity_hist": self.equity_hist[-1500:],
            "logs": self.memlog.snapshot(150),
        })

    def chart_view(self, symbol: str, source: str, tf: str | None = None):
        from dataclasses import replace
        from .strategy import entry_frame
        s = self.bot.s
        if source != "replay":
            tf = tf or next((t.entry_tf for k, t in list(self.bot.state.trades.items())
                             if (t.signal_symbol or k) == symbol and t.entry_tf), None) \
                or next((c.entry_tf for c in self.bot.candidates if c.symbol == symbol and c.entry_tf), None)
            if tf:
                s = replace(s, entry_tf=tf)
        rate = self.conf["inr_rate"]
        if source == "replay" and self._replay_obj and symbol in self._replay_obj.candles:
            df = self._replay_obj.candles[symbol]
            days = (self.replay["result"] or {}).get("days", 90)
            df_view_start = df.index[-1].timestamp() - days * 86400
        else:
            df = candles_df(self.bot.client, symbol, s.entry_tf, 400, closed_only=False)
            df_view_start = 0
        f = entry_frame(df, s)
        t = unix_seconds(df.index)
        keep = [i for i, x in enumerate(t) if x >= df_view_start]

        def series(col):
            v = f[col].values
            return [{"time": t[i], "value": float(v[i])} for i in keep if np.isfinite(v[i])]
        candles = [{"time": t[i], "open": float(df["open"].iloc[i]), "high": float(df["high"].iloc[i]),
                    "low": float(df["low"].iloc[i]), "close": float(df["close"].iloc[i])} for i in keep]
        bar = RESOLUTION_SECONDS[s.entry_tf]
        snap = lambda x: int(x) - int(x) % bar
        markers, lines = [], []
        if source == "replay" and self.replay.get("result"):
            for tr in self.replay["result"]["trades"]:
                if tr["symbol"] != symbol:
                    continue
                long = tr["side"] == "buy"
                markers.append({"time": snap(tr["entry_time"]), "position": "belowBar" if long else "aboveBar",
                                "color": "#22c55e" if long else "#ef4444", "shape": "arrowUp" if long else "arrowDown",
                                "text": f"{'BUY' if long else 'SELL'} {tr['size']}"})
                markers.append({"time": snap(tr["exit_time"]), "position": "aboveBar" if long else "belowBar",
                                "color": "#22c55e" if tr["pnl"] > 0 else "#ef4444", "shape": "circle",
                                "text": f"₹{tr['pnl'] * rate:+.0f}"})
        else:
            jp = self.journal_path
            labels = {a.name: a.label for a in getattr(self.bot, "agents", {}).values()}
            if jp.exists():
                with jp.open() as fh:
                    for r in csv.DictReader(fh):
                        if r["symbol"] != symbol:
                            continue
                        long = r["side"] == "buy"
                        et = time.mktime(time.strptime(r["entry_time"], "%Y-%m-%d %H:%M:%S"))
                        xt = time.mktime(time.strptime(r["exit_time"], "%Y-%m-%d %H:%M:%S"))
                        pnl = float(r["pnl"] or 0) * rate
                        markers.append({"time": snap(et), "position": "belowBar" if long else "aboveBar",
                                        "color": "#22c55e" if long else "#ef4444",
                                        "shape": "arrowUp" if long else "arrowDown",
                                        "text": f"{labels.get(r.get('agent') or '', '')} {r['side'].upper()}".strip()})
                        markers.append({"time": snap(xt), "position": "aboveBar" if long else "belowBar",
                                        "color": "#22c55e" if pnl > 0 else "#ef4444", "shape": "circle",
                                        "text": f"₹{pnl:+.0f}"})
            for sym, tr in list(self.bot.state.trades.items()):
                if (tr.signal_symbol or sym) != symbol:
                    continue
                long = tr.view == "buy" if tr.kind == "option" else tr.side == "buy"
                label = f"{'CALL' if long else 'PUT'} {sym}" if tr.kind == "option" else f"{'BUY' if long else 'SELL'} {tr.size}"
                if tr.agent:
                    label = f"{labels.get(tr.agent, tr.agent)} {label}"
                markers.append({"time": snap(tr.entry_time), "position": "belowBar" if long else "aboveBar",
                                "color": "#22c55e" if long else "#ef4444",
                                "shape": "arrowUp" if long else "arrowDown", "text": label})
                if tr.kind != "option":
                    lines += [{"price": tr.entry_price, "title": "ENTRY", "color": "#94a3b8"},
                              {"price": tr.stop, "title": "STOP", "color": "#ef4444"},
                              {"price": tr.target, "title": "TARGET", "color": "#22c55e"}]
        markers = [m for m in markers if not candles or m["time"] >= candles[0]["time"]]
        markers.sort(key=lambda m: m["time"])
        return _clean({"symbol": symbol, "tf": s.entry_tf, "candles": candles,
                       "ema_fast": series("ema_fast"), "ema_slow": series("ema_slow"),
                       "dc_high": series("dc_high"), "dc_low": series("dc_low"),
                       "markers": markers, "lines": lines,
                       "labels": {"ema_fast": f"EMA {s.ema_fast}", "ema_slow": f"EMA {s.ema_slow}",
                                  "dc": f"{s.donchian}-bar breakout channel"}})


APP: App | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode())

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/state":
                return self._json(APP.state_view())
            if u.path == "/api/chart":
                return self._json(APP.chart_view(q["symbol"], q.get("source", "live"), q.get("tf")))
            if u.path == "/api/replay":
                return self._json(_clean(APP.replay))
            self._json({"error": "not found"}, 404)
        except Exception as e:
            log.exception("GET %s", self.path)
            self._json({"error": str(e)}, 500)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        try:
            if self.path == "/api/bot/start":
                APP.start_bot(bool(body.get("resume")))
            elif self.path == "/api/bot/stop":
                APP.stop_bot()
            elif self.path == "/api/reset":
                APP.reset(body)
            elif self.path == "/api/connect":
                return self._json(APP.connect(body))
            elif self.path == "/api/anthropic":
                return self._json(APP.set_anthropic_key(body))
            elif self.path == "/api/replay":
                APP.run_replay(int(body.get("days", 90)))
            elif self.path == "/api/analyse":
                APP.run_analysis()
            elif self.path == "/api/plan":
                APP.run_plan()
            elif self.path in ("/api/dt/approve", "/api/dt/reject"):
                if not hasattr(APP.bot, "approve"):
                    return self._json({"ok": False, "error": "Day Trader engine is not active"})
                fn = APP.bot.approve if self.path.endswith("approve") else APP.bot.reject
                with APP.lock:
                    return self._json(fn(str(body.get("id", ""))))
            elif self.path == "/api/dt/scan":
                if hasattr(APP.bot, "step"):
                    threading.Thread(target=APP.bot.step, daemon=True).start()
            elif self.path == "/api/moonshot/analyse":
                APP.run_moon_analysis()
            elif self.path == "/api/plan/use":
                return self._json(APP.use_plan())
            else:
                return self._json({"error": "not found"}, 404)
            self._json({"ok": True})
        except Exception as e:
            log.exception("POST %s", self.path)
            self._json({"error": str(e)}, 500)


def serve(port: int = 8050, resume: bool = False):
    global APP
    from .notify import setup_logging
    setup_logging()
    APP = App()
    if resume:
        APP.start_bot(resume=True)
        log.info("Resumed the existing paper session")
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Dashboard running at http://127.0.0.1:{port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        APP.stop_bot()
