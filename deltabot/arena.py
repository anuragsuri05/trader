"""Bot arena: several trading bots are hired to reach a goal; bots that fail are fired.

Rules (paper trading, one shared pool so trades are big enough to place):
  * Each bot gets an equal share of the budget and a personal goal of the same % gain as the team.
  * A bot is FIRED immediately if it loses `death_loss` (default 20%) of its share.
  * At each performance review (default 2, evenly spaced in the session) the worst bot that
    is losing money is FIRED.
  * A fired bot's positions are closed and its budget share is split among the survivors.
  * The arena stops when the team reaches the target, the time is up, or every bot is fired.
"""
from __future__ import annotations

import csv
import logging
import time

from dataclasses import replace

from .agents import default_team, regime_on, target_team
from .claude_trader import MODEL as CLAUDE_MODEL, ClaudeTrader, snapshot_text
from . import planner as P
from .ai_traders import HOLDS, LLMTrader, duel_traders
import collections
import threading
from .broker import PaperBroker
from .client import RESOLUTION_SECONDS, DeltaClient
from .config import Settings
from .data import candles_df, load_products
from .notify import Notifier
from .risk import RiskManager, position_size
from .scanner import Candidate, liquid_universe
from .state import STATE_DIR, State
from .strategy import Signal

log = logging.getLogger("deltabot.arena")
MODE = "paper_arena"


class ArenaBot:
    def __init__(self, s: Settings, death_loss: float = 0.20, reviews: int = 2, universe_size: int = 15,
                 target_mode: bool = False, plan: dict | None = None, ai_duel: bool = False):
        if s.is_live:
            raise RuntimeError("The arena is paper-only")
        self.s = s
        self.target_mode = target_mode
        self.death_loss, self.reviews, self.universe_size = death_loss, reviews, universe_size
        self.client = DeltaClient(s.base_url)
        self.notify = Notifier(s)
        self.products = load_products(self.client)
        self.state = State.load(MODE)
        self.state.mode = MODE
        self.broker = PaperBroker(self.client, s, self.state, self.products)
        self.risk = RiskManager(s, self.state)
        self.plan = plan
        self.ai_duel = ai_duel
        self.lock = threading.RLock()
        self.ticks: dict[str, collections.deque] = {}
        if ai_duel:
            target_mode = True
            reviews = 0                                    # duel: only the -20% rule fires a trader
            self.reviews = 0
            team = duel_traders()
        elif plan:
            target_mode = self.target_mode = True          # plans are measured with market entries
            team = [P.make_agent(c) for c in plan["active"]]
        else:
            team = target_team() if target_mode else default_team()
        self.agents = {a.name: a for a in team}
        self._replanning = False
        self.claude = ClaudeTrader(target_mode=target_mode)
        if not ai_duel:
            self.agents[self.claude.name] = self.claude
        self.candidates: list[Candidate] = []
        self._cache: dict[tuple[str, str], tuple[float, object]] = {}
        self._universe: list[str] = []
        self._universe_at = 0.0
        self.state.extra.setdefault("bots", {})
        if set(self.state.extra["bots"]) != set(self.agents):      # first run or line-up changed
            self._hire_all(self.s.paper_start_equity)
        log.info("arena ready: %d bots, %d products", len(self.agents), len(self.products))

    # ------------------------------------------------------------ hiring
    def _hire_all(self, equity: float) -> None:
        share = equity / len(self.agents)
        self.state.extra["bots"] = {
            n: {"alive": True, "budget": share, "died": "", "died_at": 0.0, "promoted": False, "ideas": 0,
                "last_idea": "", "skipped_small": 0}
            for n in self.agents
        }
        self.state.extra["reviews_done"] = 0
        self.state.extra["session_start"] = time.time()
        self.state.extra["start_equity"] = equity

    def new_session(self) -> None:
        """Called when a session starts: re-hire every bot with the current balance."""
        self.state.extra["orders"] = {}
        for sym in list(self.state.trades):
            self.broker.close(self.products[sym], "new_session")
        self._hire_all(self.broker.equity())
        self.state.extra["feed"] = {}
        self.state.extra["monitor"] = {}
        self.state.extra["used_backups"] = []
        if self.plan:
            self.plan["live_active"] = []
        self.state.save()
        names = ", ".join(a.label for a in self.agents.values())
        self.notify(f"Hired {len(self.agents)} bots: {names}")

    # ------------------------------------------------------------ data
    def universe(self) -> list[str]:
        if time.time() - self._universe_at > 900 or not self._universe:
            self._universe = [sym for sym, _ in liquid_universe(self.client, self.products, self.s)[: self.universe_size]]
            self._universe_at = time.time()
        return self._universe

    def candles(self, sym: str, tf: str):
        """Cached closed candles; refetched only after a new bar of that timeframe closes."""
        step = RESOLUTION_SECONDS[tf]
        bar_close = time.time() // step * step
        hit = self._cache.get((sym, tf))
        if hit and hit[0] >= bar_close:
            return hit[1]
        df = candles_df(self.client, sym, tf, 300)
        self._cache[(sym, tf)] = (bar_close, df)
        return df

    def _frame_last(self, agent, sym):
        if isinstance(agent, (ClaudeTrader, LLMTrader)):
            return None, None
        e = self.candles(sym, agent.entry_tf)
        r = self.candles(sym, agent.regime_tf)
        if len(e) < 80 or len(r) < self.s.regime_ema_slow + 20:
            return None, None
        reg = regime_on(e, r, self.s, agent.regime_tf, agent.entry_tf)
        return agent.frame(e, reg, self.s).iloc[-1], e

    def _event(self, name: str, text: str) -> None:
        feed = self.state.extra.setdefault("feed", {}).setdefault(name, [])
        feed.append([time.time(), text])
        del feed[:-8]

    # ------------------------------------------------------------ scoring
    def bot_pnl(self) -> dict[str, dict]:
        """Realized (journal, this session) + unrealized P&L per bot, in USD."""
        start = self.state.extra.get("session_start", 0)
        out = {n: {"realized": 0.0, "unrealized": 0.0, "trades": 0, "wins": 0, "open": 0} for n in self.agents}
        path = STATE_DIR / f"trades_{MODE}.csv"
        if path.exists():
            with path.open() as f:
                for row in csv.DictReader(f):
                    a = row.get("agent") or ""
                    if a not in out:
                        continue
                    try:
                        et = time.mktime(time.strptime(row["entry_time"], "%Y-%m-%d %H:%M:%S"))
                        pnl = float(row["pnl"] or 0)
                    except (ValueError, TypeError):
                        continue
                    if et < start - 1:
                        continue
                    out[a]["realized"] += pnl
                    out[a]["trades"] += 1
                    out[a]["wins"] += pnl > 0
        for sym, t in list(self.state.trades.items()):
            if t.agent not in out:
                continue
            p = self.products[sym]
            try:
                mark = self.broker.quote(sym)["mark"]
            except Exception:
                mark = t.entry_price
            sign = 1 if t.side == "buy" else -1
            out[t.agent]["unrealized"] += sign * (mark - t.entry_price) * t.size * p.contract_value
            out[t.agent]["open"] += 1
        for v in out.values():
            v["pnl"] = v["realized"] + v["unrealized"]
        if path.exists():                              # trailing losing streak per bot (this session)
            with path.open() as f:
                rows = [r for r in csv.DictReader(f) if (r.get("agent") or "") in out]
            for n in out:
                streak = 0
                for r in reversed([r for r in rows if r["agent"] == n]):
                    try:
                        if time.mktime(time.strptime(r["entry_time"], "%Y-%m-%d %H:%M:%S")) < start - 1:
                            break
                        if float(r["pnl"] or 0) < 0:
                            streak += 1
                        else:
                            break
                    except ValueError:
                        break
                out[n]["streak"] = streak
        return out

    def _fire(self, name: str, reason: str) -> None:
        bots = self.state.extra["bots"]
        b = bots[name]
        if not b["alive"]:
            return
        for sym, t in list(self.state.trades.items()):
            if t.agent == name:
                self.broker.close(self.products[sym], "bot_fired")
        orders = self.state.extra.get("orders", {})
        for sym in [k for k, o in orders.items() if o["agent"] == name]:
            orders.pop(sym)
        b.update(alive=False, died=reason, died_at=time.time())
        self.notify(f"FIRED {self.agents[name].label}: {reason}")
        self._event(name, f"fired: {reason}")
        if self.plan and self._activate_backup(b["budget"], replacing=self.agents[name].label):
            return
        survivors = [n for n, x in bots.items() if x["alive"]]
        for n in survivors:
            bots[n]["budget"] += b["budget"] / len(survivors)

    def _activate_backup(self, budget: float, replacing: str) -> bool:
        bots = self.state.extra["bots"]
        busy = {a.symbols[0] for n, a in self.agents.items() if getattr(a, "symbols", None) and bots.get(n, {}).get("alive")}
        tried = set(self.state.extra.setdefault("used_backups", []))
        for spec in self.plan.get("backups", []):
            if spec["name"] in tried or spec["sym"] in busy:
                continue
            tried.add(spec["name"])
            self.state.extra["used_backups"] = sorted(tried)
            ag = P.make_agent(spec)
            self.agents[ag.name] = ag
            bots[ag.name] = {"alive": True, "budget": budget, "died": "", "died_at": 0.0, "promoted": False,
                             "ideas": 0, "last_idea": "", "skipped_small": 0, "backup_for": replacing}
            self.plan.setdefault("live_active", []).append(spec)
            self.notify(f"BACKUP STRATEGY ACTIVATED: {ag.label} replaces {replacing}")
            self._event(ag.name, f"activated as backup for {replacing}")
            return True
        self._start_replan()
        return False

    def _start_replan(self) -> None:
        if self._replanning or not self.plan:
            return
        self._replanning = True
        self.notify("Backups used up - re-planning on fresh data in the background")

        def work():
            try:
                eq_inr = self.broker.equity() * self.plan.get("inr_rate", 95)
                fresh = P.plan(self.client, self.products, self.s, eq_inr, self.plan["target_inr"],
                               max(5.0, (self.s.session_end - time.time()) / 60) if self.s.session_end else 60)
                busy = {a.symbols[0] for a in self.agents.values() if getattr(a, "symbols", None)}
                self.plan["backups"] = [c for c in fresh["active"] + fresh["backups"] if c["sym"] not in busy]
                for c in self.plan["backups"]:
                    c["name"] = "r" + c["name"]
                self.notify(f"Re-plan done: {len(self.plan['backups'])} new backup strategies ready")
            except Exception as e:
                log.warning("re-plan failed: %s", e)
            finally:
                self._replanning = False
        threading.Thread(target=work, daemon=True, name="replan").start()

    def _review(self) -> None:
        st, bots = self.state, self.state.extra["bots"]
        goal = self.s.target_equity / st.extra.get("start_equity", self.s.paper_start_equity) - 1 \
            if self.s.target_equity else 0.4
        pnl = self.bot_pnl()
        for n, b in bots.items():
            if not b["alive"]:
                continue
            p = pnl[n]["pnl"]
            if self.plan and pnl[n].get("streak", 0) >= 3 and n != self.claude.name:
                self._fire(n, "3 losing trades in a row")
            elif p <= -self.death_loss * b["budget"]:
                self._fire(n, f"lost {-p / b['budget']:.0%} of its budget (limit {self.death_loss:.0%})")
            elif not b["promoted"] and goal > 0 and p >= goal * b["budget"]:
                b["promoted"] = True
                self.notify(f"{self.agents[n].label} hit its personal goal (+{goal:.0%})")
        # scheduled performance reviews: fire the worst losing bot
        if self.s.session_end and self.reviews:
            start = st.extra.get("session_start", time.time())
            length = self.s.session_end - start
            due = int((time.time() - start) / (length / (self.reviews + 1))) if length > 0 else 0
            while st.extra.get("reviews_done", 0) < min(due, self.reviews):
                st.extra["reviews_done"] = st.extra.get("reviews_done", 0) + 1
                pnl = self.bot_pnl()
                losers = [(pnl[n]["pnl"], n) for n, b in bots.items() if b["alive"] and pnl[n]["pnl"] < 0]
                k = st.extra["reviews_done"]
                if losers:
                    worst = min(losers)
                    self._fire(worst[1], f"worst performer at review {k}/{self.reviews} "
                                         f"(P&L {worst[0]:+.2f} USD)")
                else:
                    self.notify(f"Review {k}/{self.reviews}: no bot is losing money, nobody fired")
        if not any(b["alive"] for b in bots.values()) and not st.halted_reason:
            st.halted_reason = "session: every bot was fired"
            self.notify("Every bot has been fired - arena stopped")

    # ------------------------------------------------------------ trading
    def _manage_exits(self) -> None:
        for sym, t in list(self.state.trades.items()):
            agent = self.agents.get(t.agent)
            if not agent:
                continue
            f, _ = self._frame_last(agent, t.signal_symbol or sym)
            reason = None
            if f is not None and (f["exit_long"] if t.side == "buy" else f["exit_short"]):
                reason = "signal_exit"
            elif time.time() - t.entry_time > agent.max_hold_bars * RESOLUTION_SECONDS[agent.entry_tf]:
                reason = "time_stop"
            if reason:
                pnl = self.broker.close(self.products[sym], reason)
                self.notify(f"{agent.label} closed {t.side.upper()} {sym} ({reason}) P&L {pnl:+.2f}")
                self._event(t.agent, f"closed {sym} ({reason}) P&L {pnl:+.3f} USD")

    def _hunt(self, equity: float) -> None:
        bots, trades = self.state.extra["bots"], self.state.trades
        orders = self.state.extra.setdefault("orders", {})
        ideas_all: list[Candidate] = []
        open_notional = sum(self.products[k].notional(t.size, t.entry_price) for k, t in trades.items())
        bot_signals: dict[str, list[str]] = {}
        for name, agent in self.agents.items():
            b = bots[name]
            if not b["alive"] or isinstance(agent, ClaudeTrader):
                continue
            ideas, radar, scanned, trending = [], [], 0, 0
            for sym in (agent.symbols or self.universe()):
                if sym in trades or sym in orders:
                    continue
                try:
                    f, e = self._frame_last(agent, sym)
                    if f is not None and hasattr(agent, "explain"):
                        scanned += 1
                        r = self.candles(sym, agent.regime_tf)
                        ex = agent.explain(e, regime_on(e, r, self.s, agent.regime_tf, agent.entry_tf), self.s)
                        if ex:
                            trending += 1
                            radar.append((ex["dist"], f"{sym}: {ex['text']}"))
                except Exception as ex:
                    log.warning("%s %s: %s", name, sym, ex)
                    continue
                if f is None or not (f["long"] or f["short"]) or not f["sl_dist"] > 0:
                    continue
                c = float(e["close"].iloc[-1])
                long = bool(f["long"])
                sig = Signal("buy" if long else "sell", c, c - f["sl_dist"] if long else c + f["sl_dist"],
                             c + f["tp_dist"] if long else c - f["tp_dist"], float(f["sl_dist"]))
                ideas.append((float(f["score"]), sym, sig))
                bot_signals.setdefault(sym, []).append(f"{agent.label}: {sig.side.upper()}")
                ideas_all.append(Candidate(sym, 1 if long else -1, float(f["score"]), 0, 0, 0, 0, "",
                                           agent.regime_tf, agent.entry_tf, agent.label))
            b["ideas"] += len(ideas)
            b["scan"] = {"time": time.time(), "scanned": scanned, "trending": trending,
                         "closest": [t for _, t in sorted(radar)[:3]]}
            if ideas:
                best = max(ideas, key=lambda x: x[0])
                b["last_idea"] = f"{best[2].side.upper()} {best[1]} ({time.strftime('%H:%M')})"
            mine = sum(1 for t in trades.values() if t.agent == name) + \
                sum(1 for o in orders.values() if o["agent"] == name)
            for score, sym, sig in sorted(ideas, key=lambda x: x[0], reverse=True):
                if mine >= agent.max_positions or len(trades) + len(orders) >= self.s.max_positions:
                    break
                if sym in trades or sym in orders:
                    continue
                product = self.products[sym]
                bot_s, own = self._bot_sizing(name, equity)
                size = position_size(product, sig, equity, own, bot_s)
                if size < 1:
                    b["skipped_small"] += 1
                    continue
                mine += 1
                if self.target_mode:      # immediate market entry
                    fill = self._market_entry(product, sig, size, name, agent.entry_tf)
                    open_notional += product.notional(size, fill)
                    continue
                # limit (maker) entry at the signal price, valid for 3 candles
                self.broker.place_limit(product, sig, size, name, agent.entry_tf,
                                        3 * RESOLUTION_SECONDS[agent.entry_tf])
                open_notional += product.notional(size, sig.price)
                self.notify(f"{agent.label} placed LIMIT {sig.side.upper()} {size} x {sym} @ {sig.price:.6g} "
                            f"(SL {sig.stop:.6g}, TP {sig.target:.6g})")
                self._event(name, f"placed limit {sig.side.upper()} {size} {sym} @ {sig.price:.6g}")
        ideas_all.sort(key=lambda c: c.score, reverse=True)
        self.candidates = ideas_all[:20]
        self.state.last_scan = time.time()
        self._claude_turn(equity, bot_signals)

    def _market_entry(self, product, sig, size: int, name: str, entry_tf: str) -> float:
        px = self.broker.quote(product.symbol)["mark"]
        # keep the planned stop/target distances around the live price
        sig = Signal(sig.side, px, px - (sig.price - sig.stop), px + (sig.target - sig.price), sig.atr)
        fill = self.broker.open(product, sig, size, signal_symbol=product.symbol)
        t = self.state.trades[product.symbol]
        t.agent, t.entry_tf = name, entry_tf
        label = self.agents[name].label
        self.notify(f"{label} BOUGHT {size} x {product.symbol} @ {fill:.6g}" if sig.side == "buy" else
                    f"{label} SOLD {size} x {product.symbol} @ {fill:.6g}")
        self._event(name, f"{'bought' if sig.side == 'buy' else 'sold'} {size} {product.symbol} @ {fill:.6g} "
                          f"(SL {sig.stop:.6g} TP {sig.target:.6g})")
        return fill

    def _bot_sizing(self, name: str, equity: float):
        """Each bot may only use its own share of the account's exposure allowance."""
        b = self.state.extra["bots"][name]
        share = min(1.0, b["budget"] / equity) if equity > 0 else 0.0
        own = sum(self.products[k].notional(t.size, t.entry_price)
                  for k, t in list(self.state.trades.items()) if t.agent == name)
        return replace(self.s, max_total_notional_x=self.s.max_total_notional_x * share), own

    # ------------------------------------------------------------ Claude
    def _claude_turn(self, equity: float, bot_signals: dict[str, list[str]]) -> None:
        ag, st = self.claude, self.state
        b = st.extra["bots"].get(ag.name)
        if not b or not b["alive"]:
            return
        if not ag.available:
            ag.last = {"time": time.time(), "error": "Claude Code CLI not found and no Anthropic API key set"}
            return
        if time.time() - st.extra.get("claude_last", 0) < ag.interval_s - 20:
            return
        st.extra["claude_last"] = time.time()
        coins = []
        for sym in self.universe():
            try:
                snap = ag.coin_snapshot(sym, self.candles(sym, "5m"), self.candles(sym, "15m"),
                                        self.candles(sym, "1h"), self.s, bot_signals.get(sym, []),
                                        self.products[sym].contract_value)
            except Exception as e:
                log.warning("claude snapshot %s: %s", sym, e)
                continue
            if snap:
                coins.append(snap)
        trades, orders = st.trades, st.extra.setdefault("orders", {})
        pnl = self.bot_pnl()[ag.name]
        marks = {}
        for sym, t in trades.items():
            if t.agent == ag.name:
                try:
                    marks[sym] = self.broker.quote(sym)["mark"]
                except Exception:
                    marks[sym] = t.entry_price
        account = {
            "equity_usd": round(equity, 2),
            "my_budget_usd": round(b["budget"], 2),
            "my_pnl_usd": round(pnl["pnl"], 3),
            "fired_if_pnl_below_usd": round(-self.death_loss * b["budget"], 2),
            "my_exposure_room_usd": round(max(0.0, (lambda bs, own: bs.max_total_notional_x * equity - own)(
                *self._bot_sizing(ag.name, equity))), 2),
            "my_positions": [{"sym": k, "side": t.side, "contracts": t.size, "entry": t.entry_price, "stop": t.stop,
                              "target": t.target, "mark": marks.get(k), "minutes_open": round((time.time() - t.entry_time) / 60)}
                             for k, t in trades.items() if t.agent == ag.name],
            "my_pending_limit_orders": [{"sym": k, "side": o["side"], "limit": o["limit"]}
                                        for k, o in orders.items() if o["agent"] == ag.name],
            "symbols_held_by_other_bots": sorted([k for k, t in trades.items() if t.agent != ag.name]
                                                 + [k for k, o in orders.items() if o["agent"] != ag.name]),
            "minutes_left_in_session": round(max(0, self.s.session_end - time.time()) / 60) if self.s.session_end else None,
            "team_target_usd": round(self.s.target_equity, 2) if self.s.target_equity else None,
        }
        t0 = time.time()
        decision, thinking, err = ag.decide(snapshot_text(coins, account))
        record = {"time": time.time(), "seconds": round(time.time() - t0, 1), "coins": len(coins),
                  "view": decision.market_view if decision else "", "thinking": thinking[:3000],
                  "error": err, "actions": []}
        by_sym = {c["sym"]: c for c in coins}
        if decision:
            for a in decision.actions:
                try:
                    verdict = self._apply_claude(a, by_sym.get(a.symbol), equity)
                except Exception as e:
                    verdict = f"error: {e}"
                record["actions"].append({**a.model_dump(), "verdict": verdict})
        ag.last = record
        st.extra["claude_last_record"] = record
        b["scan"] = {"time": record["time"], "scanned": len(coins), "trending": None, "closest": []}
        self._event(ag.name, err or ("; ".join(f"{x['action']} {x['side']} {x['symbol']}: {x['verdict']}"
                                               for x in record["actions"]) or "looked, no trade"))
        if err:
            self.notify(f"Claude AI: {err}")
        else:
            acts = "; ".join(f"{x['action']} {x['side']} {x['symbol']} -> {x['verdict']}" for x in record["actions"]) or "no trade"
            self.notify(f"Claude AI ({record['seconds']}s): {acts}")

    def _apply_claude(self, a, coin: dict | None, equity: float) -> str:
        ag, st = self.claude, self.state
        trades, orders = st.trades, st.extra.setdefault("orders", {})
        if a.action == "close":
            t = trades.get(a.symbol)
            if not t or t.agent != ag.name:
                return "rejected: not my position"
            pnl = self.broker.close(self.products[a.symbol], "claude_close")
            return f"closed, P&L {pnl:+.3f} USD"
        if a.symbol not in self.universe():
            return "rejected: not in the tradable list"
        if a.symbol in trades or a.symbol in orders:
            return "rejected: coin already held or ordered"
        mine = sum(1 for t in trades.values() if t.agent == ag.name) + \
            sum(1 for o in orders.values() if o["agent"] == ag.name)
        if mine >= ag.max_positions:
            return "rejected: already at 2 positions"
        if len(trades) + len(orders) >= self.s.max_positions:
            return "rejected: team position limit reached"
        why = ag.check(a, coin)
        if why:
            return "rejected: " + why
        px = coin["px"]
        sig = Signal(a.side, px, a.stop_price, a.target_price, coin["atr_5m"])
        bot_s, own = self._bot_sizing(ag.name, equity)
        risk_s = replace(bot_s, risk_per_trade=self.s.risk_per_trade * max(0.5, min(1.0, a.confidence)))
        product = self.products[a.symbol]
        size = position_size(product, sig, equity, own, risk_s)
        if size < 1:
            st.extra["bots"][ag.name]["skipped_small"] += 1
            return "rejected: position would be under 1 contract"
        if self.target_mode:
            fill = self._market_entry(product, sig, size, ag.name, ag.entry_tf)
            return f"{'BOUGHT' if a.side == 'buy' else 'SOLD'} {size} @ {fill:g} (market)"
        self.broker.place_limit(product, sig, size, ag.name, ag.entry_tf, 3 * RESOLUTION_SECONDS[ag.entry_tf])
        return f"placed LIMIT {a.side.upper()} {size} @ {px:g}" 

    def _check_session(self, equity: float) -> None:
        st, s = self.state, self.s
        if st.halted_reason.startswith(("target:", "session:")):
            return
        if s.target_equity and equity >= s.target_equity:
            self.flatten("target_reached")
            st.halted_reason = f"target: reached {equity:.2f} >= {s.target_equity:.2f}, all positions closed"
            self.notify("TEAM TARGET REACHED - closed everything")
        elif s.session_end and time.time() >= s.session_end:
            self.flatten("session_end")
            st.halted_reason = "session: time is up, all positions closed"
            self.notify(f"Time is up. Final balance {self.broker.equity():.2f} USD")

    def _check_orders(self) -> None:
        for msg in self.broker.check_orders():
            text, _, agent = msg.partition("|")
            self.notify(f"{self.agents[agent].label if agent in self.agents else agent}: {text}")
            self._event(agent, text)

    def _check_brackets(self) -> None:
        owners = {k: t.agent for k, t in self.state.trades.items()}
        for msg in self.broker.check_brackets():
            self.notify(msg)
            for sym, agent in owners.items():
                if sym in msg:
                    self._event(agent, msg.replace("[paper] ", ""))

    def step(self) -> None:
        self._check_orders()
        self._check_brackets()
        equity = self.broker.equity()
        for event in self.risk.update(equity):
            self.notify(event)
        self._check_session(equity)
        if self.state.killed and self.state.trades:
            self.flatten("kill_switch")
        if not self.state.halted_reason.startswith(("target:", "session:")):
            self._review()
        self._manage_exits()
        if self.risk.can_open and not self.state.halted_reason:
            self._hunt(self.broker.equity())
        alive = [self.agents[n].label for n, b in self.state.extra["bots"].items() if b["alive"]]
        log.info("arena equity=%.2f open=%d alive=%s", equity, len(self.state.trades), ", ".join(alive) or "none")
        self.state.save()

    def flatten(self, reason: str = "manual_flatten") -> None:
        self.state.extra["orders"] = {}
        for sym in list(self.state.trades):
            self.broker.close(self.products[sym], reason)
        self.state.save()

    def _fast_tick(self) -> None:
        with self.lock:
            self._fast_tick_locked()

    def _fast_tick_locked(self) -> None:
        """Every 5 seconds: live prices, stop-loss/target/hold time, session, firing/backups, live chance."""
        try:
            marks = {t["symbol"]: float(t.get("mark_price") or t.get("close") or 0)
                     for t in self.client.tickers("perpetual_futures") if t.get("symbol")}
        except Exception as e:
            log.warning("tick: prices failed: %s", e)
            return
        self.broker.set_marks(marks)
        for sym in self._universe or []:
            if marks.get(sym):
                self.ticks.setdefault(sym, collections.deque(maxlen=60)).append(marks[sym])
        for sym, t in list(self.state.trades.items()):
            m = marks.get(sym)
            if not m:
                continue
            if t.hold_until and time.time() >= t.hold_until:
                pnl = self.broker.close(self.products[sym], "hold_time_up", m)
                self.notify(f"[5s check] {sym} hold time over, closed, pnl {pnl:+.2f}")
                self._event(t.agent, f"{sym} hold time over, closed, P&L {pnl:+.3f} USD")
                continue
            long = t.side == "buy"
            if (m <= t.stop) if long else (m >= t.stop):
                pnl = self.broker.close(self.products[sym], "stop_loss", t.stop)
                self.notify(f"[5s check] {sym} stop-loss hit, pnl {pnl:+.2f}")
                self._event(t.agent, f"{sym} stop-loss hit, P&L {pnl:+.3f} USD")
            elif (m >= t.target) if long else (m <= t.target):
                pnl = self.broker.close(self.products[sym], "take_profit", t.target, maker=True)
                self.notify(f"[5s check] {sym} take-profit hit, pnl {pnl:+.2f}")
                self._event(t.agent, f"{sym} take-profit hit, P&L {pnl:+.3f} USD")
        eq = self.broker.equity()
        self._check_session(eq)
        if not self.state.halted_reason.startswith(("target:", "session:")):
            self._review()
        self._monitor(eq)
        self.state.save()

    def _monitor(self, eq: float) -> None:
        st, s = self.state, self.s
        start = st.extra.get("start_equity", eq) or eq
        rec = {"time": time.time(), "equity": eq, "checks": st.extra.get("monitor", {}).get("checks", 0) + 1}
        if s.session_end and s.target_equity:
            t0 = st.extra.get("session_start", time.time())
            elapsed = max(1.0, time.time() - t0)
            total = max(1.0, s.session_end - t0)
            need = s.target_equity - start
            expected = start + need * min(1.0, elapsed / total)
            rec["pace"] = "ahead" if eq >= expected else ("behind" if eq < start else "slightly behind")
            rec["expected_now"] = expected
            if self.plan:
                specs = [c for c in self.plan["active"] + self.plan.get("live_active", [])
                         if st.extra["bots"].get(c["name"], {}).get("alive")]
                rate = self.plan.get("inr_rate", 95)
                rec["live_chance"] = P.live_chance({"active": specs}, eq * rate, s.target_equity * rate,
                                                   max(0.0, (s.session_end - time.time()) / 60), s.risk_per_trade)
        rec["backups_left"] = (len([b for b in (self.plan or {}).get("backups", [])
                                    if b["name"] not in st.extra.get("used_backups", [])]) if self.plan else None)
        st.extra["monitor"] = rec

    def run(self, stop=None) -> None:
        if self.ai_duel:
            self._run_ai(stop)
            return
        if self.plan:
            self._run_plan(stop)
            return
        step_s = min(RESOLUTION_SECONDS[a.entry_tf] for a in self.agents.values())   # fastest bot's candle
        while not (stop and stop.is_set()):
            try:
                self.step()
            except Exception as e:
                log.exception("arena step failed: %s", e)
            sleep = step_s - (time.time() % step_s) + 8
            if self.s.session_end > time.time():
                sleep = max(1, min(sleep, self.s.session_end - time.time() + 2))
            # between candle closes, keep checking stop-loss / take-profit and the deadline every minute
            end = time.time() + sleep
            while time.time() < end and not (stop and stop.is_set()):
                (stop.wait if stop else time.sleep)(min(60, max(0.5, end - time.time())))
                if time.time() < end - 5:
                    try:
                        self._check_orders()
                        self._check_brackets()
                        eq = self.broker.equity()
                        self._check_session(eq)
                        if not self.state.halted_reason.startswith(("target:", "session:")):
                            self._review()
                        self.state.save()
                    except Exception as e:
                        log.warning("arena minute check failed: %s", e)

    # ------------------------------------------------------------ AI duel
    def _run_ai(self, stop=None) -> None:
        self.universe()
        next_minute = 0.0
        while not (stop and stop.is_set()):
            now = time.time()
            try:
                if now >= next_minute:
                    with self.lock:
                        self._check_brackets()
                    next_minute = now + 60
                self._fast_tick()
                halted = self.state.halted_reason.startswith(("target:", "session:")) or not self.risk.can_open
                for tr in self.agents.values():
                    b = self.state.extra["bots"].get(tr.name, {})
                    if halted or not b.get("alive") or tr.busy or not tr.available or time.time() < tr.next_at:
                        continue
                    tr.busy = True
                    threading.Thread(target=self._ai_turn, args=(tr,), daemon=True, name=f"ai-{tr.name}").start()
            except Exception as e:
                log.exception("ai loop: %s", e)
            (stop.wait if stop else time.sleep)(5)

    def _ai_snapshot(self, tr: LLMTrader) -> tuple[list[dict], dict]:
        from .indicators import atr as _atr, rsi as _rsi
        coins = []
        for sym in self.universe():
            try:
                c1, c5 = self.candles(sym, "1m"), self.candles(sym, "5m")
                snap = tr.coin_snapshot(sym, c5, self.candles(sym, "15m"), self.candles(sym, "1h"), self.s, [],
                                        self.products[sym].contract_value)
            except Exception as e:
                log.warning("ai snapshot %s: %s", sym, e)
                continue
            if not snap or len(c1) < 30:
                continue
            nd = snap["closes_5m"][0]
            sig = len(str(nd).split(".")[-1]) if "." in str(nd) else 2
            snap["closes_1m"] = [round(float(x), sig) for x in c1["close"].iloc[-15:]]
            snap["rsi_1m"] = round(float(_rsi(c1["close"], 14).iloc[-1]), 1)
            snap["atr_pct_1m"] = round(float(_atr(c1, 14).iloc[-1] / c1["close"].iloc[-1] * 100), 3)
            snap["ticks_5s"] = [round(x, sig) for x in list(self.ticks.get(sym, []))[-24:]]
            coins.append(snap)
        with self.lock:
            st, trades = self.state, self.state.trades
            b = st.extra["bots"][tr.name]
            equity = self.broker.equity()
            pnl = self.bot_pnl()[tr.name]
            bot_s, own = self._bot_sizing(tr.name, equity)
            recent = []
            path = STATE_DIR / f"trades_{MODE}.csv"
            if path.exists():
                with path.open() as f:
                    rows = [r for r in csv.DictReader(f) if r.get("agent") == tr.name]
                for r in rows[-5:]:
                    recent.append({"sym": r["symbol"], "side": r["side"], "pnl_usd": round(float(r["pnl"] or 0), 3),
                                   "exit": r["reason"].split(":")[-1], "opened": r["entry_time"][11:], "closed": r["exit_time"][11:]})
            account = {
                "equity_usd": round(equity, 2), "my_budget_usd": round(b["budget"], 2), "my_pnl_usd": round(pnl["pnl"], 3),
                "fired_if_pnl_below_usd": round(-self.death_loss * b["budget"], 2),
                "my_exposure_room_usd": round(max(0.0, bot_s.max_total_notional_x * equity - own), 2),
                "my_positions": [{"sym": k, "side": t.side, "contracts": t.size, "entry": t.entry_price, "stop": t.stop,
                                  "target": t.target, "mark": self.broker.marks.get(k),
                                  "seconds_left": round(t.hold_until - time.time()) if t.hold_until else None}
                                 for k, t in trades.items() if t.agent == tr.name],
                "my_recent_trades": recent,
                "held_by_other_trader": sorted(k for k, t in trades.items() if t.agent != tr.name),
                "minutes_left_in_session": round(max(0, self.s.session_end - time.time()) / 60, 1) if self.s.session_end else None,
                "team_target_usd": round(self.s.target_equity, 2) if self.s.target_equity else None,
                "gain_still_needed_pct": round((self.s.target_equity / equity - 1) * 100, 2) if self.s.target_equity else None,
            }
        return coins, account

    def _ai_turn(self, tr: LLMTrader) -> None:
        try:
            coins, account = self._ai_snapshot(tr)
            t0 = time.time()
            data, err = tr.decide(snapshot_text(coins, account))
            rec = {"time": time.time(), "seconds": round(time.time() - t0, 1), "coins": len(coins),
                   "view": (data or {}).get("market_view", ""), "error": err, "actions": []}
            wait = 120
            if data:
                wait = int(min(600, max(30, data.get("next_check_seconds") or 120)))
                by = {c["sym"]: c for c in coins}
                with self.lock:
                    for a in data.get("actions", []):
                        try:
                            verdict = self._apply_ai(tr, a, by.get(a.get("symbol")))
                        except Exception as e:
                            verdict = f"error: {e}"
                        rec["actions"].append({**a, "verdict": verdict})
            rec["next_check_s"] = wait
            tr.last = rec
            tr.next_at = time.time() + wait
            msg = err or ("; ".join(f"{x['action']} {x['side']} {x['symbol']} ({x['hold']}): {x['verdict']}" for x in rec["actions"])
                          or "looked, no trade")
            self.notify(f"{tr.label} ({rec['seconds']}s): {msg}")
            with self.lock:
                self._event(tr.name, msg)
                self.state.extra.setdefault("ai_last", {})[tr.name] = rec
                b = self.state.extra["bots"].get(tr.name)
                if b is not None:
                    b["scan"] = {"time": rec["time"], "scanned": len(coins), "trending": None, "closest": []}
                self.state.save()
        except Exception as e:
            log.exception("ai turn %s: %s", tr.name, e)
            tr.next_at = time.time() + 60
        finally:
            tr.busy = False

    def _apply_ai(self, tr: LLMTrader, a: dict, coin: dict | None) -> str:
        st = self.state
        trades = st.trades
        b = st.extra["bots"].get(tr.name, {})
        if not b.get("alive") or st.halted_reason:
            return "rejected: trading stopped"
        if a["action"] == "close":
            t = trades.get(a["symbol"])
            if not t or t.agent != tr.name:
                return "rejected: not my position"
            pnl = self.broker.close(self.products[a["symbol"]], "ai_close", self.broker.marks.get(a["symbol"]))
            return f"closed, P&L {pnl:+.3f} USD"
        if a["symbol"] not in self.universe():
            return "rejected: not in the tradable list"
        if a["symbol"] in trades:
            return "rejected: coin already held"
        if sum(1 for t in trades.values() if t.agent == tr.name) >= tr.max_positions:
            return "rejected: already at 4 positions"
        why = tr.check(a, coin)
        if why:
            return "rejected: " + why
        risk = min(8.0, max(0.5, float(a["risk_pct"]))) / 100
        equity = self.broker.equity()
        bot_s, own = self._bot_sizing(tr.name, equity)
        product = self.products[a["symbol"]]
        px = self.broker.marks.get(a["symbol"]) or coin["px"]
        sig = Signal(a["side"], px, px - (coin["px"] - a["stop_price"]), px + (a["target_price"] - coin["px"]), 0)
        size = position_size(product, sig, equity, own, replace(bot_s, risk_per_trade=risk))
        if size < 1:
            return "rejected: under 1 contract with this risk/room"
        fill = self._market_entry(product, sig, size, tr.name, "1m")
        trades[a["symbol"]].hold_until = time.time() + HOLDS[a["hold"]]
        return f"{'BOUGHT' if a['side'] == 'buy' else 'SOLD'} {size} @ {fill:g}, risk {risk:.1%}, hold {a['hold']}"

    def _run_plan(self, stop=None) -> None:
        """Plan mode: full decision cycle at every candle close of the fastest strategy,
        and a 5-second check loop in between."""
        step_s = min(RESOLUTION_SECONDS[a.entry_tf] for a in self.agents.values())
        next_step, next_minute = 0.0, 0.0
        while not (stop and stop.is_set()):
            now = time.time()
            try:
                if now >= next_step:
                    self.step()
                    step_s = min(RESOLUTION_SECONDS[a.entry_tf] for a in self.agents.values())
                    next_step = (time.time() // step_s + 1) * step_s + 6
                else:
                    if now >= next_minute:                # wick check on 1-minute candles
                        self._check_orders()
                        self._check_brackets()
                        next_minute = now + 60
                    self._fast_tick()
            except Exception as e:
                log.exception("plan loop: %s", e)
            (stop.wait if stop else time.sleep)(5)

    # ------------------------------------------------------------ view
    def arena_view(self, rate: float) -> dict:
        pnl = self.bot_pnl()
        bots = self.state.extra.get("bots", {})
        start_eq = self.state.extra.get("start_equity", self.s.paper_start_equity)
        goal = self.s.target_equity / start_eq - 1 if self.s.target_equity and start_eq else 0.4
        out = []
        for n, a in list(self.agents.items()):
            b, p = bots.get(n, {}), pnl[n]
            budget = b.get("budget", 0)
            out.append({
                "name": n, "label": a.label, "style": a.style, "tf": f"{a.regime_tf} → {a.entry_tf}",
                "alive": b.get("alive", True), "died": b.get("died", ""), "died_at": b.get("died_at", 0),
                "promoted": b.get("promoted", False),
                "budget_inr": budget * rate, "pnl_inr": p["pnl"] * rate, "realized_inr": p["realized"] * rate,
                "goal_inr": goal * budget * rate, "death_inr": -self.death_loss * budget * rate,
                "trades": p["trades"], "wins": p["wins"], "open": p["open"],
                "orders": sum(1 for o in list(self.state.extra.get("orders", {}).values()) if o["agent"] == n),
                "ideas": b.get("ideas", 0), "last_idea": b.get("last_idea", ""), "skipped_small": b.get("skipped_small", 0),
                "scan": b.get("scan", {}), "feed": self.state.extra.get("feed", {}).get(n, [])[-4:],
                "backup_for": b.get("backup_for", ""),
            })
        claude = {"available": self.claude.available, "model": CLAUDE_MODEL, "calls": self.claude.calls,
                  "target_mode": self.target_mode,
                  "backend": self.claude.backend,
                  "cost_usd": round(self.claude.cost_usd, 4), "interval_min": self.claude.interval_s // 60,
                  "last": self.claude.last or self.state.extra.get("claude_last_record") or {}}
        mon = dict(self.state.extra.get("monitor", {}))
        ai = [{"name": t.name, "label": t.label, "model": t.model, "backend": t.backend, "available": t.available,
               "calls": t.calls, "tokens": t.tokens, "cost_usd": round(t.cost_usd, 4), "busy": t.busy,
               "next_in": max(0, round(t.next_at - time.time())),
               "last": t.last or self.state.extra.get("ai_last", {}).get(t.name, {})}
              for t in list(self.agents.values()) if isinstance(t, LLMTrader)]
        return {"bots": out, "claude": None if self.ai_duel else claude, "ai": ai, "monitor": mon,
                "plan_mode": bool(self.plan), "ai_duel": self.ai_duel, "reviews_done": self.state.extra.get("reviews_done", 0), "reviews": self.reviews,
                "death_loss": self.death_loss, "goal_pct": goal,
                "session_start": self.state.extra.get("session_start", 0)}
