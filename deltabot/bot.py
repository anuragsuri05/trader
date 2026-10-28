"""The autonomous trading loop."""
from __future__ import annotations

import logging
import time
import traceback

from .broker import LiveBroker, PaperBroker
from .client import RESOLUTION_SECONDS, DeltaClient
from .config import Settings
from .data import candles_df, load_products
from .notify import Notifier
from .options import choose_future, choose_option, option_size
from .risk import RiskManager, position_size
from .scanner import Candidate, scan
from .state import State
from .strategy import Signal, entry_signal, should_exit

log = logging.getLogger(__name__)


class Bot:
    def __init__(self, s: Settings):
        self.s = s
        self.client = DeltaClient(s.base_url, s.api_key, s.api_secret)
        self.notify = Notifier(s)
        self.products = load_products(self.client)
        self.state = State.load(s.mode)
        broker_cls = LiveBroker if s.is_live else PaperBroker
        self.broker = broker_cls(self.client, s, self.state, self.products)
        self.risk = RiskManager(s, self.state)
        self.candidates: list[Candidate] = []
        kinds = {}
        for p in self.products.values():
            kinds[p.kind] = kinds.get(p.kind, 0) + 1
        log.info("loaded products %s from %s (mode=%s, instruments=%s)",
                 kinds, s.base_url, s.mode, ",".join(s.instruments))

    def _reload_products(self) -> None:
        """New option/futures expiries are listed daily - refresh, but keep anything we hold."""
        fresh = load_products(self.client)
        for sym in self.state.trades:
            if sym not in fresh and sym in self.products:
                fresh[sym] = self.products[sym]
        self.products.clear()
        self.products.update(fresh)

    # ------------------------------------------------------------------ exits
    def _exit_reason(self, sym: str, t) -> str | None:
        product = self.products[sym]
        if t.kind == "option":
            bid = self.broker.quote(sym)["bid"]
            if product.hours_to_expiry() < self.s.opt_exit_before_expiry_h:
                return "near_expiry"
            if bid <= t.stop:
                return "premium_stop"
            if bid >= t.target:
                return "premium_target"
        if t.kind == "futures" and product.hours_to_expiry() < self.s.fut_exit_before_expiry_h:
            return "expiry"
        tf = t.entry_tf or self.s.entry_tf
        df = candles_df(self.client, t.signal_symbol or sym, tf, self.s.ema_slow + 50)
        if should_exit(df, t.view or t.side, self.s):
            return "trend_break"
        # max hold scales with timeframe: 240h on 1h candles, 60h on 15m, 20h on 5m
        if time.time() - t.entry_time > self.s.max_hold_hours * RESOLUTION_SECONDS[tf]:
            return "time_stop"
        return None

    def _manage_open_trades(self) -> None:
        for sym, t in list(self.state.trades.items()):
            if sym not in self.products:
                log.warning("%s no longer listed; leaving it to check_brackets", sym)
                continue
            reason = self._exit_reason(sym, t)
            if reason:
                pnl = self.broker.close(self.products[sym], reason)
                self.notify(f"Closed {t.kind} {t.side.upper()} {sym} ({reason}), pnl {pnl:+.2f}")

    # ------------------------------------------------------------------ entries
    def _rescan_if_due(self) -> None:
        if time.time() - self.state.last_scan < self.rescan_seconds and self.candidates:
            return
        self._reload_products()
        self.candidates = scan(self.client, self.products, self.s)
        self.state.last_scan = time.time()
        if self.candidates:
            log.info("trending markets:\n  %s", "\n  ".join(str(c) for c in self.candidates[:10]))
        else:
            log.info("no trending markets right now - staying flat")

    def _try_option(self, c: Candidate, sig: Signal, equity: float) -> bool:
        q = choose_option(self.client, self.products, c.underlying, c.direction, self.s)
        if not q:
            return False
        size = option_size(q.ask, q.product.contract_value, equity, self.s)
        if size < 1:
            log.info("%s: option premium too large for account size", q.product.symbol)
            return False
        fill, filled = self.broker.open_option(q, size, sig.side, c.symbol)
        if filled < 1:
            return False
        cost = fill * filled * q.product.contract_value
        self.notify(f"Bought {filled} x {q.product.symbol} @ {fill:g} (premium ${cost:.2f}, delta {q.delta:+.2f}, "
                    f"IV {q.iv:.0%}) on {c.symbol} {'uptrend' if c.direction > 0 else 'downtrend'}")
        return True

    def _try_linear(self, product, c: Candidate, sig: Signal, equity: float, open_notional: float) -> float:
        """Perpetual or dated future. Returns notional opened (0 if nothing)."""
        if product.symbol != c.symbol:
            # dated future: shift stop/target by the same distances from the future's own price
            px = self.broker.quote(product.symbol)["mark"]
            sig = Signal(sig.side, px, px - (sig.price - sig.stop), px + (sig.target - sig.price), sig.atr)
        size = position_size(product, sig, equity, open_notional, self.s)
        if size < 1:
            log.info("%s signal but size < 1 contract (equity too small or exposure cap)", product.symbol)
            return 0.0
        fill = self.broker.open(product, sig, size, signal_symbol=c.symbol)
        self.notify(f"Opened {product.kind} {sig.side.upper()} {size} x {product.symbol} @ {fill:.6g} | "
                    f"SL {sig.stop:.6g} TP {sig.target:.6g} | trend score {c.score:.2f}")
        return product.notional(size, fill)

    def _open_new_trades(self, equity: float) -> None:
        trades = self.state.trades
        if len(trades) >= self.s.max_positions:
            return
        open_notional = sum(self.products[k].notional(t.size, t.entry_price)
                            for k, t in trades.items() if t.kind != "option" and k in self.products)
        busy = {t.signal_symbol or k for k, t in trades.items()}   # one trade per market
        for c in self.candidates:
            if len(trades) >= self.s.max_positions:
                break
            if c.symbol in busy:
                continue
            df = candles_df(self.client, c.symbol, c.entry_tf or self.s.entry_tf,
                            max(self.s.ema_slow, self.s.donchian) + 60)
            sig = entry_signal(df, c.direction, self.s)
            if not sig:
                continue
            for inst in self.s.instruments:
                if inst == "options":
                    if self._try_option(c, sig, equity):
                        break
                elif inst == "futures":
                    fut = choose_future(self.products, c.underlying, self.s)
                    if fut and (n := self._try_linear(fut, c, sig, equity, open_notional)):
                        open_notional += n
                        break
                elif inst == "perpetual":
                    if n := self._try_linear(self.products[c.symbol], c, sig, equity, open_notional):
                        open_notional += n
                        break
            for tr in trades.values():                 # remember which timeframe manages it
                if tr.signal_symbol == c.symbol and not tr.entry_tf:
                    tr.entry_tf = c.entry_tf or self.s.entry_tf
            busy.add(c.symbol)
            self.state.save()

    # ------------------------------------------------------------------ cycle
    def step(self) -> None:
        for msg in self.broker.check_brackets():
            self.notify(msg)
        equity = self.broker.equity()
        for event in self.risk.update(equity):
            self.notify(event)
        self._check_session(equity)
        if self.state.killed and self.state.trades:
            self.flatten("kill_switch")
        self._manage_open_trades()
        if self.risk.can_open:
            self._rescan_if_due()
            self._open_new_trades(equity)
        log.info("equity=%.2f open=%s peak=%.2f halted=%s", equity,
                 [f"{t.kind}:{k}" for k, t in self.state.trades.items()] or "-",
                 self.state.peak_equity, self.state.halted_reason or "no")
        self.state.save()

    def _check_session(self, equity: float) -> None:
        st, s = self.state, self.s
        if st.halted_reason.startswith(("target:", "session:")):
            return
        if s.target_equity and equity >= s.target_equity:
            self.flatten("target_reached")
            st.halted_reason = f"target: reached {equity:.2f} >= {s.target_equity:.2f}, all positions closed"
            self.notify("TARGET REACHED - closed everything and stopped trading")
        elif s.session_end and time.time() >= s.session_end:
            self.flatten("session_end")
            st.halted_reason = "session: time is up, all positions closed"
            self.notify(f"Session ended. Final equity {self.broker.equity():.2f}")

    @property
    def rescan_seconds(self) -> float:
        fastest = min(RESOLUTION_SECONDS[e] for _, e in self.s.tf_pairs())
        return min(self.s.rescan_minutes * 60, 3 * fastest)

    def flatten(self, reason: str = "manual_flatten") -> None:
        for sym in list(self.state.trades):
            pnl = self.broker.close(self.products[sym], reason)
            self.notify(f"Flattened {sym} ({reason}), pnl {pnl:+.2f}")
        self.state.save()

    # ------------------------------------------------------------------- loop
    def run(self, stop=None) -> None:
        """Loop forever (or until `stop`, a threading.Event, is set)."""
        tfs = "auto (" + ", ".join(f"{r}/{e}" for r, e in self.s.tf_pairs()) + ")" \
            if self.s.timeframe_mode == "auto" else f"{self.s.regime_tf}/{self.s.entry_tf}"
        self.notify(f"Bot started. Equity {self.broker.equity():.2f}. Instruments {','.join(self.s.instruments)}. "
                    f"Timeframes {tfs}.")
        step_s = min(RESOLUTION_SECONDS[e] for _, e in self.s.tf_pairs())
        errors = 0
        while not (stop and stop.is_set()):
            try:
                self.step()
                errors = 0
            except KeyboardInterrupt:
                raise
            except Exception as e:
                errors += 1
                log.error("step failed: %s\n%s", e, traceback.format_exc())
                if errors in (1, 5) or errors % 20 == 0:
                    self.notify(f"Error ({errors} in a row): {e}")
            # wake shortly after the next candle closes
            sleep = step_s - (time.time() % step_s) + 10
            if self.s.session_end > time.time():
                sleep = max(1, min(sleep, self.s.session_end - time.time() + 2))
            log.info("sleeping %.0fs until next candle close", sleep)
            if stop:
                stop.wait(sleep)
            else:
                time.sleep(sleep)
