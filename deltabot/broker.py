"""Order execution: a PaperBroker (simulated, real prices) and a LiveBroker (Delta Exchange).

Perpetuals / dated futures: market order + exchange-side bracket (stop-loss + take-profit).
Options (long calls/puts only): marketable IOC limit order at the ask; exits at the bid are
managed by the bot each cycle (option books are too thin for blind stop orders).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .client import DeltaClient
from .config import Settings
from .data import Product, candles_df, round_to_tick
from .options import OptionQuote, option_fee
from .state import State, TradeMeta, journal
from .strategy import Signal

log = logging.getLogger(__name__)


@dataclass
class Position:
    symbol: str
    side: str          # buy | sell
    size: int          # contracts (positive)
    entry_price: float


def _ts(t: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def _journal(mode: str, t: TradeMeta, exit_px: float | str, pnl: float | str, fees: float | str, reason: str):
    journal(mode, dict(exit_time=_ts(time.time()), mode=mode, symbol=t.symbol, side=t.side, size=t.size,
                       entry_price=t.entry_price, exit_price=exit_px,
                       pnl=round(pnl, 4) if isinstance(pnl, float) else pnl,
                       fees=round(fees, 4) if isinstance(fees, float) else fees,
                       reason=f"{t.kind}:{reason}", entry_time=_ts(t.entry_time), agent=t.agent))


class BaseBroker:
    def __init__(self, client: DeltaClient, s: Settings, state: State, products: dict[str, Product]):
        self.client, self.s, self.state, self.products = client, s, state, products

    marks: dict = {}
    marks_time: float = 0.0

    def set_marks(self, marks: dict) -> None:
        self.marks, self.marks_time = marks, time.time()

    def quote(self, symbol: str) -> dict:
        """mark / bid / ask / spot for any instrument (uses the 5-second price cache when fresh)."""
        if symbol in self.marks and time.time() - self.marks_time < 8:
            m = self.marks[symbol]
            return dict(mark=m, bid=m, ask=m, spot=m)
        t = self.client.ticker(symbol)
        q = t.get("quotes") or {}
        mark = float(t.get("mark_price") or t.get("close") or 0)
        return dict(mark=mark,
                    bid=float(q.get("best_bid") or 0) or mark,
                    ask=float(q.get("best_ask") or 0) or mark,
                    spot=float(t.get("spot_price") or 0) or mark)

    def _record(self, product: Product, side: str, size: int, fill: float, stop: float, target: float,
                signal_symbol: str, view: str) -> None:
        self.state.trades[product.symbol] = TradeMeta(
            product.symbol, side, size, fill, stop, target, time.time(),
            kind=product.kind, signal_symbol=signal_symbol, view=view, expiry=product.expiry)


class PaperBroker(BaseBroker):
    """Simulated fills with fees + slippage on real prices.
    Perp/futures SL/TP are checked against real 1m candle highs/lows."""

    def __init__(self, client, s, state, products):
        super().__init__(client, s, state, products)
        if not state.paper_cash:
            state.paper_cash = s.paper_start_equity

    def positions(self) -> dict[str, Position]:
        return {k: Position(k, t.side, t.size, t.entry_price) for k, t in self.state.trades.items()}

    def equity(self) -> float:
        eq = self.state.paper_cash
        for sym, t in self.state.trades.items():
            p = self.products.get(sym)
            if not p:
                continue
            mark = self.quote(sym)["mark"]
            if t.kind == "option":
                eq += mark * t.size * p.contract_value          # premium was paid out of cash
            else:
                sign = 1 if t.side == "buy" else -1
                eq += sign * (mark - t.entry_price) * t.size * p.contract_value
        return eq

    # ---- perpetual / futures
    def _fee(self, maker: bool) -> float:
        return (self.s.maker_fee if maker else self.s.fee_rate) * (1 + self.s.fee_tax)

    def open(self, product: Product, sig: Signal, size: int, signal_symbol: str = "", maker: bool = False) -> float:
        slip = 0.0 if maker else sig.price * self.s.slippage
        fill = sig.price + slip if sig.side == "buy" else sig.price - slip
        self.state.paper_cash -= fill * size * product.contract_value * self._fee(maker)
        self._record(product, sig.side, size, fill, sig.stop, sig.target, signal_symbol or product.symbol, sig.side)
        return fill

    # ---- resting limit (maker) entry orders
    def place_limit(self, product: Product, sig: Signal, size: int, agent: str, entry_tf: str, expire_s: float) -> None:
        self.state.extra.setdefault("orders", {})[product.symbol] = dict(
            side=sig.side, size=size, limit=sig.price, stop=sig.stop, target=sig.target, agent=agent,
            entry_tf=entry_tf, placed=time.time(), expire=time.time() + expire_s)

    def check_orders(self) -> list[str]:
        """Fill a resting limit order only if price trades THROUGH the limit (queue-safe)."""
        msgs, orders = [], self.state.extra.get("orders", {})
        for sym, od in list(orders.items()):
            product = self.products.get(sym)
            if not product or sym in self.state.trades:
                orders.pop(sym)
                continue
            df = candles_df(self.client, sym, "1m", 120, closed_only=True)
            df = df[[ts.timestamp() >= od["placed"] for ts in df.index]]   # only bars after placement
            long = od["side"] == "buy"
            for ts, bar in df.iterrows():
                if (bar["low"] < od["limit"]) if long else (bar["high"] > od["limit"]):
                    px = min(od["limit"], bar["open"]) if long else max(od["limit"], bar["open"])
                    sig = Signal(od["side"], px, od["stop"], od["target"], 0)
                    self.open(product, sig, od["size"], signal_symbol=sym, maker=True)
                    t = self.state.trades[sym]
                    t.agent, t.entry_tf, t.entry_time = od["agent"], od["entry_tf"], ts.timestamp()
                    orders.pop(sym)
                    msgs.append(f"limit filled: {od['side'].upper()} {od['size']} x {sym} @ {px:.6g}|{od['agent']}")
                    break
            else:
                if time.time() > od["expire"]:
                    orders.pop(sym)
                    msgs.append(f"limit expired unfilled: {sym}|{od['agent']}")
        return msgs

    # ---- options
    def open_option(self, q: OptionQuote, size: int, view: str, signal_symbol: str) -> tuple[float, int]:
        p = q.product
        fee = option_fee(q.ask, q.spot, size, p.contract_value, self.s)
        self.state.paper_cash -= q.ask * size * p.contract_value + fee
        self._record(p, "buy", size, q.ask, q.ask * (1 - self.s.opt_stop_loss),
                     q.ask * (1 + self.s.opt_take_profit), signal_symbol, view)
        return q.ask, size

    def close(self, product: Product, reason: str, price: float | None = None, maker: bool = False) -> float:
        t = self.state.trades.pop(product.symbol)
        cv = product.contract_value
        if t.kind == "option":
            q = self.quote(product.symbol)
            if product.hours_to_expiry() <= 0:             # settled: intrinsic value
                k = product.strike
                px = max(0.0, q["spot"] - k) if product.option_type == "call" else max(0.0, k - q["spot"])
            else:
                px = price if price is not None else q["bid"]
            fee = option_fee(px, q["spot"], t.size, cv, self.s) if px > 0 else 0.0
            self.state.paper_cash += px * t.size * cv - fee
            pnl = (px - t.entry_price) * t.size * cv - fee
        else:
            px = price if price is not None else self.quote(product.symbol)["mark"]
            slip = 0.0 if maker else px * self.s.slippage
            px = px - slip if t.side == "buy" else px + slip
            sign = 1 if t.side == "buy" else -1
            fee = px * t.size * cv * self._fee(maker)
            pnl = sign * (px - t.entry_price) * t.size * cv - fee
            self.state.paper_cash += pnl
        _journal(self.state.mode, t, px, pnl, fee, reason)
        return pnl

    def check_brackets(self) -> list[str]:
        """Simulate the exchange-side stop-loss / take-profit of perp/futures positions."""
        msgs = []
        for sym, t in list(self.state.trades.items()):
            if t.kind == "option":
                continue                      # options exits are handled by the bot loop
            df = candles_df(self.client, sym, "1m", 240, closed_only=True)
            df = df[df.index.map(lambda x: x.timestamp()) >= t.entry_time]
            for _, bar in df.iterrows():
                hit_sl = bar["low"] <= t.stop if t.side == "buy" else bar["high"] >= t.stop
                hit_tp = bar["high"] >= t.target if t.side == "buy" else bar["low"] <= t.target
                if hit_sl:   # worst case first if both inside one bar
                    pnl = self.close(self.products[sym], "stop_loss", t.stop)
                    msgs.append(f"[paper] {sym} stop-loss hit, pnl {pnl:+.2f}")
                    break
                if hit_tp:   # take-profit is a resting limit order -> maker fee, no slippage
                    pnl = self.close(self.products[sym], "take_profit", t.target, maker=True)
                    msgs.append(f"[paper] {sym} take-profit hit, pnl {pnl:+.2f}")
                    break
        return msgs


class LiveBroker(BaseBroker):
    """Real orders on Delta Exchange. Perp/futures stop-loss and take-profit live ON THE EXCHANGE
    as a bracket, so they still protect you if this bot crashes or loses internet."""

    def __init__(self, client, s, state, products):
        super().__init__(client, s, state, products)
        self._leverage_set: set[int] = set()

    def equity(self) -> float:
        data = self.client.wallet_balances()
        meta = data.get("meta") or {}
        if meta.get("net_equity") is not None:
            return float(meta["net_equity"])
        for w in data["result"]:
            if w.get("asset_symbol") == self.s.settle_asset:
                return float(w.get("balance") or 0)
        raise RuntimeError(f"no {self.s.settle_asset} wallet found; set SETTLE_ASSET in .env")

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self.client.positions():
            size = int(float(p.get("size") or 0))
            sym = p.get("product_symbol") or (p.get("product") or {}).get("symbol")
            if size == 0 or not sym:
                continue
            out[sym] = Position(sym, "buy" if size > 0 else "sell", abs(size), float(p.get("entry_price") or 0))
        return out

    @staticmethod
    def _filled(res: dict, requested: int) -> int:
        unfilled = res.get("unfilled_size")
        return requested - int(float(unfilled)) if unfilled is not None else requested

    def _cid(self, prefix: str, product: Product) -> str:
        return f"{prefix}{int(time.time())}{product.id}"[:32]

    # ---- perpetual / futures
    def open(self, product: Product, sig: Signal, size: int, signal_symbol: str = "") -> float:
        if product.id not in self._leverage_set:
            self.client.set_leverage(product.id, int(min(self.s.leverage, product.max_leverage)))
            self._leverage_set.add(product.id)
        body = {
            "product_id": product.id, "size": size, "side": sig.side, "order_type": "market_order",
            "client_order_id": self._cid("db", product),
            "bracket_stop_loss_price": str(round_to_tick(sig.stop, product.tick_size)),
            "bracket_take_profit_price": str(round_to_tick(sig.target, product.tick_size)),
            "bracket_stop_trigger_method": "mark_price",
        }
        log.info("placing order %s", body)
        res = self.client.place_order(body)
        fill = float(res.get("average_fill_price") or sig.price)
        self._record(product, sig.side, size, fill, sig.stop, sig.target, signal_symbol or product.symbol, sig.side)
        return fill

    # ---- options
    def open_option(self, q: OptionQuote, size: int, view: str, signal_symbol: str) -> tuple[float, int]:
        p = q.product
        body = {
            "product_id": p.id, "size": size, "side": "buy", "order_type": "limit_order",
            "limit_price": str(round_to_tick(q.ask, p.tick_size)), "time_in_force": "ioc",
            "client_order_id": self._cid("do", p),
        }
        log.info("placing option order %s", body)
        res = self.client.place_order(body)
        filled = self._filled(res, size)
        if filled <= 0:
            log.info("option order for %s not filled (book moved)", p.symbol)
            return 0.0, 0
        fill = float(res.get("average_fill_price") or q.ask)
        self._record(p, "buy", filled, fill, fill * (1 - self.s.opt_stop_loss),
                     fill * (1 + self.s.opt_take_profit), signal_symbol, view)
        return fill, filled

    def close(self, product: Product, reason: str, price: float | None = None) -> float:
        t = self.state.trades.get(product.symbol)
        pos = self.positions().get(product.symbol)
        if t and t.kind == "option":
            return self._close_option(product, t, pos, reason)
        self.client.cancel_all(product.id)          # remove bracket / pending orders
        exit_px = 0.0
        if pos:
            res = self.client.place_order({
                "product_id": product.id, "size": pos.size,
                "side": "sell" if pos.side == "buy" else "buy",
                "order_type": "market_order", "reduce_only": True,
                "client_order_id": self._cid("dx", product),
            })
            exit_px = float(res.get("average_fill_price") or 0)
        t = self.state.trades.pop(product.symbol, None)
        if not t:
            return 0.0
        sign = 1 if t.side == "buy" else -1
        pnl = sign * (exit_px - t.entry_price) * t.size * product.contract_value if exit_px else 0.0
        _journal("live", t, exit_px, pnl, "", reason)
        return pnl

    def _close_option(self, product: Product, t: TradeMeta, pos: Position | None, reason: str) -> float:
        if not pos:                                  # already gone (expired / closed manually)
            self.state.trades.pop(product.symbol, None)
            _journal("live", t, "", "", "", reason + "_already_closed")
            return 0.0
        bid = self.quote(product.symbol)["bid"]
        res = self.client.place_order({
            "product_id": product.id, "size": pos.size, "side": "sell", "order_type": "limit_order",
            "limit_price": str(round_to_tick(max(bid, product.tick_size), product.tick_size)),
            "time_in_force": "ioc", "reduce_only": True,
            "client_order_id": self._cid("dx", product),
        })
        filled = self._filled(res, pos.size)
        px = float(res.get("average_fill_price") or bid)
        pnl = (px - t.entry_price) * filled * product.contract_value
        if filled < pos.size:                        # thin book: keep the rest, retry next cycle
            t.size = pos.size - filled
            log.warning("%s only %d/%d sold at bid, will retry", product.symbol, filled, pos.size)
            return pnl
        self.state.trades.pop(product.symbol, None)
        _journal("live", t, px, pnl, "", reason)
        return pnl

    def check_brackets(self) -> list[str]:
        """Exchange handles SL/TP (and option settlement). Notice trades that are gone."""
        msgs = []
        live = self.positions()
        for sym in list(self.state.trades):
            if sym not in live:
                t = self.state.trades.pop(sym)
                if sym in self.products and t.kind != "option":
                    self.client.cancel_all(self.products[sym].id)   # leftover bracket leg
                _journal("live", t, "", "", "", "closed_on_exchange")
                msgs.append(f"{sym} position closed on exchange (SL/TP/expiry/manual)")
        return msgs
