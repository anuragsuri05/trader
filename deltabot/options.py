"""Options + dated futures support.

The bot only BUYS options (calls in uptrends, puts in downtrends), so the worst case per
trade is the premium paid - no unlimited-loss option selling.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from statistics import NormalDist

from .client import DeltaClient
from .config import Settings
from .data import Product

log = logging.getLogger(__name__)
N = NormalDist()


# ---------------------------------------------------------------- Black-Scholes (r = 0)
def bs_price(spot: float, strike: float, years: float, vol: float, call: bool) -> float:
    if years <= 0 or vol <= 0:
        return max(0.0, spot - strike) if call else max(0.0, strike - spot)
    sd = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if call:
        return spot * N.cdf(d1) - strike * N.cdf(d2)
    return strike * N.cdf(-d2) - spot * N.cdf(-d1)


def bs_delta(spot: float, strike: float, years: float, vol: float, call: bool) -> float:
    if years <= 0 or vol <= 0:
        return (1.0 if spot > strike else 0.0) if call else (-1.0 if spot < strike else 0.0)
    sd = vol * math.sqrt(years)
    d1 = (math.log(spot / strike) + 0.5 * sd * sd) / sd
    return N.cdf(d1) if call else N.cdf(d1) - 1


def strike_for_delta(spot: float, years: float, vol: float, abs_delta: float, call: bool) -> float:
    """Inverse of bs_delta: the strike whose |delta| equals abs_delta."""
    sd = vol * math.sqrt(years)
    d1 = N.inv_cdf(abs_delta) if call else N.inv_cdf(1 - abs_delta)
    return spot * math.exp(-d1 * sd + 0.5 * sd * sd)


# ---------------------------------------------------------------- live option chain
@dataclass
class OptionQuote:
    product: Product
    bid: float
    ask: float
    mark: float
    delta: float
    iv: float
    spot: float

    @property
    def spread(self) -> float:
        return (self.ask - self.bid) / self.ask if self.ask > 0 else 1.0

    def __str__(self) -> str:
        dte = self.product.hours_to_expiry() / 24
        return (f"{self.product.symbol:<22} bid={self.bid:<9g} ask={self.ask:<9g} delta={self.delta:+.2f} "
                f"iv={self.iv:.0%} dte={dte:.1f}d spread={self.spread:.0%}")


def option_chain(client: DeltaClient, products: dict[str, Product], underlying: str, call: bool) -> list[OptionQuote]:
    ctype = "call_options" if call else "put_options"
    out = []
    for t in client.tickers(ctype, underlying=underlying):
        p = products.get(t.get("symbol"))
        if not p or p.kind != "option":
            continue
        q, g = t.get("quotes") or {}, t.get("greeks") or {}
        try:
            out.append(OptionQuote(
                product=p,
                bid=float(q.get("best_bid") or 0),
                ask=float(q.get("best_ask") or 0),
                mark=float(t.get("mark_price") or 0),
                delta=float(g.get("delta") or 0),
                iv=float(q.get("mark_iv") or t.get("mark_vol") or 0),
                spot=float(t.get("spot_price") or g.get("spot") or 0),
            ))
        except (TypeError, ValueError):
            continue
    return out


def choose_option(client: DeltaClient, products: dict[str, Product], underlying: str,
                  direction: int, s: Settings) -> OptionQuote | None:
    """Pick the call (direction>0) or put (direction<0) closest to the target expiry and delta."""
    chain = option_chain(client, products, underlying, call=direction > 0)
    now = time.time()
    valid = [q for q in chain
             if s.opt_min_dte * 24 <= q.product.hours_to_expiry(now) <= s.opt_max_dte * 24
             and q.ask > 0 and q.bid > s.opt_min_bid and q.spread <= s.opt_max_spread and q.delta != 0]
    if not valid:
        log.info("no liquid %s option on %s within %g-%g days", "call" if direction > 0 else "put",
                 underlying, s.opt_min_dte, s.opt_max_dte)
        return None
    # 1) expiry closest to target, 2) strike with |delta| closest to target
    best_expiry = min({q.product.expiry for q in valid},
                      key=lambda e: abs((e - now) / 86400 - s.opt_target_dte))
    same = [q for q in valid if q.product.expiry == best_expiry]
    return min(same, key=lambda q: abs(abs(q.delta) - s.opt_target_delta))


def option_fee(premium: float, spot: float, contracts: int, contract_value: float, s: Settings) -> float:
    """Exchange-style option fee: % of underlying notional, capped at a % of premium. [verify on Delta's fee page]"""
    by_notional = s.opt_fee_rate * spot * contract_value * contracts
    cap = s.opt_fee_cap * premium * contract_value * contracts
    return min(by_notional, cap)


def option_size(premium: float, contract_value: float, equity: float, s: Settings) -> int:
    """Contracts so that (a) the planned stop-loss costs ~risk_per_trade of equity and
    (b) total premium never exceeds opt_max_premium_pct of equity (the worst case: expires worthless)."""
    cost = premium * contract_value
    if cost <= 0 or equity <= 0:
        return 0
    by_risk = equity * s.risk_per_trade / (cost * s.opt_stop_loss)
    by_premium = equity * s.opt_max_premium_pct / cost
    return max(0, math.floor(min(by_risk, by_premium)))


# ---------------------------------------------------------------- dated futures
def choose_future(products: dict[str, Product], underlying: str, s: Settings) -> Product | None:
    futs = [p for p in products.values()
            if p.kind == "futures" and p.underlying == underlying and p.hours_to_expiry() >= s.fut_min_dte * 24]
    return min(futs, key=lambda p: p.expiry) if futs else None
