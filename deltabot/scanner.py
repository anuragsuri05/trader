"""Picks which perpetual markets are worth trading right now."""
from __future__ import annotations

import logging
from dataclasses import dataclass

from .client import DeltaClient
from .config import Settings
from .data import Product, candles_df, perps
from .strategy import regime_frame

log = logging.getLogger(__name__)


@dataclass
class Candidate:
    symbol: str          # perpetual used for the signal
    direction: int       # +1 long bias, -1 short bias
    score: float
    adx: float
    er: float
    atr_pct: float
    turnover_usd: float
    underlying: str = ""
    regime_tf: str = ""
    entry_tf: str = ""
    agent: str = ""          # arena bot that found this idea

    def __str__(self) -> str:
        side = "LONG " if self.direction > 0 else "SHORT"
        return (f"{self.symbol:<14} {side} {self.regime_tf}/{self.entry_tf:<4} score={self.score:.2f} adx={self.adx:.1f} "
                f"er={self.er:.2f} atr%={self.atr_pct * 100:.2f} vol24h=${self.turnover_usd / 1e6:,.1f}M")


def liquid_universe(client: DeltaClient, products: dict[str, Product], s: Settings) -> list[tuple[str, float]]:
    products = perps(products)
    rows = []
    for t in client.tickers("perpetual_futures"):
        sym = t.get("symbol")
        if sym not in products or sym in s.blacklist or (s.crypto_only and products[sym].tradfi):
            continue
        turnover = float(t.get("turnover_usd") or 0)
        if turnover >= s.min_turnover_usd:
            rows.append((sym, turnover))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows[: s.universe_size]


def scan(client: DeltaClient, products: dict[str, Product], s: Settings) -> list[Candidate]:
    """Return trending markets, best first. In auto mode each market gets the timeframe
    pair (e.g. 15m/5m, 1h/15m, 4h/1h) where its trend is cleanest."""
    out = []
    for sym, turnover in liquid_universe(client, products, s):
        best = None
        for rtf, etf in s.tf_pairs():
            try:
                df = candles_df(client, sym, rtf, s.regime_ema_slow + 100)
            except Exception as e:  # one bad symbol shouldn't kill the scan
                log.warning("scan: %s %s candles failed: %s", sym, rtf, e)
                continue
            if len(df) < s.regime_ema_slow + 20:
                continue
            r = regime_frame(df, s).iloc[-1]
            if r["direction"] == 0 or (best and r["score"] <= best.score):
                continue
            best = Candidate(sym, int(r["direction"]), float(r["score"]), float(r["adx"]),
                             float(r["er"]), float(r["atr_pct"]), turnover, products[sym].underlying, rtf, etf)
        if best:
            out.append(best)
    out.sort(key=lambda c: c.score, reverse=True)
    return out
