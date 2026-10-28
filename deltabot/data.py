"""Market data helpers: candles as DataFrames and product metadata."""
from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from .client import RESOLUTION_SECONDS, DeltaClient

MAX_CANDLES_PER_CALL = 2000

KIND_BY_CONTRACT = {
    "perpetual_futures": "perpetual",
    "futures": "futures",
    "call_options": "option",
    "put_options": "option",
}


@dataclass
class Product:
    id: int
    symbol: str
    contract_value: float   # underlying units per contract (e.g. 0.001 BTC)
    tick_size: float
    taker_fee: float
    max_leverage: float
    kind: str = "perpetual"           # perpetual | futures | option
    underlying: str = ""              # e.g. BTC
    expiry: float = 0.0               # unix seconds, 0 for perpetuals
    strike: float = 0.0               # options only
    option_type: str = ""             # call | put
    tradfi: bool = False              # gold/silver/stock tokens (Delta tag "tradfi")

    def notional(self, contracts: float, price: float) -> float:
        return abs(contracts) * self.contract_value * price

    def hours_to_expiry(self, now: float | None = None) -> float:
        if not self.expiry:
            return float("inf")
        return (self.expiry - (now or time.time())) / 3600


def _parse_time(value) -> float:
    if not value:
        return 0.0
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()


def load_products(client: DeltaClient) -> dict[str, Product]:
    out = {}
    for p in client.products():
        kind = KIND_BY_CONTRACT.get(p.get("contract_type"))
        if not kind or p.get("state") != "live":
            continue
        # Only linear (vanilla) contracts: notional = contracts * contract_value * price.
        if p.get("notional_type", "vanilla") != "vanilla":
            continue
        if p.get("trading_status", "operational") != "operational":
            continue
        try:
            im = float(p.get("initial_margin") or 0)
            out[p["symbol"]] = Product(
                id=int(p["id"]),
                symbol=p["symbol"],
                contract_value=float(p["contract_value"]),
                tick_size=float(p["tick_size"]),
                taker_fee=float(p.get("taker_commission_rate") or 0.0005),
                max_leverage=(100 / im) if im > 0 else float(p.get("default_leverage") or 10),
                kind=kind,
                underlying=((p.get("underlying_asset") or {}).get("symbol") or ""),
                expiry=_parse_time(p.get("settlement_time")) if kind != "perpetual" else 0.0,
                strike=float(p.get("strike_price") or 0),
                option_type={"call_options": "call", "put_options": "put"}.get(p["contract_type"], ""),
                tradfi=(p.get("product_specs") or {}).get("top_tag") == "tradfi",
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def perps(products: dict[str, Product]) -> dict[str, Product]:
    return {k: v for k, v in products.items() if v.kind == "perpetual"}


def candles_df(client: DeltaClient, symbol: str, resolution: str, bars: int,
               end: int | None = None, closed_only: bool = True) -> pd.DataFrame:
    """Fetch the last `bars` candles (paginating backwards). Index = bar open time (UTC)."""
    step = RESOLUTION_SECONDS[resolution]
    end = int(end or time.time())
    rows: list[dict] = []
    remaining = bars + 1
    cursor = end
    while remaining > 0:
        n = min(remaining, MAX_CANDLES_PER_CALL)
        start = cursor - n * step
        chunk = client.candles(symbol, resolution, start, cursor)
        if not chunk:
            break
        rows.extend(chunk)
        remaining -= n
        cursor = start
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    df = pd.DataFrame(rows).drop_duplicates("time")
    df["time"] = pd.to_datetime(df["time"].astype(int), unit="s", utc=True)
    df = df.set_index("time").sort_index()[["open", "high", "low", "close", "volume"]].astype(float)
    if closed_only:
        # drop the still-forming bar
        now = pd.Timestamp(time.time(), unit="s", tz="UTC")
        df = df[df.index + pd.Timedelta(seconds=step) <= now]
    return df.iloc[-bars:]


def unix_seconds(index) -> list[int]:
    """Bar times as unix seconds, whatever datetime unit pandas uses (ns in pandas 2, s in pandas 3)."""
    return pd.DatetimeIndex(index).as_unit("s").asi8.tolist()


def round_to_tick(price: float, tick: float) -> float:
    return round(round(price / tick) * tick, 10)
