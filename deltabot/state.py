"""Persistent bot state (JSON) and trade journal (CSV), so restarts are safe."""
from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ROOT

STATE_DIR = Path(os.environ.get("DELTABOT_STATE_DIR") or ROOT / "state")   # separate folder = separate dashboard/agent


@dataclass
class TradeMeta:
    symbol: str
    side: str            # buy | sell
    size: int            # contracts
    entry_price: float
    stop: float
    target: float
    entry_time: float    # unix seconds
    kind: str = "perpetual"   # perpetual | futures | option
    signal_symbol: str = ""   # perpetual whose trend drove this trade
    view: str = ""            # buy (bullish) | sell (bearish) - for options the order side is always buy
    expiry: float = 0.0       # unix seconds, 0 = none
    entry_tf: str = ""        # timeframe the trade is managed on
    agent: str = ""           # which arena bot opened it
    hold_until: float = 0.0   # AI duel: auto-close after this time (0 = no limit)


@dataclass
class State:
    mode: str = "paper"
    day: str = ""
    day_start_equity: float = 0.0
    peak_equity: float = 0.0
    halted_reason: str = ""
    killed: bool = False
    trades: dict[str, TradeMeta] = field(default_factory=dict)   # open trades by symbol
    paper_cash: float = 0.0
    last_scan: float = 0.0
    extra: dict = field(default_factory=dict)   # engine-specific state (e.g. arena bots)

    @classmethod
    def load(cls, mode: str) -> "State":
        path = STATE_DIR / f"state_{mode}.json"
        if not path.exists():
            return cls(mode=mode)
        raw = json.loads(path.read_text())
        raw["trades"] = {k: TradeMeta(**v) for k, v in raw.get("trades", {}).items()}
        return cls(**raw)

    def save(self) -> None:
        STATE_DIR.mkdir(exist_ok=True)
        path = STATE_DIR / f"state_{self.mode}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        os.replace(tmp, path)   # atomic write


JOURNAL_FIELDS = ["exit_time", "mode", "symbol", "side", "size", "entry_price", "exit_price",
                  "pnl", "fees", "reason", "entry_time", "agent"]


def journal(mode: str, row: dict) -> None:
    STATE_DIR.mkdir(exist_ok=True)
    path: Path = STATE_DIR / f"trades_{mode}.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=JOURNAL_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in JOURNAL_FIELDS})
