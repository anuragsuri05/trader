"""Position sizing and account-level safety limits."""
from __future__ import annotations

import math
from datetime import datetime, timezone

from .config import Settings
from .data import Product
from .state import State
from .strategy import Signal


def position_size(product: Product, sig: Signal, equity: float, open_notional: float, s: Settings) -> int:
    """Contracts such that hitting the stop loses ~risk_per_trade of equity, within exposure caps."""
    if equity <= 0 or sig.price <= 0:
        return 0
    loss_per_contract = abs(sig.price - sig.stop) * product.contract_value
    loss_per_contract += 2 * (s.fee_rate + s.slippage) * sig.price * product.contract_value
    if loss_per_contract <= 0:
        return 0
    by_risk = (equity * s.risk_per_trade) / loss_per_contract

    notional_per_contract = product.contract_value * sig.price
    room = s.max_total_notional_x * equity - open_notional
    by_exposure = room / notional_per_contract if room > 0 else 0
    # keep required margin for this trade under ~90% of equity at the chosen leverage
    lev = min(s.leverage, product.max_leverage)
    by_margin = 0.9 * equity * lev / notional_per_contract

    return max(0, math.floor(min(by_risk, by_exposure, by_margin)))


class RiskManager:
    def __init__(self, s: Settings, state: State):
        self.s = s
        self.state = state

    def update(self, equity: float) -> list[str]:
        """Roll daily counters and evaluate limits. Returns newly triggered events."""
        st, events = self.state, []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if st.day != today:
            st.day = today
            st.day_start_equity = equity
            if st.halted_reason.startswith("daily"):
                st.halted_reason = ""
        st.peak_equity = max(st.peak_equity or equity, equity)

        if not st.killed and equity <= st.peak_equity * (1 - self.s.max_drawdown):
            st.killed = True
            st.halted_reason = (f"KILL SWITCH: equity {equity:.2f} is {self.s.max_drawdown:.0%} below "
                                f"peak {st.peak_equity:.2f}. Run `python -m deltabot reset` to resume.")
            events.append(st.halted_reason)
        elif not st.halted_reason and st.day_start_equity and \
                equity <= st.day_start_equity * (1 - self.s.daily_loss_limit):
            st.halted_reason = f"daily loss limit hit ({self.s.daily_loss_limit:.0%}); no new trades until 00:00 UTC"
            events.append(st.halted_reason)
        return events

    @property
    def can_open(self) -> bool:
        return not self.state.killed and not self.state.halted_reason
