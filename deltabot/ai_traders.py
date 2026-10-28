"""AI duel: Claude (via the local Claude Code CLI) and ChatGPT (via the OpenAI API) trade the
same target with a free hand: they pick coins, direction, stop, target, risk per trade, holding
time (15s .. 1h) and when to look again. Code only enforces structural safety (paper only).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

from .claude_trader import ClaudeTrader, find_claude_cli

log = logging.getLogger("deltabot.ai")

HOLDS = {"15s": 15, "1m": 60, "5m": 300, "15m": 900, "1h": 3600}

PROMPT = """You are "{label}", an autonomous trading agent in a PAPER-trading simulation on Delta Exchange \
(USD-settled crypto perpetual futures). Another AI model trades the same account next to you; you are both \
hired to reach the user's target balance before the session ends.

You have a free hand. Every time you are called you decide:
- which coins to trade (any coin in the snapshot), long or short, or nothing;
- stop-loss and take-profit prices;
- risk_pct: how much of the WHOLE account you lose if the stop is hit (0.5 to 8);
- hold: how long the trade may stay open before it is closed automatically: "15s", "1m", "5m", "15m" or "1h". \
Use short holds for scalps on the 5-second and 1-minute data, longer holds for swings;
- next_check_seconds: when you want to be called again (30 to 600). Call back sooner when you hold short scalps.

Mechanics you must respect:
- Orders fill immediately at market. Costs are about 0.2% of position value per round trip (fees + slippage), so a \
trade must move more than ~0.2% just to break even. Very short scalps rarely cover that unless the move is sharp.
- Stops, targets and hold times are enforced every 5 seconds.
- You are fired if your own P&L falls to -20% of your budget share. The whole account stops at -50%.
- Never trade a coin the other trader already holds, and only coins whose usd_per_contract fits your \
my_exposure_room_usd.
- You may close any of YOUR open positions at any time.

Snapshot fields per coin: px; chg_1h/chg_4h/chg_24h (%); trend_1h/trend_15m (+1 up, -1 down, 0 none) with adx/er; \
rsi_5m, atr_5m, atr_pct_5m, ema20_vs_50_5m, to_hi20/to_lo20 (ATRs to the 20-candle 5m high/low), vol_x (volume vs \
average), closes_5m (last 12 five-minute closes), closes_1m (last 15 one-minute closes), rsi_1m, atr_pct_1m, \
ticks_5s (prices sampled every 5 seconds, oldest first, when available), usd_per_contract.

Think about the target, the time left and what you have learned from your own recent trades (my_recent_trades). \
Keep each `reason` to one or two sentences citing numbers, and `market_view` under 80 words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "market_view": {"type": "string"},
        "next_check_seconds": {"type": "integer"},
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "close"]},
                "symbol": {"type": "string"},
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "stop_price": {"type": "number"},
                "target_price": {"type": "number"},
                "risk_pct": {"type": "number"},
                "hold": {"type": "string", "enum": list(HOLDS)},
                "confidence": {"type": "number"},
                "reason": {"type": "string"},
            },
            "required": ["action", "symbol", "side", "stop_price", "target_price", "risk_pct", "hold",
                         "confidence", "reason"],
            "additionalProperties": False,
        }},
    },
    "required": ["market_view", "next_check_seconds", "actions"],
    "additionalProperties": False,
}


class LLMTrader:
    """Duck-types the parts of agents.Agent the arena uses."""

    def __init__(self, name: str, label: str, backend: str, model: str):
        self.name, self.label, self.backend, self.model = name, label, backend, model
        self.style = f"Free hand · {model} · picks coins, risk 0.5-8%, holds 15s-1h"
        self.regime_tf, self.entry_tf = "1h", "1m"
        self.max_positions, self.max_hold_bars, self.symbols = 4, 60, None
        self.prompt = PROMPT.format(label=label)
        self.cli = find_claude_cli() if backend == "claude_code" else None
        self.last: dict = {}
        self.calls, self.tokens, self.cost_usd = 0, 0, 0.0
        self.next_at = 0.0
        self.busy = False

    @property
    def available(self) -> bool:
        if self.backend == "claude_code":
            return bool(self.cli)
        return bool(os.environ.get("OPENAI_API_KEY"))

    # reuse the Claude trader's per-coin snapshot + guard helpers
    coin_snapshot = staticmethod(ClaudeTrader.coin_snapshot)

    def decide(self, text: str) -> tuple[dict | None, str]:
        try:
            if self.backend == "claude_code":
                return self._claude_code(text)
            return self._openai(text)
        except Exception as e:                       # never let one bad call kill the loop
            return None, f"{type(e).__name__}: {str(e)[:200]}"

    def _claude_code(self, text: str):
        cmd = [self.cli, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--model", self.model, "--system-prompt", self.prompt, "--json-schema", json.dumps(SCHEMA)]
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        workdir = Path(__file__).resolve().parent.parent / "state"
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=300, cwd=workdir, env=env)
        try:
            out = json.loads(proc.stdout)
        except ValueError:
            lines = (proc.stderr or proc.stdout or "no output").strip().splitlines()
            return None, "Claude Code error: " + (lines[-1][:200] if lines else "unknown")
        self.calls += 1
        self.cost_usd += float(out.get("total_cost_usd") or 0)
        if out.get("is_error") or out.get("subtype") != "success":
            return None, "Claude Code: " + str(out.get("result") or out.get("subtype"))[:200]
        data = out.get("structured_output")
        if data is None:
            data = json.loads(out.get("result") or "{}")
        return data, ""

    def _openai(self, text: str):
        from openai import OpenAI, APIConnectionError, APIStatusError, AuthenticationError, RateLimitError
        try:
            r = OpenAI(timeout=180, max_retries=2).chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": self.prompt}, {"role": "user", "content": text}],
                response_format={"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": SCHEMA}},
            )
        except AuthenticationError:
            return None, "OpenAI rejected the API key"
        except RateLimitError:
            return None, "OpenAI rate limit / quota reached"
        except APIStatusError as e:
            return None, f"OpenAI error {e.status_code}"
        except APIConnectionError:
            return None, "could not reach OpenAI"
        self.calls += 1
        if r.usage:
            self.tokens += r.usage.total_tokens
        msg = r.choices[0].message
        if getattr(msg, "refusal", None):
            return None, "ChatGPT declined: " + msg.refusal[:150]
        return json.loads(msg.content or "{}"), ""

    @staticmethod
    def check(a: dict, coin: dict | None) -> str:
        if coin is None:
            return "symbol not in the snapshot"
        px = coin["px"]
        long = a["side"] == "buy"
        sd = (px - a["stop_price"]) if long else (a["stop_price"] - px)
        td = (a["target_price"] - px) if long else (px - a["target_price"])
        if sd <= 0:
            return "stop is on the wrong side of the price"
        if td <= 0:
            return "target is on the wrong side of the price"
        if sd < 0.0005 * px:
            return "stop closer than 0.05% (would be hit by noise and fees)"
        if sd > 0.15 * px:
            return "stop further than 15%"
        if a["hold"] not in HOLDS:
            return "unknown hold time"
        return ""


def duel_traders() -> list[LLMTrader]:
    return [
        LLMTrader("fable", "Claude Fable", "claude_code", "claude-fable-5-1"),
        LLMTrader("gpt", "ChatGPT", "openai", os.environ.get("OPENAI_MODEL", "gpt-5.2")),
    ]
