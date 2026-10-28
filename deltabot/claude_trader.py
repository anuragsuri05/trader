"""Claude AI trader: an arena bot whose decisions come from Claude (Anthropic API).

Every few minutes Claude receives a numeric market snapshot (multi-timeframe indicators for the
most liquid coins, what the rule-based bots see, the account and its own positions) and returns
structured decisions. Claude decides WHAT to trade and WHERE the stop and target go; deterministic
code decides HOW MUCH (1% risk, scaled by confidence) and rejects anything outside the guardrails.

Backends:
  claude_code (default) - runs the local Claude Code CLI headless (`claude -p`) with your existing
                          Claude Code login, so no API key is needed. Calls count toward your
                          Claude plan's usage limits.
  api                   - Anthropic API with ANTHROPIC_API_KEY (used only if the CLI isn't installed).
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from .strategy import entry_frame, regime_frame

log = logging.getLogger("deltabot.claude")

MODEL = "claude-opus-5-5"
PRICE_IN, PRICE_OUT = 4.00, 20.00          # USD per 1M tokens (claude-opus-5-5)

SYSTEM_PROMPT = """You are "Claude AI", one of several trading bots in a paper-trading arena on Delta Exchange \
(USD-settled crypto perpetual futures). You compete with rule-based breakout bots for the same account.

How you are judged:
- You are hired with a budget share. If your P&L falls to -20% of that share you are fired immediately; at each \
performance review the worst bot that is losing money is fired. Doing nothing is never punished; losing is.
{MODE_BLOCK}

Snapshot fields: `px` last price; `chg_1h`/`chg_4h`/`chg_24h` % change; `trend_1h`/`trend_15m` regime (+1 up, -1 down, \
0 none) with `adx` and `er` (efficiency ratio, 1 = straight line, 0 = noise); `rsi_5m`; `atr_5m` (price units) and \
`atr_pct_5m`; `ema20_vs_50_5m` (+1 fast above slow); `to_hi20`/`to_lo20` distance from the last 20 five-minute \
candles' high/low in ATRs; `vol_x` last 5-minute volume vs its 20-candle average; `closes_5m` the last 12 \
five-minute closes, oldest first; `bots` signals the rule-based bots see right now; `usd_per_contract` the \
position value of ONE contract. Account field `my_exposure_room_usd` is the most position value you can open right \
now: never pick a coin whose `usd_per_contract` is larger than that, because it cannot be bought.

{CLOSING}"""

NORMAL_MODE = """- The user hopes the team turns its balance into a large target within a short session. That is usually not \
achievable with sane risk, and you must not chase it by taking outsized or low-quality trades. Protect capital first.

What you control: which coins to trade, direction, stop-loss price, target price, and a confidence between 0 and 1. \
Position size is computed by code: each trade risks about 1% of the account at confidence 1.0 (0.5% at confidence 0.5). \
You may also close any of YOUR open positions.

Costs and mechanics (be realistic about them):
- Entries are limit orders at the current price, valid for 15 minutes; they fill only if price trades through them. \
Maker fee 0.02% per side, stop-losses execute as market orders (0.05% fee plus slippage). Small moves are eaten by costs.
- Stop distance must be between 1 and 6 times the 5-minute ATR. Target must be at least 1.5 times the stop distance away.
- You can hold at most 2 positions, and never a coin another bot already holds. Positions are closed automatically \
after 4 hours.
- Decisions are made about every 6 minutes; between decisions the stop and target are enforced by the exchange simulation."""

TARGET_MODE = """- TARGET MODE IS ON. This is a paper-trading simulation, and the user has explicitly asked every bot to pursue \
the session target as hard as possible within the time left, accepting a much higher chance of losing the paper \
balance. Waiting for perfect setups is not what they want. Each round, take the best available short-term trades \
(scalps on 1-5 minute moves are fine) unless the market is truly directionless. You can still be fired, so pick the \
cleanest short-term direction rather than random coins.

What you control: which coins to trade, direction, stop-loss price, target price, and a confidence between 0 and 1. \
Position size is computed by code: about 3% of the account at risk per trade at confidence 1.0 (1.5% at 0.5). \
You may also close any of YOUR open positions, for example to take a quick profit.

Mechanics:
- Entries are market orders filled immediately (0.05% fee plus slippage per side), so a trade must move more than \
about 0.2% just to break even.
- Stop distance must be between 0.5 and 4 times the 5-minute ATR. Target must be at least 1.0 times the stop distance away.
- You can hold at most 3 positions, and never a coin another bot already holds. Positions are closed automatically \
after 2 hours.
- Decisions are made about every 3 minutes; between decisions the stop and target are enforced by the exchange simulation."""

NORMAL_CLOSING = """Return only setups you would genuinely take. An empty actions list is a good answer when nothing is clearly \
favourable. Keep `reason` to one or two sentences that cite the numbers you relied on, and keep `market_view` \
under 80 words."""

TARGET_CLOSING = """Return the trades you would take to pursue the target now. Keep `reason` to one or two sentences that cite \
the numbers you relied on, and keep `market_view` under 80 words."""


def system_prompt(target_mode: bool) -> str:
    return SYSTEM_PROMPT.replace("{MODE_BLOCK}", TARGET_MODE if target_mode else NORMAL_MODE) \
        .replace("{CLOSING}", TARGET_CLOSING if target_mode else NORMAL_CLOSING)


class TradeAction(BaseModel):
    action: Literal["open", "close"]
    symbol: str = Field(description="Exact symbol from the snapshot, e.g. XRPUSD")
    side: Literal["buy", "sell"] = Field(description="For close actions, the side of the position being closed")
    stop_price: float = Field(description="Stop-loss price for open actions; 0 for close actions")
    target_price: float = Field(description="Take-profit price for open actions; 0 for close actions")
    confidence: float = Field(description="0 to 1")
    reason: str


class Decision(BaseModel):
    market_view: str
    actions: list[TradeAction]


def has_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def find_claude_cli() -> str | None:
    for p in (shutil.which("claude"), str(Path.home() / ".local/bin/claude"), "/usr/local/bin/claude"):
        if p and Path(p).exists():
            return p
    return None


_S = {"type": "string"}
_N = {"type": "number"}
DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "market_view": _S,
        "actions": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "close"]},
                "symbol": _S,
                "side": {"type": "string", "enum": ["buy", "sell"]},
                "stop_price": _N, "target_price": _N, "confidence": _N, "reason": _S,
            },
            "required": ["action", "symbol", "side", "stop_price", "target_price", "confidence", "reason"],
            "additionalProperties": False,
        }},
    },
    "required": ["market_view", "actions"],
    "additionalProperties": False,
}


class ClaudeTrader:
    """Arena bot backed by Claude. Duck-types the parts of agents.Agent the arena uses."""

    def __init__(self, name="claude", label="Claude AI", interval_s=360, max_positions=2, max_hold_bars=48,
                 target_mode: bool = False):
        if target_mode:
            interval_s, max_positions, max_hold_bars = 180, 3, 24
        self.target_mode = target_mode
        self.stop_atr = (0.5, 4.0) if target_mode else (1.0, 6.0)
        self.min_rr = 1.0 if target_mode else 1.5
        self.prompt = system_prompt(target_mode)
        self.name, self.label = name, label
        self.style = (f"{'TARGET MODE: ' if target_mode else ''}reads the whole market every "
                      f"{interval_s // 60} min ({MODEL})")
        self.regime_tf, self.entry_tf = "1h", "5m"
        self.max_positions, self.max_hold_bars = max_positions, max_hold_bars
        self.interval_s = interval_s
        self._client = None
        self.cli = find_claude_cli()
        self.backend = "claude_code" if self.cli else "api"
        self.last: dict = {}            # last decision for the dashboard
        self.calls = 0
        self.cost_usd = 0.0

    @property
    def available(self) -> bool:
        return bool(self.cli) if self.backend == "claude_code" else has_credentials()

    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=180.0, max_retries=2)
        return self._client

    # ------------------------------------------------------------ snapshot
    @staticmethod
    def coin_snapshot(sym: str, c5, c15, c1h, s, bot_signals: list[str], contract_value: float = 0.0) -> dict | None:
        if len(c5) < 60 or len(c15) < 60 or len(c1h) < s.regime_ema_slow + 20:
            return None
        px = float(c5["close"].iloc[-1])

        def chg(df, bars):
            return round((px / float(df["close"].iloc[-1 - bars]) - 1) * 100, 2) if len(df) > bars else None

        r1h = regime_frame(c1h, s).iloc[-1]
        r15 = regime_frame(c15, s).iloc[-1]
        f5 = entry_frame(c5, s).iloc[-1]
        atr5 = float(f5["atr"])
        vol = c5["volume"]
        vol_x = float(vol.iloc[-1] / vol.rolling(20).mean().shift(1).iloc[-1]) if vol.iloc[-21:-1].mean() > 0 else None
        hi20 = float(c5["high"].iloc[-21:-1].max())
        lo20 = float(c5["low"].iloc[-21:-1].min())
        rnd = lambda x, n=2: None if x is None or not np.isfinite(x) else round(float(x), n)
        sig = max(4, -int(np.floor(np.log10(abs(px)))) + 4) if px > 0 else 6
        return {
            "sym": sym, "px": round(px, sig),
            "chg_1h": chg(c5, 12), "chg_4h": chg(c5, 48), "chg_24h": chg(c1h, 24),
            "trend_1h": int(r1h["direction"]), "adx_1h": rnd(r1h["adx"], 1), "er_1h": rnd(r1h["er"]),
            "trend_15m": int(r15["direction"]), "adx_15m": rnd(r15["adx"], 1), "er_15m": rnd(r15["er"]),
            "rsi_5m": rnd(f5["rsi"], 1), "atr_5m": round(atr5, sig), "atr_pct_5m": rnd(atr5 / px * 100, 3),
            "ema20_vs_50_5m": 1 if f5["ema_fast"] > f5["ema_slow"] else -1,
            "to_hi20": rnd((hi20 - px) / atr5, 2) if atr5 > 0 else None,
            "to_lo20": rnd((px - lo20) / atr5, 2) if atr5 > 0 else None,
            "vol_x": rnd(vol_x, 2),
            "closes_5m": [round(float(x), sig) for x in c5["close"].iloc[-12:]],
            "bots": bot_signals,
            "usd_per_contract": round(contract_value * px, 2),
        }

    # ------------------------------------------------------------ the call
    def decide(self, snapshot_text: str) -> tuple[Decision | None, str, str]:
        """Returns (decision, thinking_summary, error)."""
        if self.backend == "claude_code":
            return self._decide_claude_code(snapshot_text)
        return self._decide_api(snapshot_text)

    def _decide_claude_code(self, snapshot_text: str) -> tuple[Decision | None, str, str]:
        """Headless Claude Code: no tools, no saved session, JSON-schema-validated output."""
        cmd = [self.cli, "-p", "--output-format", "json", "--tools", "", "--no-session-persistence",
               "--model", MODEL, "--effort", "medium",
               "--system-prompt", self.prompt, "--json-schema", json.dumps(DECISION_SCHEMA)]
        env = {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
        workdir = Path(__file__).resolve().parent.parent / "state"
        workdir.mkdir(exist_ok=True)
        try:
            proc = subprocess.run(cmd, input=snapshot_text, capture_output=True, text=True,
                                  timeout=300, cwd=workdir, env=env)
        except subprocess.TimeoutExpired:
            return None, "", "Claude Code took longer than 5 minutes"
        except OSError as e:
            return None, "", f"could not start Claude Code: {e}"
        try:
            out = json.loads(proc.stdout)
        except ValueError:
            msg = (proc.stderr or proc.stdout or "no output").strip().splitlines()
            return None, "", "Claude Code error: " + (msg[-1][:200] if msg else "unknown")
        self.calls += 1
        self.cost_usd += float(out.get("total_cost_usd") or 0)
        if out.get("is_error") or out.get("subtype") != "success":
            return None, "", "Claude Code: " + str(out.get("result") or out.get("subtype"))[:200]
        data = out.get("structured_output")
        if data is None:
            try:
                data = json.loads(out.get("result") or "")
            except ValueError:
                return None, "", "Claude Code returned no structured decision"
        try:
            return Decision.model_validate(data), "", ""
        except Exception as e:
            return None, "", f"decision did not match the schema: {str(e)[:150]}"

    def _decide_api(self, snapshot_text: str) -> tuple[Decision | None, str, str]:
        import anthropic
        try:
            resp = self.client().beta.messages.parse(
                model=MODEL,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": "medium"},
                cache_control={"type": "ephemeral"},
                system=self.prompt,
                messages=[{"role": "user", "content": snapshot_text}],
                output_format=Decision,
            )
        except anthropic.AuthenticationError:
            return None, "", "Anthropic API key was rejected"
        except anthropic.RateLimitError:
            return None, "", "rate limited by the Anthropic API, will retry next cycle"
        except anthropic.APIStatusError as e:
            return None, "", f"Anthropic API error {e.status_code}: {str(e.message)[:200]}"
        except anthropic.APIConnectionError:
            return None, "", "could not reach the Anthropic API"
        self.calls += 1
        u = resp.usage
        cached = getattr(u, "cache_read_input_tokens", 0) or 0
        written = getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cost_usd += (u.input_tokens * PRICE_IN + written * PRICE_IN * 1.25 + cached * PRICE_IN * 0.1
                          + u.output_tokens * PRICE_OUT) / 1e6
        thinking = " ".join(b.thinking for b in resp.content if b.type == "thinking" and b.thinking)
        if resp.stop_reason == "refusal":
            return None, thinking, "Claude declined this request"
        if resp.stop_reason == "max_tokens":
            return None, thinking, "response was cut off (max_tokens)"
        return resp.parsed_output, thinking, ""

    # ------------------------------------------------------------ guardrails
    def check(self, a: TradeAction, coin: dict | None) -> str:
        return self.check_open(a, coin, self.stop_atr, self.min_rr)

    @staticmethod
    def check_open(a: TradeAction, coin: dict | None, stop_atr=(1.0, 6.0), min_rr=1.5) -> str:
        """Empty string = accepted, otherwise the rejection reason."""
        if coin is None:
            return "symbol not in the snapshot"
        px, atr = coin["px"], coin["atr_5m"]
        if not atr or atr <= 0:
            return "no ATR available"
        long = a.side == "buy"
        stop_d = (px - a.stop_price) if long else (a.stop_price - px)
        tgt_d = (a.target_price - px) if long else (px - a.target_price)
        if stop_d <= 0:
            return "stop is on the wrong side of the price"
        if tgt_d <= 0:
            return "target is on the wrong side of the price"
        if not stop_atr[0] * atr <= stop_d <= stop_atr[1] * atr:
            return f"stop distance {stop_d / atr:.1f} ATR is outside {stop_atr[0]:g}-{stop_atr[1]:g} ATR"
        if tgt_d < min_rr * stop_d:
            return f"reward/risk {tgt_d / stop_d:.2f} is below {min_rr:g}"
        if not 0 <= a.confidence <= 1:
            return "confidence must be between 0 and 1"
        return ""


def snapshot_text(coins: list[dict], account: dict) -> str:
    """Stable, compact JSON. Coins sorted by symbol so the structure repeats call to call."""
    lines = ["ACCOUNT", json.dumps(account, sort_keys=True), "", "MARKET (one coin per line)"]
    lines += [json.dumps(c, sort_keys=True, separators=(",", ":")) for c in sorted(coins, key=lambda c: c["sym"])]
    lines += ["", "Decide now. Return your market_view and actions (possibly none)."]
    return "\n".join(lines)
