"""Settings loaded from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BASE_URLS = {
    ("india", False): "https://api.india.delta.exchange",
    ("india", True): "https://cdn-ind.testnet.deltaex.org",
    ("global", False): "https://api.delta.exchange",
    ("global", True): "https://testnet-api.delta.exchange",
}


# (regime timeframe, entry timeframe) pairs the bot can trade on
TF_PAIRS = {"4h/1h": ("4h", "1h"), "1h/15m": ("1h", "15m"), "15m/5m": ("15m", "5m")}


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in ("1", "true", "yes", "y")


@dataclass
class Settings:
    # --- connection ---
    api_key: str = ""
    api_secret: str = ""
    region: str = "india"            # india | global
    testnet: bool = False
    mode: str = "paper"              # paper | live
    live_confirmed: bool = False     # must be true to send real orders
    settle_asset: str = "USD"        # wallet asset used for equity

    # --- market selection ---
    min_turnover_usd: float = 2_000_000   # 24h liquidity filter
    universe_size: int = 25               # most liquid N perps analysed
    regime_tf: str = "4h"                 # timeframe for picking markets
    entry_tf: str = "1h"                  # timeframe for entries/exits
    rescan_minutes: int = 60
    # "auto" = for each market pick whichever timeframe pair shows the cleanest trend,
    # or a fixed pair such as "4h/1h", "1h/15m", "15m/5m"
    timeframe_mode: str = "4h/1h"
    blacklist: list[str] = field(default_factory=list)
    crypto_only: bool = True              # skip gold/silver/stock tokens

    # --- strategy ---
    ema_fast: int = 20
    ema_slow: int = 50
    regime_ema_fast: int = 50
    regime_ema_slow: int = 200
    donchian: int = 20
    atr_len: int = 14
    adx_min: float = 20.0
    er_min: float = 0.25
    sl_atr: float = 3.0
    tp_atr: float = 8.0
    max_hold_hours: float = 240
    rsi_max_long: float = 75
    rsi_min_short: float = 25

    # --- risk ---
    risk_per_trade: float = 0.01     # fraction of equity lost if stop hits
    max_positions: int = 3
    leverage: int = 5                # leverage set on each product
    max_total_notional_x: float = 3  # total open notional <= X * equity
    daily_loss_limit: float = 0.04   # stop opening trades for the day
    max_drawdown: float = 0.20       # kill switch: flatten + halt
    fee_rate: float = 0.0005         # taker fee per side, used in paper/backtest
    maker_fee: float = 0.0002        # limit-order (maker) fee per side
    fee_tax: float = 0.18            # GST on trading fees (India), paper only
    slippage: float = 0.0005         # per side, used in paper/backtest

    # --- instruments ---
    # Order = preference. For each trending market the bot uses the first type that is available:
    #   options   -> buy a call (uptrend) / put (downtrend) on the underlying (BTC, ETH, XAUT)
    #   futures   -> dated futures contract of the underlying (if Delta lists any)
    #   perpetual -> perpetual futures (always available)
    instruments: list[str] = field(default_factory=lambda: ["perpetual"])

    # --- options (long calls / puts only: max loss = premium paid) ---
    opt_target_dte: float = 14        # preferred days to expiry
    opt_min_dte: float = 4
    opt_max_dte: float = 35
    opt_target_delta: float = 0.45    # |delta| of the strike to buy
    opt_max_spread: float = 0.12      # skip if (ask - bid) / ask is wider
    opt_min_bid: float = 0.0          # skip strikes with no bid
    opt_stop_loss: float = 0.5        # exit if premium falls 50%
    opt_take_profit: float = 1.5      # exit if premium rises 150%
    opt_exit_before_expiry_h: float = 48
    opt_max_premium_pct: float = 0.03 # max premium in one trade, as fraction of equity
    opt_fee_rate: float = 0.0003      # fee on underlying notional per side (capped, see broker)
    opt_fee_cap: float = 0.10         # fee never more than 10% of premium

    # --- dated futures ---
    fut_min_dte: float = 5            # don't open futures expiring sooner than this
    fut_exit_before_expiry_h: float = 24

    # --- session goal (0 = off): stop + close everything when reached ---
    target_equity: float = 0.0        # in settle currency (USD)
    session_end: float = 0.0          # unix seconds

    # --- paper ---
    paper_start_equity: float = 1000.0

    # --- alerts ---
    telegram_token: str = ""
    telegram_chat_id: str = ""

    @property
    def base_url(self) -> str:
        return BASE_URLS[(self.region, self.testnet)]

    def tf_pairs(self) -> list[tuple[str, str]]:
        if self.timeframe_mode == "auto":
            return list(TF_PAIRS.values())
        return [TF_PAIRS[self.timeframe_mode]]

    def apply_timeframe_mode(self) -> None:
        if self.timeframe_mode != "auto":
            self.regime_tf, self.entry_tf = TF_PAIRS[self.timeframe_mode]

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


def load_settings() -> Settings:
    load_dotenv()
    s = Settings()
    s.api_key = _env("DELTA_API_KEY", "")
    s.api_secret = _env("DELTA_API_SECRET", "")
    s.region = _env("DELTA_REGION", s.region).lower()
    s.testnet = _bool("DELTA_TESTNET", s.testnet)
    s.mode = _env("BOT_MODE", s.mode).lower()
    s.live_confirmed = _bool("LIVE_TRADING_CONFIRMED", False)
    s.settle_asset = _env("SETTLE_ASSET", s.settle_asset)
    s.min_turnover_usd = float(_env("MIN_TURNOVER_USD", str(s.min_turnover_usd)))
    s.universe_size = int(_env("UNIVERSE_SIZE", str(s.universe_size)))
    s.crypto_only = _bool("CRYPTO_ONLY", s.crypto_only)
    s.blacklist = [x.strip() for x in _env("BLACKLIST", "").split(",") if x.strip()]
    s.risk_per_trade = float(_env("RISK_PER_TRADE", str(s.risk_per_trade)))
    s.max_positions = int(_env("MAX_POSITIONS", str(s.max_positions)))
    s.leverage = int(_env("LEVERAGE", str(s.leverage)))
    s.max_total_notional_x = float(_env("MAX_TOTAL_NOTIONAL_X", str(s.max_total_notional_x)))
    s.daily_loss_limit = float(_env("DAILY_LOSS_LIMIT", str(s.daily_loss_limit)))
    s.max_drawdown = float(_env("MAX_DRAWDOWN", str(s.max_drawdown)))
    s.paper_start_equity = float(_env("PAPER_START_EQUITY", str(s.paper_start_equity)))
    s.instruments = [x.strip().lower() for x in _env("INSTRUMENTS", ",".join(s.instruments)).split(",") if x.strip()]
    s.opt_target_dte = float(_env("OPT_TARGET_DTE", str(s.opt_target_dte)))
    s.opt_min_dte = float(_env("OPT_MIN_DTE", str(s.opt_min_dte)))
    s.opt_max_dte = float(_env("OPT_MAX_DTE", str(s.opt_max_dte)))
    s.opt_target_delta = float(_env("OPT_TARGET_DELTA", str(s.opt_target_delta)))
    s.opt_max_spread = float(_env("OPT_MAX_SPREAD", str(s.opt_max_spread)))
    s.opt_stop_loss = float(_env("OPT_STOP_LOSS", str(s.opt_stop_loss)))
    s.opt_take_profit = float(_env("OPT_TAKE_PROFIT", str(s.opt_take_profit)))
    s.opt_max_premium_pct = float(_env("OPT_MAX_PREMIUM_PCT", str(s.opt_max_premium_pct)))
    s.timeframe_mode = _env("TIMEFRAME_MODE", s.timeframe_mode)
    if s.timeframe_mode != "auto" and s.timeframe_mode not in TF_PAIRS:
        raise ValueError(f"TIMEFRAME_MODE must be auto or one of {list(TF_PAIRS)}")
    s.apply_timeframe_mode()
    bad = set(s.instruments) - {"perpetual", "futures", "options"}
    if bad or not s.instruments:
        raise ValueError(f"INSTRUMENTS must be a comma list of perpetual,futures,options (got {bad or 'nothing'})")
    s.telegram_token = _env("TELEGRAM_TOKEN", "")
    s.telegram_chat_id = _env("TELEGRAM_CHAT_ID", "")
    if s.region not in ("india", "global"):
        raise ValueError("DELTA_REGION must be 'india' or 'global'")
    if s.mode not in ("paper", "live"):
        raise ValueError("BOT_MODE must be 'paper' or 'live'")
    return s
