"""Offline tests: run with `python3 -m pytest tests` or `python3 tests/test_core.py`."""
import hashlib
import hmac
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deltabot.backtest import align_regime, simulate
from deltabot.client import sign
from deltabot.config import Settings
from deltabot.data import Product, round_to_tick
from deltabot.risk import RiskManager, position_size
from deltabot.state import State
from deltabot.strategy import Signal, entry_frame

BTC = Product(id=27, symbol="BTCUSD", contract_value=0.001, tick_size=0.5, taker_fee=0.0005, max_leverage=100)


def make_df(prices, freq="1h"):
    idx = pd.date_range("2025-01-01", periods=len(prices), freq=freq, tz="UTC")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p * 1.002, "low": p * 0.998, "close": p, "volume": 1.0}, index=idx)


def test_signature_matches_spec():
    expected = hmac.new(b"secret", b"GET1700000000/v2/orders?product_id=1", hashlib.sha256).hexdigest()
    assert sign("secret", "GET", "1700000000", "/v2/orders", "?product_id=1", "") == expected


def test_position_size_risks_one_percent():
    s = Settings(risk_per_trade=0.01, fee_rate=0, slippage=0, max_total_notional_x=100, leverage=100)
    sig = Signal("buy", 100_000, 99_000, 103_000, 500)
    # $10,000 equity -> risk $100; each contract loses 1000 * 0.001 = $1 -> 100 contracts
    assert position_size(BTC, sig, 10_000, 0, s) == 100


def test_position_size_respects_exposure_cap():
    s = Settings(risk_per_trade=0.5, fee_rate=0, slippage=0, max_total_notional_x=1, leverage=100)
    sig = Signal("buy", 100_000, 99_000, 103_000, 500)
    # notional cap = 1x of $1000 -> 1000 / (0.001 * 100000) = 10 contracts
    assert position_size(BTC, sig, 1_000, 0, s) == 10


def test_risk_daily_limit_and_kill_switch():
    s = Settings(daily_loss_limit=0.05, max_drawdown=0.2)
    st = State()
    rm = RiskManager(s, st)
    rm.update(1000)
    assert rm.can_open
    rm.update(940)                       # -6% today
    assert not rm.can_open and st.halted_reason.startswith("daily")
    rm.update(790)                       # -21% from peak
    assert st.killed


def test_donchian_has_no_lookahead():
    df = make_df(np.linspace(100, 200, 120))
    f = entry_frame(df, Settings())
    # channel at bar i must equal the max of bars i-20..i-1, never include bar i
    i = 80
    assert f["dc_high"].iloc[i] == df["high"].iloc[i - 20:i].max()


def test_regime_alignment_uses_closed_bars_only():
    s = Settings(regime_tf="4h", entry_tf="1h")
    edf = make_df(np.ones(10) * 100, "1h")
    rdf = make_df(np.ones(3) * 100, "4h")
    d = align_regime(edf, rdf, s)
    assert len(d) == len(edf)   # 4h bar opening 00:00 is only usable from the 1h bar closing at 04:00


def test_simulate_uptrend_makes_long_trades():
    s = Settings(regime_tf="4h", entry_tf="1h", fee_rate=0, slippage=0)
    prices = 100 * np.exp(np.cumsum(np.r_[np.zeros(60), np.full(400, 0.002)] + 0.003 * np.sin(np.arange(460))))
    df = make_df(prices)
    direction = pd.Series(1, index=df.index)
    trades, curve = simulate(df, direction, BTC, s, 1000)
    assert trades and all(t["side"] == "buy" for t in trades)
    assert curve.iloc[-1] > 1000


def test_round_to_tick():
    assert round_to_tick(100.26, 0.5) == 100.5
    assert round_to_tick(0.123456, 0.0001) == 0.1235


# ---------------------------------------------------------------- options / futures
from deltabot.options import bs_delta, bs_price, choose_future, option_fee, option_size, strike_for_delta


def test_put_call_parity():
    c = bs_price(100, 105, 0.1, 0.6, True)
    p = bs_price(100, 105, 0.1, 0.6, False)
    assert abs((c - p) - (100 - 105)) < 1e-9


def test_strike_for_delta_inverts_delta():
    for call, d in ((True, 0.45), (False, 0.30)):
        k = strike_for_delta(84000, 14 / 365, 0.4, d, call)
        assert abs(abs(bs_delta(84000, k, 14 / 365, 0.4, call)) - d) < 1e-6


def test_option_size_caps_premium_and_risk():
    s = Settings(risk_per_trade=0.01, opt_stop_loss=0.5, opt_max_premium_pct=0.03)
    # premium $2000/BTC, contract 0.001 BTC -> $2 per contract; equity $1000
    # by risk: 10 / (2 * 0.5) = 10; by premium: 30 / 2 = 15 -> 10
    assert option_size(2000, 0.001, 1000, s) == 10


def test_option_fee_is_capped():
    s = Settings(opt_fee_rate=0.0003, opt_fee_cap=0.10)
    assert option_fee(1.0, 84000, 10, 0.001, s) == 0.10 * 1.0 * 0.001 * 10      # cheap option -> cap
    assert abs(option_fee(5000, 84000, 10, 0.001, s) - 0.0003 * 84000 * 0.001 * 10) < 1e-12


def test_choose_future_skips_near_expiry():
    import time as _t
    now = _t.time()
    near = Product(1, "BTC-FUT-NEAR", 0.001, 0.5, 0.0005, 100, kind="futures", underlying="BTC", expiry=now + 2 * 86400)
    far = Product(2, "BTC-FUT-FAR", 0.001, 0.5, 0.0005, 100, kind="futures", underlying="BTC", expiry=now + 30 * 86400)
    assert choose_future({p.symbol: p for p in (near, far)}, "BTC", Settings(fut_min_dte=5)).symbol == "BTC-FUT-FAR"


def test_unix_seconds_any_pandas_unit():
    from deltabot.data import unix_seconds
    idx = pd.to_datetime(pd.Series([1761000000, 1761003600]), unit="s", utc=True)
    assert unix_seconds(idx) == [1761000000, 1761003600]
    assert unix_seconds(pd.DatetimeIndex(idx).as_unit("ns")) == [1761000000, 1761003600]


def test_options_backtest_charges_time_decay():
    """Regression: option time-to-expiry must shrink bar by bar (it was frozen by a timestamp-unit bug)."""
    from deltabot.backtest import simulate_options
    s = Settings(entry_tf="1h", opt_target_dte=14, max_hold_hours=10_000, opt_stop_loss=0.99,
                 opt_take_profit=100, opt_exit_before_expiry_h=0)
    t = np.arange(760)
    rise = 100 * np.exp(t * 0.0008 + 0.004 * np.sin(t * 0.7))
    df = make_df(np.r_[rise, np.full(500, rise[-1] * 0.995)])     # trend, then flat until expiry
    df["high"] = df["low"] = df["close"]
    trades, _ = simulate_options(df, pd.Series(1, index=df.index), s, 1000, 0.001)
    held = [x for x in trades if x["reason"] == "near_expiry"]
    assert held, "scenario should hold an option to expiry"
    # at expiry only intrinsic value is left; with frozen time it would be worth ~2x entry
    assert all(x["exit"] < x["entry"] * 1.2 for x in held)


# ---------------------------------------------------------------- paper limit orders
class _StubClient:
    def __init__(self, bars):
        self.bars = bars

    def candles(self, symbol, resolution, start, end):
        return [b for b in self.bars if start <= b["time"] <= end]

    def ticker(self, symbol):
        return {"mark_price": self.bars[-1]["close"]}


def _paper(bars, **kw):
    from deltabot.broker import PaperBroker
    s = Settings(paper_start_equity=1000, fee_rate=0.0005, maker_fee=0.0002, fee_tax=0.0, slippage=0.001, **kw)
    st = State(mode="paper_test")
    return PaperBroker(_StubClient(bars), s, st, {"BTCUSD": BTC}), st


def test_limit_order_fills_only_when_price_trades_through():
    import time as _t
    now = int(_t.time()) // 60 * 60
    placed = now - 600
    mk = lambda ts, lo: {"time": ts, "open": 100.0, "high": 100.5, "low": lo, "close": 100.0, "volume": 1}
    # touches 99.0 exactly (no fill), then trades through to 98.9 (fill)
    bars = [mk(placed + 60 * i, lo) for i, lo in enumerate([99.5, 99.0, 98.9, 99.6])]
    br, st = _paper(bars)
    br.place_limit(BTC, Signal("buy", 99.0, 97.0, 104.0, 1), 10, "brk5", "5m", 3600)
    st.extra["orders"]["BTCUSD"]["placed"] = placed          # pretend it was placed 10 minutes ago
    msgs = br.check_orders()
    assert "BTCUSD" in st.trades and st.trades["BTCUSD"].entry_price == 99.0
    assert st.trades["BTCUSD"].entry_time == placed + 120 and st.trades["BTCUSD"].agent == "brk5"
    assert abs(st.paper_cash - (1000 - 99.0 * 10 * 0.001 * 0.0002)) < 1e-9     # maker fee, no slippage
    assert msgs and msgs[0].startswith("limit filled")


def test_limit_order_expires_unfilled():
    import time as _t
    now = int(_t.time()) // 60 * 60
    bars = [{"time": now - 300 + 60 * i, "open": 100, "high": 101, "low": 99.5, "close": 100, "volume": 1} for i in range(4)]
    br, st = _paper(bars)
    br.place_limit(BTC, Signal("buy", 99.0, 97.0, 104.0, 1), 10, "brk5", "5m", -1)   # already expired
    assert br.check_orders()[0].startswith("limit expired") and not st.trades and not st.extra["orders"]


# ---------------------------------------------------------------- Claude guardrails
def test_claude_guardrails():
    from deltabot.claude_trader import ClaudeTrader, TradeAction
    coin = {"px": 100.0, "atr_5m": 1.0}
    mk = lambda side, sl, tp, conf=0.8: TradeAction(action="open", symbol="X", side=side, stop_price=sl,
                                                   target_price=tp, confidence=conf, reason="")
    chk = ClaudeTrader.check_open
    assert chk(mk("buy", 98, 104), coin) == ""                       # 2 ATR stop, 2:1
    assert "wrong side" in chk(mk("buy", 101, 104), coin)
    assert "wrong side" in chk(mk("sell", 98, 96), coin)             # stop below price on a short
    assert "outside 1-6 ATR" in chk(mk("buy", 99.5, 104), coin)      # 0.5 ATR too tight
    assert "outside 1-6 ATR" in chk(mk("buy", 90, 130), coin)        # 10 ATR too wide
    assert "reward/risk" in chk(mk("buy", 98, 101), coin)            # 1:2 reward
    assert "confidence" in chk(mk("sell", 102, 96, 1.5), coin)
    assert chk(mk("buy", 98, 104), None) == "symbol not in the snapshot"


def test_monte_carlo_bounds():
    from deltabot.analyse import monte_carlo
    always_win = [{"rate": 600.0, "rets": np.array([0.02])}]                # +2% per trade, ~100 trades in 10 min
    r = monte_carlo(always_win, 10, 5000, 5500, 0.5, sims=500)
    assert r["p_target"] == 1.0 and r["p_loss"] == 0.0                    # 5 wins of 2% clears +10%
    always_lose = [{"rate": 60.0, "rets": np.array([-0.01])}]
    r = monte_carlo(always_lose, 10, 5000, 5500, 0.5, sims=500)
    assert r["p_target"] == 0.0 and r["p_loss"] == 1.0
    assert monte_carlo([], 10, 5000, 5500, 0.5)["p_target"] == 0.0


# ---------------------------------------------------------------- all-weather portfolio
def _aw_prices(n=400, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    cols = ["BTC", "ETH", "XRP", "LTC"]
    return pd.DataFrame({c: 100 * np.exp(np.cumsum(rng.normal(0.001 * (i - 1), 0.03, n))) for i, c in enumerate(cols)}, index=idx)


def test_allweather_trend_has_no_lookahead():
    import deltabot.allweather as AW
    P = _aw_prices()
    w1 = AW.trend_weights(P, list(P.columns))
    P2 = P.copy()
    P2.iloc[300:] *= 3.0                                    # change only the FUTURE after day 300
    w2 = AW.trend_weights(P2, list(P.columns))
    assert np.allclose(w1.iloc[:300].values, w2.iloc[:300].values)


def test_allweather_trend_follows_direction():
    import deltabot.allweather as AW
    idx = pd.date_range("2022-01-01", periods=300, freq="D", tz="UTC")
    P = pd.DataFrame({"BTC": 100 * np.exp(np.linspace(0, 1, 300)) * (1 + 0.01 * np.sin(np.arange(300))),
                      "ETH": 100 * np.exp(np.linspace(0, -1, 300)) * (1 + 0.01 * np.cos(np.arange(300)))}, index=idx)
    w = AW.trend_weights(P, ["BTC", "ETH"]).iloc[-1]
    assert w["BTC"] > 0 and w["ETH"] < 0                    # long the riser, short the faller


def test_allweather_carry_hysteresis():
    import deltabot.allweather as AW
    idx = pd.date_range("2022-01-01", periods=40, freq="D", tz="UTC")
    P = pd.DataFrame({"BTC": np.full(40, 100.0)}, index=idx)
    daily = np.r_[np.full(10, 0.0003), np.full(10, 0.00009), np.full(20, -0.0003)]   # per day (3 x per-8h)
    F = pd.DataFrame({"BTC": daily}, index=idx)
    on = AW.carry_states(P, F, ["BTC"])["BTC"]
    assert on.iloc[12] and on.iloc[18]                      # stays on while funding still positive (above exit)
    assert not on.iloc[-1]                                  # off once funding turns negative


def test_allweather_drawdown_brake():
    import deltabot.allweather as AW
    idx = pd.date_range("2022-01-01", periods=10, freq="D", tz="UTC")
    carry = pd.DataFrame({"ret": np.zeros(10)}, index=idx)
    trend = pd.DataFrame({"ret": np.r_[np.full(5, -0.12), np.full(5, 0.01)]}, index=idx)
    port = AW.portfolio(carry, trend)
    assert port["braked"].iloc[-1]                          # deep drawdown -> trend risk halved
    assert abs(port["ret"].iloc[-1] - 0.5 * 0.5 * 0.01) < 1e-12


# ---------------------------------------------------------------- survival agent (Kelly + 8-point edge rule)
def test_survival_kelly_and_edge_rule():
    from deltabot.survival import kelly, EDGE_MIN
    k = kelly(0.5, 100.0, 99.0, 102.0)                       # 1% stop vs 2% target: costs lift break-even 33% -> ~41%
    assert 0.40 < k["p_be"] < 0.42 and k["edge"] > EDGE_MIN and 0.15 < k["f"] < 0.17
    k2 = kelly(0.45, 100.0, 99.0, 102.0)                     # above break-even but by less than 8 points -> no trade
    assert k2["edge"] < EDGE_MIN
    k3 = kelly(0.9, 100.0, 99.0, 100.1)                      # target inside the costs -> never a trade
    assert k3["edge"] < 0 and k3["f"] == 0


# ---------------------------------------------------------------- day trader (ORB + Time Machine)
def _dt_frame(days=8, breakout=True, seed=3):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-10-01", periods=days * 288, freq="5min", tz="UTC")
    p = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    df = pd.DataFrame({"open": p, "high": p * 1.001, "low": p * 0.999, "close": p, "volume": 1.0}, index=idx)
    if breakout:                                            # last day: big push up right after the opening range
        last = idx[-288].normalize() + pd.Timedelta(hours=13, minutes=45)
        df.loc[last:last + pd.Timedelta(minutes=10), ["open", "high", "low", "close"]] *= 1.02
    return df


def test_daytrader_orb_trigger_and_costs():
    from deltabot.daytrader import DTConfig, day_bars, orb_trigger, walk, cost_r
    cfg = DTConfig()
    df = _dt_frame()
    b = day_bars(df, df.index[-1], cfg)
    tr = orb_trigger(b, cfg)
    assert tr and tr["side"] == 1 and tr["slot"] == 3        # first 5-min close after the 15-min range
    assert abs(tr["stop"] - (tr["or_high"] + tr["or_low"]) / 2) < 1e-9
    assert abs(tr["target"] - (tr["entry"] + tr["risk"])) < 1e-9
    _, r = walk(b, tr["slot"] + 1, 1, tr["entry"], tr["stop"], tr["target"], tr["risk"], 1.0)
    assert r <= 1.0 - cost_r(tr["entry"], tr["risk"]) + 1e-9  # every R is after costs


def test_daytrader_time_machine_odds_add_up():
    from deltabot.daytrader import DTConfig, day_bars, orb_trigger, time_machine
    cfg = DTConfig(replays=2000)
    df = _dt_frame()
    tr = orb_trigger(day_bars(df, df.index[-1], cfg), cfg)
    tm = time_machine(df, tr, cfg)
    assert abs(tm["target_first"] + tm["stop_first"] + tm["neither"] - 1) < 1e-9
    assert tm["sessions"] >= 5 and tm["verdict"]
    assert time_machine(df, tr, cfg)["target_first"] == tm["target_first"]   # same seed -> same answer


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
