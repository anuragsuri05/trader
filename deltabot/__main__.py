"""CLI:  python -m deltabot <command>

  check      verify connection, API keys, balance, open positions
  scan       show which markets are trending right now
  options    show the option the bot would buy for each trending market
  backtest   test the strategy on past data   (--days 90 --top 10 | --symbols BTCUSD,ETHUSD
             --instrument perpetual|options)
  run        start the autonomous bot (paper or live, per BOT_MODE in .env)
  dashboard  web dashboard for paper testing  ->  http://127.0.0.1:8050
  status     show saved bot state
  flatten    close every position the bot opened
  reset      clear the kill switch / daily halt
"""
from __future__ import annotations

import argparse
import logging
import sys

from .client import DeltaClient
from .config import load_settings
from .data import load_products
from .notify import setup_logging
from .state import State


def cmd_check(s):
    c = DeltaClient(s.base_url, s.api_key, s.api_secret)
    products = load_products(c)
    print(f"Endpoint : {s.base_url}  (region={s.region}, testnet={s.testnet})")
    print(f"Mode     : {s.mode.upper()}")
    kinds = {}
    for p in products.values():
        kinds[p.kind] = kinds.get(p.kind, 0) + 1
    print(f"Markets  : {kinds}")
    print(f"Trading  : {', '.join(s.instruments)} (set INSTRUMENTS in .env)")
    if not (s.api_key and s.api_secret):
        print("API keys : not set (fine for paper mode / scan / backtest)")
        return
    data = c.wallet_balances()
    print(f"API keys : OK")
    meta = data.get("meta") or {}
    if "net_equity" in meta:
        print(f"Equity   : {meta['net_equity']}")
    for w in data["result"]:
        if float(w.get("balance") or 0):
            print(f"Wallet   : {w['asset_symbol']} balance={w['balance']} available={w.get('available_balance')}")
    pos = [p for p in c.positions() if float(p.get("size") or 0)]
    print(f"Positions: {len(pos)}")
    for p in pos:
        print(f"   {p.get('product_symbol')} size={p['size']} entry={p.get('entry_price')} upnl={p.get('unrealized_pnl')}")


def cmd_scan(s):
    from .scanner import scan
    c = DeltaClient(s.base_url)
    cands = scan(c, load_products(c), s)
    if not cands:
        print("No market is trending cleanly right now. The bot would stay flat.")
    for i, cand in enumerate(cands, 1):
        print(f"{i:>2}. {cand}")


def cmd_options(s):
    from .options import choose_option
    from .scanner import scan
    c = DeltaClient(s.base_url)
    products = load_products(c)
    with_options = {p.underlying for p in products.values() if p.kind == "option"}
    print(f"Underlyings with options: {', '.join(sorted(with_options)) or 'none'}")
    cands = [x for x in scan(c, products, s) if x.underlying in with_options]
    if not cands:
        print("None of them is trending right now, so the bot would not buy options.")
        for u in sorted(with_options):     # still show what it WOULD pick
            for d in (1, -1):
                q = choose_option(c, products, u, d, s)
                print(f"  if {u} trended {'up  ' if d > 0 else 'down'}: {q or 'no liquid contract'}")
        return
    for x in cands:
        print(f"{x}\n   -> {choose_option(c, products, x.underlying, x.direction, s) or 'no liquid contract'}")


def cmd_backtest(s, args):
    from .backtest import run_backtest
    from .scanner import liquid_universe
    c = DeltaClient(s.base_url)
    products = load_products(c)
    if args.symbols:
        symbols = [x.strip().upper() for x in args.symbols.split(",")]
    elif args.instrument == "options":
        with_options = {p.underlying for p in products.values() if p.kind == "option"}
        symbols = [sym for sym, _ in liquid_universe(c, products, s) if products[sym].underlying in with_options]
    else:
        symbols = [sym for sym, _ in liquid_universe(c, products, s)[: args.top]]
    costs = (f"model IV = realized x1.15, 3% bid/ask spread, delta {s.opt_target_delta}, {s.opt_target_dte:g}-day expiry"
             if args.instrument == "options" else
             f"fees {s.fee_rate:.3%} + slippage {s.slippage:.3%} per side")
    print(f"Backtesting {args.instrument} on {len(symbols)} markets over {args.days} days "
          f"({costs}, risk {s.risk_per_trade:.1%}/trade)\n")
    df = run_backtest(c, products, symbols, args.days, s, instrument=args.instrument)
    if df.empty:
        print("No results.")
        return
    print("\nSummary (each market simulated alone, $1000 start):")
    print(f"  markets profitable : {(df['return'] > 0).sum()}/{len(df)}")
    print(f"  average return     : {df['return'].mean():+.1%}   (buy & hold avg {df['buy_hold'].mean():+.1%})")
    print(f"  average max DD     : {df['max_dd'].mean():.1%}   worst {df['max_dd'].min():.1%}")
    print(f"  total trades       : {int(df['trades'].sum())}")
    print("\nPast results do not predict future results. Paper trade before going live.")


def cmd_allweather():
    from .allweather import run
    r = run()
    pct = lambda x: f"{x * 100:+.1f}%"
    print(f"\nAll-weather portfolio backtest {r['period']} (Binance public data, Delta fees, no tuning)")
    print(f"{'':<10} {'per year':>9} {'vol':>5} {'Sharpe':>6} {'worst drop':>10} {'+months':>8}")
    for k in ("portfolio", "carry", "trend", "btc_hold"):
        x = r[k]
        print(f"{k:<10} {pct(x['cagr']):>9} {x['vol'] * 100:>4.0f}% {x['sharpe']:>6.2f} {pct(x['max_dd']):>10} {x['pos_months']:>7.0%}")
    for y, v in r["years"].items():
        print(f"  {y}: portfolio {pct(v['portfolio'])}   BTC {pct(v['btc_hold'])}")


def cmd_train(s, args):
    from . import ml
    from .scanner import liquid_universe
    c = DeltaClient(s.base_url)
    products = load_products(c)
    symbols = [sym for sym, _ in liquid_universe(c, products, s)[: args.coins]]
    tfs = tuple(tuple(x.split(":")) for x in args.tfs.split(","))
    print(f"Building training data: {len(symbols)} coins x {args.tfs}, {args.entry} entries, last {args.days} days ...")
    data = ml.build_dataset(c, products, symbols, s, args.days, tfs=tfs, entry_mode=args.entry)
    if data.empty:
        sys.exit("no data")
    res = ml.train(data)
    r = res.report
    fmt = lambda x: f"{x['trades']:>4} trades  win {x['win_rate']:.0%}  avg {x['avg_r']:+.3f}R  total {x['total_r']:+.1f}R"
    print(f"\nthreshold chosen on validation: {r['threshold']:.2f}")
    print("VALIDATION  all signals :", fmt(r["validation_all"]))
    print("VALIDATION  model filter:", fmt(r["validation_model"]))
    print("TEST (unseen) all signals :", fmt(r["test_all"]))
    print("TEST (unseen) model filter:", fmt(r["test_model"]))
    for tf, x in r["test_by_tf"].items():
        print(f"   test {tf:<4}", fmt(x))
    for wk, x in r["test_by_week"].items():
        print(f"   week {wk}", fmt(x))
    path = ml.save(res, {"days": args.days, "symbols": symbols, "candidates": len(data), "tfs": args.tfs, "entry": args.entry})
    print(f"\nsaved {path}")
    passed = r["test_model"]["total_r"] > 0 and r["test_model"]["trades"] >= 30
    print("PASSED: profitable on unseen data" if passed else "FAILED: not profitable on unseen data - do not trade it")


def cmd_run(s):
    from .bot import Bot
    if s.is_live:
        if not (s.api_key and s.api_secret):
            sys.exit("Live mode needs DELTA_API_KEY and DELTA_API_SECRET in .env")
        if not s.live_confirmed:
            sys.exit("Live mode is locked. Set LIVE_TRADING_CONFIRMED=yes in .env once you have paper-traded "
                     "and accept that you can lose the money in this account.")
        print("*** LIVE TRADING WITH REAL MONEY ***")
    Bot(s).run()


def cmd_status(s):
    st = State.load(s.mode)
    print(f"mode={st.mode} day={st.day} day_start_equity={st.day_start_equity:.2f} peak={st.peak_equity:.2f}")
    print(f"halted={st.halted_reason or 'no'} killed={st.killed}")
    if st.mode == "paper":
        print(f"paper cash={st.paper_cash:.2f}")
    for sym, t in st.trades.items():
        print(f"  {sym} {t.side} {t.size} @ {t.entry_price:.6g}  SL {t.stop:.6g}  TP {t.target:.6g}")


def cmd_flatten(s):
    from .bot import Bot
    bot = Bot(s)
    bot.flatten()
    bot.state.halted_reason = "manual flatten; run `python -m deltabot reset` to resume"
    bot.state.save()
    print("All bot positions closed and bot halted.")


def cmd_reset(s):
    st = State.load(s.mode)
    st.killed, st.halted_reason, st.peak_equity = False, "", 0.0
    st.save()
    print("Kill switch / halt cleared. Peak equity reset.")


def main(argv=None):
    p = argparse.ArgumentParser(prog="deltabot", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("check", "scan", "options", "run", "status", "flatten", "reset"):
        sub.add_parser(name)
    sub.add_parser("allweather", help="backtest the all-weather portfolio 2020-today")
    trn = sub.add_parser("train")
    trn.add_argument("--days", type=int, default=60)
    trn.add_argument("--coins", type=int, default=15)
    trn.add_argument("--tfs", default="15m:1h,1h:4h", help="entry:regime pairs")
    trn.add_argument("--entry", choices=["market", "limit"], default="limit")
    db = sub.add_parser("dashboard")
    db.add_argument("--port", type=int, default=8050)
    db.add_argument("--resume", action="store_true", help="continue the saved paper session on start")
    bt = sub.add_parser("backtest")
    bt.add_argument("--days", type=int, default=90)
    bt.add_argument("--top", type=int, default=10, help="test the N most liquid perps")
    bt.add_argument("--symbols", default="", help="comma list, e.g. BTCUSD,ETHUSD")
    bt.add_argument("--instrument", choices=["perpetual", "options"], default="perpetual")
    args = p.parse_args(argv)

    setup_logging(logging.INFO if args.cmd == "run" else logging.WARNING)
    s = load_settings()
    if args.cmd == "allweather":
        cmd_allweather()
    elif args.cmd == "train":
        cmd_train(s, args)
    elif args.cmd == "dashboard":
        from .dashboard import serve
        serve(args.port, args.resume)
    elif args.cmd == "backtest":
        cmd_backtest(s, args)
    else:
        globals()[f"cmd_{args.cmd}"](s)


if __name__ == "__main__":
    main()
