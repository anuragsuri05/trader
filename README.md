# deltabot: an automatic trading bot for Delta Exchange (perpetuals, futures and options)

The bot runs without supervision. It:
1. **Picks markets.** Every hour it looks at the 25 most liquid perpetual futures on Delta Exchange. It keeps only the ones in a clean trend on the 4-hour chart, judged by the trend's direction, how strong it is (ADX), and how steadily it moves (efficiency ratio).
2. **Enters trades.** On the 1-hour chart, it buys when the price breaks above its recent 20-bar high in an uptrend, or sells when the price breaks below its 20-bar low in a downtrend.
3. **Protects every trade.** The stop-loss and take-profit are placed **on the exchange** as a bracket order, so they keep working even if the bot or your internet goes down.
4. **Exits.** A trade closes on its stop-loss, its take-profit, a break in the short-term trend, or after 10 days at most.
5. **Sizes and limits risk.** Each trade risks about 1% of equity. The bot holds at most 3 positions. It stops opening trades for the day after a 4% loss, and it closes everything and halts after a 20% drawdown.
6. **Records everything.** It writes a log, a trade journal (CSV), and optional Telegram alerts. Its state survives restarts.

## Instruments: perpetuals, futures and options

Set `INSTRUMENTS` in `.env`. It's a list in order of preference. For each trending market, the bot uses the first instrument type that is available.

| Type | What the bot does | Worst case per trade | Where the stop lives |
|---|---|---|---|
| `perpetual` | Long or short the perpetual future | Stop-loss (~1% of equity), plus any price gap | On the exchange (bracket order) |
| `futures` | Same, on the nearest dated future that is more than 5 days from expiry. Closes 24h before expiry. | Same | On the exchange (bracket order) |
| `options` | **Buys** a call in an uptrend or a put in a downtrend: ~14 days to expiry, delta ~0.45, only if the bid/ask spread is under 12% | **The premium paid**, capped at 3% of equity | The bot sells at the bid on −50% / +150% premium, a trend break, or 48h before expiry |

- The trend signal always comes from the **perpetual** chart (BTCUSD, ETHUSD…). Options and futures are just different ways of acting on it.
- Delta India currently lists options on **BTC, ETH and XAUT** only, and **no dated futures** (as of Oct 2026). With `INSTRUMENTS=options,perpetual`, the bot buys options for those three coins and trades perpetuals for everything else. The futures code switches on automatically if Delta lists dated futures again.
- The bot **never sells (writes) options**. Selling options can lose far more than you collect.
- `python3 -m deltabot options` shows which contract the bot would buy right now.

### Options backtest (model-based)
Delta's history does not provide clean prices for strikes that have already expired, so the options backtest *models* the prices with Black-Scholes. Implied volatility is set to 30-day realized volatility × 1.15, and it pays a 3% bid/ask spread plus fees. Results on BTC and ETH:

| Window | Buying options | Perpetuals (same signals) | Buy & hold |
|---|---|---|---|
| 1 year | **+4.6%** (BTC +5.2%, ETH +4.1%; max DD ~−3%) | +4.1% | −30.2% |
| 180 days | **+3.7%** | +1.0% | +8.5% |
| 90 days | −1.0% | 0.0% | +31.2% |

*(Corrected 2026-10-22: an earlier version of this table came from a timestamp bug that froze option time decay. These numbers are from the fixed code, BTC and ETH only.)*

The options version looks somewhat better because a trend strategy suits the options payoff: losses are capped while winners can run. But there are only 3–18 trades per market, so this is a hint, not proof. Real implied volatility and fills will differ from the model.

```bash
python3 -m deltabot backtest --instrument options --days 365
```

## Recommended: the All-weather portfolio

`python3 -m deltabot allweather` (backtest) · dashboard → Settings → Engine → **All-weather portfolio** (paper). Code: `deltabot/allweather.py` (strategy + backtest), `deltabot/allweather_live.py` (live paper engine).

Two return sources that tend to fail at different times, plus risk control:
1. **Carry (market-neutral):** long spot + short perpetual on BTC/ETH/SOL/XRP while funding pays (on above 0.005% per 8h as a 7-day average, off when it turns negative).
2. **Trend:** long/short the 20 largest coins *as of January 2020* (including later losers such as EOS, DASH and NEO, to avoid survivorship bias) in the direction of their 20/60/120-day moves. Each position is sized by its volatility, the sleeve is volatility-targeted to 20%, gross exposure is capped at 2×, and a no-trade band skips small rebalances.
3. **AI risk committee (daily):** Claude Fable 5.1 searches the web for concrete danger (hacks, insolvencies, depegs, bans) and ChatGPT reviews the numbers and Claude's headlines. They can only *reduce* risk: the lower trend scale wins, and carry stays on only if both agree.
4. **Drawdown brake:** trend risk is halved below a −20% drawdown until the portfolio is back above −10%.

Every setting was fixed in advance from published research (time-series momentum, crypto carry); nothing was tuned on the backtest.

**Backtest, Jan 2020 – Oct 2026** (Binance public daily prices + funding, Delta fees):

| | Per year | Worst drop | Sharpe | Profitable years |
|---|---|---|---|---|
| **Portfolio (balanced 50/50)** | **+11.5%** | **−18.2%** | 0.94 | 6 of 7 (2022: −3.8% while BTC −64%) |
| Conservative 70/30 | +10.8% | −9.2% | 1.39 | 6 of 7 |
| Holding BTC | +42.0% | −76.7% | 0.89 | 5 of 7 |

- **By market condition:** in bear months the portfolio averaged **+1.8%** (positive 69% of the time) while BTC averaged −19%.
- **Robustness:** moving every setting ±50% kept the result positive (+7.4% to +13.3% a year).
- **Delta's own data 2024–26:** +9.9% a year, close to Binance's +12.4% over the same dates.

**Limits:**
- Not every month is profitable: about 54% are, and the worst month was −4.8%.
- The carry backtest assumes a perfect hedge, so spot/perp basis and exchange risk are not modelled.
- Copying the portfolio exactly with Delta's minimum contract sizes needs a large account (the panel shows the amount).
- The AI committee cannot be backtested fairly; judge it only on live results.

## Day Trader pipeline (adapted from day-trader-skill)

```bash
DELTABOT_STATE_DIR=state/daytrader python3 -m deltabot dashboard --port 8052
```
`deltabot/daytrader.py`, paper only. The method of [jakeschincariol/day-trader-skill](https://github.com/Jakeschincariol/day-trader-skill) (MIT; the original is in `external/day-trader-skill`) ported from US stocks to Delta perpetuals:

`scanner → plan → Time Machine → budget → risk check → your Approve click → paper order → stop/target/flat-by alerts`

- **Session:** crypto never closes, so the session opens at the US equity open (13:30 UTC / 19:00 IST), when crypto volume peaks.
- **Setup:** the opening range is the first 15 minutes. A setup triggers on the first 5-minute close beyond the range within 2 hours. The stop is the range midpoint, the target is 1R, and positions close 6.5 hours after the open.
- **Time Machine:** 10,000 replays of the rest of the session, drawn from the coin's own 5-minute bars at the same time of day over 60 days, plus the same rule on every past day. Unlike the stock original, **every R includes Delta's fees**.
- **Honest finding:** across 15 coins over the last 60 days (780 trades), this setup won 43% of the time and averaged **−0.63R after costs**. The opening range is narrow in crypto, so fees can eat up to ~0.9R on BTC. The Time Machine usually says "coin flip" or "leans toward the stop", which is exactly what it's for.

## Survival agent ("pay for yourself or die")

```bash
DELTABOT_STATE_DIR=state/survival python3 -m deltabot dashboard --port 8051   # its own dashboard + state
```
`deltabot/survival.py`, paper only. Inspired by viral "AI bot turned $50 into $5,000" posts, but built honestly on Delta Exchange. Polymarket, which those posts use, was unreachable from this connection, and real-money prediction-market betting is likely illegal for residents of India.

- **Survival rules:** starts with $50 and pays rent (default $200/month, like an AI subscription). At $0 it **dies**. Below 50% of its start it enters **survival mode**: the AI desks pause and only the core trades.
- **Desks:**
  - **CARRY** and **TREND:** the tested all-weather core, with 60% of the money.
  - **SCOUT:** Claude Fable 5.1 searches live news and X sentiment every 30 min.
  - **QUANT:** ChatGPT reads 1m–1h data and 5-second prices every 10 min.
- **AI desk rules** (from the post): each trade must come with a probability that the target is hit before the stop. Code computes the break-even probability after costs, trades only with an **edge of 8 points or more**, and sizes by **Kelly, capped at 6% of the bankroll at risk**. A desk that loses 25% of its budget is **shut down** and its money moves to the core.
- **Telegram alerts:** set `TELEGRAM_TOKEN` and `TELEGRAM_CHAT_ID` in `.env` to get every event on your phone.
- **Expectations:** the core earns roughly +10%/year in backtests, which does **not** cover $200/month on a $50 bankroll. With that rent the agent is expected to die unless the AI desks find real edges, and our tests found none so far. Set rent to 0 to watch the desks without the rent pressure.

## Read this first: what the backtest showed

This bot **does not guarantee profit**, and no bot does. Here are honest results on Delta Exchange history, with fees and slippage included, for the 11–12 most liquid markets:

| Version | 1 year | 180 days | 90 days | Avg max drawdown |
|---|---|---|---|---|
| 15m entries / 1h trend (first version) | −14.7% | −11.3% | −5.0% | −9 to −20% |
| **1h entries / 4h trend (current default)** | **+1.1%** | **−0.7%** | **0.0%** | **−2 to −4%** |
| Buy & hold over the same period | +117% | +148% | +132% | much larger |

The current version keeps losses small, but it has **not shown a reliable edge**. Over this period, simply holding the coins beat it by a wide margin. Treat it as a well-built, safe framework for testing ideas. It is not a money machine.

Re-run the backtest yourself at any time:
```bash
python3 -m deltabot backtest --days 365 --top 12
```

## Dashboard (paper test with a ₹ balance)

```bash
python3 -m deltabot dashboard          # then open http://127.0.0.1:8050
```
- **Live paper test:** the bot runs on real Delta prices with a pretend ₹5,000 (adjustable in Settings), aiming for a ₹7,000 target. The page shows its market analysis, a chart with the indicator lines (EMA 20/50 and the breakout channel), entry/stop/target lines, every booked trade, a balance curve and an activity log.
- **Auto timeframe** (default in the dashboard, `TIMEFRAME_MODE=auto` in `.env`): for each coin the bot compares 15m/5m, 1h/15m and 4h/1h charts and trades on whichever shows the cleanest trend. It re-checks every 5 minutes. On the last 30 days, faster timeframes produced more trades but not more profit (5m: −1.1% over 200 trades; 15m: +0.2%; 1h: −0.3%).
- **Bot arena** (default engine in the dashboard): breakout bots on 3m, 5m and 15m charts (`deltabot/agents.py`) are hired with equal budget shares and trade one shared paper pool in parallel. They enter with **limit (maker) orders**: 0.02% fee instead of 0.05%, and no slippage. A limit order fills only if the price trades *through* it within 3 candles. A bot is fired if it loses 20% of its share; at each of 2 performance reviews the worst losing bot is fired; a fired bot's budget goes to the survivors.
  - Why this line-up: a walk-forward test on 12 coins showed 1-minute trading loses heavily (fees exceed the moves), and that Pullback, Mean Reversion and Momentum styles lose even at near-zero cost. With realistic limit-order fills, breakouts were the only style that came out positive: +0.2% to +0.6% per coin over 10 days and +0.6% to +1.0% over 30 days, at 1% risk per trade. That is a small edge, not a fast one. The original 5-style team is kept as `classic_team()`.
- **Claude AI bot** (`deltabot/claude_trader.py`): a 4th arena bot whose decisions come from Claude (`claude-opus-5-5` via the Anthropic API, adaptive thinking, structured output). About every 6 minutes it receives a numeric snapshot of the 15 most liquid coins (multi-timeframe trend, ADX, efficiency ratio, RSI, ATR, distance to 20-candle high/low, volume, last 12 five-minute closes, what the breakout bots see) plus its account and positions. It returns trades with a stop, target, confidence and reason. Code enforces the guardrails: stop between 1 and 6 ATR, reward/risk at least 1.5, at most 2 positions, size fixed at 1% risk × confidence (minimum 0.5), limit-order entries, and the same firing rules as the other bots. By default it runs through the local **Claude Code CLI** (`claude -p`, headless, no tools, no saved session, JSON-schema output) using your existing Claude Code login, so **no API key is needed**. Calls count toward your Claude plan's usage limits (about $0.07 per decision at API prices). If the CLI isn't installed it falls back to `ANTHROPIC_API_KEY`. There is no backtest for it: Claude's knowledge of past prices would make one misleading, so judge it only on live paper results.
- **Target mode** (Settings → Mode, **paper only**: the arena refuses to run live): bots stop waiting for tested setups and chase the target. Four *Momentum Chaser* bots (1m/3m/5m/15m) trade any coin whose fast EMA, price and last 3 candles all point the same way. Entries are market orders, with a 1.5 ATR stop and a 2.5 ATR target. Risk is 3% per trade, with up to 10 positions, up to 10× total exposure and no daily stop; the kill switch is at −50%. Claude gets a target-chasing mandate, decides every 3 minutes, and may use stops of 0.5–4 ATR with reward/risk ≥ 1. Each bot may only use its own share of the exposure allowance. **This trades far more and swings far harder; nothing about it was shown to be profitable.**
- **Target chance** (header card → **Analyse**, before you start): replays the selected mode's rule bots on the last 5–14 days of real Delta candles (fees included) to measure trade frequency and per-trade results, then simulates 20,000 sessions of your length (`deltabot/analyse.py`). It shows the chance of reaching the target, the chance of ending in profit or loss, the most likely end balance and the 90% range. Claude is not included because it has no history to replay.
- **Strategy plan** (header → **Plan & analyse**, then **Use this plan**; `deltabot/planner.py`): tests 5 strategy styles × 4 timeframes (1m/3m/5m/15m) × the 10 most liquid coins on recent real candles. For each risk level (1–8% per trade) it greedily builds the team that most raises the chance of *your* target in *your* session length, **using only the older ~70% of the data**. The headline chance comes from the newest ~30%, which the plan never saw. It also shows the optimistic chance (if the past repeated exactly), the chance of a loss or of losing 30%+, the realistic targets, and 8 backup strategies. Live (paper, market entries): a check every 5 seconds (prices, stop/target, session, firing, pace and a live re-estimated chance). A strategy is replaced by the next backup when it loses 20% of its share, loses 3 trades in a row, or is worst at a review; when the backups run out it re-plans in the background.
- **AI duel** (Settings → Engine → *AI duel*; `deltabot/ai_traders.py`, paper only): **Claude Fable 5.1** (`claude-fable-5-1` through the local Claude Code CLI and your login, no API key) and **ChatGPT** (`gpt-5.2` through the OpenAI API, `OPENAI_API_KEY` in `.env`; override with `OPENAI_MODEL`) trade one shared target with a free hand. Each picks coins, side, stop, target, risk per trade (0.5–8%), a hold time of 15s/1m/5m/15m/1h, and when to look again (30 s–10 min). Each decides on its own thread from a snapshot of 1m/5m/15m/1h data plus 5-second prices and its own recent trades. A 5-second loop enforces stops, targets and hold times. Code checks only that the stop and target are on the correct sides (stop 0.05–15% away), that the coin is affordable, the −20% firing rule per trader and the −50% account stop. OpenAI usage is billed to that key; Claude Code usage counts against your Claude plan.
- **Moonshot** (Settings → Engine → *Moonshot*; `deltabot/moonshot.py`, **paper only**): all-in on every trade with leverage, one goal: the target. Every win compounds; a liquidation ends the session at ₹0. **Analyse odds** replays the 1-minute momentum-burst entry on 5 days of real Delta data at 5×–200× leverage, with fees charged on the full leveraged position, and picks the best chance. For $69 → $7,000 in one night the best was 20× at about a **0.6% chance (≈1 in 170 nights)**, with a **99.2% chance of liquidation**. Maximum leverage is worse because fees take ~18% of the balance per trade at 100×.
- **Session goal:** the bot stops and closes everything when the paper balance reaches the target, or when the session time (default 30 minutes, set in Settings) runs out.
- **Replay on past data:** runs the same rules over the last 30–365 days on one shared ₹ account and shows every trade on the chart.
- **Connect Delta** (optional) only *reads* your real balance with an API key. The dashboard **never places real orders**.
- A ₹5,000 account (~$53) is too small for 1 BTC, SOL or BNB contract at 1% risk. The bot skips those and trades smaller-contract coins (XRP, DOGE, UNI…).

## Setup

```bash
cd trader
pip3 install -r requirements.txt
cp .env.example .env
python3 -m deltabot check      # connectivity
python3 -m deltabot scan       # which markets are trending right now
```

### The only thing you need from Delta Exchange: an API key
A bot can't log in with your password. It uses an **API key** instead:
1. Log in to Delta Exchange and go to **Account → API Keys → Create**.
2. Turn on **Trading** permission only. **Never turn on withdrawals.**
3. Whitelist the IP address of the machine or server the bot runs on.
4. Paste the key and secret into `.env` as `DELTA_API_KEY` and `DELTA_API_SECRET`.
5. Run `python3 -m deltabot check`. It should show your balance.

## Running it, in the safe order

| Step | `.env` setting | Command | How long |
|---|---|---|---|
| 1. Paper (fake money, real prices, no key needed) | `BOT_MODE=paper` | `python3 -m deltabot run` | 2–4+ weeks |
| 2. Testnet (Delta's demo account and testnet keys) | `BOT_MODE=live`, `DELTA_TESTNET=true`, `LIVE_TRADING_CONFIRMED=yes` | `python3 -m deltabot run` | 1–2 weeks |
| 3. Live, small | `BOT_MODE=live`, `DELTA_TESTNET=false`, `LIVE_TRADING_CONFIRMED=yes` | `python3 -m deltabot run` | Use money you can afford to lose |

The bot needs to run 24/7 because crypto never closes. Use a small cloud server (e.g. AWS Mumbai or Tokyo) with a static IP, and keep the bot running with `systemd`, `tmux`, or:
```bash
nohup python3 -m deltabot run > logs/stdout.log 2>&1 &
```

## Commands
| Command | What it does |
|---|---|
| `check` | Tests the connection, keys, balance and positions |
| `scan` | Lists the markets that are trending now, ranked |
| `options` | Shows the call or put the bot would buy for each trending coin |
| `backtest --days 365 --top 12` | Tests the strategy on past data (perpetuals) |
| `backtest --instrument options --days 365` | Model-based options backtest (BTC/ETH/XAUT) |
| `run` | Starts the bot |
| `status` | Shows saved state and open trades |
| `flatten` | Emergency stop: closes all the bot's positions and halts it |
| `reset` | Clears the kill switch or daily halt |

## Files
```
deltabot/
  client.py     Delta REST API + HMAC signing
  data.py       candles, product metadata
  indicators.py EMA, ATR, RSI, ADX, efficiency ratio
  strategy.py   regime + entry/exit rules (shared by bot and backtest)
  options.py    option chain selection, Black-Scholes, option sizing, dated-futures selection
  scanner.py    picks which markets to trade
  risk.py       position sizing, daily loss limit, kill switch
  broker.py     PaperBroker (simulated) + LiveBroker (perps/futures: bracket orders; options: IOC limit orders)
  bot.py        the main loop
  backtest.py   historical simulation
state/          bot state + trade journals (created at runtime)
logs/           bot.log
tests/          offline unit tests:  python3 tests/test_core.py
```

## Known limits and what to verify
- **Option orders** are IOC limit orders at the ask to buy and at the bid to sell. On a thin book a sell may fill only partly; the bot keeps the rest and retries every hour. Option fees are modelled as 0.03% of the underlying's value, capped at 10% of the premium. **[verify on Delta's fee page]**
- The **live order path has not been run against a real account yet**. The paper mode and the public data endpoints were tested against the real API. Run it on the **testnet first** to confirm the bracket order fields, fill prices, and the cancel-on-close behaviour work with your account.
- Each market is backtested on its own, so the backtest does not model the 3-position limit or the hourly re-ranking.
- Delta charges GST on top of trading fees in India, and crypto profits are taxed at 30% with no loss set-off, plus 1% TDS. The backtest ignores tax.
- Some listed perpetuals are gold (XAUT, PAXG) or tokenized stocks (e.g. TSLAX). Add them to `BLACKLIST` if you only want crypto.
