# Trading Bots — End-to-End Knowledge Base

> Reference file for building, testing, deploying, and running automated trading bots.
> Compiled: 2026-10-22. Library versions, broker API terms, fees, and regulations change often, so check anything marked **[verify]** against the official source before relying on it.
> Nothing in this file is financial advice. Most retail trading bots lose money once costs are counted. Treat every strategy as unproven until it has been tested out-of-sample and with live paper trading.

---

## Table of Contents
1. [What a Trading Bot Is](#1-what-a-trading-bot-is)
2. [Bot Types and Strategy Families](#2-bot-types-and-strategy-families)
3. [System Architecture](#3-system-architecture)
4. [Markets, Brokers and Exchange APIs](#4-markets-brokers-and-exchange-apis)
5. [Market Data](#5-market-data)
6. [Tech Stack and Libraries](#6-tech-stack-and-libraries)
7. [Strategy Development Workflow](#7-strategy-development-workflow)
8. [Backtesting: How to Do It and Where It Goes Wrong](#8-backtesting-how-to-do-it-and-where-it-goes-wrong)
9. [Performance Metrics](#9-performance-metrics)
10. [Risk Management and Position Sizing](#10-risk-management-and-position-sizing)
11. [Order Execution](#11-order-execution)
12. [Machine Learning and AI/LLM Bots](#12-machine-learning-and-aillm-bots)
13. [Infrastructure, Deployment and Ops](#13-infrastructure-deployment-and-ops)
14. [Monitoring, Logging and Alerting](#14-monitoring-logging-and-alerting)
15. [Security](#15-security)
16. [Regulation and Tax](#16-regulation-and-tax)
17. [Reference Project Structure](#17-reference-project-structure)
18. [Code Skeletons](#18-code-skeletons)
19. [Build Roadmap (Phased)](#19-build-roadmap-phased)
20. [Checklists](#20-checklists)
21. [Common Failure Modes](#21-common-failure-modes)
22. [Glossary](#22-glossary)
23. [Learning Resources](#23-learning-resources)

---

## 1. What a Trading Bot Is

A trading bot is a program that does some or all of the following without a human in the loop:
1. **Ingests data**: prices, order books, fundamentals, news, on-chain data.
2. **Generates signals**: decides to buy, sell, or hold based on rules or models.
3. **Manages risk**: sizes positions and enforces limits and stops.
4. **Executes orders** through a broker or exchange API.
5. **Tracks state**: positions, PnL, and open orders, reconciled against the broker.
6. **Reports and alerts**: logs, dashboards, notifications.

**Spectrum of automation**
| Level | Description |
|---|---|
| Alert bot | Sends signals, and a human places the trades |
| Semi-auto | Bot places trades, and a human approves or kills them |
| Fully automated | Runs on its own and stops itself through hard risk limits and kill switches |

**Holding-period spectrum**: HFT (microseconds to milliseconds), intraday (minutes to hours), swing (days to weeks), position (weeks to months), and rebalancing (monthly or quarterly).
Retail traders realistically compete at **intraday through position/rebalancing** timeframes. HFT needs colocation, FPGA hardware, and direct market access, none of which a retail account can get.

---

## 2. Bot Types and Strategy Families

### 2.1 Trend Following / Momentum
- **Idea**: assets that have been going up tend to keep going up (time-series momentum), or the recent winners keep outperforming the losers (cross-sectional momentum).
- **Signals**: moving-average crossovers (e.g. 50/200 SMA), Donchian channel breakouts (Turtle system), ADX filter, MACD, rate of change.
- **Profile**: low win rate (30–45%) offset by large winners. Loses money in choppy markets. Needs discipline through long drawdowns.

### 2.2 Mean Reversion
- **Idea**: prices that stretch far from a "fair" level tend to come back.
- **Signals**: RSI(2) extremes, Bollinger Band touches, z-score of price against its moving average, VWAP deviation, Internal Bar Strength.
- **Profile**: high win rate with occasional very large losses (left-tail risk). A stop-loss and a regime filter are essential.

### 2.3 Statistical Arbitrage / Pairs Trading
- **Idea**: two cointegrated assets, such as KO/PEP or HDFCBANK/ICICIBANK, drift apart and then reconverge.
- **Tools**: Engle-Granger or Johansen cointegration tests, Kalman filter hedge ratio, half-life from an Ornstein-Uhlenbeck fit, and spread z-score entry/exit (e.g. enter at ±2, exit at 0).
- **Risk**: cointegration can break permanently (regime change).

### 2.4 Market Making
- **Idea**: quote both a bid and an ask and earn the spread while managing inventory.
- **Models**: Avellaneda-Stoikov (inventory-skewed quotes), Guéant-Lehalle-Fernandez-Tapia.
- **Risks**: adverse selection (informed traders pick off your quotes), inventory risk, and latency. Hummingbot is the standard open-source tool for crypto.

### 2.5 Arbitrage
- **Cross-exchange**: buy on exchange A and sell on B. Withdrawal times, fees, and pre-funding on both sides eat most of the edge.
- **Triangular**: BTC→ETH→USDT→BTC within one exchange.
- **Funding-rate / basis arbitrage**: long spot and short perpetual to collect the funding rate (crypto), or cash-and-carry with futures.
- **DEX/CEX and on-chain (MEV)**: very competitive and dominated by specialized searchers.
- Pure arbitrage is mostly competed away for retail. Basis and funding trades remain workable but carry exchange counterparty risk.

### 2.6 Grid Bots
- Place a ladder of buy orders below the price and sell orders above it, so the bot profits from oscillation inside a range.
- Popular on Binance, KuCoin, Pionex, and Bybit.
- **Risk**: a trending breakout leaves you holding a falling asset (downside) or fully sold out (upside).

### 2.7 DCA (Dollar-Cost Averaging) Bots
- Buy fixed amounts at intervals, sometimes adding "safety orders" as the price falls (3Commas style).
- Martingale-style DCA, which raises order size on each drop, can wipe out an account.

### 2.8 Breakout / Opening Range Breakout (ORB)
- Trade the break of the first N minutes' high or low. This is very common in Indian intraday index options and futures (NIFTY, BANKNIFTY).

### 2.9 Options Strategies
- Short straddles/strangles (theta decay), iron condors, delta-neutral hedging, and volatility arbitrage (implied vs. realized volatility).
- You need the Greeks (via `py_vollib` or `QuantLib`), margin awareness, and tail-risk controls. Selling options can produce unlimited losses on gap moves.

### 2.10 Event / News / Sentiment Driven
- Earnings, macro releases (CPI, FOMC, RBI policy), and news headlines through NLP or LLM classification.
- Latency matters a great deal, and retail traders arrive late to most news.

### 2.11 Factor / Portfolio Rebalancing
- Rank a universe by factors (value, momentum, quality, low volatility, size) and rebalance monthly.
- This is the most robust family for retail. Turnover and costs are low, and the academic evidence behind it is long.

### 2.12 Copy / Signal Bots
- Execute signals from TradingView webhooks, Telegram channels, or other traders.
- The quality of the signal source is the entire edge. Be wary of paid signal groups.

### 2.13 Execution Algorithms (not alpha, but bots)
- **TWAP**: time-weighted slicing. **VWAP**: volume-weighted. **POV**: percentage of volume. **Implementation Shortfall** (Almgren-Chriss).
- Used to reduce market impact on large orders.

---

## 3. System Architecture

### 3.1 Core Components
```
          ┌────────────────────┐
          │   Data Feeds        │  REST (historical) + WebSocket (live)
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Data Handler / Bars │  normalize, resample, store
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Strategy / Signals  │  indicators, models → target positions
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Portfolio & Risk    │  sizing, limits, exposure, kill switch
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Execution / OMS     │  order routing, retries, fills, slippage
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Broker / Exchange   │
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ State Store + Recon │  DB of orders/fills/positions; reconcile with broker
          └─────────┬──────────┘
                    ▼
          ┌────────────────────┐
          │ Monitoring & Alerts │  logs, metrics, Telegram/Slack/email
          └────────────────────┘
```

### 3.2 Design Patterns
- **Event-driven architecture**: `MarketEvent → SignalEvent → OrderEvent → FillEvent`. The same code path runs in backtest and live, which removes a whole class of backtest/live discrepancies. This is the most important design decision in the system.
- **Broker abstraction layer**: one interface (`place_order`, `cancel`, `positions`, `balance`) with an adapter per broker (paper, Zerodha, Alpaca, Binance). Swapping paper for live then means changing a config value.
- **Target-position model**: the strategy outputs desired positions, and the execution layer works out the orders needed to get there. This makes the system self-healing after crashes.
- **Idempotency**: use a client order ID (`clientOrderId` / `tag`) on every order so a retry never creates a duplicate order.
- **Reconciliation loop**: compare local state with broker state every N seconds. Treat the broker as the source of truth.
- **State machine for orders**: `NEW → SUBMITTED → PARTIALLY_FILLED → FILLED / CANCELED / REJECTED / EXPIRED`.
- **Config-driven**: symbols, parameters, and risk limits live in YAML/TOML or env vars, not in code.
- **Separation of research and production**: notebooks for research, and a tested package for production.

### 3.3 Sync vs Async
- Polling REST every N seconds is simple and fine for timeframes of 1 minute or longer.
- `asyncio` with WebSockets suits real-time and multi-symbol bots. Use `websockets`, `aiohttp`, or `ccxt.pro`.
- Rate limits apply to every broker, so build a token-bucket limiter.

---

## 4. Markets, Brokers and Exchange APIs

### 4.1 India (NSE/BSE: equities, F&O, currency, commodities on MCX)
| Broker | API | Notes |
|---|---|---|
| Zerodha | **Kite Connect** (`kiteconnect` Python) | Most popular. Paid monthly subscription **[verify current price]**. WebSocket ticker. Daily login/access token required, so plan for a daily re-auth flow. |
| Upstox | Upstox API v2/v3 | Free API access **[verify]**. Python SDK, WebSocket (protobuf). |
| Angel One | SmartAPI (`smartapi-python`) | Free. TOTP login. |
| Dhan | DhanHQ API (`dhanhq`) | Free trading API, data API may be paid **[verify]**. Good options data. |
| Fyers | Fyers API v3 | Free. Popular with algo traders. |
| ICICI Direct | Breeze API | Free. Some historical data. |
| Kotak Neo, 5paisa, Shoonya (Finvasia), Alice Blue | Various | Shoonya was known for zero brokerage plus free API **[verify]**. |

**India-specific constraints**
- **SEBI retail algo framework** (circular issued Feb 2025, phased implementation through 2025–26 **[verify current status]**): every algo order must carry an exchange-provided **algo ID**. Brokers act as principals and must register algos with the exchange. There is an orders-per-second threshold above which a retail strategy must be registered. API access must be restricted by **static IP whitelisting**, and API keys must be unique per client. "White-box" (rules disclosed) and "black-box" algos (which require an RA-registered provider) are treated differently. Open APIs without these controls are being phased out. **Read your broker's current implementation notes before building.**
- Market hours: 09:15–15:30 IST (equity and F&O). Pre-open session 09:00–09:15.
- Weekly index option expiries were rationalized by SEBI/exchanges in 2024–25. Contract lot sizes and the expiry days of NIFTY and SENSEX have changed several times **[verify current expiry calendar]**.
- Costs: STT (raised on F&O in Oct 2024), exchange transaction charges, SEBI fee, stamp duty, GST, and brokerage. **Model all of them in backtests.** On small intraday options trades these costs often exceed the gross edge.
- Peak margin rules mean intraday leverage is limited.
- Circuit limits (price bands) apply to individual stocks, and F&O stocks have dynamic bands.

### 4.2 United States (equities and options)
| Broker | API | Notes |
|---|---|---|
| Alpaca | REST + WebSocket, `alpaca-py` | Commission-free stocks and crypto. Excellent paper trading. Best starter API. |
| Interactive Brokers | TWS API / Client Portal API, `ib_async` (successor to `ib_insync`) | Global markets. The pro choice. Steeper learning curve. |
| Tradier | REST | Options-friendly. |
| Charles Schwab | Schwab Trader API (replaced TD Ameritrade API), `schwab-py` | |
| tastytrade | REST/streaming | Options. |
- **Pattern Day Trader rule**: accounts under $25k are limited to 3 day trades in 5 days on margin accounts. FINRA approved changes to PDT in 2025–26 **[verify current status]**.
- Market hours: 09:30–16:00 ET, with pre- and post-market sessions.

### 4.3 Crypto (24/7)
| Exchange | Notes |
|---|---|
| Binance | Deepest liquidity. Spot plus USDⓈ-M and COIN-M futures. Testnet available. Restricted in some jurisdictions. |
| Coinbase (Advanced Trade API) | US-regulated. |
| Kraken | Spot and futures. Good reputation. |
| Bybit, OKX, Bitget, KuCoin | Derivatives-heavy. Check availability in your jurisdiction. |
| Hyperliquid, dYdX | On-chain perpetuals DEXs. |
| India: CoinDCX, WazirX, Mudrex, CoinSwitch | Must be FIU-IND registered. Indian tax rules apply (§16.2). |
- **CCXT** is a unified API for 100+ exchanges (`pip install ccxt`). The WebSocket streams that used to be in `ccxt.pro` are now included in the open-source package.
- Crypto-specific risks: exchange insolvency (FTX in 2022), API key theft, liquidation cascades, funding-rate swings, and weekend illiquidity.

### 4.4 Forex / CFDs
- **OANDA** (v20 REST API, `oandapyV20`), **MetaTrader 5** (`MetaTrader5` Python package, Windows only; Expert Advisors are written in MQL5), **cTrader** (cAlgo in C#), and IG.
- For Indian residents, trading forex outside INR pairs on recognized exchanges, or through offshore CFD brokers, violates FEMA. The RBI publishes an "Alert List" of unauthorized platforms. **Indian residents should only trade INR currency pairs on NSE/BSE/MCX-SX.**

### 4.5 No-code / Low-code Platforms
- **TradingView** (Pine Script plus webhook alerts to a broker bridge), **Streak** (Zerodha), **Tradetron**, **AlgoTest**, **QuantConnect** (LEAN, cloud), **3Commas**, **Cryptohopper**, **Pionex** (built-in grid bots), **Composer** (US).
- Useful for prototyping. You give up control and transparency, and you pay subscription costs.

---

## 5. Market Data

### 5.1 Data Types
- **OHLCV bars** (1m, 5m, 1h, 1d), **ticks/trades**, **Level 1 quotes** (best bid/ask), **Level 2 / order book depth**, **options chains** (with IV and Greeks), **fundamentals**, **corporate actions** (splits, dividends, bonuses), **news/sentiment**, **alternative data**, and **on-chain data** (crypto).

### 5.2 Sources
| Source | Coverage | Cost |
|---|---|---|
| `yfinance` | Global EOD and limited intraday | Free, unofficial, breaks often, research only |
| Broker APIs (Kite, Upstox, Dhan, Alpaca, IBKR) | Their markets | Free or included |
| NSE bhavcopy / `jugaad-data`, `nsepython` | India EOD, F&O | Free (scraping, so fragile) |
| CCXT `fetch_ohlcv` | Crypto | Free |
| Binance public data (data.binance.vision) | Crypto tick and klines bulk | Free |
| Polygon.io (rebranded Massive in 2025 **[verify]**), Tiingo, Alpha Vantage, Twelve Data, EODHD, Financial Modeling Prep | US/global | Freemium |
| Databento, Norgate Data, QuantQuote, CSI | Institutional-grade, survivorship-free | Paid |
| TrueData, Global Datafeeds, Accelpix | India real-time and historical tick data | Paid |
| Tardis.dev, Kaiko, CoinAPI | Crypto tick and order book history | Paid |
| FRED | Macro | Free |

### 5.3 Data Quality Rules
- **Adjust for splits and dividends**, but know which adjustment method your source uses.
- **Survivorship bias**: include delisted symbols. A backtest on today's NIFTY 50 constituents over 10 years is badly biased.
- **Point-in-time data**: fundamentals must be the values *as known then*, not restated figures.
- **Timezones**: store everything in UTC and convert only at display time. Mark bar timestamps as either open time or close time, consistently.
- **Missing bars and bad ticks**: detect gaps and spikes, and forward-fill only where that is appropriate.
- **Storage**: Parquet files for research (fast and columnar). For live data use **TimescaleDB/PostgreSQL**, **QuestDB**, **DuckDB** (embedded analytics), **ArcticDB** (pandas-native), or **InfluxDB**. Redis works for hot state.

---

## 6. Tech Stack and Libraries

### 6.1 Language Choice
- **Python**: the default for research and retail bots. Huge ecosystem, fast enough for anything at 1 second or slower.
- **Rust / C++**: low latency and the core of high-performance engines (NautilusTrader's core is Rust).
- **C#**: QuantConnect LEAN and cTrader. **MQL5**: MetaTrader. **Pine Script**: TradingView only. **Go/Java**: some production systems.

### 6.2 Python Libraries
**Data and numerics**: `pandas`, `polars` (faster), `numpy`, `scipy`, `numba` (JIT for loops), `pyarrow`, `duckdb`.

**Technical indicators**
- `TA-Lib` (C library plus wrapper; wheels now available on pip), `pandas-ta` (pure Python; check maintenance status **[verify]**), `ta`, `tulipy`, `finta`.

**Backtesting frameworks**
| Framework | Style | Notes |
|---|---|---|
| **vectorbt** / vectorbt PRO | Vectorized | Extremely fast parameter sweeps. Great for research. |
| **backtrader** | Event-driven | Mature with lots of tutorials. Largely unmaintained but stable. |
| **backtesting.py** | Event-driven (simple) | Very easy, good plots, single asset. |
| **NautilusTrader** | Event-driven, Rust core | Production-grade. Same code for backtest and live. Many adapters (Binance, IBKR, Bybit, Databento…). |
| **QuantConnect LEAN** | Event-driven (C#/Python) | Cloud or local, multi-asset, big data library. |
| **zipline-reloaded** | Event-driven | Quantopian legacy. Pairs with `pyfolio-reloaded` and `alphalens-reloaded`. |
| **bt** | Portfolio/rebalance | Good for allocation strategies. |
| **Freqtrade** | Crypto bot framework | Backtest, hyperopt, dry-run, live, Telegram, FreqAI (ML). Excellent. |
| **Hummingbot** | Crypto market making / arbitrage | CEX plus DEX connectors. |
| **Jesse** | Crypto | Clean API. |
| **OctoBot** | Crypto | UI-based. |

**Broker SDKs**: `kiteconnect`, `upstox-python-sdk`, `smartapi-python`, `dhanhq`, `fyers-apiv3`, `breeze-connect`, `alpaca-py`, `ib_async`, `ccxt`, `python-binance`, `oandapyV20`, `MetaTrader5`.

**Analytics and reports**: `quantstats` (tear sheets), `pyfolio-reloaded`, `empyrical-reloaded`, `alphalens-reloaded`.

**Statistics and ML**: `statsmodels` (cointegration, ARIMA, OLS), `arch` (GARCH), `scikit-learn`, `lightgbm`, `xgboost`, `catboost`, `pytorch`, `optuna` (hyperparameter search), `mlfinlab` (Lopez de Prado techniques; now commercial), `sktime`, `hmmlearn` (regime detection).

**Options**: `py_vollib`, `QuantLib`, `mibian`, `opstrat`.

**Portfolio optimization**: `PyPortfolioOpt`, `riskfolio-lib`, `cvxpy`.

**Infrastructure**: `asyncio`, `websockets`, `aiohttp`, `httpx`, `APScheduler`, `pydantic` (config/validation), `SQLAlchemy`, `loguru`/`structlog`, `python-telegram-bot`, `tenacity` (retries), `python-dotenv`.

**Visualization and UI**: `plotly`, `matplotlib`, `mplfinance`, `lightweight-charts` (TradingView's open-source charts), `streamlit`/`dash` for dashboards, Grafana.

---

## 7. Strategy Development Workflow

1. **Hypothesis**: a reason the edge should exist, such as a behavioral bias, a structural flow, a risk premium, or a liquidity provision. "The backtest looks good" is not a reason.
2. **Data**: collect clean, survivorship-free, point-in-time data.
3. **Exploration**: plot it, look at distributions, check that the effect exists in raw statistics.
4. **Prototype backtest**: vectorized and fast, with realistic costs.
5. **Robustness**: parameter sensitivity, walk-forward, out-of-sample (OOS), multiple markets and timeframes, Monte Carlo.
6. **Event-driven backtest**: the same engine you will run live.
7. **Paper trade**: at least 1–3 months, or 50–100+ trades. Compare paper fills against the backtest's assumed fills.
8. **Small live**: real money at 5–10% of target size. Measure slippage and latency.
9. **Scale up gradually** while monitoring for decay.
10. **Retire** the strategy when live performance falls outside its expected distribution, for example when the drawdown exceeds 1.5–2× the backtest's maximum.

**Strong signs of a real edge**: it works across related instruments, it is insensitive to small parameter changes, it has an economic rationale, it shows a reasonable number of trades (more than 100), and the OOS results are close to the in-sample results.

---

## 8. Backtesting: How to Do It and Where It Goes Wrong

### 8.1 The Biases
| Bias | What it is | Fix |
|---|---|---|
| **Look-ahead bias** | Using information not available at the decision time, such as the close of the same bar, future-adjusted data, or `shift` errors | Signal on bar *t* and execute at the open of *t+1*. Use `.shift(1)` and audit carefully. |
| **Survivorship bias** | Only testing on stocks that exist today | Use survivorship-free datasets that include delisted stocks |
| **Overfitting / data snooping** | Tuning until it fits the noise | OOS hold-out, walk-forward, fewer parameters, Deflated Sharpe Ratio, combinatorial purged CV |
| **Selection bias** | Reporting the best of 500 tried variants | Track the number of trials and adjust (Bailey & López de Prado) |
| **Transaction cost neglect** | Ignoring fees, spread, slippage, taxes | Model them all, and stress-test at 2× costs |
| **Liquidity / market impact** | Assuming you can fill any size | Cap trade size at a small % of ADV. Model impact. |
| **Fill assumptions** | Assuming limit orders fill when price touches the level | Require price to trade *through* the limit, or model queue position |
| **Bar-internal ambiguity** | Stop and target both inside one bar: which hit first? | Assume the worst case, or use lower-timeframe data |
| **Regime bias** | Testing only in a bull market | Cover multiple regimes (2008, 2020, 2022 for crypto) |
| **Corporate actions / rollovers** | Futures roll gaps, dividends | Use continuous back-adjusted futures and handle dividends |
| **Short-selling assumptions** | Assuming every stock can be shorted for free | Model borrow availability and fees. In India, intraday short or F&O only. |

### 8.2 Validation Methods
- **Train / validation / test split** by time, never randomly shuffled.
- **Walk-forward optimization**: optimize on a rolling window and test on the following window, then roll forward.
- **Purged and embargoed K-fold CV** (López de Prado) for ML, which prevents label leakage across overlapping windows.
- **Monte Carlo**: resample the trade sequence or returns to get a distribution of drawdowns and outcomes.
- **Parameter heatmaps**: you want a broad plateau, not a sharp peak.
- **Deflated Sharpe Ratio / Probability of Backtest Overfitting (PBO)**.
- **Noise test**: add noise to the prices and see whether the strategy survives.

### 8.3 Realistic Cost Model (minimum)
`net_return = gross − commission − spread/2 × 2 − slippage − taxes/levies − funding/borrow`
- Crypto: maker/taker fees of roughly 0.02–0.1%, plus funding on perpetuals.
- India F&O: brokerage, STT, exchange charges, GST, SEBI fee, and stamp duty. Use your broker's brokerage calculator to calibrate.
- Slippage: start at 1–2 ticks for liquid instruments and more for illiquid ones. Calibrate against paper and live fills.

---

## 9. Performance Metrics

| Metric | Formula / meaning | Rough "good" for retail |
|---|---|---|
| CAGR | Compound annual growth rate | Compare against buy-and-hold |
| Volatility | Std of returns × √252 (daily) or × √365 (crypto) | — |
| **Sharpe ratio** | (Return − risk-free) / volatility, annualized | >1 good, >2 suspicious unless HF/MM |
| **Sortino ratio** | Uses downside deviation only | >1.5 |
| **Max drawdown (MDD)** | Largest peak-to-trough decline | Must be survivable psychologically and financially |
| **Calmar ratio** | CAGR / MDD | >1 |
| Win rate | % profitable trades | Meaningless alone |
| Profit factor | Gross profit / gross loss | >1.3 after costs |
| Expectancy | (Win% × avg win) − (Loss% × avg loss) | >0 after costs |
| Avg win / avg loss (payoff ratio) | — | — |
| Exposure % | Time in the market | — |
| Turnover | Trading volume / capital | Lower means more robust to costs |
| Number of trades | Statistical significance | >100 preferred |
| Ulcer index, time to recovery | Drawdown pain | — |
| Alpha / beta | Against a benchmark | — |
| Tail metrics | Skew, kurtosis, VaR, CVaR/Expected Shortfall | Watch negative skew |

---

## 10. Risk Management and Position Sizing

> More bots die from poor risk management than from bad signals.

### 10.1 Position Sizing Methods
- **Fixed fractional**: risk X% of equity per trade, typically 0.25–1%.
  `qty = (equity × risk_pct) / (entry − stop)`
- **Volatility targeting / ATR sizing**: `qty = (equity × risk_pct) / (k × ATR)`. This equalizes risk across assets.
- **Kelly criterion**: `f* = (bp − q) / b`. Use **fractional Kelly** (¼–½), because full Kelly is far too aggressive given estimation error.
- **Risk parity / inverse-volatility weighting** for portfolios.
- **Fixed notional**: simple, but it ignores volatility.

### 10.2 Limits to Hard-Code
- Max risk per trade (e.g. 1% of equity).
- Max position size per symbol (e.g. 10–20% of equity).
- Max total gross and net exposure, and max leverage.
- **Daily loss limit** (e.g. −2% to −3%), after which the bot stops trading for the day.
- **Max drawdown kill switch** (e.g. −15% to −20% from peak): stop and alert a human.
- Max orders per minute, a protection against runaway loops.
- Max open positions, and sector or correlation caps.
- **Fat-finger checks**: reject orders above X notional or with a price more than Y% from the last traded price.
- No trading during the first and last N minutes, or around known events (optional).

### 10.3 Stops and Exits
- Hard stop-loss: price-based, ATR-based, or volatility-based.
- Trailing stop, chandelier exit, time stop (exit after N bars).
- Take-profit and scaling out.
- **Place stops on the exchange/broker side** where possible (bracket/OCO/GTT orders) so they still work if the bot crashes.

### 10.4 Systemic Risks
- Gap risk (overnight, weekends, circuit halts), liquidity evaporation, exchange or broker outage, API changes, correlation spikes in crises, leverage liquidation, and counterparty failure.

---

## 11. Order Execution

### 11.1 Order Types
- **Market**: immediate fill, unknown price. **Limit**: known price, uncertain fill.
- **Stop / stop-limit**, **trailing stop**.
- **Bracket / OCO / cover orders**: entry plus stop plus target.
- **GTT (Good Till Triggered)** in India, **GTC/IOC/FOK/Day** time-in-force.
- **Post-only** for makers (crypto), **reduce-only** for futures.
- **Iceberg** orders.

### 11.2 Execution Best Practices
- Use limit orders (passive or marketable limits) to control slippage.
- Always set and track a `client_order_id`.
- Handle **partial fills**, **rejections** (margin, price band, freeze quantity), and **timeouts**. On a timeout, query the order status before retrying, so a retry never double-fills.
- Respect **rate limits** with exponential backoff (`tenacity`).
- Handle **tick size** and **lot size** rounding, and **freeze quantity** limits (NSE F&O caps quantity per order, so large orders must be sliced).
- Handle **WebSocket reconnects** and resubscribe. Detect stale data, for example no tick for N seconds.
- Synchronize your clock with NTP.
- Record **expected vs. actual fill price** on every order to measure slippage.

### 11.3 Latency (for context)
- Retail Python bot through a REST API: about 50–500 ms round trip.
- Cloud VM in the same region as the exchange (e.g. AWS Tokyo for Binance, AWS Mumbai `ap-south-1` for Indian brokers): lower and more stable.
- Colocation: microseconds, institutional only.

---

## 12. Machine Learning and AI/LLM Bots

### 12.1 Where ML Helps
- Feature-rich cross-sectional ranking (gradient-boosted trees such as LightGBM are the workhorse).
- Regime detection (HMMs, clustering) to switch strategies or reduce exposure.
- Meta-labeling (López de Prado): a primary rule-based model gives direction, and an ML model decides whether to take the trade and how big.
- Volatility forecasting (GARCH, HAR, ML), which helps sizing more than direction.
- Execution optimization and fill probability.
- NLP sentiment from news, filings, and social media.

### 12.2 Pitfalls Specific to ML
- Financial data has a very low signal-to-noise ratio, is non-stationary, and has few independent samples.
- **Leakage**: features built with future data, scaling fit on the whole dataset, or overlapping labels. Use purged CV.
- Use **fractionally differentiated** features to keep memory while making the series stationary.
- Labeling: the **triple-barrier method** beats fixed-horizon labels.
- Deep learning (LSTM, Transformers) on raw prices rarely beats simple baselines. Be skeptical of YouTube "95% accuracy" claims, which are almost always leakage.
- **Reinforcement learning** (FinRL, stable-baselines3) is research-grade and overfits easily.

### 12.3 LLM-Powered Bots (2024–2026 trend)
- Uses: news and earnings-call summarization and sentiment, parsing filings, converting natural-language strategies into code, agentic research assistants, and trade journaling and analysis.
- Risks: hallucination, latency, cost, non-determinism, and prompt injection from scraped news text. **An LLM should not have unchecked authority to place orders.** Route its output through deterministic risk checks and limits.
- Pattern: the LLM produces a structured score (JSON with a schema). That score becomes one feature among many, then passes through the risk engine before reaching execution.
- Backtesting LLM signals is hard, because the model's training data leaks the future (look-ahead through pretraining). Only forward-test LLM signals on events after the model's training cutoff.

---

## 13. Infrastructure, Deployment and Ops

### 13.1 Where to Run
- **Local machine**: fine for research and paper trading. Not reliable for live trading because of sleep, power loss, and network drops.
- **VPS / cloud VM**: AWS EC2 (Mumbai `ap-south-1` for India, `ap-northeast-1` Tokyo for Binance), GCP, Azure, DigitalOcean, Hetzner, Oracle Cloud free tier. A small VM (2 vCPU / 4 GB) is enough for most bots.
- **India**: the SEBI framework requires a **static IP** registered with your broker, so use a cloud VM with an Elastic/static IP.
- **Windows VPS** is required for MetaTrader 5.

### 13.2 Deployment Practices
- **Docker** plus `docker compose` with services for the bot, the DB, Grafana, and Redis.
- Process supervision: `systemd`, `supervisord`, or Docker `restart: unless-stopped`.
- Schedulers: `cron`, `APScheduler`, or systemd timers for daily login, data downloads, and start/stop around market hours.
- **Environment separation**: `backtest`, `paper`, and `live` configs, with live gated behind an explicit flag.
- CI: GitHub Actions running tests and lint. Never deploy untested changes during market hours.
- Version control everything except secrets. Tag the exact commit that is running live.
- **Graceful shutdown**: on SIGTERM, stop new orders, optionally cancel open orders, persist state, then exit.
- **Restart safety**: on boot, load state from the DB, reconcile with the broker, then resume.
- Automate the daily auth flow (e.g. Kite access token, TOTP) where the broker's terms allow it.

### 13.3 Testing
- Unit tests for indicators, sizing, and the risk checks (the most important tests).
- Integration tests against a broker **sandbox/testnet** (Binance testnet, Alpaca paper, IBKR paper).
- Replay tests that feed recorded live data through the strategy and compare its decisions.
- Chaos tests that kill the process mid-order, drop the WebSocket, or return API errors.

---

## 14. Monitoring, Logging and Alerting

- **Structured logs** (JSON) for every signal, order, fill, error, and risk breach. Keep them for audit purposes, which SEBI also requires brokers and algos to maintain.
- **Metrics**: equity, PnL (realized and unrealized), positions, exposure, latency, order reject rate, slippage, heartbeat. Use Prometheus plus Grafana, or a simple Streamlit dashboard.
- **Alerts**: Telegram bot (most popular), Discord, Slack, email, or SMS/PagerDuty for critical failures.
  - Alert on: errors, rejected orders, risk limit hits, kill switch, no heartbeat for N minutes, data staleness, reconciliation mismatch, and a daily PnL summary.
- **Heartbeat / dead-man's switch**: an external monitor (e.g. healthchecks.io, Uptime Kuma) alerts if the bot stops pinging.
- **Remote kill switch**: a Telegram command or file flag that flattens positions and halts the bot.
- **Trade journal**: export trades for later analysis and tax reporting.

---

## 15. Security

- **Never commit API keys.** Use `.env` plus `.gitignore`, or a secrets manager (AWS Secrets Manager, Vault, 1Password CLI).
- **Least-privilege API keys**: enable trading but **disable withdrawals** (crypto), and IP-whitelist the key.
- 2FA on every account. Keep TOTP secrets secure if you automate login.
- Harden the server: SSH keys only, no root login, firewall (ufw), fail2ban, automatic security updates.
- Pin dependency versions and beware typosquatted PyPI packages (fake `ccxt`/`binance` packages have been used to steal keys).
- Do not run untrusted "free bot" code or strategies from Telegram or YouTube. Scams and key-stealers are common.
- **Scams to avoid**: guaranteed-return bots, "AI arbitrage" platforms that hold your funds, MEV-bot YouTube scams (copy-paste smart contracts that drain your wallet), and Ponzi-style "trading bot" investment schemes.

---

## 16. Regulation and Tax

### 16.1 India **[verify all against current SEBI/IT Dept notices]**
- **SEBI retail algo framework** (2025): see §4.1. Key points are broker-mediated API access, static IP, an algo ID tag on orders, registration above an OPS threshold, and research-analyst registration for those selling black-box algos.
- Selling strategies or signals to the public may require **SEBI Research Analyst / Investment Adviser** registration.
- **Tax**:
  - Intraday equity is a **speculative business income**, taxed at slab rates.
  - F&O is **non-speculative business income**, taxed at slab rates. A **tax audit** may apply depending on turnover and profit; F&O turnover is calculated in a particular way (sum of absolute profits and losses).
  - Delivery equity: STCG and LTCG rates were revised in the July 2024 budget (STCG 20%, LTCG 12.5% above the ₹1.25L exemption) **[verify current]**.
  - **Crypto (VDA)**: flat 30% plus cess, **no loss set-off**, and 1% TDS on transfers (Sec 194S).
  - Keep records of every trade. Brokers provide tax P&L reports.
- Forex: see §4.4 (FEMA restrictions).

### 16.2 United States
- SEC/FINRA rules: the PDT rule, wash-sale rule (applies to securities; crypto treatment may change **[verify]**), and Form 1099-B / 1099-DA (crypto, from 2025).
- Section 1256 contracts (futures, index options) get 60/40 long-term/short-term treatment.
- Trader Tax Status (TTS) and the Mark-to-market election (Sec 475(f)).

### 16.3 General
- Market manipulation is illegal everywhere: spoofing, layering, wash trading, pump-and-dump. A bot can do these by accident (for example by placing and canceling orders rapidly), so design to avoid it.
- Exchange terms of service may prohibit certain automation or scraping.

---

## 17. Reference Project Structure

```
trader/
├── README.md
├── TRADING_BOTS_KNOWLEDGE.md      ← this file
├── pyproject.toml                 # deps pinned
├── .env.example                   # API_KEY=, API_SECRET=, TELEGRAM_TOKEN=
├── .gitignore                     # .env, data/, logs/, *.db
├── config/
│   ├── base.yaml                  # symbols, timeframes, risk limits
│   ├── paper.yaml
│   └── live.yaml
├── data/                          # parquet cache (gitignored)
├── notebooks/                     # research only
├── src/trader/
│   ├── __init__.py
│   ├── config.py                  # pydantic settings
│   ├── core/
│   │   ├── events.py              # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   │   ├── engine.py              # event loop (shared by backtest + live)
│   │   └── clock.py
│   ├── data/
│   │   ├── base.py                # DataFeed interface
│   │   ├── historical.py          # download + cache
│   │   └── live.py                # websocket feed
│   ├── strategies/
│   │   ├── base.py                # Strategy interface → target positions
│   │   ├── sma_cross.py
│   │   └── rsi_mean_reversion.py
│   ├── risk/
│   │   ├── sizing.py
│   │   └── limits.py              # pre-trade checks, kill switch
│   ├── execution/
│   │   ├── broker_base.py         # Broker interface
│   │   ├── paper_broker.py        # simulated fills + costs
│   │   ├── zerodha_broker.py
│   │   ├── alpaca_broker.py
│   │   └── ccxt_broker.py
│   ├── portfolio/
│   │   └── portfolio.py           # positions, PnL, reconciliation
│   ├── storage/
│   │   └── db.py                  # SQLite/Postgres: orders, fills, equity
│   ├── monitoring/
│   │   ├── logger.py
│   │   └── telegram.py
│   ├── backtest/
│   │   ├── runner.py
│   │   └── costs.py
│   └── main.py                    # CLI: backtest | paper | live
├── tests/
│   ├── test_indicators.py
│   ├── test_sizing.py
│   └── test_risk_limits.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── scripts/
    ├── download_data.py
    └── daily_login.py
```

---

## 18. Code Skeletons

### 18.1 Broker Interface
```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"

@dataclass
class Order:
    symbol: str
    side: Side
    qty: float
    order_type: str = "LIMIT"     # MARKET | LIMIT | SL | SL-M
    limit_price: float | None = None
    stop_price: float | None = None
    client_id: str = ""           # idempotency key

class Broker(ABC):
    @abstractmethod
    def place_order(self, order: Order) -> str: ...
    @abstractmethod
    def cancel_order(self, order_id: str) -> None: ...
    @abstractmethod
    def get_order(self, order_id: str) -> dict: ...
    @abstractmethod
    def positions(self) -> dict[str, float]: ...
    @abstractmethod
    def balance(self) -> float: ...
```

### 18.2 Strategy Interface (target positions)
```python
import pandas as pd

class Strategy:
    def __init__(self, params: dict):
        self.params = params

    def target_weights(self, bars: dict[str, pd.DataFrame]) -> dict[str, float]:
        """Return desired weight per symbol in [-1, 1]. Must only use data up to the last closed bar."""
        raise NotImplementedError

class SmaCross(Strategy):
    def target_weights(self, bars):
        out = {}
        for sym, df in bars.items():
            fast = df["close"].rolling(self.params["fast"]).mean()
            slow = df["close"].rolling(self.params["slow"]).mean()
            out[sym] = 1.0 if fast.iloc[-1] > slow.iloc[-1] else 0.0
        return out
```

### 18.3 Vectorized Backtest (no look-ahead)
```python
import numpy as np
import pandas as pd

def backtest_sma(df: pd.DataFrame, fast=20, slow=100, cost_bps=10):
    px = df["close"]
    signal = (px.rolling(fast).mean() > px.rolling(slow).mean()).astype(float)
    position = signal.shift(1).fillna(0)            # act on NEXT bar → no look-ahead
    ret = px.pct_change().fillna(0)
    turnover = position.diff().abs().fillna(0)
    strat = position * ret - turnover * cost_bps / 1e4
    equity = (1 + strat).cumprod()
    dd = equity / equity.cummax() - 1
    ann = 252
    sharpe = np.sqrt(ann) * strat.mean() / strat.std()
    cagr = equity.iloc[-1] ** (ann / len(equity)) - 1
    return {"CAGR": cagr, "Sharpe": sharpe, "MaxDD": dd.min(), "equity": equity}
```

### 18.4 Position Sizing and Pre-Trade Risk Check
```python
def size_by_risk(equity, entry, stop, risk_pct=0.01, lot=1):
    risk_per_unit = abs(entry - stop)
    if risk_per_unit <= 0:
        return 0
    qty = (equity * risk_pct) / risk_per_unit
    return int(qty // lot) * lot

class RiskManager:
    def __init__(self, max_pos_pct=0.2, daily_loss_limit=0.03, max_dd=0.15, max_order_notional=None):
        self.max_pos_pct = max_pos_pct
        self.daily_loss_limit = daily_loss_limit
        self.max_dd = max_dd
        self.max_order_notional = max_order_notional
        self.halted = False

    def check(self, order_notional, equity, start_of_day_equity, peak_equity, last_price, order_price):
        if self.halted:
            return False, "halted"
        if equity <= start_of_day_equity * (1 - self.daily_loss_limit):
            self.halted = True; return False, "daily loss limit"
        if equity <= peak_equity * (1 - self.max_dd):
            self.halted = True; return False, "max drawdown kill switch"
        if order_notional > equity * self.max_pos_pct:
            return False, "position too large"
        if self.max_order_notional and order_notional > self.max_order_notional:
            return False, "fat finger notional"
        if order_price and abs(order_price / last_price - 1) > 0.05:
            return False, "price too far from LTP"
        return True, "ok"
```

### 18.5 CCXT Live Loop (crypto, polling)
```python
import ccxt, time, os
import pandas as pd

ex = ccxt.binance({
    "apiKey": os.environ["BINANCE_KEY"],
    "secret": os.environ["BINANCE_SECRET"],
    "enableRateLimit": True,
})
ex.set_sandbox_mode(True)  # testnet first!

SYMBOL, TF = "BTC/USDT", "15m"
while True:
    try:
        ohlcv = ex.fetch_ohlcv(SYMBOL, TF, limit=200)
        df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "vol"])
        df = df.iloc[:-1]                          # drop the still-forming candle
        # ... compute signal → target → diff vs current position → order
    except ccxt.NetworkError as e:
        print("network", e); time.sleep(5)
    except ccxt.ExchangeError as e:
        print("exchange", e)                       # alert
    time.sleep(ex.parse_timeframe(TF) - (time.time() % ex.parse_timeframe(TF)) + 2)
```

### 18.6 Zerodha Kite Connect (outline)
```python
from kiteconnect import KiteConnect, KiteTicker
kite = KiteConnect(api_key=API_KEY)
# 1) user logs in at kite.login_url() → request_token
# 2) data = kite.generate_session(request_token, api_secret=API_SECRET)
kite.set_access_token(data["access_token"])   # valid for the trading day
kite.place_order(
    variety=kite.VARIETY_REGULAR, exchange=kite.EXCHANGE_NSE,
    tradingsymbol="INFY", transaction_type=kite.TRANSACTION_TYPE_BUY,
    quantity=1, product=kite.PRODUCT_MIS, order_type=kite.ORDER_TYPE_LIMIT,
    price=1500.0, tag="bot1",
)
```

### 18.7 Alpaca (paper)
```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
client = TradingClient(KEY, SECRET, paper=True)
client.submit_order(MarketOrderRequest(symbol="SPY", qty=1, side=OrderSide.BUY,
                                       time_in_force=TimeInForce.DAY, client_order_id="bot-001"))
```

### 18.8 Telegram Alert
```python
import httpx, os
def alert(msg: str):
    httpx.post(f"https://api.telegram.org/bot{os.environ['TG_TOKEN']}/sendMessage",
               data={"chat_id": os.environ["TG_CHAT"], "text": msg}, timeout=10)
```

---

## 19. Build Roadmap (Phased)

| Phase | Goal | Deliverables |
|---|---|---|
| 0. Decide | Market, broker, timeframe, capital, strategy family | One-page spec |
| 1. Data | Reliable historical data pipeline | Downloader, Parquet cache, data quality checks |
| 2. Research | Find an edge | Notebooks, vectorized backtests including costs |
| 3. Validate | Robustness | Walk-forward, OOS, Monte Carlo, parameter heatmaps |
| 4. Engine | Event-driven core shared by backtest and live | Events, strategy and broker interfaces, paper broker |
| 5. Risk | Hard limits | RiskManager, kill switch, unit tests |
| 6. Broker adapter | Real API | Auth, orders, positions, reconciliation, WebSocket |
| 7. Ops | Run unattended | Docker, VPS with static IP, systemd, logs, Telegram alerts, heartbeat |
| 8. Paper trade | 1–3 months | Compare against the backtest, measure slippage |
| 9. Live (small) | Real fills | Daily review, journal |
| 10. Scale and iterate | More capital or strategies | Portfolio of uncorrelated strategies, dashboards |

**Suggested first bot**: daily or hourly trend-following or a mean-reversion rule on a liquid index (NIFTY/BANKNIFTY futures, an index ETF, or BTC/ETH), run through Alpaca paper, Binance testnet, or a Zerodha paper adapter. Learn the plumbing before chasing edge.

---

## 20. Checklists

### Pre-Live Checklist
- [ ] Backtest includes all costs, slippage, and realistic fills
- [ ] No look-ahead (signal on closed bar, execute next bar)
- [ ] OOS and walk-forward results acceptable, and parameters robust
- [ ] Paper traded for at least N weeks with results matching expectations
- [ ] Risk limits hard-coded and unit tested (per-trade, daily, drawdown, fat-finger)
- [ ] Exchange-side stop-losses where possible
- [ ] Idempotent orders (client IDs), handling for partial fills and rejects
- [ ] Reconciliation with broker on startup and periodically
- [ ] Graceful shutdown and safe restart
- [ ] Alerts for errors, fills, limit hits, heartbeat, and daily summary
- [ ] Remote kill switch tested
- [ ] Secrets in env/secret manager, withdrawals disabled, IP whitelisted
- [ ] Regulatory requirements met (India: broker algo registration, static IP)
- [ ] Starting capital is money you can afford to lose

### Daily Ops Checklist
- [ ] Auth token refreshed and bot healthy before market open
- [ ] Data feed live, not stale
- [ ] Positions match broker
- [ ] Review previous day's trades, slippage, and errors
- [ ] Check for corporate actions, expiry, holidays, and events

---

## 21. Common Failure Modes

1. Overfit backtest → loses money live.
2. Ignored costs → a "profitable" high-frequency strategy is negative net.
3. Look-ahead bug (`shift` missing, using the current candle's close).
4. Bot crashed while holding a position with no exchange-side stop.
5. Duplicate orders from retrying without idempotency.
6. WebSocket silently died and the bot traded on stale prices.
7. Infinite order loop from a logic bug, so no rate or order-count limit stopped it.
8. Wrong symbol, lot size, or tick size, or an expired contract.
9. Daily token expiry caused a missed session (India brokers).
10. Timezone bugs (IST vs UTC vs exchange time).
11. Holiday or half-day calendar not handled.
12. Leverage plus a gap move led to liquidation.
13. API change or deprecation broke the bot. Subscribe to the broker's developer changelog.
14. API keys leaked through a GitHub commit or a malicious package.
15. Strategy decay as the edge was arbitraged away, with nobody monitoring live against expected performance.
16. Emotional override, where a human intervened at the worst time.

---

## 22. Glossary

- **ADV**: average daily volume. **ATR**: average true range (volatility).
- **Alpha**: excess return. **Beta**: market sensitivity.
- **Basis**: futures minus spot. **Funding rate**: periodic payment between longs and shorts on perpetuals.
- **Bps**: basis point = 0.01%.
- **Drawdown**: decline from peak equity.
- **Fill**: execution of an order. **Partial fill**: a portion of the order executed.
- **Latency**: delay between event and action.
- **LTP**: last traded price. **MIS/NRML/CNC**: Zerodha product types (intraday, carry-forward F&O, delivery).
- **Maker/Taker**: adds liquidity (limit resting) or removes it (marketable).
- **OCO**: one-cancels-other. **GTT**: good-till-triggered.
- **OHLCV**: open/high/low/close/volume.
- **OOS**: out-of-sample. **IS**: in-sample. **WFO**: walk-forward optimization.
- **OMS / EMS**: order and execution management systems.
- **Slippage**: difference between expected and actual fill price.
- **Spread**: ask minus bid.
- **Tick size / lot size**: minimum price increment and minimum quantity.
- **VWAP / TWAP**: volume- and time-weighted average price.
- **Perp**: perpetual futures (no expiry, crypto).
- **Greeks**: delta, gamma, theta, vega, rho (options sensitivities). **IV**: implied volatility.
- **Freeze quantity**: maximum quantity per single order on NSE F&O.
- **STT**: Securities Transaction Tax (India).

---

## 23. Learning Resources

**Books**
- *Algorithmic Trading* and *Quantitative Trading*, Ernest Chan (practical, good starting point)
- *Advances in Financial Machine Learning* and *Machine Learning for Asset Managers*, Marcos López de Prado
- *Machine Learning for Algorithmic Trading* (2nd ed.), Stefan Jansen (Python-heavy, with a GitHub repo)
- *Trading and Exchanges*, Larry Harris (market microstructure)
- *Algorithmic and High-Frequency Trading*, Cartea, Jaimungal & Penalva
- *Systematic Trading* and *Leveraged Trading*, Robert Carver (sizing, robustness; `pysystemtrade` on GitHub)
- *Following the Trend*, Andreas Clenow; *Stocks on the Move*, Clenow
- *Evidence-Based Technical Analysis*, David Aronson
- *Option Volatility and Pricing*, Sheldon Natenberg
- *Fooled by Randomness*, Nassim Taleb (mindset)

**Papers and concepts**: "The Deflated Sharpe Ratio" and "The Probability of Backtest Overfitting" (Bailey & López de Prado); "Time Series Momentum" (Moskowitz, Ooi, Pedersen); Avellaneda & Stoikov (2008), "High-frequency trading in a limit order book"; Almgren & Chriss (2000), optimal execution.

**Sites and communities**: QuantConnect docs and forums, Quantpedia (strategy encyclopedia), Robot Wealth, Quantocracy (blog aggregator), r/algotrading, the Freqtrade docs and Discord, the Zerodha TradingQ&A / Kite Connect forum, the Hummingbot docs, the NautilusTrader docs, and SSRN for papers.

**Courses**: QuantInsti EPAT (India), Coursera/WorldQuant University (free MSc in Financial Engineering), and QuantConnect Bootcamp.

---

*End of knowledge base. Update the **[verify]** items and add project-specific decisions (chosen market, broker, strategy) as the project progresses.*
