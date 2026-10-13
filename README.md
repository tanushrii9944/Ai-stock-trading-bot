[README (1).md](https://github.com/user-attachments/files/33366326/README.1.md)[Upl# AI Stock Trading Bot (Java)

A dependency-free Java bot that learns to predict next-day price direction and trades on it in a **paper-trading backtest**. It is meant as a learning project and a starting point, not a production trading system.

> ⚠️ **Disclaimer:** For educational purposes only. This is not financial advice. Backtest results do not guarantee future performance. Never trade real money with code you haven't thoroughly tested.

## How it works

1. **Data:** loads closing prices from a CSV, or generates synthetic demo data if none is given.
2. **Features:** 1-day return, 5-day return, SMA10/SMA30 ratio, RSI(14), 10-day volatility.
3. **AI model:** logistic regression (trained with SGD) that outputs `P(next-day close is up)`.
4. **Strategy (long-only):**
   - Buy when `P(up) > 0.55`
   - Sell when `P(up) < 0.45`
5. **Walk-forward testing:** the model is pre-trained on the first 60% of the data, then trades the remaining 40% while continuing to learn online from each new day's outcome (no look-ahead in training).
6. **Report:** direction accuracy, number of trades, final equity, return vs. buy-and-hold, Sharpe ratio, max drawdown.

## Requirements

- Java 17 or newer (JDK)
- No external libraries

## Build and run

```bash
# Compile
javac TradingBot.java

# Run with built-in synthetic data
java TradingBot

# Run with your own data
java TradingBot prices.csv
```

### CSV format

A header row, then one row per trading day, with the closing price in the second column:

```csv
date,close
2024-01-02,185.64
2024-01-03,184.25
2024-01-04,181.91
```

You need at least ~130 rows. You can download historical data from sources such as Yahoo Finance or Stooq.

## Example output

```
=== Backtest Results (out-of-sample) ===
Test days:           599
Direction accuracy:  5x.xx%
Trades executed:     ...
Final equity:        $...
Bot return:          ...%
Buy & hold return:   ...%
Sharpe (annualized): ...
Max drawdown:        ...%
```

(Exact numbers depend on your data.)

## Configuration

Tweak the constants at the top of `TradingBot.java`:

| Constant | Meaning | Default |
|---|---|---|
| `INITIAL_CASH` | Starting capital | 10,000 |
| `COMMISSION` | Fee per trade | 0.1% |
| `BUY_THRESHOLD` / `SELL_THRESHOLD` | Probability cutoffs | 0.55 / 0.45 |
| `TRAIN_SPLIT` | Fraction of data used for initial training | 0.6 |
| `EPOCHS` | Initial training passes | 30 |
| `LEARNING_RATE` / `ONLINE_LEARNING_RATE` | SGD step sizes | 0.05 / 0.01 |

## Known simplifications

- Signals are computed at the close and filled at that same close (no slippage or intraday modeling).
- Single asset, long-only, all-in / all-out position sizing.
- No stop-loss, risk management, or position sizing logic.
- Prices only; no volume, fundamentals, or news.

## Ideas for extension

- Connect to a broker API (e.g. Alpaca paper trading) by replacing `PaperBroker`.
- Add features: MACD, Bollinger Bands, volume, sentiment.
- Swap in a stronger model (gradient boosting, LSTM via Deep Java Library).
- Add stop-loss, volatility-based position sizing, and multi-asset portfolios.
- Add unit tests and cross-validation over multiple time periods.

## Project structure

```
.
├── TradingBot.java   # all code: data loading, features, model, broker, backtest
└── README.md
```

## License

MIT. Use at your own risk.
oading README (1).md…]()

