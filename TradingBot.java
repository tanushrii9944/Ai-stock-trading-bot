import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * AI Stock Trading Bot (paper trading / backtest only)
 *
 * - Model: logistic regression trained with SGD (predicts P(next-day close is up))
 * - Features: 1d return, 5d return, SMA10/SMA30 ratio, RSI(14), 10d volatility
 * - Strategy: long-only. Buy when P(up) > BUY_THRESHOLD, sell when P(up) < SELL_THRESHOLD
 * - Walk-forward: model is pre-trained on the first part of data, then keeps
 *   learning online as each new day is revealed.
 *
 * No external dependencies. Requires Java 17+.
 *
 * Usage:
 *   javac TradingBot.java
 *   java TradingBot                # uses synthetic demo data
 *   java TradingBot prices.csv     # CSV with header, columns: date,close
 *
 * DISCLAIMER: Educational use only. Not financial advice.
 */
public class TradingBot {

    // ---------- Configuration ----------
    static final double INITIAL_CASH = 10_000.0;
    static final double COMMISSION = 0.001;      // 0.1% per trade
    static final double BUY_THRESHOLD = 0.55;
    static final double SELL_THRESHOLD = 0.45;
    static final double TRAIN_SPLIT = 0.6;       // first 60% of data = initial training
    static final int EPOCHS = 30;
    static final double LEARNING_RATE = 0.05;
    static final double ONLINE_LEARNING_RATE = 0.01;
    static final int WARMUP = 30;                // days needed before first feature row

    // ---------- Data loading ----------
    static double[] loadPrices(String path) throws IOException {
        List<Double> prices = new ArrayList<>();
        List<String> lines = Files.readAllLines(Paths.get(path));
        for (int i = 1; i < lines.size(); i++) {      // skip header
            String line = lines.get(i).trim();
            if (line.isEmpty()) continue;
            String[] parts = line.split(",");
            try {
                prices.add(Double.parseDouble(parts[parts.length > 1 ? 1 : 0].trim()));
            } catch (NumberFormatException ignored) { }
        }
        return prices.stream().mapToDouble(Double::doubleValue).toArray();
    }

    static double[] syntheticPrices(int n, long seed) {
        Random rnd = new Random(seed);
        double[] p = new double[n];
        p[0] = 100;
        double lastRet = 0;
        for (int i = 1; i < n; i++) {
            // small drift + mild momentum + noise
            double ret = 0.0003 + 0.15 * lastRet + rnd.nextGaussian() * 0.012;
            p[i] = p[i - 1] * (1 + ret);
            lastRet = ret;
        }
        return p;
    }

    // ---------- Feature engineering ----------
    static final int NUM_FEATURES = 5;

    static double sma(double[] p, int end, int window) {
        double s = 0;
        for (int i = end - window + 1; i <= end; i++) s += p[i];
        return s / window;
    }

    static double rsi(double[] p, int end, int window) {
        double gain = 0, loss = 0;
        for (int i = end - window + 1; i <= end; i++) {
            double d = p[i] - p[i - 1];
            if (d > 0) gain += d; else loss -= d;
        }
        if (loss == 0) return 100;
        double rs = (gain / window) / (loss / window);
        return 100 - 100 / (1 + rs);
    }

    static double volatility(double[] p, int end, int window) {
        double mean = 0;
        double[] r = new double[window];
        for (int k = 0; k < window; k++) {
            int i = end - window + 1 + k;
            r[k] = p[i] / p[i - 1] - 1;
            mean += r[k];
        }
        mean /= window;
        double var = 0;
        for (double x : r) var += (x - mean) * (x - mean);
        return Math.sqrt(var / window);
    }

    static double clip(double x) { return Math.max(-3, Math.min(3, x)); }

    static double[] features(double[] p, int i) {
        double ret1 = p[i] / p[i - 1] - 1;
        double ret5 = p[i] / p[i - 5] - 1;
        double smaRatio = sma(p, i, 10) / sma(p, i, 30) - 1;
        double rsiN = (rsi(p, i, 14) - 50) / 50.0;
        double vol = volatility(p, i, 10);
        return new double[] {
            clip(ret1 * 20), clip(ret5 * 10), clip(smaRatio * 15), clip(rsiN), clip(vol * 30)
        };
    }

    // ---------- AI model: logistic regression ----------
    static class LogisticModel {
        final double[] w = new double[NUM_FEATURES];
        double b = 0;

        double predict(double[] x) {
            double z = b;
            for (int i = 0; i < w.length; i++) z += w[i] * x[i];
            return 1.0 / (1.0 + Math.exp(-z));
        }

        void update(double[] x, int y, double lr) {
            double err = predict(x) - y;
            for (int i = 0; i < w.length; i++) w[i] -= lr * err * x[i];
            b -= lr * err;
        }

        void train(List<double[]> X, List<Integer> Y, int epochs, double lr) {
            for (int e = 0; e < epochs; e++)
                for (int i = 0; i < X.size(); i++) update(X.get(i), Y.get(i), lr);
        }
    }

    // ---------- Paper broker ----------
    static class PaperBroker {
        double cash;
        double shares = 0;
        int trades = 0;

        PaperBroker(double cash) { this.cash = cash; }

        void buyAll(double price) {
            if (cash <= 0) return;
            double qty = cash / (price * (1 + COMMISSION));
            shares += qty;
            cash = 0;
            trades++;
        }

        void sellAll(double price) {
            if (shares <= 0) return;
            cash += shares * price * (1 - COMMISSION);
            shares = 0;
            trades++;
        }

        double equity(double price) { return cash + shares * price; }
        boolean inPosition() { return shares > 0; }
    }

    // ---------- Metrics ----------
    static double sharpe(List<Double> equity) {
        int n = equity.size() - 1;
        double[] r = new double[n];
        double mean = 0;
        for (int i = 0; i < n; i++) {
            r[i] = equity.get(i + 1) / equity.get(i) - 1;
            mean += r[i];
        }
        mean /= n;
        double var = 0;
        for (double x : r) var += (x - mean) * (x - mean);
        double sd = Math.sqrt(var / n);
        return sd == 0 ? 0 : mean / sd * Math.sqrt(252);
    }

    static double maxDrawdown(List<Double> equity) {
        double peak = equity.get(0), mdd = 0;
        for (double e : equity) {
            peak = Math.max(peak, e);
            mdd = Math.max(mdd, (peak - e) / peak);
        }
        return mdd;
    }

    // ---------- Main ----------
    public static void main(String[] args) throws IOException {
        double[] prices = args.length > 0 ? loadPrices(args[0]) : syntheticPrices(1500, 42);
        if (args.length == 0) System.out.println("No CSV given - using synthetic demo data.\n");

        int n = prices.length;
        if (n < WARMUP + 100) {
            System.err.println("Need at least " + (WARMUP + 100) + " price points.");
            return;
        }
        int trainEnd = (int) (n * TRAIN_SPLIT);

        // 1) Initial training
        List<double[]> X = new ArrayList<>();
        List<Integer> Y = new ArrayList<>();
        for (int i = WARMUP; i < trainEnd; i++) {
            X.add(features(prices, i));
            Y.add(prices[i + 1] > prices[i] ? 1 : 0);
        }
        LogisticModel model = new LogisticModel();
        model.train(X, Y, EPOCHS, LEARNING_RATE);

        // 2) Walk-forward trading with online learning
        PaperBroker broker = new PaperBroker(INITIAL_CASH);
        List<Double> equity = new ArrayList<>();
        equity.add(INITIAL_CASH);
        int correct = 0, total = 0;

        for (int i = trainEnd; i < n - 1; i++) {
            double[] x = features(prices, i);
            double pUp = model.predict(x);

            if (pUp > BUY_THRESHOLD && !broker.inPosition()) broker.buyAll(prices[i]);
            else if (pUp < SELL_THRESHOLD && broker.inPosition()) broker.sellAll(prices[i]);

            equity.add(broker.equity(prices[i + 1]));

            int label = prices[i + 1] > prices[i] ? 1 : 0;
            if ((pUp >= 0.5 ? 1 : 0) == label) correct++;
            total++;
            model.update(x, label, ONLINE_LEARNING_RATE);   // learn from today's outcome
        }

        // 3) Report
        double finalEq = equity.get(equity.size() - 1);
        double buyHold = INITIAL_CASH * prices[n - 1] / prices[trainEnd];
        System.out.println("=== Backtest Results (out-of-sample) ===");
        System.out.printf("Test days:           %d%n", total);
        System.out.printf("Direction accuracy:  %.2f%%%n", 100.0 * correct / total);
        System.out.printf("Trades executed:     %d%n", broker.trades);
        System.out.printf("Final equity:        $%.2f (start $%.2f)%n", finalEq, INITIAL_CASH);
        System.out.printf("Bot return:          %.2f%%%n", 100 * (finalEq / INITIAL_CASH - 1));
        System.out.printf("Buy & hold return:   %.2f%%%n", 100 * (buyHold / INITIAL_CASH - 1));
        System.out.printf("Sharpe (annualized): %.2f%n", sharpe(equity));
        System.out.printf("Max drawdown:        %.2f%%%n", 100 * maxDrawdown(equity));
    }
}
