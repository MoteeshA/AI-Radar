from flask import Flask, render_template
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# 📄 Load stocks from CSV
def get_stock_list():
    try:
        df = pd.read_csv("nifty50.csv")
        return df["symbol"].dropna().tolist()
    except Exception as e:
        print("Error loading CSV:", e)
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS"]


# 🔥 Main Scanner Function
def get_opportunities():
    stocks = get_stock_list()
    results = []
    fallback_data = []

    try:
        data = yf.download(
            tickers=stocks,
            period="3mo",  # slightly more data for better analysis
            interval="1d",
            group_by='ticker',
            threads=True
        )
    except Exception as e:
        print("Error fetching stock data:", e)
        return [{
            "stock": "ERROR",
            "price": "-",
            "score": 0,
            "signals": "Data fetch failed"
        }]

    for stock in stocks:
        try:
            if stock not in data:
                continue

            df = data[stock].copy()
            df.dropna(inplace=True)

            if len(df) < 50:
                continue

            # 🔥 Indicators
            df['MA20'] = df['Close'].rolling(20).mean()
            df['MA50'] = df['Close'].rolling(50).mean()

            latest = df.iloc[-1]
            prev = df.iloc[:-1]

            latest_close = float(latest['Close'])
            latest_volume = float(latest['Volume'])

            max_20 = float(prev['Close'].rolling(20).max().iloc[-1])
            avg_vol_10 = float(prev['Volume'].rolling(10).mean().iloc[-1])
            avg_price = float(prev['Close'].mean())

            ma20 = float(latest['MA20'])
            ma50 = float(latest['MA50'])

            # 🔥 Conditions
            breakout = latest_close > max_20
            volume_spike = latest_volume > 2 * avg_vol_10
            uptrend = ma20 > ma50
            momentum = latest_close > df['Close'].iloc[-5]

            # 🧠 Improved scoring
            score = 0
            signals = []

            if breakout:
                score += 35
                signals.append("Breakout")

            if volume_spike:
                score += 25
                signals.append("Volume Spike")

            if uptrend:
                score += 20
                signals.append("MA Uptrend")

            if momentum:
                score += 10
                signals.append("Momentum")

            if latest_close > avg_price:
                score += 10
                signals.append("Above Avg Price")

            # 🧠 Save fallback always
            fallback_data.append({
                "stock": stock.replace(".NS", ""),
                "price": round(latest_close, 2),
                "score": score,
                "signals": ", ".join(signals) if signals else "Neutral"
            })

            # 🎯 Strong opportunities
            if score >= 25:
                results.append({
                    "stock": stock.replace(".NS", ""),
                    "price": round(latest_close, 2),
                    "score": score,
                    "signals": ", ".join(signals)
                })

        except Exception as e:
            print(f"Skipping {stock}: {e}")
            continue

    # 🔥 CASE 1: Strong signals
    if len(results) > 0:
        return sorted(results, key=lambda x: x["score"], reverse=True)[:10]

    # 🔥 CASE 2: fallback (always show something useful)
    if len(fallback_data) > 0:
        fallback_sorted = sorted(fallback_data, key=lambda x: x["score"], reverse=True)[:10]

        for item in fallback_sorted:
            item["signals"] = item["signals"] + " (Weak Signal)"

        return fallback_sorted

    # 🔥 CASE 3: total failure
    return [{
        "stock": "MARKET",
        "price": "-",
        "score": 0,
        "signals": "Market data unavailable"
    }]


# 🏠 Home Route
@app.route("/")
def home():
    data = get_opportunities()
    return render_template("index.html", data=data)


# ▶️ Run App
if __name__ == "__main__":
    app.run(debug=True)