from flask import Flask, render_template
import yfinance as yf
import pandas as pd

app = Flask(__name__)

def get_opportunities():
    stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]
    results = []

    for stock in stocks:
        df = yf.download(stock, period="2mo", interval="1d")

        # 🔥 FIX: flatten columns
        df.columns = df.columns.get_level_values(0)

        df.dropna(inplace=True)

        if len(df) < 20:
            continue

        latest = df.iloc[-1]
        prev = df.iloc[:-1]

        latest_close = float(latest['Close'])
        latest_volume = float(latest['Volume'])

        max_20 = float(prev['Close'].rolling(20).max().iloc[-1])
        avg_vol_10 = float(prev['Volume'].rolling(10).mean().iloc[-1])

        breakout = latest_close > max_20
        volume_spike = latest_volume > 2 * avg_vol_10

        score = 0
        signals = []

        if breakout:
            score += 50
            signals.append("Breakout")

        if volume_spike:
            score += 50
            signals.append("Volume Spike")

        if score > 0:
            results.append({
                "stock": stock,
                "price": round(latest_close, 2),
                "score": score,
                "signals": ", ".join(signals)
            })

    results = sorted(results, key=lambda x: x["score"], reverse=True)[:3]
    return results


@app.route("/")
def home():
    data = get_opportunities()
    return render_template("index.html", data=data)


if __name__ == "__main__":
    app.run(debug=True)