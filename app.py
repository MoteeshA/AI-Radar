from flask import Flask, render_template, request, jsonify, session
import yfinance as yf
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from functools import lru_cache
import time
from datetime import datetime, timedelta
import json
import hashlib
import random
from collections import OrderedDict

# ==============================
# ✅ ENV & CONFIG
# ==============================
load_dotenv()

# ==============================
# 🤖 OPENAI
# ==============================
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")

USE_AI = False
client = None

if api_key:
    try:
        client = OpenAI(api_key=api_key)
        USE_AI = True
        print("✅ AI Enabled with GPT-4o-mini")
    except:
        print("⚠️ OpenAI initialization failed")

print(f"🤖 AI Status: {'ENABLED' if USE_AI else 'DISABLED'}")

# ==============================
# 🚀 APP INITIALIZATION
# ==============================
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here")
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Simple cache for AI responses
ai_cache = OrderedDict()
CACHE_SIZE = 100
CACHE_TTL = 3600

def get_cache_key(prefix, data):
    key_str = f"{prefix}_{json.dumps(data, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()

def get_cached_ai_response(key):
    if key in ai_cache:
        timestamp, response = ai_cache[key]
        if time.time() - timestamp < CACHE_TTL:
            return response
        else:
            del ai_cache[key]
    return None

def cache_ai_response(key, response):
    ai_cache[key] = (time.time(), response)
    while len(ai_cache) > CACHE_SIZE:
        ai_cache.popitem(last=False)

# ==============================
# 📄 COMPREHENSIVE STOCK LIST
# ==============================
def get_stock_list():
    """Returns comprehensive list of Indian stocks"""
    comprehensive_stocks = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
        "BAJFINANCE.NS", "LT.NS", "WIPRO.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "MARUTI.NS", "SUNPHARMA.NS", "HCLTECH.NS", "ULTRACEMCO.NS", "TITAN.NS",
        "ADANIPORTS.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS", "TATAMOTORS.NS",
        "TATASTEEL.NS", "JSWSTEEL.NS", "TECHM.NS", "INDUSINDBK.NS", "NESTLEIND.NS",
        "VEDL.NS", "DABUR.NS", "TORNTPHARM.NS", "MUTHOOTFIN.NS", "NAUKRI.NS",
        "BANKBARODA.NS", "CANBK.NS", "PNB.NS", "FEDERALBNK.NS", "IDFCFIRSTB.NS",
        "LTIM.NS", "MPHASIS.NS", "PERSISTENT.NS", "COFORGE.NS", "ZOMATO.NS",
        "DLF.NS", "GODREJPROP.NS", "HAL.NS", "BEL.NS", "COALINDIA.NS"
    ]
    return comprehensive_stocks

# ==============================
# 🔧 SAFE SERIES EXTRACTION
# ==============================
def get_series(df, col):
    if col not in df.columns:
        return pd.Series([0] * len(df))
    data = df[col]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data

# ==============================
# 📊 TECHNICAL INDICATORS
# ==============================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def calculate_bollinger_bands(series, period=20, std_dev=2):
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    return upper_band, lower_band, sma

def calculate_atr(df, period=14):
    high = get_series(df, "High")
    low = get_series(df, "Low")
    close = get_series(df, "Close")
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr.fillna(0)

def detect_advanced_patterns(df):
    patterns = []
    close = get_series(df, "Close")
    high = get_series(df, "High")
    low = get_series(df, "Low")
    open_prices = get_series(df, "Open")
    
    if len(close) < 30:
        return patterns
    
    resistance = close.rolling(20).max()
    support = close.rolling(20).min()
    
    if close.iloc[-1] > resistance.iloc[-2] * 0.99:
        patterns.append("🚀 Breakout")
    elif close.iloc[-1] < support.iloc[-2] * 1.01:
        patterns.append("📉 Breakdown")
    
    body = abs(close - open_prices)
    avg_body = body.rolling(20).mean()
    upper_shadow = high - np.maximum(open_prices, close)
    lower_shadow = np.minimum(open_prices, close) - low
    
    if body.iloc[-1] < avg_body.iloc[-1] * 0.3:
        patterns.append("⚖️ Doji")
    
    if (lower_shadow.iloc[-1] > 2 * body.iloc[-1] and 
        upper_shadow.iloc[-1] < body.iloc[-1] * 0.5):
        patterns.append("🔨 Hammer")
    
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    
    if len(ma20) > 50 and len(ma50) > 50:
        if ma20.iloc[-1] > ma50.iloc[-1] and ma20.iloc[-5] < ma50.iloc[-5]:
            patterns.append("📈 Golden Cross")
    
    return patterns

# ==============================
# 🤖 AI FUNCTIONS
# ==============================
def ai_trade_explanation(data):
    cache_key = get_cache_key("explain", data)
    cached = get_cached_ai_response(cache_key)
    if cached:
        return cached
    
    if not USE_AI or not client:
        return generate_fallback_explanation(data)
    
    try:
        prompt = f"""
        Act like a professional trader. Give concise analysis:

        Stock: {data['stock']}
        Price: ₹{data['price']}
        RSI: {data['rsi']:.1f}
        Pattern: {data['pattern']}
        Signal: {data['signal']}
        Score: {data['score']}/100

        Respond in 2 lines:
        - Why this trade exists
        - Entry zone and exit strategy
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
        result = response.choices[0].message.content.strip()
        cache_ai_response(cache_key, result)
        return result
    except Exception as e:
        print(f"AI Error: {e}")
        return generate_fallback_explanation(data)

def generate_fallback_explanation(data):
    if data['signal'] == 'BUY':
        return f"📈 Bullish momentum | Entry: ₹{data['price']} | Target: +5% | Stop: -3%"
    elif data['signal'] == 'SELL':
        return f"📉 Bearish pressure | Exit above resistance | Watch for reversal"
    else:
        return f"⏸️ Consolidation | Wait for breakout above ₹{data['price'] * 1.02:.2f}"

def market_sentiment_ai():
    cache_key = "market_sentiment"
    cached = get_cached_ai_response(cache_key)
    if cached:
        sentiment, reason = cached.split("|", 1)
        return sentiment, reason
    
    if not USE_AI or not client:
        return get_fallback_sentiment()
    
    try:
        nifty = yf.download("^NSEI", period="5d", interval="1d", progress=False)
        if not nifty.empty:
            close = get_series(nifty, "Close")
            change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
            
            prompt = f"""
            NIFTY changed {change:.2f}% today.
            Analyze Indian market sentiment.
            Return: SENTIMENT|REASON
            Sentiment: Bullish/Bearish/Sideways
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=60,
                temperature=0.6
            )
            
            result = response.choices[0].message.content.strip()
            if "|" in result:
                cache_ai_response(cache_key, result)
                sentiment, reason = result.split("|", 1)
                return sentiment.strip(), reason.strip()
    except:
        pass
    
    return get_fallback_sentiment()

def get_fallback_sentiment():
    return "Sideways", "Market consolidating ahead of key levels"

def backtest(df):
    trades = 0
    wins = 0
    total_return = 0
    max_drawdown = 0
    peak = 0
    
    close = get_series(df, "Close")
    signals = df.get("Signal", pd.Series(["HOLD"] * len(df)))
    
    position = None
    entry_price = 0
    
    for i in range(1, len(df)):
        current_signal = signals.iloc[i] if i < len(signals) else "HOLD"
        
        if position is None:
            if current_signal == "BUY":
                position = "LONG"
                entry_price = close.iloc[i]
                trades += 1
            elif current_signal == "SELL":
                position = "SHORT"
                entry_price = close.iloc[i]
                trades += 1
        else:
            exit_condition = (position == "LONG" and current_signal == "SELL") or \
                            (position == "SHORT" and current_signal == "BUY") or \
                            i == len(df) - 1
            
            if exit_condition:
                exit_price = close.iloc[i]
                if position == "LONG":
                    pnl = (exit_price - entry_price) / entry_price
                else:
                    pnl = (entry_price - exit_price) / entry_price
                
                total_return += pnl
                if pnl > 0:
                    wins += 1
                position = None
        
        if close.iloc[i] > peak:
            peak = close.iloc[i]
        drawdown = (peak - close.iloc[i]) / peak if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
    
    accuracy = (wins / trades * 100) if trades > 0 else 50
    return round(accuracy, 2), round(total_return * 100, 2), round(max_drawdown * 100, 2)

# ==============================
# ⚡ OPTIMIZED DATA FETCHING
# ==============================
@lru_cache(maxsize=1)
def get_market_data_cached():
    stocks = get_stock_list()
    all_data = {}
    batch_size = 25
    
    for i in range(0, len(stocks), batch_size):
        batch = stocks[i:i+batch_size]
        try:
            data = yf.download(
                tickers=batch,
                period="3mo",
                interval="1d",
                group_by='ticker',
                threads=True,
                progress=False
            )
            if not data.empty:
                if len(batch) == 1:
                    all_data[batch[0]] = data
                else:
                    for ticker in batch:
                        if ticker in data.columns.levels[0]:
                            all_data[ticker] = data[ticker]
        except Exception as e:
            print(f"Batch error: {e}")
        time.sleep(0.3)
    
    return all_data

# ==============================
# 🔥 ENHANCED SCANNER
# ==============================
def get_opportunities():
    stocks = get_stock_list()
    results = []
    
    try:
        market_data = get_market_data_cached()
    except Exception as e:
        print(f"Data fetch error: {e}")
        return []
    
    processed_stocks = random.sample(stocks, min(80, len(stocks)))
    
    for stock in processed_stocks:
        try:
            if stock not in market_data or market_data[stock].empty:
                continue
            
            df = market_data[stock].dropna()
            if len(df) < 30:
                continue
            
            close = get_series(df, "Close")
            volume = get_series(df, "Volume")
            high = get_series(df, "High")
            low = get_series(df, "Low")
            
            df["MA20"] = close.rolling(20).mean()
            df["MA50"] = close.rolling(50).mean()
            df["RSI"] = calculate_rsi(close)
            
            macd, macd_signal = calculate_macd(close)
            df["MACD"] = macd
            df["MACD_SIGNAL"] = macd_signal
            
            upper_bb, lower_bb, _ = calculate_bollinger_bands(close)
            df["BB_UPPER"] = upper_bb
            df["BB_LOWER"] = lower_bb
            
            df["ATR"] = calculate_atr(df)
            patterns = detect_advanced_patterns(df)
            
            latest_close = float(close.iloc[-1])
            latest_volume = float(volume.iloc[-1])
            avg_volume = float(volume.rolling(20).mean().iloc[-1])
            rsi = float(df["RSI"].iloc[-1])
            
            # Scoring System
            score = 50
            
            if df["MA20"].iloc[-1] > df["MA50"].iloc[-1]:
                score += 15
            else:
                score -= 5
            
            if 60 < rsi < 80:
                score += 15
            elif rsi > 50:
                score += 8
            elif rsi < 30:
                score -= 8
            
            if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:
                score += 12
            else:
                score -= 5
            
            volume_ratio = latest_volume / avg_volume if avg_volume > 0 else 1
            if volume_ratio > 1.5:
                score += 12
            elif volume_ratio > 1.2:
                score += 6
            
            if latest_close > df["BB_UPPER"].iloc[-1]:
                score += 10
            elif latest_close < df["BB_LOWER"].iloc[-1]:
                score -= 8
            
            pattern_bonus = 0
            for pattern in patterns:
                if "Breakout" in pattern:
                    pattern_bonus += 12
                elif "Golden Cross" in pattern:
                    pattern_bonus += 10
                elif "Hammer" in pattern:
                    pattern_bonus += 6
            score += min(pattern_bonus, 15)
            score = max(0, min(100, score))
            
            if score >= 70:
                final_signal = "BUY"
            elif score <= 35:
                final_signal = "SELL"
            else:
                final_signal = "WATCH"
            
            confidence = score
            atr_value = float(df["ATR"].iloc[-1]) if not pd.isna(df["ATR"].iloc[-1]) else latest_close * 0.02
            stop_loss = round(latest_close - (atr_value * 1.5), 2)
            target_1 = round(latest_close + (atr_value * 2), 2)
            
            insight = ai_trade_explanation({
                'stock': stock.replace(".NS", ""),
                'price': latest_close,
                'rsi': rsi,
                'pattern': ", ".join(patterns[:2]) if patterns else "Sideways",
                'signal': final_signal,
                'score': confidence
            })
            
            results.append({
                "stock": stock.replace(".NS", ""),
                "price": round(latest_close, 2),
                "score": score,
                "confidence": confidence,
                "signals": final_signal,
                "pattern": ", ".join(patterns[:2]) if patterns else "Sideways",
                "insight": insight,
                "sl": stop_loss,
                "target": target_1,
                "rsi": round(rsi, 1),
                "volume_ratio": round(volume_ratio, 2)
            })
            
        except Exception as e:
            continue
    
    return sorted(results, key=lambda x: x["score"], reverse=True)[:30]

# ==============================
# 🚀 CRITICAL API ROUTES (FIXED)
# ==============================

@app.route("/api/stock-data/<symbol>")
def api_stock_data(symbol):
    """CRITICAL FIX: This is the endpoint your frontend calls!"""
    period = request.args.get("period", "3mo")
    interval = request.args.get("interval", "1d")
    
    print(f"📊 API called for: {symbol} with period={period}, interval={interval}")
    
    try:
        ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        
        if df.empty:
            return jsonify({"error": f"No data found for {symbol}"}), 404
        
        df.reset_index(inplace=True)
        close = get_series(df, "Close")
        
        # Calculate indicators
        df["MA20"] = close.rolling(20).mean()
        df["EMA9"] = close.ewm(span=9, adjust=False).mean()
        df["RSI"] = calculate_rsi(close)
        
        macd, macd_signal = calculate_macd(close)
        df["MACD"] = macd
        df["MACD_SIGNAL"] = macd_signal
        
        upper_bb, lower_bb, _ = calculate_bollinger_bands(close)
        df["BB_UPPER"] = upper_bb
        df["BB_LOWER"] = lower_bb
        
        # Generate signals
        signals = []
        for i in range(len(df)):
            if i == 0:
                signals.append("HOLD")
                continue
            
            buy_conditions = (
                df["EMA9"].iloc[i] > df["MA20"].iloc[i] and 
                df["RSI"].iloc[i] > 50 and
                df["MACD"].iloc[i] > df["MACD_SIGNAL"].iloc[i]
            )
            
            sell_conditions = (
                df["RSI"].iloc[i] > 70 or
                (df["EMA9"].iloc[i] < df["MA20"].iloc[i] and df["RSI"].iloc[i] < 45)
            )
            
            if buy_conditions:
                signals.append("BUY")
            elif sell_conditions:
                signals.append("SELL")
            else:
                signals.append("HOLD")
        
        df["Signal"] = signals
        patterns = detect_advanced_patterns(df)
        
        date_column = "Date" if "Date" in df.columns else df.columns[0]
        
        response_data = {
            "dates": df[date_column].astype(str).tolist(),
            "open": get_series(df, "Open").tolist(),
            "high": get_series(df, "High").tolist(),
            "low": get_series(df, "Low").tolist(),
            "close": close.tolist(),
            "volume": get_series(df, "Volume").tolist(),
            "ma20": df["MA20"].fillna(0).tolist(),
            "ema9": df["EMA9"].fillna(0).tolist(),
            "rsi": df["RSI"].fillna(50).tolist(),
            "signals": signals,
            "current_price": float(close.iloc[-1]),
            "support": float(close.rolling(20).min().iloc[-1]),
            "resistance": float(close.rolling(20).max().iloc[-1]),
            "pattern": ", ".join(patterns[:2]) if patterns else "Sideways"
        }
        
        print(f"✅ API response prepared for {symbol}: {len(response_data['dates'])} candles")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ API ERROR for {symbol}: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/live-price/<symbol>")
def api_live_price(symbol):
    """Lightweight live price endpoint"""
    try:
        ticker = symbol + ".NS" if not symbol.endswith(".NS") else symbol
        df = yf.download(ticker, period="1d", interval="5m", progress=False)
        
        if df.empty:
            return jsonify({"error": "No data"}), 404
        
        close = get_series(df, "Close")
        volume = get_series(df, "Volume")
        
        return jsonify({
            "price": float(close.iloc[-1]),
            "volume": float(volume.iloc[-1]),
            "change": float(((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100) if len(close) > 1 else 0,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==============================
# 🏠 HTML ROUTES
# ==============================
@app.route("/")
def home():
    opportunities = get_opportunities()
    market_sentiment, sentiment_reason = market_sentiment_ai()
    total_stocks = len(get_stock_list())
    
    return render_template("index.html", 
                          data=opportunities,
                          market_sentiment=market_sentiment,
                          sentiment_reason=sentiment_reason,
                          total_stocks=total_stocks,
                          ai_enabled=USE_AI)

@app.route("/stock/<symbol>")
def stock_detail(symbol):
    """Stock detail page with chart"""
    return render_template("stock.html", symbol=symbol.upper())

@app.route("/market")
def market_overview():
    indices = {
        "NIFTY 50": "^NSEI",
        "BANK NIFTY": "^NSEBANK",
        "SENSEX": "^BSESN"
    }
    
    market_data = {}
    for name, symbol in indices.items():
        try:
            data = yf.download(symbol, period="5d", interval="1d", progress=False)
            if not data.empty:
                close = get_series(data, "Close")
                change = ((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]) * 100
                market_data[name] = {
                    "value": round(close.iloc[-1], 2),
                    "change": round(change, 2)
                }
        except:
            market_data[name] = {"value": "N/A", "change": 0}
    
    market_sentiment, sentiment_reason = market_sentiment_ai()
    return render_template("market.html", 
                          market_data=market_data,
                          market_sentiment=market_sentiment,
                          sentiment_reason=sentiment_reason)

@app.route("/api/stocks")
def api_stocks():
    """API endpoint for stock list"""
    stocks = get_stock_list()
    return jsonify([s.replace(".NS", "") for s in stocks])

@app.route("/api/opportunities")
def api_opportunities():
    """API endpoint for opportunities"""
    opportunities = get_opportunities()
    return jsonify(opportunities)

@app.route("/chatbot", methods=["POST"])
def chatbot():
    """AI Chatbot API endpoint"""
    data = request.json
    user_message = data.get("message", "")
    stock_symbol = data.get("stock", None)
    
    if not USE_AI:
        return jsonify({"response": "AI is currently disabled. Please set OPENAI_API_KEY to enable AI features."})
    
    stock_data = None
    if stock_symbol:
        try:
            ticker = stock_symbol + ".NS" if not stock_symbol.endswith(".NS") else stock_symbol
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1d")
            if not hist.empty:
                stock_data = {
                    'symbol': stock_symbol,
                    'price': round(hist['Close'].iloc[-1], 2)
                }
        except:
            pass
    
    try:
        prompt = f"""
        You are a trading assistant. Answer professionally:
        
        {f"Stock: {stock_data['symbol']} at ₹{stock_data['price']}" if stock_data else ""}
        
        User: {user_message}
        
        Assistant (concise, helpful):
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.8
        )
        
        return jsonify({"response": response.choices[0].message.content.strip()})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# ==============================
# ▶️ RUN APPLICATION
# ==============================
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)