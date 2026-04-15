from fastapi import FastAPI, HTTPException
from indicators import calculate_rsi, moving_average, calculate_volatility, determine_trend, rule_engine
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import yfinance as yf
from services.pipeline import run_finrobot_pipeline
import os
import datetime
import pandas as pd
import numpy as np
import traceback
import requests

app = FastAPI()

DEFAULT_COMPANIES = [
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"symbol": "INFY.NS", "name": "Infosys"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
    {"symbol": "ITC.NS", "name": "ITC"},
    {"symbol": "SBIN.NS", "name": "State Bank of India"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
]

NIFTY_50_FALLBACK = [
    {"symbol": "ADANIENT.NS", "name": "Adani Enterprises"},
    {"symbol": "ADANIPORTS.NS", "name": "Adani Ports"},
    {"symbol": "APOLLOHOSP.NS", "name": "Apollo Hospitals"},
    {"symbol": "ASIANPAINT.NS", "name": "Asian Paints"},
    {"symbol": "AXISBANK.NS", "name": "Axis Bank"},
    {"symbol": "BAJAJ-AUTO.NS", "name": "Bajaj Auto"},
    {"symbol": "BAJFINANCE.NS", "name": "Bajaj Finance"},
    {"symbol": "BAJAJFINSV.NS", "name": "Bajaj Finserv"},
    {"symbol": "BPCL.NS", "name": "Bharat Petroleum"},
    {"symbol": "BHARTIARTL.NS", "name": "Bharti Airtel"},
    {"symbol": "BRITANNIA.NS", "name": "Britannia Industries"},
    {"symbol": "CIPLA.NS", "name": "Cipla"},
    {"symbol": "COALINDIA.NS", "name": "Coal India"},
    {"symbol": "DIVISLAB.NS", "name": "Divi's Laboratories"},
    {"symbol": "DRREDDY.NS", "name": "Dr. Reddy's Laboratories"},
    {"symbol": "EICHERMOT.NS", "name": "Eicher Motors"},
    {"symbol": "GRASIM.NS", "name": "Grasim Industries"},
    {"symbol": "HCLTECH.NS", "name": "HCL Technologies"},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "HDFCLIFE.NS", "name": "HDFC Life"},
    {"symbol": "HEROMOTOCO.NS", "name": "Hero MotoCorp"},
    {"symbol": "HINDALCO.NS", "name": "Hindalco Industries"},
    {"symbol": "HINDUNILVR.NS", "name": "Hindustan Unilever"},
    {"symbol": "ICICIBANK.NS", "name": "ICICI Bank"},
    {"symbol": "ITC.NS", "name": "ITC Limited"},
    {"symbol": "INDUSINDBK.NS", "name": "IndusInd Bank"},
    {"symbol": "INFY.NS", "name": "Infosys"},
    {"symbol": "JSWSTEEL.NS", "name": "JSW Steel"},
    {"symbol": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank"},
    {"symbol": "LT.NS", "name": "Larsen & Toubro"},
    {"symbol": "LTIM.NS", "name": "LTIMindtree"},
    {"symbol": "M&M.NS", "name": "Mahindra & Mahindra"},
    {"symbol": "MARUTI.NS", "name": "Maruti Suzuki"},
    {"symbol": "NTPC.NS", "name": "NTPC"},
    {"symbol": "NESTLEIND.NS", "name": "Nestle India"},
    {"symbol": "ONGC.NS", "name": "ONGC"},
    {"symbol": "POWERGRID.NS", "name": "Power Grid Corporation"},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "SBILIFE.NS", "name": "SBI Life Insurance"},
    {"symbol": "SBIN.NS", "name": "State Bank of India"},
    {"symbol": "SUNPHARMA.NS", "name": "Sun Pharma"},
    {"symbol": "TCS.NS", "name": "Tata Consultancy Services"},
    {"symbol": "TATACONSUM.NS", "name": "Tata Consumer Products"},
    {"symbol": "TATAMOTORS.NS", "name": "Tata Motors"},
    {"symbol": "TATASTEEL.NS", "name": "Tata Steel"},
    {"symbol": "TECHM.NS", "name": "Tech Mahindra"},
    {"symbol": "TITAN.NS", "name": "Titan Company"},
    {"symbol": "ULTRACEMCO.NS", "name": "UltraTech Cement"},
    {"symbol": "UPL.NS", "name": "UPL"},
    {"symbol": "WIPRO.NS", "name": "Wipro"}
]

NEWS_API_KEY = "3378abff54944530a21836a131f8ad57"


def get_news_headlines(query: str, limit: int = 3) -> list:
    try:
        search_term = query.split('.')[0]
        url = f"https://newsapi.org/v2/everything?q={search_term}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        response = requests.get(url, timeout=5)
        data = response.json()
        return [a["title"] for a in data.get("articles", [])[:limit]]
    except:
        return []


def generate_market_times() -> list:
    """Generate all 30-minute market time slots from 9:15 to 15:30 (NSE)."""
    times = []
    h, m = 9, 15
    while True:
        times.append(f"{h:02d}:{m:02d}")
        if h == 15 and m == 30:
            break
        m += 30
        if m >= 60:
            m -= 60
            h += 1
        if h > 15 or (h == 15 and m > 30):
            break
    return times


def predict_next_interval(prices: list, highs: list = None, lows: list = None,
                          rsi_vals: list = None, ma_vals: list = None) -> float:
    """
    Adaptive 15-minute NSE price predictor.

    Tuned for 15-minute candles (half the volatility of 30m candles):

    ① ATR from High/Low history (last ≤10 candles) – 15m ATR is typically
       ₹1.5–5 for large-cap NSE stocks.
       Default = 0.15% of current price if H/L unavailable.

    ② Weighted Linear Regression (WLR) on last ≤10 closes.
       More candles = better trend signal at 15m granularity.

    ③ Slope clamped to ±0.35 × ATR – tighter than 30m because 15m moves
       are smaller; prevents over-shooting on momentum candles.

    ④ MA mean-reversion target – 10% pull toward 20-period MA.

    ⑤ RSI-adaptive blend (same logic, more reliable with 15m RSI):
         RSI < 25  → 10% momentum, 90% reversion
         RSI 25-35 → 25% / 75%
         RSI 35-45 → 45% / 55%
         RSI 45-55 → 72% / 28%  (neutral: momentum-biased)
         RSI 55-65 → 60% / 40%
         RSI 65-75 → 25% / 75%
         RSI > 75  → 10% / 90%

    ⑥ Reversal detector – if last candle > 1.5 × ATR, apply 30% bounce.
       Threshold raised vs 30m version to avoid over-correcting.

    ⑦ Hard cap – ±0.6 × ATR (tighter than 30m's 0.75, appropriate for
       shorter interval where large moves are less expected).
    """
    if not prices:
        return 0.0
    current = float(prices[-1])
    if len(prices) < 2:
        return current

    # Use up to 10 candles for WLR (more data at 15m granularity)
    n = min(10, len(prices))
    recent = np.array(prices[-n:], dtype=float)

    # ① ATR — use up to 10 candles; default = 0.15% of price
    atr = max(current * 0.0015, 0.5)
    if highs and lows and len(highs) >= 2 and len(lows) >= 2:
        n_atr = min(10, len(highs))
        trs = []
        for i in range(1, n_atr):
            hi = float(highs[-n_atr + i])
            lo = float(lows[-n_atr + i])
            pc = float(prices[-n_atr + i - 1])
            trs.append(max(hi - lo, abs(hi - pc), abs(lo - pc)))
        if trs:
            atr = max(float(np.mean(trs)), 0.5)

    # ② WLR slope
    wts = np.arange(1, n + 1, dtype=float)
    ws  = wts.sum()
    x   = np.arange(n, dtype=float)
    xwm = (wts * x).sum() / ws
    ywm = (wts * recent).sum() / ws
    num = (wts * (x - xwm) * (recent - ywm)).sum()
    den = (wts * (x - xwm) ** 2).sum()
    slope = float(num / den) if den != 0 else 0.0

    # ③ Clamp slope to ±0.35 × ATR (tighter than 30m)
    slope = max(-0.35 * atr, min(0.35 * atr, slope))
    momentum_target = current + slope

    # ④ MA mean-reversion target (20-period MA context)
    if ma_vals:
        try:
            mv = float(ma_vals[-1])
            ma = mv if not np.isnan(mv) else float(np.mean(recent))
        except (ValueError, TypeError):
            ma = float(np.mean(recent))
    else:
        ma = float(np.mean(recent))
    reversion_target = current + (ma - current) * 0.10

    # ⑤ RSI-adaptive blend
    rsi = 50.0
    if rsi_vals:
        try:
            rv = float(rsi_vals[-1])
            if not np.isnan(rv):
                rsi = rv
        except (ValueError, TypeError):
            pass

    if   rsi < 25: alpha = 0.10
    elif rsi < 35: alpha = 0.25
    elif rsi < 45: alpha = 0.45
    elif rsi < 55: alpha = 0.72
    elif rsi < 65: alpha = 0.60
    elif rsi < 75: alpha = 0.25
    else:          alpha = 0.10

    blended = alpha * momentum_target + (1 - alpha) * reversion_target

    # ⑥ Reversal detector (threshold = 1.5 ATR for 15m)
    if len(prices) >= 2:
        last_move = current - float(prices[-2])
        if abs(last_move) > 1.5 * atr:
            blended += -last_move * 0.30

    # ⑦ Hard cap: ±0.6 × ATR (tighter for 15m)
    delta = max(-0.6 * atr, min(0.6 * atr, blended - current))
    return round(current + delta, 2)


def calc_r2(actuals: list, predicted: list):
    """
    Calculate R² (coefficient of determination).
    Returns None if fewer than 2 data pairs.
    """
    if len(actuals) < 2:
        return None
    a = np.array(actuals, dtype=float)
    p = np.array(predicted, dtype=float)
    ss_res = float(np.sum((a - p) ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    if ss_tot == 0:
        return 1.0
    return round(float(1 - ss_res / ss_tot), 4)


@app.get("/api/companies")
async def get_companies():
    return {"global": DEFAULT_COMPANIES, "nifty50": NIFTY_50_FALLBACK}


class PredictionRequest(BaseModel):
    symbol: str


# ================= DAILY =================
@app.post("/api/predict/daily")
async def predict_daily(req: PredictionRequest):
    try:
        ticker = yf.Ticker(req.symbol)
        hist = ticker.history(period="30d")

        if hist.empty:
            raise HTTPException(status_code=400, detail="No data")

        df_full = hist.copy()
        rsi_window = min(14, len(df_full) - 1) if len(df_full) > 1 else 1
        df_full['RSI'] = calculate_rsi(df_full['Close'], window=rsi_window)
        ma_window = min(10, len(df_full))
        df_full['MA'] = moving_average(df_full['Close'], window=ma_window)

        df = df_full.tail(10).copy()

        prices = df['Close'].tolist()
        dates = [d.strftime("%Y-%m-%d") for d in df.index]

        rsi_val = float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else 50.0
        ma_val = float(df['MA'].iloc[-1]) if not pd.isna(df['MA'].iloc[-1]) else prices[-1]
        vol_val = calculate_volatility(df['Close'])
        price_change = ((prices[-1] - prices[0]) / prices[0]) * 100 if prices[0] != 0 else 0.0
        trend_val = determine_trend(prices, ma_val)

        technical_data = {"rsi": rsi_val, "ma": ma_val}
        engine_result = rule_engine(price_change, rsi_val, trend_val, vol_val)
        headlines = get_news_headlines(req.symbol)

        ai_result = run_finrobot_pipeline(
            prices, "daily", headlines=headlines, indicators=technical_data
        )

        return {
            "symbol": req.symbol,
            "prices": prices,
            "dates": dates,
            "ai_insight": ai_result,
            "rule_engine": engine_result,
            "technicals": {
                "rsi": round(rsi_val, 2),
                "ma": round(ma_val, 2),
                "volatility": round(vol_val, 2),
                "price_change_pct": round(price_change, 2),
                "trend": trend_val
            }
        }

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error")


# ================= INTRADAY =================
@app.post("/api/predict/intraday")
async def predict_intraday(req: PredictionRequest):
    try:
        ticker = yf.Ticker(req.symbol)

        # Fetch 5 days of 15-minute interval data for indicator context
        hist = ticker.history(period="5d", interval="15m")

        if hist.empty:
            raise HTTPException(status_code=400, detail="No intraday data available. Market may be closed.")

        # Convert to IST
        if hist.index.tz is not None:
            hist.index = hist.index.tz_convert('Asia/Kolkata')

        # Always use latest available trading date
        unique_dates = sorted(list(set(hist.index.date)))
        target_date = unique_dates[-1]
        today_date_str = target_date.strftime("%d %B %Y")

        # Calculate RSI and MA on full 5-day data for accuracy
        df_full = hist.copy()
        rsi_window = min(14, len(df_full) - 1) if len(df_full) > 1 else 1
        df_full['RSI'] = calculate_rsi(df_full['Close'], window=rsi_window)
        ma_window = min(20, len(df_full))
        df_full['MA'] = moving_average(df_full['Close'], window=ma_window)

        # Filter to today only.
        # yfinance 15m candles: filter at >= 09:00 to include it.
        market_filter_time = datetime.time(9, 0)
        df_today = df_full[
            (df_full.index.date == target_date) &
            (df_full.index.time >= market_filter_time)
        ].copy()

        if len(df_today) < 1:
            raise HTTPException(
                status_code=400,
                detail="No market data from 9:15 onwards. Market may not have opened yet."
            )

        # Build clean arrays from actual yfinance data (no predicted values used as input)
        actual_times  = [d.strftime("%H:%M") for d in df_today.index]
        actual_prices = [round(float(p), 2) for p in df_today['Close'].tolist()]
        actual_highs  = [round(float(h), 2) for h in df_today['High'].tolist()]
        actual_lows   = [round(float(l), 2) for l in df_today['Low'].tolist()]
        rsi_series = []
        ma_series  = []

        for i, (rsi_v, ma_v) in enumerate(zip(df_today['RSI'].tolist(), df_today['MA'].tolist())):
            rsi_series.append(float(rsi_v) if not pd.isna(rsi_v) else 50.0)
            ma_series.append(float(ma_v)   if not pd.isna(ma_v)  else actual_prices[i])

        # ── BUILD DYNAMIC MARKET SCHEDULE ──────────────────────────────────────
        # yfinance 15m candles are open-stamped (9:15, 9:30, 9:45 …).
        # We use them exactly as returned. Build the schedule from actual
        # timestamps and extend forward in 15-min steps to market close.
        MARKET_CLOSE = datetime.time(15, 30)
        INTERVAL_MINUTES = 15

        # Get the actual times as datetime.time objects for arithmetic
        actual_time_objs = [d.time() for d in df_today.index]

        # Determine the step interval from data (should be 30 min)
        # Build the extended schedule: actual times + future slots
        if actual_time_objs:
            last_time_obj = actual_time_objs[-1]
            schedule_times = list(actual_time_objs)  # already in order
            # Extend forward
            t = last_time_obj
            while True:
                dt = datetime.datetime(2000, 1, 1, t.hour, t.minute) + datetime.timedelta(minutes=INTERVAL_MINUTES)
                t = dt.time()
                if t > MARKET_CLOSE:
                    break
                schedule_times.append(t)
                if t == MARKET_CLOSE:
                    break
            all_market_times = [f"{t.hour:02d}:{t.minute:02d}" for t in schedule_times]
        else:
            all_market_times = []

        # ── PREDICTION CHAIN ──────────────────────────────────────────────────
        # Rule: At step i (time T), predict price at T+15min.
        # ONLY use actual prices[:i+1] as input — never use predicted prices.
        # If T+15min has actual data → compute error and cumulative R².
        # If T+15min is in the future → store as a future prediction only.
        # ─────────────────────────────────────────────────────────────────────
        prediction_chain = []
        completed_actuals = []    # ground-truth targets that we predicted
        completed_predicted = []  # our predictions for those targets

        # Build a lookup of actual data by time string
        actual_lookup = {t: p for t, p in zip(actual_times, actual_prices)}

        for i, (current_time, current_price) in enumerate(zip(actual_times, actual_prices)):
            # Find where this time sits in the full dynamic schedule
            if current_time not in all_market_times:
                continue
            t_idx = all_market_times.index(current_time)

            # There must be a next slot to predict
            if t_idx >= len(all_market_times) - 1:
                break

            target_time = all_market_times[t_idx + 1]

            # Predict using ONLY actual prices up to and including current step
            pred_price = predict_next_interval(
                actual_prices[:i + 1],
                actual_highs[:i + 1],
                actual_lows[:i + 1],
                rsi_series[:i + 1],
                ma_series[:i + 1]
            )

            # Check if the target time has already occurred (actual data available)
            actual_at_target = actual_lookup.get(target_time, None)
            error = None
            r2 = None

            if actual_at_target is not None:
                error = round(abs(actual_at_target - pred_price), 2)
                completed_actuals.append(actual_at_target)
                completed_predicted.append(pred_price)
                r2 = calc_r2(completed_actuals, completed_predicted)

            prediction_chain.append({
                "from_time": current_time,
                "target_time": target_time,
                "actual_price_at_from": current_price,
                "predicted_price": pred_price,
                "actual_price_at_target": actual_at_target,
                "error": error,
                "r2_score": r2,
                "is_future": actual_at_target is None
            })


        # ── OVERALL TECHNICALS & SIDEBAR ──────────────────────────────────────
        rsi_val = rsi_series[-1] if rsi_series else 50.0
        ma_val = ma_series[-1] if ma_series else actual_prices[-1]
        vol_val = calculate_volatility(df_today['Close'])
        price_change = (
            ((actual_prices[-1] - actual_prices[0]) / actual_prices[0]) * 100
            if actual_prices[0] != 0 else 0.0
        )
        trend_val = determine_trend(actual_prices, ma_val)

        technical_data = {"rsi": rsi_val, "ma": ma_val}
        engine_result = rule_engine(price_change, rsi_val, trend_val, vol_val)
        headlines = get_news_headlines(req.symbol)

        ai_result = run_finrobot_pipeline(
            actual_prices, "intraday_today",
            headlines=headlines, indicators=technical_data
        )

        # Attach the deterministic next-30m and closing predictions to ai_result
        future_chain = [c for c in prediction_chain if c["is_future"]]
        if future_chain:
            ai_result["next_30m_price_prediction"] = future_chain[0]["predicted_price"]
            ai_result["closing_price_prediction"] = future_chain[-1]["predicted_price"]
        elif prediction_chain:
            # Market is closed — show last prediction as closing
            ai_result["next_30m_price_prediction"] = prediction_chain[-1]["predicted_price"]
            ai_result["closing_price_prediction"] = prediction_chain[-1]["predicted_price"]

        return {
            "symbol": req.symbol,
            "today_date": today_date_str,
            "all_market_times": all_market_times,
            "actual_times": actual_times,
            "actual_prices": actual_prices,
            "prediction_chain": prediction_chain,
            "ai_insight": ai_result,
            "rule_engine": engine_result,
            "technicals": {
                "rsi": round(rsi_val, 2),
                "ma": round(ma_val, 2),
                "volatility": round(vol_val, 2),
                "price_change_pct": round(price_change, 2),
                "trend": trend_val
            }
        }

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Error")


# ================= STATIC =================
os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", reload=True)