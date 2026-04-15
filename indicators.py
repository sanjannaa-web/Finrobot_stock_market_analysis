import pandas as pd
import numpy as np

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def moving_average(data, window=10):
    return data.rolling(window).mean()

def calculate_volatility(data):
    """Calculate annualized volatility from price series."""
    returns = data.pct_change().dropna()
    if len(returns) < 2:
        return 0.0
    return float(returns.std() * 100)  # as percentage

def determine_trend(prices: list, ma: float) -> str:
    """Determine trend using last price vs Moving Average."""
    if len(prices) < 2:
        return "sideways"
    last_price = prices[-1]
    if last_price > ma * 1.001:
        return "up"
    elif last_price < ma * 0.999:
        return "down"
    else:
        return "sideways"

def rule_engine(price_change: float, rsi: float, trend: str, volatility: float) -> dict:
    """
    Deterministic rule-based financial engine.
    No AI, no reasoning, no guessing — strict numeric execution only.

    INPUT:
        price_change (float): percentage change over period
        rsi (float): 0–100
        trend (str): "up", "down", or "sideways"
        volatility (float): numeric

    PROCESS:
        Initialize score = 0
        1. RSI:     rsi < 30 → +25  |  rsi > 70 → -25
        2. Price:   change > 2 → +20 | change < -2 → -20
        3. Trend:   "up" → +15 | "down" → -15
        4. Volatility: > 5 → -10

    DECISION:
        score >= 30 → BUY
        score <= -30 → SELL
        else → HOLD

    CONFIDENCE:
        abs(score) * 2, capped at 100

    OUTPUT: dict with decision, confidence, score
    """

    # Initialize
    score = 0

    # Rule 1: RSI
    if rsi < 30:
        score += 25
    elif rsi > 70:
        score -= 25

    # Rule 2: Price Change
    if price_change > 2:
        score += 20
    elif price_change < -2:
        score -= 20

    # Rule 3: Trend
    if trend == "up":
        score += 15
    elif trend == "down":
        score -= 15

    # Rule 4: Volatility
    if volatility > 5:
        score -= 10

    # Decision
    if score >= 30:
        decision = "BUY"
    elif score <= -30:
        decision = "SELL"
    else:
        decision = "HOLD"

    # Confidence
    confidence = max(15, abs(score) * 2)
    if confidence > 100:
        confidence = 100

    return {
        "decision": decision,
        "confidence": confidence,
        "score": score
    }