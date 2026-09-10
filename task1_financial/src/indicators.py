"""
Technical indicators computed from first principles (no TA-Lib).

Each function takes a pandas Series/DataFrame of OHLCV data and returns
a pandas Series aligned to the input index. NaNs are left in place for
the warm-up window rather than dropped, so callers can decide how to
handle them (see data_pipeline.py for the null-handling policy).
"""
import pandas as pd
import numpy as np


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple Moving Average."""
    return close.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index using Wilder's smoothing (the standard
    definition -- a plain rolling mean of gains/losses is a common but
    incorrect shortcut).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    # Where avg_loss is 0 and avg_gain > 0, RSI is 100 by definition
    rsi_series = rsi_series.where(avg_loss != 0, 100.0)
    return rsi_series


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    MACD line, signal line, and histogram.
    Returns a DataFrame with columns: macd, signal, histogram.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "histogram": histogram}
    )


def bollinger_bands(close: pd.Series, window: int = 20, num_std: float = 2.0):
    """
    Bollinger Bands. Returns a DataFrame with columns:
    middle_band, upper_band, lower_band, percent_b.
    """
    middle = close.rolling(window=window, min_periods=window).mean()
    std = close.rolling(window=window, min_periods=window).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    # %B: where price sits within the bands, useful as a momentum signal
    percent_b = (close - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame(
        {
            "middle_band": middle,
            "upper_band": upper,
            "lower_band": lower,
            "percent_b": percent_b,
        }
    )


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given an OHLCV DataFrame with a 'Close' column, attach all required
    indicators as new columns and return the enriched DataFrame.
    """
    out = df.copy()
    out["SMA_50"] = sma(out["Close"], 50)
    out["SMA_200"] = sma(out["Close"], 200)
    out["RSI_14"] = rsi(out["Close"], 14)

    macd_df = macd(out["Close"], 12, 26, 9)
    out["MACD"] = macd_df["macd"]
    out["MACD_signal"] = macd_df["signal"]
    out["MACD_hist"] = macd_df["histogram"]

    bb_df = bollinger_bands(out["Close"], 20, 2.0)
    out["BB_mid"] = bb_df["middle_band"]
    out["BB_upper"] = bb_df["upper_band"]
    out["BB_lower"] = bb_df["lower_band"]
    out["BB_percent_b"] = bb_df["percent_b"]

    return out


def derive_momentum_signal(latest_row: pd.Series) -> str:
    """
    Combine SMA crossover, RSI, and MACD histogram into a simple
    rule-based momentum label. This is a *feature*, not the final
    Buy/Hold/Sell call -- that reasoning is delegated to the LLM in
    Task 1B, which reasons over these signals rather than restating them.
    """
    score = 0
    if pd.notna(latest_row.get("SMA_50")) and pd.notna(latest_row.get("SMA_200")):
        score += 1 if latest_row["SMA_50"] > latest_row["SMA_200"] else -1
    if pd.notna(latest_row.get("RSI_14")):
        if latest_row["RSI_14"] > 70:
            score -= 1  # overbought
        elif latest_row["RSI_14"] < 30:
            score += 1  # oversold, potential reversal up
    if pd.notna(latest_row.get("MACD_hist")):
        score += 1 if latest_row["MACD_hist"] > 0 else -1

    if score >= 2:
        return "bullish"
    if score <= -2:
        return "bearish"
    return "neutral"
