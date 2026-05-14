"""
Stocks & Commodity Price Forecaster — Streamlit dashboard.
Run from project root: streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched via Streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import io
from datetime import datetime, timezone, timedelta
import pytz

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf
import datetime as dt

from config import (
    ASSETS,
    DEFAULT_FORECAST_DAYS,
    DEFAULT_INTERVAL,
    DEFAULT_PERIOD,
)
from src.data_loader import load_commodity_history
from src.model_ensemble import attach_forecast, train_and_backtest

st.set_page_config(
    page_title="Stocks & Commodity Price Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=Poppins:wght@400;600&display=swap');

    .title-wrapper {
        display: flex;
        justify-content: center;
        padding: 0 0 0.05rem;
        margin-bottom: 0.5rem;
    }
    .main-header {
        width: min(90vw, 900px);
        font-size: 1.75rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        line-height: 1;
        margin: 0;
        padding: 0.12rem 0.08rem 0.15rem;
        color: #2c1f14;
        background: linear-gradient(135deg, #fbf7f1 0%, #f4ede5 100%);
        border: 1px solid rgba(108, 88, 68, 0.18);
        border-radius: 2.4rem;
        box-shadow: 0 20px 36px rgba(69, 49, 30, 0.08);
        text-align: center;
        font-family: 'Poppins', sans-serif;
        position: relative;
        overflow: hidden;
    }
    .main-header span {
        display: block;
        font-family: 'Poppins', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        text-transform: none;
        margin-bottom: 0.05rem;
        color: #24180f;
        line-height: 1.02;
    }
    .main-header strong {
        display: block;
        font-family: 'Poppins', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #7d664d;
        line-height: 1.1;
        margin-bottom: 0;
    }
    @keyframes headerLift {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-1.8px); }
        100% { transform: translateY(0px); }
    }
    .subtle { color: #5E412F; font-size: 0.95rem; margin-bottom: 1.25rem; font-family: 'Georgia', serif; }
    
    /* All metric boxes and containers */
    div[data-testid="stMetric"] { 
        background: linear-gradient(135deg, #FFF3D6 0%, #FFE8B8 100%) !important;
        border: 2px solid #8B4513 !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        color: #4A2E1A !important;
    }
    
    /* Reduce font size for metric values */
    div[data-testid="stMetric"] div:last-child {
        font-size: 0.85rem !important;
    }
    
    /* Card styling */
    .reportview-container .metric-container { 
        background: linear-gradient(135deg, #FFF4DD 0%, #FFE6C0 100%);
        border: 3px solid #8B4513;
        border-radius: 12px;
    }
    
    /* All section blocks */
    section[data-testid="stVerticalBlock"] div:has(> [data-testid="stMetric"]) {
        background: transparent !important;
    }
    
    body { background: linear-gradient(135deg, #FFFCFA 0%, #FFF9F4 60%, #FFFCFA 100%); font-family: 'Georgia', serif; margin: 0; }
    .stApp { background: linear-gradient(135deg, #FFFCFA 0%, #FFF9F4 60%, #FFFCFA 100%); font-family: 'Georgia', serif; }
    main > div.block-container { padding: 0.05rem 0.75rem 0.25rem; }
    section[data-testid="stVerticalBlock"] { padding: 0.05rem 0 0.25rem; margin: 0; }
    h1, h2, h3, h4, h5, h6 { color: #5B3C2B; font-family: 'Georgia', serif; }
    p, div, span, label { color: #4B2E2A; font-family: 'Georgia', serif; }
    
    .stSlider .rc-slider-handle,
    .stSlider .rc-slider-handle:focus,
    .stSlider .rc-slider-handle:hover,
    .stSlider .rc-slider-handle:active {
        box-shadow: none !important;
        outline: none !important;
    }
    
    /* Reduce header space but keep action buttons visible */
    header[data-testid="stHeader"] {
        min-height: 2rem !important;
        height: 2rem !important;
    }
    .stApp header {
        padding-top: 0.5rem !important;
        padding-bottom: 0.5rem !important;
        margin-bottom: 0 !important;
    }
    
    .stTabs [data-baseweb="tab-list"] { background-color: transparent; border: none; }
    .stTabs [data-baseweb="tab"] { color: #654321; font-family: 'Georgia', serif; background-color: transparent; border: none; padding: 0.5rem 1rem; }
    .stTabs [data-baseweb="tab"][aria-selected="true"] { 
        background: linear-gradient(145deg, #FFD1B6 0%, #FFB179 100%);
        color: #654321;
        border: 2px solid #DAA520;
        border-radius: 8px;
    }
    
    .stSidebar { 
        background: linear-gradient(180deg, #FFFDF8 0%, #FFF6E8 100%);
        color: #3E2F1F;
        font-family: 'Poppins', sans-serif;
        border: 1px solid rgba(108, 88, 68, 0.16);
        box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.7);
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar p, .stSidebar div, .stSidebar label, .stSidebar span { 
        color: #3E2F1F;
        font-family: 'Poppins', sans-serif;
        font-size: 0.95rem;
    }
    .stSidebar label {
        font-weight: 600;
        letter-spacing: 0.01em;
    }
    .stSidebar .stTextInput > div > label,
    .stSidebar .stNumberInput > div > label,
    .stSidebar .stSelectbox > div > label,
    .stSidebar .stSlider > label,
    .stSidebar .stCheckbox > label {
        color: #3E2F1F !important;
        font-family: 'Poppins', sans-serif !important;
        font-size: 0.95rem !important;
    }
    .stSidebar .css-1adrfps {
        background: transparent !important;
    }
    
    .stButton>button { 
        background: linear-gradient(145deg, #FF6347 0%, #FF7F50 100%);
        color: white;
        border: 2px solid #CD5C5C;
        border-radius: 8px;
        font-family: 'Georgia', serif;
        padding: 0.35rem 0.7rem !important;
        font-size: 0.95rem !important;
        min-height: auto !important;
    }
    
    .stButton>button:hover { 
        background: linear-gradient(145deg, #FF7F50 0%, #FF6347 100%);
    }
    
    .stSelectbox, .stSlider, .stCheckbox { 
        font-family: 'Georgia', serif;
        color: #4B2E2A;
    }
    
    .stSelectbox label, .stSlider label, .stCheckbox label { 
        color: #4B2E2A;
        font-family: 'Georgia', serif;
    }
    
    /* Dropdown styling with contrasting colors */
    .stSelectbox [data-baseweb="select"] {
        background: linear-gradient(135deg, #FF8C69 0%, #FF7F50 100%) !important;
        border: 2px solid #654321 !important;
        border-radius: 8px !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: white !important;
    }
    
    .stSelectbox [data-baseweb="select"] svg {
        fill: white !important;
    }
    
    /* Input fields styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        background: linear-gradient(135deg, #FF8C69 0%, #FF7F50 100%) !important;
        border: 2px solid #654321 !important;
        color: white !important;
    }

    /* Restore slider track while keeping the handle clean */
    .stSlider .rc-slider-rail {
        background: linear-gradient(90deg, #FFB6B6 0%, #FFA07A 100%) !important;
        opacity: 0.45 !important;
    }
    .stSlider .rc-slider-track {
        background: linear-gradient(90deg, #FF6347 0%, #FF7F50 100%) !important;
    }
    .stSlider .rc-slider-handle {
        background: #ffffff !important;
        border: 2px solid #654321 !important;
        box-shadow: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)





@st.cache_data(ttl=3600, show_spinner=False)
def _load_data(ticker: str, period: str, interval: str, use_cache: bool):
    return load_commodity_history(ticker, period=period, interval=interval, use_cache=use_cache)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_news(ticker: str) -> list[dict[str, str]]:
    try:
        ticker_obj = yf.Ticker(ticker)
        raw_news = ticker_obj.news or []
        headlines = []
        for item in raw_news[:8]:
            content = item.get("content", {})
            title = content.get("title", "")
            provider = content.get("provider", {}).get("displayName", "")
            pub_date = content.get("pubDate", "")
            published = ""
            if pub_date:
                try:
                    # Parse ISO format date
                    date_obj = dt.datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
                    published = date_obj.strftime("%Y-%m-%d %H:%M")
                except:
                    published = pub_date
            canonical_url = content.get("canonicalUrl", {}).get("url", "")
            headlines.append({"title": title, "provider": provider, "published": published, "link": canonical_url})
        return headlines
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []


def get_currency_symbol(asset_key: str) -> str:
    """Return currency symbol based on asset type.
    
    Args:
        asset_key: The asset key from ASSETS dictionary (e.g., "Indian: ...", "American: ...", "Commodity: ...")
    
    Returns:
        Currency symbol: "₹" for Indian stocks, "$" for others
    """
    if "Indian:" in asset_key:
        return "₹"
    return "$"


def _is_market_holiday(date: pd.Timestamp) -> bool:
    """Check if a date is a US market holiday."""
    # US Market holidays (simplified list for common holidays)
    us_holidays = {
        (1, 1),    # New Year's Day
        (1, 20),   # MLK Day (3rd Monday of January) - approximate
        (2, 17),   # Presidents' Day (3rd Monday of February) - approximate
        (3, 17),   # Good Friday (varies) - approximate
        (5, 26),   # Memorial Day (last Monday of May) - approximate
        (7, 4),    # Independence Day
        (9, 1),    # Labor Day (1st Monday of September) - approximate
        (11, 28),  # Thanksgiving (4th Thursday of November) - approximate
        (12, 25),  # Christmas
    }
    return (date.month, date.day) in us_holidays or date.weekday() >= 5  # 5=Saturday, 6=Sunday


def _get_closing_price_with_timestamp(df: pd.DataFrame, asset_key: str) -> tuple[float, str, bool, str]:
    """Get the previous trading day's closing price with timestamp and timezone, and check if market was closed.
    
    Returns:
        Tuple of (price, formatted_timestamp_with_tz, is_holiday_or_weekend, holiday_message)
    """
    try:
        # Determine timezone based on asset type
        if "Indian:" in asset_key:
            market_tz = pytz.timezone('Asia/Kolkata')
            tz_label = "IST"
        else:
            market_tz = pytz.timezone('US/Eastern')
            tz_label = "EDT"
        
        # Get the PREVIOUS trading day (index -2, since -1 is the last available)
        if len(df) >= 2:
            prev_date = df.index[-2]
            price = df["close"].iloc[-2]
        else:
            # If only 1 day, get that day
            prev_date = df.index[-1]
            price = df["close"].iloc[-1]
        
        # Localize the date to market timezone with market close time
        # Set close time: IST 15:30 (3:30 PM), EDT 16:00 (4:00 PM)
        close_hour = 15 if "Indian:" in asset_key else 16
        close_minute = 30 if "Indian:" in asset_key else 0
        
        # Create datetime with market close time
        prev_datetime = prev_date.replace(hour=close_hour, minute=close_minute, second=0)
        
        if df.index.tz is None:
            prev_date_tz = market_tz.localize(prev_datetime)
        else:
            prev_date_tz = pd.Timestamp(prev_datetime).tz_localize(None).tz_localize(market_tz)
        
        formatted_timestamp = prev_date_tz.strftime("%Y-%m-%d %H:%M:%S %Z")
        
        # Check if it's a holiday or weekend
        today_aware = datetime.now(tz=pytz.UTC).astimezone(market_tz)
        today_ts = pd.Timestamp(today_aware.date())
        is_today_holiday = _is_market_holiday(today_ts)
        is_yesterday_holiday = _is_market_holiday(prev_date)
        
        holiday_message = ""
        if is_today_holiday:
            holiday_message = "Today is a market holiday"
        elif is_yesterday_holiday:
            holiday_message = "Yesterday was a market holiday"
        elif prev_date.weekday() >= 5:
            holiday_message = "Last trading day was a weekend"
        
        return float(price), formatted_timestamp, (is_today_holiday or is_yesterday_holiday), holiday_message
    except Exception as e:
        print(f"Error getting closing price: {e}")
        return float(df["close"].iloc[-1]), "N/A", False, ""


def _generate_backtest_report() -> tuple[io.BytesIO, str]:
    """Generate comprehensive backtest report for all assets across all history windows.
    
    Returns:
        Tuple of (BytesIO object with Excel file, filename string)
    """
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.maximum(np.abs(y_true), 1e-8)
        return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    
    def get_approx_return(close: pd.Series) -> float:
        """Calculate approximate 1-year return (252 trading days)."""
        if len(close) < 252:
            daily_return = close.pct_change().mean()
            return float(daily_return * 252 * 100) if np.isfinite(daily_return) else 0.0
        else:
            return float((close.iloc[-1] / close.iloc[-252] - 1) * 100)
    
    history_windows = ["2y", "3y", "4y"]
    results = []
    
    for asset_display, ticker in sorted(ASSETS.items()):
        for window in history_windows:
            try:
                df = load_commodity_history(ticker, period=window, interval=DEFAULT_INTERVAL, use_cache=True)
                close = df["close"].astype(float)
                
                if len(close) < 30:
                    continue
                
                n_obs = len(close)
                approx_return = get_approx_return(close)
                
                fit = train_and_backtest(close)
                bt = fit.backtest
                
                result = {
                    "Asset": asset_display,
                    "History Window": window,
                    "Observations": n_obs,
                    "Approx 1Y Return (%)": round(approx_return, 2),
                    "MAE": round(bt.mae, 4),
                    "RMSE": round(bt.rmse, 4),
                    "MAPE (%)": round(bt.mape, 2),
                }
                results.append(result)
            except Exception:
                continue
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(by=["Asset", "History Window"]).reset_index(drop=True)
    
    # Export to Excel
    output_buffer = io.BytesIO()
    with pd.ExcelWriter(output_buffer, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Results", index=False)
        worksheet = writer.sheets["Results"]
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    output_buffer.seek(0)
    filename = f"backtest_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return output_buffer, filename


def _clean_asset_display_name(asset_key: str) -> str:
    """Create a clean readable asset name for compare labels.

    Removes category prefixes like "Indian:", "American:", "Commodity:" and drops ticker symbols
    in parentheses such as "(AAPL)".
    """
    name = asset_key
    for prefix in ("Commodity: ", "American: ", "Indian: "):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    if name.endswith(")") and " (" in name:
        name = name[: name.rfind(" (")]
    return name


def _score_headlines_sentiment(headlines: list[dict[str, str]]) -> tuple[float, str]:
    positive = ["up", "rise", "gain", "beat", "strong", "bull", "positive", "surge", "jump", "record"]
    negative = ["down", "fall", "loss", "weak", "miss", "sell", "bear", "drop", "cut", "decline"]
    score = 0.0
    for item in headlines:
        text = item["title"].lower()
        for word in positive:
            if word in text:
                score += 1
        for word in negative:
            if word in text:
                score -= 1
    if not headlines:
        return 0.0, "Neutral"
    normalized = score / max(len(headlines), 1)
    label = "Neutral"
    if normalized >= 0.75:
        label = "Strong Positive"
    elif normalized >= 0.25:
        label = "Positive"
    elif normalized <= -0.75:
        label = "Strong Negative"
    elif normalized <= -0.25:
        label = "Negative"
    return normalized, label


def _headline_sentiment_score(title: str) -> tuple[float, str]:
    positive = ["up", "rise", "gain", "beat", "strong", "bull", "positive", "surge", "jump", "record"]
    negative = ["down", "fall", "loss", "weak", "miss", "sell", "bear", "drop", "cut", "decline"]
    text = title.lower()
    positive_count = sum(1 for word in positive if word in text)
    negative_count = sum(1 for word in negative if word in text)
    score = positive_count - negative_count
    if score >= 2:
        label = "Strong Positive"
    elif score == 1:
        label = "Positive"
    elif score == 0:
        label = "Neutral"
    elif score == -1:
        label = "Negative"
    else:
        label = "Strong Negative"
    normalized = max(min(score / 3.0, 1.0), -1.0)
    return normalized, label


def _fig_news_sentiment_overlay(close: pd.Series, headlines: list[dict[str, str]], currency_symbol: str) -> go.Figure:
    dates = []
    prices = []
    sentiment_scores = []
    sentiment_labels = []
    colors = []

    for item in headlines:
        published = item.get("published", "")
        try:
            date_obj = dt.datetime.fromisoformat(published.replace("Z", "+00:00"))
            headline_date = pd.Timestamp(date_obj.date())
        except Exception:
            continue

        price = close.asof(headline_date)
        if pd.isna(price):
            continue

        score, label = _headline_sentiment_score(item.get("title", ""))
        color = "#2E8B57" if score > 0 else "#8B4513" if score == 0 else "#B22222"

        dates.append(headline_date)
        prices.append(price)
        sentiment_scores.append(score)
        sentiment_labels.append(f"{item.get('title', '')} ({label})")
        colors.append(color)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Price",
            line=dict(color="#FF7F50", width=1.8),
        ),
        secondary_y=False,
    )

    if dates:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="markers",
                name="News events",
                marker=dict(color=colors, size=12, symbol="diamond"),
                hovertext=sentiment_labels,
                hoverinfo="text",
            ),
            secondary_y=False,
        )
        fig.add_trace(
            go.Bar(
                x=dates,
                y=sentiment_scores,
                name="Sentiment score",
                marker_color=["#2E8B57" if s > 0 else "#8B4513" if s == 0 else "#B22222" for s in sentiment_scores],
                opacity=0.6,
            ),
            secondary_y=True,
        )

    fig.update_layout(
        title=dict(text="News & Sentiment Overlay", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    fig.update_yaxes(title_text=f"Price ({currency_symbol})", secondary_y=False, showgrid=True, gridcolor="rgba(139, 69, 19, 0.3)")
    fig.update_yaxes(title_text="Sentiment score", secondary_y=True, range=[-1, 1], showgrid=False)
    return fig


def _fig_asset_comparison(
    primary_close: pd.Series,
    primary_name: str,
    compare_close: pd.Series,
    compare_name: str,
) -> go.Figure:
    df = pd.DataFrame({primary_name: primary_close, compare_name: compare_close}).dropna()
    normalized = df / df.iloc[0] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[primary_name],
            mode="lines",
            name=primary_name,
            line=dict(color="#FF7F50", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=normalized.index,
            y=normalized[compare_name],
            mode="lines",
            name=compare_name,
            line=dict(color="#8B4513", width=2),
        )
    )
    fig.update_layout(
        title=dict(text=f"{primary_name} vs {compare_name} — normalized comparison", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    fig.update_xaxes(title_text="Date", showgrid=True, gridcolor="rgba(139, 69, 19, 0.3)")
    fig.update_yaxes(title_text="Normalized performance (100 = start)", showgrid=True, gridcolor="rgba(139, 69, 19, 0.3)")
    return fig


def _compose_trade_signal(forecast_change_pct: float, sentiment_score: float, forecast_vol: float) -> tuple[str, str]:
    if forecast_change_pct > 1 and sentiment_score > 0.25:
        return "Strong Buy", "Forecast momentum and sentiment are aligned. Consider adding on dips."
    if forecast_change_pct > 0.5 and sentiment_score >= 0:
        return "Buy", "Moderate upside expected with neutral-to-positive news sentiment."
    if forecast_change_pct < -1 and sentiment_score < -0.25:
        return "Strong Sell", "Negative sentiment and forecast direction suggest significant downside risk."
    if forecast_change_pct < -0.5 and sentiment_score <= 0:
        return "Sell", "Forecast points down and headlines are cautious."
    if abs(forecast_change_pct) < 0.4 and abs(sentiment_score) < 0.25:
        return "Hold", "The model and news are mixed. Wait for clearer direction."
    if forecast_change_pct > 0:
        return "Watch / Hold", "Forecast is positive but sentiment is muted. Monitor the next data points."
    return "Watch / Hold", "Forecast is negative but sentiment is not strongly bearish. Avoid committing too soon."


def _relative_comparison_recommendation(
    primary_name: str,
    primary_forecast_pct: float,
    primary_sentiment_label: str,
    compare_name: str,
    compare_forecast_pct: float,
    compare_sentiment_label: str,
) -> tuple[str, str]:
    if not np.isfinite(primary_forecast_pct) and not np.isfinite(compare_forecast_pct):
        return (
            "Comparison unavailable",
            "Both asset forecasts are unavailable. Review the market data and retry later.",
        )

    def sentiment_value(label: str) -> float:
        return {
            "Strong Positive": 1.0,
            "Positive": 0.5,
            "Neutral": 0.0,
            "Negative": -0.5,
            "Strong Negative": -1.0,
        }.get(label, 0.0)

    primary_score = (primary_forecast_pct if np.isfinite(primary_forecast_pct) else 0.0) + sentiment_value(primary_sentiment_label) * 1.2
    compare_score = (compare_forecast_pct if np.isfinite(compare_forecast_pct) else 0.0) + sentiment_value(compare_sentiment_label) * 1.2

    if primary_score > compare_score + 0.8:
        return (
            f"{primary_name} appears stronger for near-term opportunity",
            f"{primary_name} has a more favorable forecast move and sentiment profile than {compare_name}. Consider this asset first if you are looking for a higher momentum candidate.",
        )
    if compare_score > primary_score + 0.8:
        return (
            f"{compare_name} appears stronger for near-term opportunity",
            f"{compare_name} has a more favorable forecast move and sentiment profile than {primary_name}. It may offer a better risk/reward profile over the selected window.",
        )
    return (
        "Assets are closely matched",
        f"Both {primary_name} and {compare_name} show similar forecast potential and sentiment. Use risk tolerance and portfolio balance to decide between them.",
    )


def _fig_price_history(close: pd.Series, name: str, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close",
            line=dict(color="#FF7F50", width=1.8),
        )
    )
    fig.update_layout(
        title=dict(text=f"{name} — historical close", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_backtest(actual: pd.Series, predicted: pd.Series, name: str, currency_symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="Actual",
            line=dict(color="#8B4513", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=predicted.index,
            y=predicted.values,
            mode="lines",
            name="Predicted",
            line=dict(color="#FFA07A", width=2),
        )
    )
    fig.update_layout(
        title=dict(text=f"{name} — backtest actual vs predicted", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=380,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_forecast(
    history: pd.Series,
    fc_index: pd.DatetimeIndex,
    fc_values: np.ndarray,
    name: str,
    currency_symbol: str,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history.index,
            y=history.values,
            mode="lines",
            name="History",
            line=dict(color="#FF7F50", width=1.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fc_index,
            y=fc_values,
            mode="lines",
            name="Forecast",
            line=dict(color="#8B4513", width=2),
        ),
    )
    fig.add_vline(
        x=history.index[-1],
        line_width=1,
        line_dash="dash",
        line_color="#8B4513",
    )
    fig.update_layout(
        title=dict(text=f"{name} — history + recursive multi-day forecast", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=400,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#3D9024"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="h", yanchor="bottom", y=0.02, xanchor="left", x=0.02, bgcolor="rgba(255, 255, 255, 0.9)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_importance(model, feature_names: list[str]) -> go.Figure:
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        fig = go.Figure()
        fig.update_layout(title="Feature importance (n/a)", template="plotly_white", height=200)
        return fig
    order = np.argsort(imp)[::-1][:15]
    fig = go.Figure(
        go.Bar(
            x=imp[order],
            y=[feature_names[i] for i in order],
            orientation="h",
            marker_color="#FFA07A",
        )
    )
    fig.update_layout(
        title=dict(text="Top feature importances (ensemble method)", font=dict(size=14, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        margin=dict(l=120, r=60, t=45, b=40),
        height=360,
        xaxis_title="Importance",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
    )
    return fig


def _fig_models_backtest_comparison(fit, actual: pd.Series, currency_symbol: str, name: str) -> go.Figure:
    """Compare backtest predictions from all 4 models vs actual prices."""
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(
        go.Scatter(
            x=actual.index,
            y=actual.values,
            mode="lines",
            name="Actual",
            line=dict(color="#000000", width=3),
            opacity=0.8,
        )
    )
    
    # Colors for each model
    model_colors = {
        "gbr": "#FF6B6B",
        "lightgbm": "#4ECDC4",
        "catboost": "#FFE66D",
        "xgb": "#95E1D3",
    }
    
    # Add predictions from individual models
    individual_models = fit.individual_models
    for model_name in ["gbr", "lightgbm", "catboost", "xgb"]:
        if model_name in individual_models:
            model_result = individual_models[model_name]
            bt = model_result.backtest
            fig.add_trace(
                go.Scatter(
                    x=bt.y_pred.index,
                    y=bt.y_pred.values,
                    mode="lines",
                    name=model_name.upper(),
                    line=dict(color=model_colors.get(model_name, "#808080"), width=1.8),
                    opacity=0.7,
                )
            )
    
    # Add ensemble prediction
    ensemble_bt = fit.backtest
    fig.add_trace(
        go.Scatter(
            x=ensemble_bt.y_pred.index,
            y=ensemble_bt.y_pred.values,
            mode="lines",
            name="Ensemble",
            line=dict(color="#FF7F50", width=2.5, dash="dash"),
        )
    )
    
    fig.update_layout(
        title=dict(text=f"{name} — Model Backtest Comparison (Actual vs Predicted)", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=450,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.02, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_models_forecast_comparison(fit, currency_symbol: str, name: str) -> go.Figure:
    """Compare forecast predictions from all 4 models."""
    fig = go.Figure()
    
    # Add history
    fig.add_trace(
        go.Scatter(
            x=fit.history_close.index,
            y=fit.history_close.values,
            mode="lines",
            name="History",
            line=dict(color="#654321", width=2),
        )
    )
    
    # Colors for each model
    model_colors = {
        "gbr": "#FF6B6B",
        "lightgbm": "#4ECDC4",
        "catboost": "#FFE66D",
        "xgb": "#95E1D3",
    }
    
    # Add forecasts from individual models
    individual_models = fit.individual_models
    for model_name in ["gbr", "lightgbm", "catboost", "xgb"]:
        if model_name in individual_models:
            model_result = individual_models[model_name]
            if len(model_result.forecast_close) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=model_result.forecast_index,
                        y=model_result.forecast_close,
                        mode="lines+markers",
                        name=f"{model_name.upper()} Forecast",
                        line=dict(color=model_colors.get(model_name, "#808080"), width=2),
                        marker=dict(size=6),
                    )
                )
    
    # Add ensemble forecast
    if len(fit.forecast_close) > 0:
        fig.add_trace(
            go.Scatter(
                x=fit.forecast_index,
                y=fit.forecast_close,
                mode="lines+markers",
                name="Ensemble Forecast",
                line=dict(color="#FF7F50", width=3, dash="dash"),
                marker=dict(size=8, symbol="star"),
            )
        )
    
    # Add vertical line at last history point
    if len(fit.history_close) > 0:
        fig.add_vline(
            x=fit.history_close.index[-1],
            line_width=1,
            line_dash="dot",
            line_color="#8B4513",
        )
    
    fig.update_layout(
        title=dict(text=f"{name} — Model Forecast Comparison", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        hovermode="x unified",
        margin=dict(l=40, r=60, t=50, b=40),
        height=450,
        xaxis_title="Date",
        yaxis_title=f"Price ({currency_symbol})",
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.02, bgcolor="rgba(255, 255, 255, 0.95)", bordercolor="#654321", borderwidth=1),
    )
    return fig


def _fig_models_performance_metrics(fit) -> go.Figure:
    """Create a bar chart comparing performance metrics of all models."""
    models = ["gbr", "lightgbm", "catboost", "xgb", "ensemble"]
    mae_values = []
    rmse_values = []
    mape_values = []
    
    # Get metrics for individual models
    individual_models = fit.individual_models
    for model_name in ["gbr", "lightgbm", "catboost", "xgb"]:
        if model_name in individual_models:
            bt = individual_models[model_name].backtest
            mae_values.append(bt.mae)
            rmse_values.append(bt.rmse)
            mape_values.append(bt.mape)
    
    # Get ensemble metrics
    ensemble_bt = fit.backtest
    mae_values.append(ensemble_bt.mae)
    rmse_values.append(ensemble_bt.rmse)
    mape_values.append(ensemble_bt.mape)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("MAE", "RMSE", "MAPE (%)"),
        horizontal_spacing=0.12,
    )
    
    # Colors for models
    colors = ["#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#FF7F50"]
    
    # Add MAE
    fig.add_trace(
        go.Bar(x=models, y=mae_values, name="MAE", marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # Add RMSE
    fig.add_trace(
        go.Bar(x=models, y=rmse_values, name="RMSE", marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # Add MAPE
    fig.add_trace(
        go.Bar(x=models, y=mape_values, name="MAPE", marker_color=colors, showlegend=False),
        row=1, col=3
    )
    
    fig.update_yaxes(title_text="MAE", row=1, col=1)
    fig.update_yaxes(title_text="RMSE", row=1, col=2)
    fig.update_yaxes(title_text="MAPE (%)", row=1, col=3)
    
    fig.update_layout(
        title=dict(text="Model Performance Metrics Comparison", font=dict(size=16, color="#654321", family="Georgia, serif")),
        template="plotly_white",
        plot_bgcolor="#FAF0E6",
        paper_bgcolor="#FFFFF0",
        height=400,
        showlegend=False,
    )
    
    return fig


def _get_report_file_path() -> Path:
    """Get the path where the comprehensive backtest report should be stored."""
    report_dir = ROOT / "data"
    report_dir.mkdir(exist_ok=True)
    return report_dir / "backtest_report_cache.xlsx"


def _report_exists_and_fresh(max_age_hours: int = 24) -> bool:
    """Check if a cached report exists and is fresh (not older than max_age_hours).
    
    Args:
        max_age_hours: Maximum age of the report in hours (default: 24)
    
    Returns:
        True if report exists and is fresh, False otherwise
    """
    report_path = _get_report_file_path()
    if not report_path.exists():
        return False
    
    file_age_seconds = (datetime.now(tz=timezone.utc) - datetime.fromtimestamp(report_path.stat().st_mtime, tz=timezone.utc)).total_seconds()
    file_age_hours = file_age_seconds / 3600
    
    return file_age_hours < max_age_hours


def _get_or_generate_report(force_regenerate: bool = False) -> tuple[io.BytesIO, str]:
    """Get existing report or generate a new one if needed.
    
    Args:
        force_regenerate: If True, always generate a new report
    
    Returns:
        Tuple of (BytesIO object with Excel file, filename string)
    """
    report_path = _get_report_file_path()
    
    # Use cached report if it exists and is fresh, unless force_regenerate is True
    if not force_regenerate and report_path.exists() and _report_exists_and_fresh():
        # Read from cache
        output_buffer = io.BytesIO(report_path.read_bytes())
        return output_buffer, report_path.name
    
    # Generate new report
    output_buffer, _ = _generate_backtest_report()
    
    # Save to cache
    report_path.write_bytes(output_buffer.getvalue())
    output_buffer.seek(0)
    
    return output_buffer, report_path.name


def main():
    st.markdown(
        '<div class="title-wrapper"><h1 class="main-header"><span>STOCKS & COMMODITIES</span><strong>PRICE FORECASTING ENGINE</strong></h1></div>',
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.header("Parameters")
        asset_labels = []
        asset_keys = list(ASSETS.keys())
        for k in asset_keys:
            currency_sym = get_currency_symbol(k)
            asset_labels.append(f"{currency_sym} {k}")
        # Default to first asset
        asset_label = st.selectbox("Asset", asset_labels, index=0)
        # Map back to the real asset key
        if asset_label.startswith("₹ "):
            asset = asset_label[2:]
        elif asset_label.startswith("$ "):
            asset = asset_label[2:]
        else:
            asset = asset_label
        ticker = ASSETS[asset]
        asset_name = asset.split(" - ", 1)[1] if " - " in asset else asset
        currency_symbol = get_currency_symbol(asset)
        asset_display_name = _clean_asset_display_name(asset)

        period = "2y"
        forecast_days = st.slider("Forecast horizon (trading days)", 5, 15, DEFAULT_FORECAST_DAYS)
        
        # Refresh button with round arrow icon
        col_refresh1, col_refresh2 = st.columns([1, 4])
        with col_refresh1:
            refresh = st.button("🔄", help="Refresh data", key="refresh_btn")
        with col_refresh2:
            st.markdown("<p style='margin: 0.5rem 0; color: #5B3C2B;'>Refresh data</p>", unsafe_allow_html=True)
        
        st.markdown(
            """
            **IMPORTANT DISCLAIMER & SEBI COMPLIANCE NOTICE:**
            
            This platform is for **informational and educational purposes only**. The forecasts, analyses, 
            and information provided here **do NOT constitute financial advice, investment recommendations, 
            or a solicitation to buy or sell any financial instrument**.
            
            **Regulatory Notice (SEBI Compliance):**
            - This platform is **not authorized or regulated by SEBI** (Securities and Exchange Board of India).
            - **No SEBI investor protections apply** to this platform or its services.
            - The predictions and models herein are **not approved by SEBI** and carry significant risk.
            - Past performance is **not indicative of future results**.
            - All market forecasts are subject to model drift and market anomalies.
            
            **Risk Disclosure:**
            - **Capital at risk**: All investments in stocks, commodities, and derivatives carry substantial risk of loss.
            - The accuracy of price forecasts is **not guaranteed** and may be significantly inaccurate.
            - Market volatility, geopolitical events, and systemic shocks can render predictions obsolete.
            - Leverage and derivatives amplify both gains and losses.
            
            **Your Responsibilities:**
            - **Always consult a qualified SEBI-registered financial advisor** before making any investment decisions.
            - Conduct your own due diligence and validate all information independently.
            - Never invest capital you cannot afford to lose.
            - Ensure compliance with applicable tax and regulatory obligations.
            
            **Limitation of Liability:**
            The creators of this platform assume **no responsibility** for any financial losses, incorrect forecasts, 
            or damages arising from the use of this information. Use at your own risk and discretion.
            """
        )


    try:
        df = _load_data(ticker, period, DEFAULT_INTERVAL, use_cache=not refresh)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    close = df["close"].astype(float)
    
    # Get closing price with timestamp and timezone (pass asset key, not ticker)
    closing_price, timestamp_str, is_holiday, holiday_msg = _get_closing_price_with_timestamp(df, asset)
    
    # Display metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        price_display = f"{currency_symbol}{closing_price:,.2f}"
        if is_holiday and holiday_msg:
            st.metric("Previous Close Price", price_display)
            st.warning(f"🔴 {holiday_msg}", icon="⚠️")
        else:
            st.metric("Previous Close Price", price_display)
    with c2:
        st.metric("Last Observed Day", timestamp_str)
    with c3:
        ret_1y = close.pct_change(252).iloc[-1] if len(close) > 252 else float("nan")
        st.metric("Estimated 1yr Return", f"{ret_1y * 100:.1f}%" if np.isfinite(ret_1y) else "—")

    try:
        fit = train_and_backtest(close)
        fc = attach_forecast(fit, horizon_days=forecast_days)
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

    forecast_change = fc.forecast_close[-1] - close.iloc[-1]
    forecast_change_pct = (forecast_change / close.iloc[-1]) * 100
    avg_forecast = fc.forecast_close.mean()
    forecast_vol = np.std(fc.forecast_close)
    forecast_sentiment = "📈 BULLISH" if forecast_change > 0 else "📉 BEARISH"
    returns = close.pct_change() * 100
    sentiment_score = 0.0
    news_headlines = _fetch_news(ticker)
    sentiment_score, sentiment_label = _score_headlines_sentiment(news_headlines)

    # Compute additional metrics for report
    returns_pct = close.pct_change().dropna()
    var_95 = returns_pct.quantile(0.05)
    cvar_95 = returns_pct[returns_pct <= var_95].mean()
    max_dd = ((close.cummax() - close) / close.cummax()).max()
    sharpe = (returns_pct.mean() / returns_pct.std()) * np.sqrt(252) if returns_pct.std() > 0 else 0
    signal_text, signal_note = _compose_trade_signal(forecast_change_pct, sentiment_score, forecast_vol)

    tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g, tab_h, tab_i, tab_j = st.tabs(
        [
            "Overview & forecast",
            "Backtest",
            "Insights",
            "Statistics",
            "News & Sentiment",
            "Trading Signals",
            "Technical Analysis",
            "Risk Metrics",
            "Compare assets",
            "Model Comparison",
        ]
    )

    with tab_a:
        st.markdown("---")
        
        col_left, col_center, col_right = st.columns((0.15, 0.7, 0.15))
        with col_center:
            st.plotly_chart(
                _fig_forecast(fc.history_close, fc.forecast_index, fc.forecast_close, asset_name, currency_symbol),
                use_container_width=True,
            )
            st.markdown(
                '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Historical closing prices with the model&apos;s multi-day forecast shown beyond the latest observed value. Use this chart to compare recent price behavior against the projected trend.</p>',
                unsafe_allow_html=True,
            )
            
            # Forecast table
            st.subheader("Forecasted Prices")
            forecast_df = pd.DataFrame({
                "Date": fc.forecast_index.strftime("%Y-%m-%d"),
                "Forecasted Price": [f"{currency_symbol}{price:,.2f}" for price in fc.forecast_close],
            })
            st.dataframe(forecast_df, use_container_width=True)
            
            st.plotly_chart(_fig_importance(fc.model, fc.feature_names), use_container_width=True)
            st.markdown(
                '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Feature importance ranks the variables driving the forecast. Taller bars indicate features that had stronger influence on model predictions.</p>',
                unsafe_allow_html=True,
            )

    with tab_b:
        bt = fit.backtest
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (holdout)", f"{bt.mae:,.4f}")
        m2.metric("RMSE (holdout)", f"{bt.rmse:,.4f}")
        m3.metric("MAPE (holdout)", f"{bt.mape:.2f}%")
        st.plotly_chart(_fig_backtest(fit.history_close, bt.y_pred, asset_name, currency_symbol), use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Backtest comparison of actual prices versus model predictions on the validation set. Smaller gaps and closer alignment show stronger historical fit.</p>',
            unsafe_allow_html=True,
        )

    with tab_c:
        st.subheader("Forecast Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FFF4FB 0%, #F7E8F5 100%); color: #3A1D41; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(141, 76, 131, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#6B2B70;">1️⃣ Forecast Direction</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.8rem 0; font-size:0.98rem; color:#5F3A72;">Sentiment: <strong>{forecast_sentiment}</strong></p>
                        <p style="margin:0.2rem 0 0.4rem 0; font-size:0.96rem; color:#4F2B5F;">The <strong>{forecast_days}-day forecast</strong> shows a {('bullish' if forecast_change > 0 else 'bearish')} bias with an expected move of <strong>{forecast_change:+.2f}</strong> ({forecast_change_pct:+.2f}%).</p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.88); border-radius: 12px; color:#3C1F44; font-weight:700;">Current price: {currency_symbol}{close.iloc[-1]:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FFF6F9 0%, #F9E9F4 100%); color: #3D1D46; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(161, 85, 143, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#701F6B;">3️⃣ Forecast Range & Confidence</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.85rem 0; color:#5B2C63;">Price Range: <strong>{currency_symbol}{fc.forecast_close.min():,.2f}</strong> → <strong>{currency_symbol}{fc.forecast_close.max():,.2f}</strong></p>
                        <p style="margin:0.2rem 0 0.8rem 0; color:#52315A;">Mean Forecast: <strong>{currency_symbol}{avg_forecast:,.2f}</strong> | Volatility: <strong>±{currency_symbol}{forecast_vol:.2f}</strong></p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.8rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#45225C; font-weight:700;">Tighter range = higher model confidence.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FFF7FA 0%, #F6E8F4 100%); color: #3E1C42; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(158, 86, 141, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#6F1F6C;">4️⃣ Model Accuracy</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.75rem 0; color:#5C2E69;">Backtest MAPE: <strong>{bt.mape:.2f}%</strong></p>
                        <p style="margin:0.2rem 0 0.85rem 0; color:#50315A;">This measures recent holdout fit, not future performance.</p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#40224A; font-weight:700;">Use as a signal guide, not a certainty.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with col2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE7F4 0%, #EFD2EE 100%); color: #3D1D41; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(163, 94, 141, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#772A71;">2️⃣ Short-term Momentum</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.8rem 0; color:#5B2F61;">The model extends recent 1–7 day patterns forward.</p>
                        <p style="margin:0.2rem 0 0.8rem 0; color:#4F2A57;">Best during trending markets; may lag when conditions shift rapidly.</p>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#3D1F42; font-weight:700;">Monitor short-term trend strength and momentum.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE8F7 0%, #E9D4F5 100%); color: #3E1E44; padding: 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 14px 30px rgba(170, 96, 150, 0.08); display: flex; flex-direction: column; justify-content: space-between; min-height: 220px;">
                    <h4 style="margin-top:0; color:#752C6A;">5️⃣ Practical Use</h4>
                    <div>
                        <p style="margin:0.4rem 0 0.8rem 0; color:#5D2F64;">Use this forecast as a tactical baseline signal.</p>
                        <ul style="margin:0.4rem 0 0 1rem; padding:0; color:#4E2853;">
                            <li>Combine with fundamental analysis</li>
                            <li>Monitor inventory, demand, and macro risk</li>
                            <li>Always manage position risk</li>
                        </ul>
                    </div>
                    <div style="margin-top:0.8rem; padding:0.85rem; background: rgba(255,255,255,0.86); border-radius: 12px; color:#3F1D43; font-weight:700;">Treat this as a supporting analysis layer.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #F8E0F0 0%, #E9D4F7 100%); color: #3B1B43; padding: 1.2rem; border-radius: 18px; margin-top:0.5rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 34px rgba(175, 93, 150, 0.1);">
                <h4 style="margin-top:0; color:#7C2F70;">🔔 Reliability & Guidance</h4>
                <p style="margin:0.4rem 0 0.7rem 0; color:#5F2F67;">Peak forecast accuracy is strongest in days 1–5. The signal weakens on longer horizons.</p>
                <p style="margin:0.2rem 0 0.8rem 0; color:#4D2751;">This is a model-based guide, not investment advice. Use it only as part of broader research and risk controls.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")

    with tab_d:
        st.subheader("Price Statistics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Mean Price", f"{currency_symbol}{close.mean():,.2f}")
        col2.metric("Median Price", f"{currency_symbol}{close.median():,.2f}")
        col3.metric("Standard Deviation", f"{currency_symbol}{close.std():,.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Min Price", f"{currency_symbol}{close.min():,.2f}")
        col2.metric("Max Price", f"{currency_symbol}{close.max():,.2f}")
        col3.metric("Range", f"{currency_symbol}{close.max() - close.min():,.2f}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Skewness", f"{close.skew():.3f}")
        col2.metric("Kurtosis", f"{close.kurtosis():.3f}")
        col3.metric("Coefficient of Variation (%)", f"{(close.std() / close.mean()) * 100:.2f}%")
        
        st.markdown("---")
        st.subheader("Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(
            go.Histogram(
                x=close.values,
                nbinsx=50,
                name="Price",
                marker_color="#FF7F50",
                opacity=0.7,
            )
        )
        fig_dist.update_layout(
            title=dict(text=f"{asset_name} — Price Distribution", font=dict(color="#654321", family="Georgia, serif")),
            xaxis_title="Price (USD)",
            yaxis_title="Frequency",
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            height=400,
        )
        st.plotly_chart(fig_dist, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Histogram of historical closing prices for the selected asset. This highlights where the price has spent most of its time and how concentrated the levels are.</p>',
            unsafe_allow_html=True,
        )

    with tab_e:
        st.subheader("News & Sentiment")

        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #FFF8F0 0%, #F5E6D3 100%); color: #3B1B43; padding: 1.4rem; border-radius: 18px; margin: 1rem 0; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 34px rgba(175, 93, 150, 0.1);">
                <h4 style="margin-top:0; color:#7C2F70;">📰 News Sentiment Summary</h4>
                <p style="margin:0.4rem 0 0.8rem 0; color:#5F2F67; font-size:1rem;">We aggregate recent headlines and show how market news aligns with the current price action.</p>
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap;">
                    <div style="font-size:1.2rem; font-weight:700; color:#3C1D3F;">Overall sentiment: <span style='color:#572E74;'>{sentiment_label}</span></div>
                    <div style="font-size:1.35rem; font-weight:800; color:#3C1D3F;">Headline score: <span style='color:#572E74;'>{sentiment_score:+.2f}</span></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if news_headlines:
            for headline in news_headlines[:8]:
                sentiment_value, sentiment_label_item = _headline_sentiment_score(headline.get("title", ""))
                color = "#2E8B57" if sentiment_value > 0 else "#8B4513" if sentiment_value == 0 else "#B22222"
                st.markdown(
                    f"<div style='background: linear-gradient(135deg, #FFF8F0 0%, #F8E3D8 100%); border:3px solid {color}; padding:0.72rem 0.9rem; border-radius:14px; margin-bottom:0.75rem; font-family: Poppins, sans-serif;'>"
                    f"<a href='{headline['link']}' target='_blank' style='text-decoration: none; color:#4F1D3A; font-size:1.18rem; font-weight:700; line-height:1.2;'><strong>{headline['title']}</strong></a><br>"
                    f"<span style='color:#7A4B2E; font-size:0.95rem; letter-spacing:0.01em;'>{headline['provider']} — {headline['published']}</span><br>"
                    f"<span style='display:inline-block;margin-top:0.4rem;padding:0.35rem 0.7rem;background:{color};color:white;border-radius:999px;font-size:0.9rem;'>Sentiment: {sentiment_label_item}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No recent headlines found for this asset. Try a different ticker or wait for new market updates.")

    with tab_i:
        st.subheader("Compare assets")
        st.markdown(
            "Choose a second asset here to compare price moves, forecast strength, sentiment bias, and buy/sell signals. "
            "The primary asset remains selected above in the parameters panel."
        )
        compare_name_map = {
            _clean_asset_display_name(key): key
            for key in asset_keys
            if key != asset
        }
        compare_options = ["Select second asset"] + list(compare_name_map.keys())
        compare_label = st.selectbox("Second asset", compare_options, index=0, key="compare_asset_select")

        if compare_label == "Select second asset":
            st.info("Choose a second asset in this tab to compare price, news sentiment, model signal, and risk metrics.")
        else:
            compare_asset_key = compare_name_map.get(compare_label, compare_label)
            compare_ticker = ASSETS.get(compare_asset_key)
            compare_asset_name = _clean_asset_display_name(compare_asset_key)
            compare_currency_symbol = get_currency_symbol(compare_asset_key)
            asset_display_name = _clean_asset_display_name(asset)

            try:
                compare_df = _load_data(compare_ticker, period, DEFAULT_INTERVAL, use_cache=not refresh)
                compare_close = compare_df["close"].astype(float)
            except Exception as e:
                st.error(f"Failed to load comparison asset data: {e}")
                compare_close = None

            if compare_close is not None:
                comp_df = pd.DataFrame({asset_display_name: close, compare_asset_name: compare_close}).dropna()
                corr = comp_df.pct_change().corr().iloc[0, 1]
                primary_return = close.iloc[-1] / close.iloc[0] - 1
                compare_return = compare_close.iloc[-1] / compare_close.iloc[0] - 1
                primary_vol = close.pct_change().std() * np.sqrt(252)
                compare_vol = compare_close.pct_change().std() * np.sqrt(252)
                primary_sharpe = (close.pct_change().mean() / close.pct_change().std()) * np.sqrt(252) if close.pct_change().std() > 0 else 0
                compare_sharpe = (compare_close.pct_change().mean() / compare_close.pct_change().std()) * np.sqrt(252) if compare_close.pct_change().std() > 0 else 0
                volatility_ratio = primary_vol / compare_vol if compare_vol != 0 else float("nan")

                compare_headlines = _fetch_news(compare_ticker)
                compare_sentiment_score, compare_sentiment_label = _score_headlines_sentiment(compare_headlines)

                try:
                    compare_fit = train_and_backtest(compare_close)
                    compare_fc = attach_forecast(compare_fit, horizon_days=forecast_days)
                    compare_forecast_change = compare_fc.forecast_close[-1] - compare_close.iloc[-1]
                    compare_forecast_change_pct = (compare_forecast_change / compare_close.iloc[-1]) * 100
                    compare_signal_text, compare_signal_note = _compose_trade_signal(
                        compare_forecast_change_pct,
                        compare_sentiment_score,
                        np.std(compare_fc.forecast_close),
                    )
                except Exception as e:
                    compare_fc = None
                    compare_forecast_change_pct = float("nan")
                    compare_signal_text = "Model unavailable"
                    compare_signal_note = f"Comparison model failed: {e}"

                primary_signal_text, primary_signal_note = _compose_trade_signal(
                    forecast_change_pct,
                    sentiment_score,
                    forecast_vol,
                )
                recommendation_title, recommendation_note = _relative_comparison_recommendation(
                    asset_display_name,
                    forecast_change_pct,
                    sentiment_label,
                    compare_asset_name,
                    compare_forecast_change_pct,
                    compare_sentiment_label,
                )

                st.plotly_chart(
                    _fig_asset_comparison(close, asset_display_name, compare_close, compare_asset_name),
                    use_container_width=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric(f"{asset_display_name} total return", f"{primary_return * 100:+.2f}%")
                m2.metric(f"{compare_asset_name} total return", f"{compare_return * 100:+.2f}%")
                m3.metric("Correlation", f"{corr:.2f}")

                m4, m5, m6, m7 = st.columns(4)
                m4.metric(f"{asset_display_name} forecast move", f"{forecast_change_pct:+.2f}%")
                m5.metric(f"{compare_asset_name} forecast move", f"{compare_forecast_change_pct:+.2f}%")
                m6.metric(f"{asset_display_name} signal", primary_signal_text)
                m7.metric(f"{compare_asset_name} signal", compare_signal_text)

                m8, m9, m10, m11 = st.columns(4)
                m8.metric(f"{asset_display_name} Sharpe", f"{primary_sharpe:.2f}")
                m9.metric(f"{compare_asset_name} Sharpe", f"{compare_sharpe:.2f}")
                m10.metric(f"{asset_display_name} sentiment", f"{sentiment_label}")
                m11.metric(f"{compare_asset_name} sentiment", f"{compare_sentiment_label}")

                st.markdown(
                    f"<div style='background: linear-gradient(135deg, #FFF8F0 0%, #F5E6D3 100%); padding:1rem; border-radius:16px; border:1px solid #D7C2A2; margin-top:1rem;'>"
                    f"<h4 style='margin:0 0 0.4rem 0; color:#5F3A72;'>{recommendation_title}</h4>"
                    f"<p style='margin:0; color:#4B2E2A;'>{recommendation_note}</p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"<p style='margin-top:0.8rem; color:#3F2F1F; font-size:0.95rem;'>Comparison uses {len(comp_df):,} common trading days. Prices are normalized to a starting value of 100 for both assets to show relative performance.</p>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "<div style='background: linear-gradient(135deg, #F0F8FF 0%, #F4F1FF 100%); color:#2F286A; padding:1rem; border-radius:16px; border:1px solid #C7C0E8; margin-top:1rem;'>"
                    "<h4 style='margin:0 0 0.6rem 0;'>Comparison highlights</h4>"
                    "<ul style='margin:0; padding-left:1.1rem; line-height:1.6;'>"
                    f"<li><strong>{asset_display_name} forecast move</strong>: {forecast_change_pct:+.2f}% over the next {forecast_days} trading days.</li>"
                    f"<li><strong>{compare_asset_name} forecast move</strong>: {compare_forecast_change_pct:+.2f}% over the next {forecast_days} trading days.</li>"
                    f"<li><strong>{asset_display_name} signal</strong>: {primary_signal_text}.</li>"
                    f"<li><strong>{compare_asset_name} signal</strong>: {compare_signal_text}.</li>"
                    "</ul>"
                    "</div>",
                    unsafe_allow_html=True,
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"<h4 style='color:#5F3A72;'>{asset_display_name} news & sentiment</h4>", unsafe_allow_html=True)
                    if news_headlines:
                        for headline in news_headlines[:4]:
                            sentiment_value, sentiment_label_item = _headline_sentiment_score(headline.get("title", ""))
                            color = "#2E8B57" if sentiment_value > 0 else "#8B4513" if sentiment_value == 0 else "#B22222"
                            st.markdown(
                                f"<div style='background: linear-gradient(135deg, #FFF8F0 0%, #F8E3D8 100%); border:3px solid {color}; padding:0.7rem 0.9rem; border-radius:12px; margin-bottom:0.7rem; font-family: Poppins, sans-serif;'>"
                                f"<strong>{headline['title']}</strong><br>"
                                f"<span style='color:#7A4B2E; font-size:0.92rem;'>{headline['provider']} — {headline['published']}</span><br>"
                                f"<span style='display:inline-block;margin-top:0.35rem;padding:0.3rem 0.55rem;background:{color};color:white;border-radius:999px;font-size:0.85rem;'>Sentiment: {sentiment_label_item}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.write("No recent headlines found for this asset.")
                with col2:
                    st.markdown(f"<h4 style='color:#5F3A72;'>{compare_asset_name} news & sentiment</h4>", unsafe_allow_html=True)
                    if compare_headlines:
                        for headline in compare_headlines[:4]:
                            sentiment_value, sentiment_label_item = _headline_sentiment_score(headline.get("title", ""))
                            color = "#2E8B57" if sentiment_value > 0 else "#8B4513" if sentiment_value == 0 else "#B22222"
                            st.markdown(
                                f"<div style='background: linear-gradient(135deg, #FFF8F0 0%, #F8E3D8 100%); border:3px solid {color}; padding:0.7rem 0.9rem; border-radius:12px; margin-bottom:0.7rem; font-family: Poppins, sans-serif;'>"
                                f"<strong>{headline['title']}</strong><br>"
                                f"<span style='color:#7A4B2E; font-size:0.92rem;'>{headline['provider']} — {headline['published']}</span><br>"
                                f"<span style='display:inline-block;margin-top:0.35rem;padding:0.3rem 0.55rem;background:{color};color:white;border-radius:999px;font-size:0.85rem;'>Sentiment: {sentiment_label_item}</span>"
                                f"</div>",
                                unsafe_allow_html=True,
                            )
                    else:
                        st.write("No recent headlines found for this comparison asset.")


    with tab_f:
        st.subheader("Trading Signals")
        signal_text, signal_note = _compose_trade_signal(forecast_change_pct, sentiment_score, forecast_vol)

        st.metric("Recommended Action", signal_text)
        st.markdown(
            f"<div style='background:#E8F6EF; border:1px solid #A6D4B8; padding:1rem; border-radius:16px; margin-bottom:1rem;'>"
            f"<p style='margin:0; color:#1F4D3D; font-size:1.1rem; font-weight:700;'><strong>Signal:</strong> {signal_text}</p>"
            f"<p style='margin:0.75rem 0 0 0; color:#2D5C48; font-size:1.05rem; font-weight:700;'>Why we prefer this action:</p>"
            f"<p style='margin:0.35rem 0 0 0; color:#2F3E46; font-size:0.98rem; line-height:1.55;'>"
            f"Current price is <strong>{currency_symbol}{close.iloc[-1]:,.2f}</strong>, and the model forecasts a <strong>{forecast_change_pct:+.2f}%</strong> move over the next <strong>{forecast_days}</strong> trading days." 
            f"News sentiment is <strong>{sentiment_label.lower()}</strong> ({sentiment_score:+.2f}), so this recommendation is built from the stock's current momentum and the latest headline bias." 
            "</p>"
            f"<p style='margin:0.8rem 0 0 0; color:#2F3E46; font-size:0.95rem; line-height:1.5;'>"
            f"Compared with alternative actions, this option is preferred because the stock's forecast move and sentiment signal are aligned and the implied volatility range is manageable at ±{currency_symbol}{forecast_vol:.2f}."
            f"</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        
        st.info(
            "This signal is an informational overlay, not financial advice. Use it together with technical context and portfolio risk rules."
        )

    with tab_g:
        st.subheader("Technical Analysis")
        
        # Calculate technical indicators
        returns = close.pct_change() * 100
        sma_20 = close.rolling(window=20).mean()
        sma_50 = close.rolling(window=50).mean()
        rolling_vol = returns.rolling(window=20).std()
        avg_return = returns.mean()
        momentum_value = f"{((close.iloc[-1] / close.iloc[-5]) - 1) * 100:+.2f}%" if len(close) > 5 else "N/A"
        
        st.markdown("Technical Overview")
        row1, row2 = st.columns(2)
        
        with row1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F9E6F8 0%, #F1D9F4 100%); color: #3A1D41; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(148, 86, 157, 0.08);">
                    <h4 style="margin-top:0; color:#6A236D;">📊 Daily Volatility (20d)</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5D2F64;">Standard deviation of returns over 20 days. Higher values indicate more uncertainty.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3C1D3F; font-size:1.5rem; font-weight:700;">{rolling_vol.iloc[-1]:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE9F6 0%, #F4DAF1 100%); color: #3C1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(170, 96, 155, 0.08);">
                    <h4 style="margin-top:0; color:#7A2F72;">📈 SMA 20</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5E2F65;">Smoothed average price over 20 days. Crossovers with longer MAs indicate trend shifts.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3F1F43; font-size:1.5rem; font-weight:700;">{currency_symbol}{sma_20.iloc[-1]:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F6DDF5 0%, #E9CCF2 100%); color: #3D1C40; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(164, 88, 146, 0.08);">
                    <h4 style="margin-top:0; color:#6B226A;">💹 Avg Daily Return</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2E64;">Mean daily percent change. Reflects the average short-term momentum in the asset.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3E1C41; font-size:1.5rem; font-weight:700;">{avg_return:+.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with row2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F8E3F4 0%, #EAD5F3 100%); color: #3C1D43; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(162, 86, 143, 0.08);">
                    <h4 style="margin-top:0; color:#702F71;">📉 SMA 50</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5B2E62;">Longer-term moving average. Prices above it often signal the persistence of an uptrend.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3F1F44; font-size:1.5rem; font-weight:700;">{currency_symbol}{sma_50.iloc[-1]:,.2f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F9E7F3 0%, #EBD3F2 100%); color: #3D1C44; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(159, 88, 143, 0.08);">
                    <h4 style="margin-top:0; color:#722F6F;">📈 Annualized Return</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2F65;">Projected yearly return if the average daily drift continues. Use this as a directional reference.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3D1D42; font-size:1.5rem; font-weight:700;">{avg_return * 252:+.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F7DFF2 0%, #E7D2F3 100%); color: #3D1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 32px rgba(161, 88, 146, 0.08);">
                    <h4 style="margin-top:0; color:#762E71;">🚀 Price Momentum (5d)</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2E64;">Recent 5-day price change. Positive values indicate upward short-term pressure.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.82); border-radius: 12px; color:#3F1D43; font-size:1.5rem; font-weight:700;">{momentum_value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown("---")
        st.subheader("Moving Averages & Volatility")
        fig_ma = go.Figure()
        fig_ma.add_trace(
            go.Scatter(
                x=close.index,
                y=close.values,
                mode="lines",
                name="Close",
                line=dict(color="#FF7F50", width=1.5),
            )
        )
        fig_ma.add_trace(
            go.Scatter(
                x=sma_20.index,
                y=sma_20.values,
                mode="lines",
                name="SMA 20",
                line=dict(color="#8B4513", width=2, dash="dash"),
            )
        )
        fig_ma.add_trace(
            go.Scatter(
                x=sma_50.index,
                y=sma_50.values,
                mode="lines",
                name="SMA 50",
                line=dict(color="#FFA07A", width=2, dash="dash"),
            )
        )
        fig_ma.update_layout(
            title=dict(text=f"{asset_name} — Price with Moving Averages", font=dict(color="#654321", family="Georgia, serif")),
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            hovermode="x unified",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            height=400,
            xaxis_title="Date",
            yaxis_title=f"Price ({currency_symbol})",
        )
        st.plotly_chart(fig_ma, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Price chart with 20-day and 50-day moving averages. Watch for crossovers as potential trend signals and to see how the price interacts with shorter- and longer-term averages.</p>',
            unsafe_allow_html=True,
        )

    with tab_h:
        st.subheader("Risk Metrics")
        
        # Calculate risk metrics
        returns = close.pct_change().dropna()
        var_95 = returns.quantile(0.05)
        cvar_95 = returns[returns <= var_95].mean()
        max_dd = ((close.cummax() - close) / close.cummax()).max()
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        row1, row2 = st.columns(2)
        
        with row1:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F9E6F7 0%, #F2D9F2 100%); color: #3B1E41; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(148, 86, 157, 0.08);">
                    <h4 style="margin-top:0; color:#6F2670;">⚠️ Value at Risk (95%)</h4>
                    <p style="margin:0.35rem 0 0.85rem 0; color:#5A2D60;">Worst expected daily loss 95% of the time.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3D1C41; font-size:1.4rem; font-weight:700;">{var_95:.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE8F7 0%, #E9D4F5 100%); color: #3E1E44; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(170, 96, 150, 0.08);">
                    <h4 style="margin-top:0; color:#742E72;">🔻 Max Drawdown</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5A2D62;">Largest peak-to-trough decline in the series.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3F1D44; font-size:1.4rem; font-weight:700;">{max_dd * 100:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        with row2:
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #FCE9F6 0%, #F4DAF1 100%); color: #3C1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(170, 96, 155, 0.08);">
                    <h4 style="margin-top:0; color:#7A2F72;">📉 Conditional VaR</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5E2F65;">Average loss on the worst 5% of days.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3F1F43; font-size:1.4rem; font-weight:700;">{cvar_95:.3f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #F7E2F3 0%, #E7D2F3 100%); color: #3D1B42; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(161, 88, 146, 0.08);">
                    <h4 style="margin-top:0; color:#752F6F;">⭐ Sharpe Ratio</h4>
                    <p style="margin:0.35rem 0 0.9rem 0; color:#5C2E64;">Risk-adjusted return relative to volatility.</p>
                    <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3E1D43; font-size:1.4rem; font-weight:700;">{sharpe:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #F8E0F0 0%, #E9D4F7 100%); color: #3B1B43; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(175, 93, 150, 0.1);">
                <h4 style="margin-top:0; color:#7C2F70;">📈 Volatility (Annual)</h4>
                <p style="margin:0.35rem 0 0.9rem 0; color:#5F2F67;">Annualized standard deviation of returns.</p>
                <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3C1E42; font-size:1.4rem; font-weight:700;">{returns.std() * np.sqrt(252):.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #F8EAF3 0%, #E7DFF5 100%); color: #3B1C43; padding: 1rem 1.1rem; border-radius: 18px; margin-bottom: 1rem; border: 2px solid #8B4513; font-family: 'Georgia', serif; box-shadow: 0 16px 30px rgba(170, 92, 150, 0.1);">
                <h4 style="margin-top:0; color:#742F72;">📊 Days Up (%)</h4>
                <p style="margin:0.35rem 0 0.9rem 0; color:#5F2F67;">Percentage of trading days with positive returns.</p>
                <div style="padding:0.85rem 1rem; background: rgba(255,255,255,0.84); border-radius: 12px; color:#3C1D43; font-size:1.4rem; font-weight:700;">{(returns > 0).sum() / len(returns) * 100:.1f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        st.subheader("Drawdown Analysis")
        fig_dd = go.Figure()
        cummax = close.cummax()
        drawdown = (close - cummax) / cummax * 100
        
        fig_dd.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill="tozeroy",
                name="Drawdown",
                line=dict(color="#8B4513"),
                fillcolor="rgba(139, 69, 19, 0.3)",
            )
        )
        fig_dd.update_layout(
            title=dict(text=f"{asset_name} — Drawdown from Peak", font=dict(color="#654321", family="Georgia, serif")),
            template="plotly_white",
            plot_bgcolor="#FAF0E6",
            paper_bgcolor="#FFFFF0",
            height=400,
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            xaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
            yaxis=dict(showgrid=True, gridwidth=1, gridcolor="rgba(139, 69, 19, 0.3)", title_font=dict(color="#654321"), tickfont=dict(color="#654321")),
        )
        st.plotly_chart(fig_dd, use_container_width=True)
        st.markdown(
            '<p style="margin-top:0.25rem; margin-bottom:1rem; color:#3F2F1F; font-size:0.95rem;">Drawdown tracks the percentage decline from the peak price. Deeper and longer drawdowns indicate higher drawdown risk. This helps identify periods of maximum loss from peak values.</p>',
            unsafe_allow_html=True,
        )

    with tab_j:
        st.subheader("Model Comparison")
        st.markdown(
            "This section compares the performance of all four boosting models: **XGBoost**, **LightGBM**, **Gradient Boosting Regressor (GBR)**, and **CatBoost**. "
            "The ensemble method combines predictions from all four using inverse-MAE weighting to produce the final forecast."
        )
        
        st.markdown("---")
        st.markdown("### Performance Metrics Comparison")
        st.markdown(
            "This comparison shows three key performance metrics for each model:\n"
            "- **MAE**: Mean Absolute Error (lower is better)\n"
            "- **RMSE**: Root Mean Squared Error (lower is better)\n"
            "- **MAPE**: Mean Absolute Percentage Error (lower is better)"
        )
        st.plotly_chart(
            _fig_models_performance_metrics(fit),
            use_container_width=True,
        )
        
        st.markdown("---")
        st.markdown("### Model Weights & Metrics")
        
        # Create a detailed metrics table
        metrics_data = []
        for model_name in ["gbr", "lightgbm", "catboost", "xgb"]:
            if model_name in fit.individual_models:
                model_result = fit.individual_models[model_name]
                bt = model_result.backtest
                metrics_data.append({
                    "Model": model_name.upper(),
                    "Weight": f"{fit.weights[model_name]:.4f}",
                    "MAE": f"{bt.mae:.6f}",
                    "RMSE": f"{bt.rmse:.6f}",
                    "MAPE (%)": f"{bt.mape:.2f}",
                })
        
        # Add ensemble metrics
        ensemble_bt = fit.backtest
        metrics_data.append({
            "Model": "ENSEMBLE",
            "Weight": "—",
            "MAE": f"{ensemble_bt.mae:.6f}",
            "RMSE": f"{ensemble_bt.rmse:.6f}",
            "MAPE (%)": f"{ensemble_bt.mape:.2f}",
        })
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        st.markdown(
            "<p style='margin-top:1rem; color:#3F2F1F; font-size:0.95rem;'>"
            "<strong>Note on weights:</strong> Models are weighted inversely to their MAE (Mean Absolute Error) from the backtest period. "
            "A model with lower error gets a higher weight in the ensemble. This ensures more accurate models contribute more to the final prediction."
            "</p>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()