"""Generate comprehensive backtest report for all assets across all history windows."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ASSETS, DEFAULT_INTERVAL
from src.data_loader import load_commodity_history
from src.model_ensemble import train_and_backtest


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


def get_approx_return(close: pd.Series) -> float:
    """Calculate approximate 1-year return (252 trading days)."""
    if len(close) < 252:
        # If less than 1 year, annualize what we have
        daily_return = close.pct_change().mean()
        return float(daily_return * 252 * 100) if np.isfinite(daily_return) else 0.0
    else:
        return float((close.iloc[-1] / close.iloc[-252] - 1) * 100)


def process_asset(
    asset_name: str,
    ticker: str,
    history_window: str,
    results: list[dict],
) -> None:
    """
    Process a single asset-window combination.
    
    Args:
        asset_name: Display name of asset (e.g., "Commodity: Gold")
        ticker: Yahoo Finance ticker
        history_window: History window (e.g., "2y", "5y", "10y")
        results: List to append results to
    """
    try:
        print(f"Processing {asset_name} ({ticker}) — {history_window}...", end=" ", flush=True)
        
        # Load data
        df = load_commodity_history(ticker, period=history_window, interval=DEFAULT_INTERVAL, use_cache=True)
        close = df["close"].astype(float)
        
        if len(close) < 30:
            print("⚠ Insufficient data")
            return
        
        # Calculate metrics
        n_obs = len(close)
        approx_return = get_approx_return(close)
        
        # Train model and backtest
        fit = train_and_backtest(close)
        bt = fit.backtest
        
        # Append result
        result = {
            "Asset": asset_name,
            "History Window": history_window,
            "Observations": n_obs,
            "Approx 1Y Return (%)": round(approx_return, 2),
            "MAE": round(bt.mae, 4),
            "RMSE": round(bt.rmse, 4),
            "MAPE (%)": round(bt.mape, 2),
        }
        results.append(result)
        print("✓")
        
    except Exception as e:
        print(f"✗ Error: {str(e)[:60]}")


def main():
    """Generate backtest report for all assets across all history windows."""
    print("=" * 80)
    print("COMMODITY & STOCK FORECASTER — BACKTEST REPORT GENERATOR")
    print("=" * 80)
    print()
    
    history_windows = ["2y"]
    results = []
    
    total = len(ASSETS) * len(history_windows)
    current = 0
    
    print(f"Total combinations to process: {total}")
    print("-" * 80)
    print()
    
    for asset_display, ticker in sorted(ASSETS.items()):
        for window in history_windows:
            current += 1
            print(f"[{current:3d}/{total}] ", end="")
            process_asset(asset_display, ticker, window, results)
    
    print()
    print("=" * 80)
    print("GENERATING EXCEL REPORT...")
    print("=" * 80)
    print()
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    # Sort by asset name
    df_results = df_results.sort_values(by=["Asset", "History Window"]).reset_index(drop=True)
    
    # Export to Excel with formatting
    output_file = ROOT / "backtest_report.xlsx"
    
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        df_results.to_excel(writer, sheet_name="Results", index=False)
        
        # Get the worksheet to apply formatting
        worksheet = writer.sheets["Results"]
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"✓ Report saved to: {output_file}")
    print()
    print(f"Total records: {len(df_results)}")
    print(f"Assets: {df_results['Asset'].nunique()}")
    print(f"History windows: {df_results['History Window'].nunique()}")
    print()
    print("Summary Statistics:")
    print("-" * 80)
    print(df_results[[
        "History Window", "Observations", "Approx 1Y Return (%)", "MAE", "RMSE", "MAPE (%)"
    ]].groupby("History Window").agg({
        "Observations": ["min", "mean", "max"],
        "Approx 1Y Return (%)": ["min", "mean", "max"],
        "MAE": ["min", "mean", "max"],
        "RMSE": ["min", "mean", "max"],
        "MAPE (%)": ["min", "mean", "max"],
    }).round(2))
    print()
    print("=" * 80)
    print("Report generation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
