"""Visualization module for stock predictions and training metrics.

Generates publication-quality charts for the README and analysis.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec


# Style settings for clean, dark-themed charts
STYLE = {
    "bg_color": "#1a1a2e",
    "text_color": "#e0e0e0",
    "grid_color": "#2a2a4a",
    "accent1": "#00d4ff",       # Cyan (actual)
    "accent2": "#ff6b6b",       # Red (predicted)
    "accent3": "#51cf66",       # Green (train)
    "accent4": "#ffd43b",       # Yellow (highlights)
}


def _setup_dark_style():
    """Apply dark theme to matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": STYLE["bg_color"],
        "axes.facecolor": STYLE["bg_color"],
        "axes.edgecolor": STYLE["grid_color"],
        "axes.labelcolor": STYLE["text_color"],
        "text.color": STYLE["text_color"],
        "xtick.color": STYLE["text_color"],
        "ytick.color": STYLE["text_color"],
        "grid.color": STYLE["grid_color"],
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    })


def plot_training_history(history: dict, save_path: str = "plots/training_history.png"):
    """Plot training and test loss curves.

    Shows how the model improved over time and if it overfit.
    """
    _setup_dark_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    ax1.plot(epochs, history["train_loss"], color=STYLE["accent3"],
             label="Train Loss", linewidth=2)
    ax1.plot(epochs, history["test_loss"], color=STYLE["accent2"],
             label="Test Loss", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training & Test Loss")
    ax1.legend(framealpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(epochs, history["learning_rate"], color=STYLE["accent4"],
             linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale("log")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_predictions(
    actual_prices: np.ndarray,
    predicted_prices: np.ndarray,
    dates: list,
    ticker: str,
    save_path: str = "plots/predictions.png",
    train_prices: np.ndarray = None,
    train_dates: list = None,
):
    """Plot actual vs predicted stock prices.

    The main chart showing how well the model predicts.
    """
    _setup_dark_style()

    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.3)

    # --- Main Price Chart ---
    ax1 = fig.add_subplot(gs[0])

    # Plot training data if provided
    if train_prices is not None and train_dates is not None:
        ax1.plot(train_dates, train_prices, color=STYLE["accent3"],
                 alpha=0.4, linewidth=1, label="Train Data")

    # Actual vs Predicted
    ax1.plot(dates, actual_prices, color=STYLE["accent1"],
             linewidth=2, label="Actual Price")
    ax1.plot(dates, predicted_prices, color=STYLE["accent2"],
             linewidth=2, linestyle="--", label="Predicted Price")

    # Highlight prediction accuracy
    ax1.fill_between(
        dates, actual_prices, predicted_prices,
        alpha=0.15, color=STYLE["accent2"]
    )

    ax1.set_title(f"{ticker} Stock Price Prediction (LSTM)", fontsize=16, fontweight="bold")
    ax1.set_ylabel("Price ($)")
    ax1.legend(loc="upper left", framealpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # --- Error Chart ---
    ax2 = fig.add_subplot(gs[1])
    errors = actual_prices - predicted_prices
    colors = [STYLE["accent3"] if e >= 0 else STYLE["accent2"] for e in errors]
    ax2.bar(dates, errors, color=colors, alpha=0.7, width=2)
    ax2.axhline(y=0, color=STYLE["text_color"], linewidth=0.5)
    ax2.set_ylabel("Error ($)")
    ax2.set_title("Prediction Error (Actual - Predicted)")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance(
    model,
    X_test: np.ndarray,
    feature_names: list,
    device,
    save_path: str = "plots/feature_importance.png",
):
    """Estimate feature importance using permutation.

    For each feature, shuffle its values and measure how much
    the prediction changes. More change = more important feature.
    """
    import torch

    _setup_dark_style()

    model.eval()
    X_tensor = torch.FloatTensor(X_test).to(device)

    # Get baseline predictions
    with torch.no_grad():
        baseline = model(X_tensor).cpu().numpy()

    importances = []
    for i, name in enumerate(feature_names):
        # Shuffle this feature across all samples
        X_shuffled = X_tensor.clone()
        perm = torch.randperm(X_shuffled.size(0))
        X_shuffled[:, :, i] = X_shuffled[perm, :, i]

        with torch.no_grad():
            shuffled_pred = model(X_shuffled).cpu().numpy()

        # Importance = how much predictions changed
        importance = np.mean(np.abs(baseline - shuffled_pred))
        importances.append(importance)

    # Sort by importance
    sorted_idx = np.argsort(importances)
    sorted_names = [feature_names[i] for i in sorted_idx]
    sorted_importance = [importances[i] for i in sorted_idx]

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(sorted_names, sorted_importance, color=STYLE["accent1"], alpha=0.8)

    # Highlight top 3
    for bar in bars[-3:]:
        bar.set_color(STYLE["accent4"])

    ax.set_xlabel("Importance (Mean Prediction Change)")
    ax.set_title("Feature Importance (Permutation-based)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_technical_indicators(
    df: pd.DataFrame,
    ticker: str,
    save_path: str = "plots/technical_indicators.png",
):
    """Plot the stock data with technical indicators.

    Shows what the model actually sees as input features.
    """
    _setup_dark_style()

    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle(f"{ticker} Technical Indicators (Model Input Features)",
                 fontsize=16, fontweight="bold", y=0.98)

    # Limit to last 252 days (1 year) for clarity
    plot_df = df.tail(252)

    # 1. Price + Moving Averages + Bollinger Bands
    ax = axes[0]
    ax.plot(plot_df.index, plot_df["Close"], color=STYLE["accent1"],
            linewidth=2, label="Close")
    ax.plot(plot_df.index, plot_df["SMA_20"], color=STYLE["accent4"],
            linewidth=1, alpha=0.7, label="SMA 20")
    ax.plot(plot_df.index, plot_df["SMA_50"], color=STYLE["accent2"],
            linewidth=1, alpha=0.7, label="SMA 50")
    ax.fill_between(plot_df.index, plot_df["BB_upper"], plot_df["BB_lower"],
                     alpha=0.1, color=STYLE["accent1"], label="Bollinger Bands")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # 2. Volume
    ax = axes[1]
    ax.bar(plot_df.index, plot_df["Volume"], color=STYLE["accent1"], alpha=0.5, width=1)
    ax.set_ylabel("Volume")
    ax.grid(True, alpha=0.3)

    # 3. RSI
    ax = axes[2]
    ax.plot(plot_df.index, plot_df["RSI"], color=STYLE["accent1"], linewidth=1.5)
    ax.axhline(y=70, color=STYLE["accent2"], linestyle="--", alpha=0.5, label="Overbought (70)")
    ax.axhline(y=30, color=STYLE["accent3"], linestyle="--", alpha=0.5, label="Oversold (30)")
    ax.fill_between(plot_df.index, 70, plot_df["RSI"],
                     where=plot_df["RSI"] > 70, alpha=0.2, color=STYLE["accent2"])
    ax.fill_between(plot_df.index, 30, plot_df["RSI"],
                     where=plot_df["RSI"] < 30, alpha=0.2, color=STYLE["accent3"])
    ax.set_ylabel("RSI")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
    ax.grid(True, alpha=0.3)

    # 4. MACD
    ax = axes[3]
    ax.plot(plot_df.index, plot_df["MACD"], color=STYLE["accent1"], linewidth=1.5, label="MACD")
    ax.axhline(y=0, color=STYLE["text_color"], linewidth=0.5)
    colors = [STYLE["accent3"] if v >= 0 else STYLE["accent2"] for v in plot_df["MACD"]]
    ax.bar(plot_df.index, plot_df["MACD"], color=colors, alpha=0.3, width=1)
    ax.set_ylabel("MACD")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
