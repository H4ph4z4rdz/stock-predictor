"""Stock Price Predictor -- Gradio Web UI.

Launch with:
    python src/app.py

Then open http://localhost:7863 in your browser.
Select a stock ticker and see predictions!
"""

import os
import sys

import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from core.data import StockDataPipeline
from core.model import StockLSTM
from utils.helpers import load_config, get_device


# Initialize
config = load_config()
device = get_device()


def predict_stock(ticker: str, period: str = "2y") -> tuple:
    """Fetch data, run prediction, return chart and metrics."""
    if not ticker.strip():
        return None, "Please enter a stock ticker."

    ticker = ticker.strip().upper()

    try:
        # Setup pipeline with custom ticker/period
        pred_config = config.copy()
        pred_config["data"] = config["data"].copy()
        pred_config["data"]["ticker"] = ticker
        pred_config["data"]["period"] = period

        pipeline = StockDataPipeline(pred_config)
        data = pipeline.prepare_data(ticker)

        # Load model
        num_features = len(config["data"]["features"])
        model = StockLSTM(
            input_size=num_features,
            hidden_size=config["model"]["hidden_size"],
            num_layers=config["model"]["num_layers"],
            dropout=config["model"]["dropout"],
            bidirectional=config["model"]["bidirectional"],
        ).to(device)

        model_path = os.path.join(config["output"]["model_dir"], "best_model.pt")
        if not os.path.exists(model_path):
            return None, "No trained model found. Run 'python src/train.py' first."

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # Predict on test set
        X_test_tensor = torch.FloatTensor(data["X_test"]).to(device)
        with torch.no_grad():
            predictions = model(X_test_tensor).cpu().numpy()

        actual = pipeline.inverse_transform_price(data["y_test"])
        predicted = pipeline.inverse_transform_price(predictions)

        # Metrics
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        actual_dir = np.diff(actual) > 0
        pred_dir = np.diff(predicted) > 0
        dir_acc = np.mean(actual_dir == pred_dir) * 100

        # Create chart
        fig, ax = plt.subplots(figsize=(14, 6))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#1a1a2e")

        ax.plot(data["dates_test"], actual, color="#00d4ff",
                linewidth=2, label="Actual")
        ax.plot(data["dates_test"], predicted, color="#ff6b6b",
                linewidth=2, linestyle="--", label="Predicted")
        ax.fill_between(data["dates_test"], actual, predicted,
                        alpha=0.15, color="#ff6b6b")

        ax.set_title(f"{ticker} Price Prediction", color="#e0e0e0",
                     fontsize=14, fontweight="bold")
        ax.set_ylabel("Price ($)", color="#e0e0e0")
        ax.legend(framealpha=0.3)
        ax.grid(True, alpha=0.2, color="#2a2a4a")
        ax.tick_params(colors="#e0e0e0")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.tight_layout()

        metrics_text = (
            f"Ticker: {ticker}\n"
            f"MAE: ${mae:.2f}\n"
            f"RMSE: ${rmse:.2f}\n"
            f"MAPE: {mape:.2f}%\n"
            f"Direction Accuracy: {dir_acc:.1f}%\n"
            f"Test Period: {data['dates_test'][0].strftime('%Y-%m-%d')} to "
            f"{data['dates_test'][-1].strftime('%Y-%m-%d')}\n"
            f"Current Price: ${actual[-1]:.2f}\n"
            f"Predicted: ${predicted[-1]:.2f}"
        )

        return fig, metrics_text

    except Exception as e:
        return None, f"Error: {str(e)}"


# Build the Gradio UI
with gr.Blocks(title="Stock Price Predictor") as demo:
    gr.Markdown(
        """
        # Stock Price Predictor
        ### LSTM Neural Network trained on historical market data

        Enter any stock ticker to see how well the model predicts prices.
        The model uses 12 technical indicators over 60-day windows.

        **DISCLAIMER:** This is an educational project. Do NOT use for real trading.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            ticker_input = gr.Textbox(
                label="Stock Ticker",
                value="AAPL",
                placeholder="e.g., AAPL, MSFT, GOOGL",
            )
            period_input = gr.Dropdown(
                label="Data Period",
                choices=["1y", "2y", "5y"],
                value="2y",
            )
            predict_btn = gr.Button("Predict", variant="primary")
            metrics_output = gr.Textbox(
                label="Metrics",
                interactive=False,
                lines=8,
            )

            gr.Markdown("### Quick Picks")
            for t in config["ui"]["default_tickers"]:
                gr.Button(t, size="sm").click(
                    predict_stock, [gr.State(t), period_input],
                    [gr.Plot(label="Prediction Chart"), metrics_output],
                )

        with gr.Column(scale=3):
            chart_output = gr.Plot(label="Prediction Chart")

    gr.Markdown(
        """
        ---
        **How it works:**
        - Downloads real-time data from Yahoo Finance
        - Calculates 12 technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
        - Feeds 60-day windows through a 2-layer LSTM neural network
        - Predicts the closing price 5 days ahead

        **Metrics explained:**
        - **MAE** (Mean Absolute Error): Average dollar amount the prediction is off
        - **RMSE** (Root Mean Squared Error): Penalizes large errors more
        - **MAPE** (Mean Absolute Percentage Error): Error as a percentage of price
        - **Direction Accuracy**: How often it predicts up/down correctly

        *Built with PyTorch, yfinance, matplotlib & Gradio*
        """
    )

    predict_btn.click(predict_stock, [ticker_input, period_input],
                      [chart_output, metrics_output])


if __name__ == "__main__":
    print(f"\nLaunching Stock Price Predictor...")
    print(f"   Open http://localhost:{config['ui']['port']} in your browser\n")
    demo.launch(
        server_name="0.0.0.0",
        server_port=config["ui"]["port"],
        share=False,
        theme=gr.themes.Soft(primary_hue="indigo"),
    )
