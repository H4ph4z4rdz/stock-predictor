"""Train the stock price prediction model.

Usage:
    python src/train.py
    python src/train.py --ticker MSFT
    python src/train.py --ticker NVDA --epochs 200

This script:
1. Downloads historical stock data from Yahoo Finance
2. Engineers technical indicator features
3. Trains an LSTM model to predict future prices
4. Generates visualization charts
5. Saves the model and plots
"""

import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from core.data import StockDataPipeline
from core.model import StockLSTM
from core.trainer import train_model
from core.visualize import (
    plot_training_history,
    plot_predictions,
    plot_feature_importance,
    plot_technical_indicators,
)
from utils.helpers import load_config, get_device


def main():
    parser = argparse.ArgumentParser(description="Train stock price predictor")
    parser.add_argument("--ticker", type=str, default=None, help="Stock ticker symbol")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs")
    args = parser.parse_args()

    # Load config
    config = load_config()
    if args.ticker:
        config["data"]["ticker"] = args.ticker
    if args.epochs:
        config["training"]["epochs"] = args.epochs

    ticker = config["data"]["ticker"]
    plots_dir = config["output"]["plots_dir"]

    print("=" * 60)
    print(f"  STOCK PRICE PREDICTOR - Training on {ticker}")
    print("=" * 60)

    # Setup device
    device = get_device()

    # Prepare data
    pipeline = StockDataPipeline(config)
    data = pipeline.prepare_data()

    # Plot technical indicators (what the model sees)
    plot_technical_indicators(
        data["df"], ticker,
        save_path=os.path.join(plots_dir, "technical_indicators.png"),
    )

    # Create model
    num_features = len(config["data"]["features"])
    model = StockLSTM(
        input_size=num_features,
        hidden_size=config["model"]["hidden_size"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
        bidirectional=config["model"]["bidirectional"],
    ).to(device)

    # Train
    history = train_model(
        model=model,
        train_loader=data["train_loader"],
        test_loader=data["test_loader"],
        config=config,
        device=device,
    )

    # Plot training history
    plot_training_history(
        history,
        save_path=os.path.join(plots_dir, "training_history.png"),
    )

    # Load best model for evaluation
    model_path = os.path.join(config["output"]["model_dir"], "best_model.pt")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Generate predictions on test set
    X_test_tensor = torch.FloatTensor(data["X_test"]).to(device)
    with torch.no_grad():
        test_predictions = model(X_test_tensor).cpu().numpy()

    # Convert back to dollar prices
    actual_prices = pipeline.inverse_transform_price(data["y_test"])
    predicted_prices = pipeline.inverse_transform_price(test_predictions)

    # Also get training predictions for the full chart
    X_train_tensor = torch.FloatTensor(data["X_train"]).to(device)
    with torch.no_grad():
        train_predictions = model(X_train_tensor).cpu().numpy()
    train_actual = pipeline.inverse_transform_price(data["y_train"])

    # Calculate metrics
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    rmse = np.sqrt(np.mean((actual_prices - predicted_prices) ** 2))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100

    # Direction accuracy: did we predict the right direction?
    actual_direction = np.diff(actual_prices) > 0
    predicted_direction = np.diff(predicted_prices) > 0
    direction_acc = np.mean(actual_direction == predicted_direction) * 100

    print(f"\n{'='*50}")
    print(f"  TEST RESULTS - {ticker}")
    print(f"{'='*50}")
    print(f"  MAE:  ${mae:.2f}")
    print(f"  RMSE: ${rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Direction Accuracy: {direction_acc:.1f}%")
    print(f"  Price Range: ${actual_prices.min():.2f} - ${actual_prices.max():.2f}")

    # Plot predictions
    plot_predictions(
        actual_prices=actual_prices,
        predicted_prices=predicted_prices,
        dates=data["dates_test"],
        ticker=ticker,
        save_path=os.path.join(plots_dir, "predictions.png"),
        train_prices=train_actual,
        train_dates=data["dates_train"],
    )

    # Plot feature importance
    plot_feature_importance(
        model=model,
        X_test=data["X_test"],
        feature_names=config["data"]["features"],
        device=device,
        save_path=os.path.join(plots_dir, "feature_importance.png"),
    )

    # Save metrics to file
    with open(os.path.join(plots_dir, "metrics.txt"), "w") as f:
        f.write(f"Ticker: {ticker}\n")
        f.write(f"MAE: ${mae:.2f}\n")
        f.write(f"RMSE: ${rmse:.2f}\n")
        f.write(f"MAPE: {mape:.2f}%\n")
        f.write(f"Direction Accuracy: {direction_acc:.1f}%\n")
        f.write(f"Test samples: {len(actual_prices)}\n")
        f.write(f"Epochs trained: {len(history['train_loss'])}\n")

    print(f"\nCharts saved to {plots_dir}/")
    print(f"Model saved to {config['output']['model_dir']}/")
    print(f"\nRun the Gradio UI: python src/app.py")

    # Save pipeline config for the UI
    torch.save({
        "scaler_params": {
            "data_min": pipeline.scaler.data_min_.tolist(),
            "data_max": pipeline.scaler.data_max_.tolist(),
            "data_range": pipeline.scaler.data_range_.tolist(),
        },
        "price_scaler_params": {
            "data_min": pipeline.price_scaler.data_min_.tolist(),
            "data_max": pipeline.price_scaler.data_max_.tolist(),
            "data_range": pipeline.price_scaler.data_range_.tolist(),
        },
        "config": config,
    }, os.path.join(config["output"]["model_dir"], "pipeline_config.pt"))


if __name__ == "__main__":
    main()
