"""Training loop for the stock price predictor.

TRAINING A TIME SERIES MODEL:
Similar to our CNN and BERT training, but with key differences:

1. LOSS FUNCTION: MSE (Mean Squared Error) instead of Cross-Entropy
   - Classification: "Is this positive or negative?" → Cross-Entropy
   - Regression: "What will the price be?" → MSE
   - MSE = average of (predicted - actual)^2

2. EARLY STOPPING: Very important for stock prediction
   - Stocks are noisy — there's randomness that can't be predicted
   - Training too long = model memorizes noise (overfitting)
   - We stop when validation loss stops improving

3. LEARNING RATE SCHEDULING: Reduce LR when stuck
   - Start with LR=0.001 for fast learning
   - If loss plateaus, reduce to 0.0005, then 0.00025, etc.
   - Helps the model fine-tune in the loss landscape

4. GRADIENT CLIPPING: LSTMs can have exploding gradients
   - Long sequences = many multiplications during backprop
   - Gradients can grow exponentially → NaN/Inf
   - Clipping caps the gradient magnitude
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> dict:
    """Train the LSTM stock predictor.

    Args:
        model: StockLSTM model.
        train_loader: Training DataLoader.
        test_loader: Test/validation DataLoader.
        config: Training configuration.
        device: Device to train on.

    Returns:
        Training history dictionary.
    """
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    weight_decay = config["training"]["weight_decay"]
    patience = config["training"]["patience"]
    min_delta = config["training"]["min_delta"]
    model_dir = config["output"]["model_dir"]

    # Loss function: Mean Squared Error (for regression)
    criterion = nn.MSELoss()

    # Optimizer: Adam
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Learning rate scheduler: reduce LR when loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"],
        verbose=True,
    )

    # Training history
    history = {
        "train_loss": [],
        "test_loss": [],
        "learning_rate": [],
    }

    # Early stopping
    best_test_loss = float("inf")
    patience_counter = 0

    print(f"\nStarting training for up to {epochs} epochs...")
    print(f"  Early stopping patience: {patience}")
    print(f"  Device: {device}\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        train_losses = []

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevent exploding gradients in LSTM)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # --- Evaluation Phase ---
        model.eval()
        test_losses = []

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                test_losses.append(loss.item())

        avg_test_loss = np.mean(test_losses)
        epoch_time = time.time() - epoch_start

        # Update scheduler
        scheduler.step(avg_test_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(avg_train_loss)
        history["test_loss"].append(avg_test_loss)
        history["learning_rate"].append(current_lr)

        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Test Loss: {avg_test_loss:.6f} | "
                f"LR: {current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

        # Early stopping check
        if avg_test_loss < best_test_loss - min_delta:
            best_test_loss = avg_test_loss
            patience_counter = 0
            # Save best model
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete! Best test loss: {best_test_loss:.6f}")
    return history
