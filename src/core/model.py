"""LSTM model for stock price prediction.

WHAT IS AN LSTM?
LSTM = Long Short-Term Memory. It's a type of Recurrent Neural Network (RNN)
designed to learn patterns in sequential data (data with a time component).

WHY NOT A REGULAR NEURAL NETWORK?
A regular (feedforward) network sees all inputs simultaneously — it has no
concept of "before" and "after". For stock data, ORDER MATTERS:
  [100, 105, 110] → trending UP
  [110, 105, 100] → trending DOWN
Same numbers, different meaning. LSTMs understand this because they process
data one step at a time, maintaining a "memory" of what came before.

HOW LSTM WORKS (simplified):
An LSTM cell has 3 gates that control information flow:

1. FORGET GATE: "What old information should I discard?"
   - Example: "That price spike 50 days ago isn't relevant anymore"

2. INPUT GATE: "What new information should I remember?"
   - Example: "RSI just dropped below 30 — that's important!"

3. OUTPUT GATE: "What should I output right now?"
   - Example: "Based on everything I've seen, predict price goes up"

The magic is that these gates are LEARNED during training — the model
figures out on its own what to remember and what to forget.

ARCHITECTURE:
  Input: [batch_size, 60 days, 12 features]
      |
  [LSTM Layer 1] — processes sequence step by step, outputs hidden states
      |            each step sees: current day's features + previous hidden state
  [Dropout]     — regularization (prevent overfitting)
      |
  [LSTM Layer 2] — deeper pattern recognition on Layer 1's outputs
      |
  [Final Hidden State] — the LSTM's "summary" of the entire 60-day sequence
      |
  [Linear Layer] — maps hidden state → single price prediction
      |
  Output: predicted price (1 number)
"""

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """LSTM-based stock price prediction model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        """Initialize the LSTM model.

        Args:
            input_size: Number of input features per time step (12 for our indicators).
            hidden_size: Number of LSTM hidden units (the model's "memory capacity").
                        More units = can learn more complex patterns, but risks overfitting.
            num_layers: Number of stacked LSTM layers. Layer 1 learns basic patterns,
                       Layer 2 learns patterns-of-patterns (higher-level abstractions).
            dropout: Dropout rate between LSTM layers (regularization).
            bidirectional: If True, process sequence both forward and backward.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # LSTM layers
        # input_size = number of features per day (12)
        # hidden_size = internal memory size (128)
        # The LSTM outputs a hidden state at each time step
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,           # Input shape: [batch, seq_len, features]
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        # Fully connected layers to map LSTM output → price prediction
        fc_input_size = hidden_size * self.num_directions
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),           # Output: single price value
        )

        # Count parameters
        total = sum(p.numel() for p in self.parameters())
        print(f"  Model parameters: {total:,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Step by step:
        1. LSTM processes the sequence day-by-day
           - At each step, it sees: [today's 12 features] + [yesterday's hidden state]
           - It updates its internal memory (cell state) using the 3 gates
           - It outputs a new hidden state

        2. We take the LAST hidden state (after seeing all 60 days)
           - This is the LSTM's "summary" of the entire sequence
           - It encodes patterns like trends, momentum, volatility changes

        3. Feed through fully connected layers to get price prediction

        Args:
            x: Input tensor [batch_size, seq_length, num_features]
               Example: [32, 60, 12] = 32 samples, 60 days, 12 features each

        Returns:
            Predicted prices [batch_size, 1]
        """
        # LSTM forward pass
        # lstm_out shape: [batch_size, seq_length, hidden_size * num_directions]
        # h_n shape: [num_layers * num_directions, batch_size, hidden_size]
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take the last time step's output
        # This contains the LSTM's "understanding" after seeing all 60 days
        # Shape: [batch_size, hidden_size * num_directions]
        if self.bidirectional:
            # Concatenate forward and backward final hidden states
            last_hidden = torch.cat(
                (h_n[-2, :, :], h_n[-1, :, :]), dim=1
            )
        else:
            last_hidden = h_n[-1, :, :]

        # Map to prediction
        prediction = self.fc(last_hidden)
        return prediction.squeeze(-1)
