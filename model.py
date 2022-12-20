import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2class = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.tensor):
        lstm_out, _ = self.lstm(sequence)
        logits = self.hidden2class(lstm_out[-1].view(1, -1))
        return logits
