import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int):
        """

        :param input_dim: Number of spectrogram frequency bins
        :param hidden_dim: hidden layer size
        """
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2class = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.tensor):
        """

        :param sequence: torch.tensor(shape=[sequence_len, 1, 80])
        :return: torch.tensor(shape=[1, 1])
        """
        lstm_out, _ = self.lstm(sequence)
        logits = self.hidden2class(lstm_out[-1].view(1, -1))
        return logits


class LSTMDenoiser(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, spectrogram_resolution: int):
        """

        :param input_dim: number of frequency bins in input spectrogram
        :param hidden_dim: hidden layer size
        :param spectrogram_resolution: number of frequency bins in output spectrogram
        """
        super(LSTMDenoiser, self).__init__()
        self.hidden_dim = hidden_dim
        self.spectrogram_resolution = spectrogram_resolution

        self.lstm = nn.LSTM(input_dim, hidden_dim)

        self.hidden2spectrogram = nn.Linear(hidden_dim, spectrogram_resolution)

    def forward(self, noisy: torch.tensor):
        """

        :param noisy: torch.tensor(shape=[sequence_len, 1, 80])
        :return: torch.tensor(shape=[sequence_len, 80])
        """
        lstm_out, _ = self.lstm(noisy)
        predicted = self.hidden2spectrogram(lstm_out.view(-1, self.hidden_dim))
        return predicted
