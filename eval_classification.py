import os
import torch

import config
from dataloader import load_classification_dataloader
from model import LSTMClassifier


dataloader = load_classification_dataloader("./data/val")

# Evaluate the model
n_correct = 0
n_eval = 400
for sequence, label in dataloader:
    prediction = model(sequence.permute(1, 0, 2)).item()
    if prediction >= 0:
        prediction = 1
    else:
        prediction = 0
    if label[0] == prediction:
        n_correct += 1
    print("%i; Predicted: %.3f" % (label[0].item(), prediction))

print("Accuracy: %.3f" % (n_correct / n_eval * 100))
# show_spectrograms(clean_data[20], noisy_data[20])