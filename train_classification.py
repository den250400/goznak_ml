import os
import torch

import config
from dataloader import load_classification_dataloader
from model import LSTMClassifier
from procedures import train_classification

model = LSTMClassifier(input_dim=80, hidden_dim=32)

dataloader = load_classification_dataloader("./data/val")

train_classification(model, dataloader, n_epochs=1)

if not os.path.exists(config.MODEL_SAVE_DIR):
    os.makedirs(config.MODEL_SAVE_DIR)

torch.save(model.state_dict(), config.MODEL_SAVE_DIR)

