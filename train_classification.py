import os
import torch

import config
from modules.dataloader import load_classification_dataloader
from modules.models import LSTMClassifier
from modules.procedures import train_classification

model = LSTMClassifier(input_dim=80, hidden_dim=64)

dataloader = load_classification_dataloader("./data/train")
validation_dataloader = load_classification_dataloader("./data/val")

train_classification(model, dataloader, validation_dataloader, n_epochs=50)

if not os.path.exists(config.MODEL_SAVE_DIR):
    os.makedirs(config.MODEL_SAVE_DIR)

model.to('cpu')
torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'classifier2.pth'))

