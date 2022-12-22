import os
import torch

import config
from modules.dataloader import load_classification_dataloader
from modules.models import LSTMClassifier
from modules.procedures import eval_classification


model = LSTMClassifier(input_dim=80, hidden_dim=64)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, 'classifier2.pth')))

dataloader = load_classification_dataloader("./data/val")

eval_classification(model, dataloader)
