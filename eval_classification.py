import os
import argparse
import torch

import config
from modules.dataloader import load_classification_dataloader
from modules.models import LSTMClassifier
from modules.procedures import eval_classification


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to directory with test data", default='./data/val')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='classifier.pth')
args = parser.parse_args()

model = LSTMClassifier(input_dim=80, hidden_dim=64)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, args.model_filename)))

dataloader = load_classification_dataloader(args.data_path)

eval_classification(model, dataloader)
