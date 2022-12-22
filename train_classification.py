import os
import argparse
import torch

import config
from modules.dataloader import load_classification_dataloader
from modules.models import LSTMClassifier
from modules.procedures import train_classification


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, help="Number of training epochs", default=50)
parser.add_argument("--dataset_path", type=str, help="Path to directory with train and val folders", default='./data')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='classifier.pth')
args = parser.parse_args()

model = LSTMClassifier(input_dim=80, hidden_dim=64)

print("Loading the training data")
dataloader = load_classification_dataloader(os.path.join(args.dataset_path, "train"))
print("Loading the validation data")
validation_dataloader = load_classification_dataloader(os.path.join(args.dataset_path, "val"))

# Train
train_classification(model, dataloader, validation_dataloader, n_epochs=args.epochs)

# Save the model
if not os.path.exists(config.MODEL_SAVE_DIR):
    os.makedirs(config.MODEL_SAVE_DIR)

model.to('cpu')
torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, args.model_filename))

