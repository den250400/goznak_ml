import os
import argparse
import torch

import config
from modules.dataloader import load_denoising_dataloader
from modules.models import LSTMDenoiser
from modules.procedures import train_denoising


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, help="Number of training epochs", default=10)
parser.add_argument("--dataset_path", type=str, help="Path to directory with train and val folders", default='./data')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='denoiser.pth')
args = parser.parse_args()

model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)

# Load data
dataloader = load_denoising_dataloader(os.path.join(args.dataset_path, "train"))
validation_dataloader = load_denoising_dataloader(os.path.join(args.dataset_path, "val"))

# Train
train_denoising(model, dataloader, validation_dataloader, n_epochs=args.epochs)

# Save the model
if not os.path.exists(config.MODEL_SAVE_DIR):
    os.makedirs(config.MODEL_SAVE_DIR)

model.to('cpu')
torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, args.model_filename))
