import os
import torch

import config
from modules.dataloader import load_denoising_dataloader
from modules.models import LSTMDenoiser
from modules.procedures import train_denoising

model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)

# Load data
dataloader = load_denoising_dataloader("./data/train")
validation_dataloader = load_denoising_dataloader("./data/val")

# Train
train_denoising(model, dataloader, validation_dataloader, n_epochs=10)

# Save the model
if not os.path.exists(config.MODEL_SAVE_DIR):
    os.makedirs(config.MODEL_SAVE_DIR)

model.to('cpu')
torch.save(model.state_dict(), os.path.join(config.MODEL_SAVE_DIR, 'denoiser.pth'))
