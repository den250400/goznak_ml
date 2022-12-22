import os
import torch

import config
from modules.dataloader import load_denoising_dataloader
from modules.models import LSTMDenoiser
from modules.procedures import eval_denoising, show_denoising_spectrograms


model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, 'denoiser.pth')))

dataloader = load_denoising_dataloader("./data/val")

eval_denoising(model, dataloader)
show_denoising_spectrograms(model, dataloader)

