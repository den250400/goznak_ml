import os
import argparse
import torch

import config
from modules.dataloader import load_denoising_dataloader
from modules.models import LSTMDenoiser
from modules.procedures import eval_denoising, show_denoising_spectrograms


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to directory with test data", default='./data/val')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='denoiser.pth')
args = parser.parse_args()

model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, args.model_filename)))

dataloader = load_denoising_dataloader(args.dataset_path)

eval_denoising(model, dataloader)
show_denoising_spectrograms(model, dataloader)

