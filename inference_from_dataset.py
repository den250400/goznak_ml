import torch
import argparse
import os

from modules.models import LSTMDenoiser
from modules.dataloader import load_denoising_dataloader
from modules.audioutils import mel2audio
import config

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, help="Path to directory with test data", default='./data/val')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='denoiser.pth')
parser.add_argument("--clean_path", type=str, help="Clean audio save path", default='./data/clean.wav')
parser.add_argument("--noisy_path", type=str, help="Noisy audio save path", default='./data/noisy.wav')
parser.add_argument("--predicted_path", type=str, help="Predicted (denoised) audio save path", default='./data/predicted.wav')
args = parser.parse_args()


model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, args.model_filename)))

dataloader = load_denoising_dataloader(args.dataset_path)

for noisy, clean in dataloader:
    predicted = model(noisy.permute(1, 0, 2))

    mel2audio(clean.squeeze().numpy().T, args.clean_path)
    mel2audio(noisy.squeeze().numpy().T, args.noisy_path)
    mel2audio(predicted.detach().squeeze().numpy().T, args.predicted_path)
    break
