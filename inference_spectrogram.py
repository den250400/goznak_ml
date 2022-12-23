import torch
import argparse
import os
import numpy as np

from modules.models import LSTMDenoiser
from modules.plotting import show2spectrograms
import config


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path to input MEL-spectrogram", default='./data/noisy.npy')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='denoiser.pth')
parser.add_argument("--output_path", type=str, help="Output spectrogram save path", default='./data/predicted.npy')
args = parser.parse_args()

model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, args.model_filename)))

spectrogram = np.load(args.input_path)
spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(1)

predicted = model(spectrogram_tensor)
predicted_np = predicted.detach().squeeze().numpy()

np.save(args.output_path, predicted_np)
show2spectrograms(predicted_np, spectrogram)

