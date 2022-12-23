import torch
import argparse
import os

from modules.models import LSTMDenoiser
from modules.audioutils import mel2audio, audio2mel
import config


parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Path to input audio file", default='./data/noisy.wav')
parser.add_argument("--model_filename", type=str, help="Filename of model state dict", default='denoiser.pth')
parser.add_argument("--output_path", type=str, help="Output audio save path", default='./data/predicted.wav')
args = parser.parse_args()


model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, args.model_filename)))

spectrogram, samplerate = audio2mel(args.input_path)
spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(1)

predicted = model(spectrogram_tensor)

mel2audio(predicted.detach().squeeze().numpy().T, args.output_path, samplerate=samplerate)
