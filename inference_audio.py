import torch
import os

from modules.models import LSTMDenoiser
from modules.dataloader import load_denoising_dataloader
import config
from modules.audioutils import mel2audio, audio2mel

PATH = "./data/noisy.wav"

model = LSTMDenoiser(input_dim=80, hidden_dim=80, spectrogram_resolution=80)
model.load_state_dict(torch.load(os.path.join(config.MODEL_SAVE_DIR, 'denoiser.pth')))


spectrogram, samplerate = audio2mel(PATH)
spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(1)

predicted = model(spectrogram_tensor)

mel2audio(predicted.detach().squeeze().numpy().T, "./data/predicted.wav", samplerate=samplerate)

"""
dataloader = load_denoising_dataloader("./data/val")

for noisy, clean in dataloader:
    predicted = model(noisy.permute(1, 0, 2))

    mel2audio(clean.squeeze().numpy().T, "./data/clean.wav")
    mel2audio(noisy.squeeze().numpy().T, "./data/noisy.wav")
    mel2audio(predicted.detach().squeeze().numpy().T, "./data/predicted.wav")
    break
"""