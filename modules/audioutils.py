import librosa
import soundfile as sf
import numpy as np


def audio2mel(path: str):
    """
    Read audio from file and convert it to log-normalized MEL-spectrogram

    :param path: path to audio file
    :return:
    """
    audio, samplerate = sf.read(path)
    spectrogram = 1 + np.log(1.e-12 + librosa.feature.melspectrogram(audio, sr=samplerate, n_fft=1024, hop_length=256, fmin=20, fmax=8000, n_mels=80)).T / 10.

    return spectrogram, samplerate


def mel2audio(spectrogram: np.array, path: str, samplerate: int = 16000):
    """
    Transform MEL-spectrogram to audio using the Griffin-Lim algorithm and save it to disk

    :param spectrogram: np.array(shape=[80, sequence_len])
    :param path: path for saving the audiofile
    :param samplerate: output sample rate
    :return:
    """
    spectrogram = np.exp((spectrogram - 1) * 10)
    audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr=samplerate, n_fft=1024, hop_length=256, fmin=20, fmax=8000)
    sf.write(path, audio, samplerate=samplerate)
