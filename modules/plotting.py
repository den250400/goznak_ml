import numpy as np
from matplotlib import pyplot as plt


def show2spectrograms(clean: np.array, noisy: np.array, colormap: str = 'jet'):
    """
    Show noisy and clean spectrograms

    :param clean: np.array(shape=[sequence_len, 80])
    :param noisy: np.array(shape=[sequence_len, 80])
    :param colormap: matplotlib colormap
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(2, 1, 1)
    img = clean - clean.min()
    img = img / img.max() * 255
    plt.title('Clean')
    plt.imshow(img.astype(np.uint8).T, cmap=colormap)

    fig.add_subplot(2, 1, 2)
    img = noisy - noisy.min()
    img = img / img.max() * 255
    plt.title('Noisy')
    plt.imshow(img.astype(np.uint8).T, cmap=colormap)

    plt.show()


def show3spectrograms(clean: np.array, noisy: np.array, prediction: np.array, colormap: str = 'jet'):
    """
    Show noisy, predicted and ground-truth spectrogram

    :param clean: np.array(shape=[sequence_len, 80])
    :param noisy: np.array(shape=[sequence_len, 80])
    :param prediction: np.array(shape=[sequence_len, 80])
    :param colormap: matplotlib colormap
    :return:
    """
    fig = plt.figure(figsize=(8, 8))
    fig.add_subplot(3, 1, 1)
    img = noisy - noisy.min()
    img = img / img.max() * 255
    plt.title('Noisy input')
    plt.imshow(img.astype(np.uint8).T, cmap=colormap)

    fig.add_subplot(3, 1, 2)
    img = prediction - prediction.min()
    img = img / img.max() * 255
    plt.title('Denoised prediction')
    plt.imshow(img.astype(np.uint8).T, cmap=colormap)

    fig.add_subplot(3, 1, 3)
    img = clean - clean.min()
    img = img / img.max() * 255
    plt.title('True label')
    plt.imshow(img.astype(np.uint8).T, cmap=colormap)

    plt.show()
