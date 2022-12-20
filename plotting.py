import numpy as np
from matplotlib import pyplot as plt


def show_spectrograms(clean, noisy, colormap='jet'):
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