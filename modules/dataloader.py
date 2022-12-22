from tqdm import tqdm
import numpy as np
import os
import torch


class CustomDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, _dataset):
        self.dataset = _dataset

    def __getitem__(self, index):
        example, target = self.dataset[index]
        return torch.tensor(example, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def __len__(self):
        return len(self.dataset)


def load_data(path: str):
    """
    Load lists of numpy arrays with noisy, clean data, and generate their corresponding noisy/clean labels
    :param path: path to directory with data. Should contain 'clean' and 'noisy' subdirectories
    :return:
    """
    clean_dirs = sorted([os.path.join(os.path.join(path, "clean"), p) for p in os.listdir(os.path.join(path, "clean"))])
    clean_data = []
    clean_labels = []
    noisy_data = []
    noisy_labels = []

    for clean_dir in tqdm(clean_dirs):
        clean_paths = sorted([os.path.join(clean_dir, p) for p in os.listdir(clean_dir)])
        noisy_paths = [p.replace('clean', 'noisy') for p in clean_paths]
        for clean_path, noisy_path in zip(clean_paths, noisy_paths):
            clean_data.append(np.load(clean_path))
            clean_labels.append(np.array([1]))
            noisy_data.append(np.load(noisy_path))
            noisy_labels.append(np.array([0]))

    return clean_data, clean_labels, noisy_data, noisy_labels


def load_classification_dataloader(path: str, shuffle: bool = True):
    """
    Load data from disk and transform it to dataloader for classification

    :param path: path to directory with data. Should contain 'clean' and 'noisy' subdirectories
    :param shuffle: shuffle arg of torch.utils.data.DataLoader
    :return:
    """
    clean_data, clean_labels, noisy_data, noisy_labels = load_data(path)
    data = list(zip(clean_data, clean_labels))
    data.extend(list(zip(noisy_data, noisy_labels)))

    dataloader = torch.utils.data.DataLoader(dataset=CustomDataset(data),
                                             batch_size=1,
                                             shuffle=shuffle)

    return dataloader


def load_denoising_dataloader(path: str, shuffle: bool = True):
    """
    Load data from disk and transform it to dataloader for denoising

    :param path: path to directory with data. Should contain 'clean' and 'noisy' subdirectories
    :param shuffle: shuffle arg of torch.utils.data.DataLoader
    :return:
    """
    clean_data, clean_labels, noisy_data, noisy_labels = load_data(path)
    data = list(zip(noisy_data, clean_data))

    dataloader = torch.utils.data.DataLoader(dataset=CustomDataset(data),
                                             batch_size=1,
                                             shuffle=shuffle)

    return dataloader
