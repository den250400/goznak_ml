import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from modules.plotting import show3spectrograms


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_classification(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                         validation_dataloader: torch.utils.data.DataLoader, n_epochs: int = 10):
    """
    Train the classification model

    :param model: Neural network
    :param dataloader: dataloader with training dataset
    :param validation_dataloader: dataloader with validation dataset
    :param n_epochs: number of training epochs
    :return:
    """
    # Select device for training
    device = select_device()
    print(f"Training device: {device}")
    model.to(device)

    # Train the models
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}/{n_epochs}")
        model.train()
        for sequence, label in tqdm(dataloader):
            sequence, label = sequence.to(device), label.to(device)

            model.zero_grad()

            scores = model(sequence.permute(1, 0, 2))

            loss = loss_fn(scores, label)
            loss.backward()
            optimizer.step()

        eval_classification(model, validation_dataloader)


def eval_classification(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    """
    Compute and print the accuracy of model

    :param model: Neural network
    :param dataloader: dataloader with test dataset
    :return:
    """
    device = select_device()
    print(f"Eval device: {device}")
    model.to(device)
    model.eval()

    # Evaluate the models
    n_correct = 0
    for sequence, label in tqdm(dataloader):
        sequence, label = sequence.to(device), label.to(device)

        prediction = model(sequence.permute(1, 0, 2)).item()
        if prediction >= 0:
            prediction = 1
        else:
            prediction = 0
        if label[0] == prediction:
            n_correct += 1

    print("Accuracy: %.3f%%" % (n_correct / len(dataloader) * 100))


def train_denoising(model: nn.Module, dataloader: torch.utils.data.DataLoader,
                    validation_dataloader: torch.utils.data.DataLoader, n_epochs: int = 10):
    """
    Train the denoising model

    :param model: Neural network
    :param dataloader: dataloader with training dataset
    :param validation_dataloader: dataloader with validation dataset
    :param n_epochs: number of training epochs
    :return:
    """
    # Select device for training
    device = select_device()
    print(f"Training device: {device}")
    model.to(device)

    # Train the models
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}/{n_epochs}")
        model.train()
        for noisy, clean in tqdm(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)

            model.zero_grad()

            predicted = model(noisy.permute(1, 0, 2))

            loss = loss_fn(predicted[None, :, :], clean)
            loss.backward()
            optimizer.step()

        eval_denoising(model, validation_dataloader)


def eval_denoising(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    """
    Compute and print mean MSE for test data
    :param model: Neural network
    :param dataloader: dataloader with test data
    :return:
    """
    device = select_device()
    print(f"Eval device: {device}")
    model.to(device)
    model.eval()

    # Evaluate the models
    loss_fn = nn.MSELoss()
    mse = 0
    for noisy, clean in tqdm(dataloader):
        noisy, clean = noisy.to(device), clean.to(device)

        predicted = model(noisy.permute(1, 0, 2))
        loss = loss_fn(predicted[None, :, :], clean)

        mse += loss.item()

    print("Mean MSE: %.3f" % (mse / len(dataloader)))


def show_denoising_spectrograms(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    """
    Plot the noisy input, denoised and ground-truth spectrograms

    :param model: Neural network
    :param dataloader: dataloader with test data
    :return:
    """
    device = select_device()
    print(f"Eval device: {device}")
    model.to(device)
    model.eval()

    with torch.no_grad():
        for noisy, clean in tqdm(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)

            predicted = model(noisy.permute(1, 0, 2))

            clean = clean.squeeze().cpu().numpy()
            noisy = noisy.squeeze().cpu().numpy()
            predicted = predicted.cpu().numpy()

            show3spectrograms(clean, noisy, predicted)
