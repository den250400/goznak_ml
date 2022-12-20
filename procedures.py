import torch
import torch.nn as nn
import torch.optim as optim


def select_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_classification(model: nn.Module, dataloader: torch.utils.data.DataLoader, n_epochs: int = 10):
    # Select device for training
    device = select_device()
    print(f"Training device: {device}")
    model.to(device)
    model.train()

    # Train the model
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(n_epochs):
        for sequence, label in dataloader:
            sequence, label = sequence.to(device), label.to(device)

            model.zero_grad()

            scores = model(sequence.permute(1, 0, 2))

            loss = loss_fn(scores, label)
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item()}")


def eval_classification(model: nn.Module, dataloader: torch.utils.data.DataLoader):
    device = select_device()
    print(f"Eval device: {device}")
    model.to(device)
    model.eval()

    # Evaluate the model
    n_correct = 0
    for sequence, label in dataloader:
        prediction = model(sequence.permute(1, 0, 2)).item()
        if prediction >= 0:
            prediction = 1
        else:
            prediction = 0
        if label[0] == prediction:
            n_correct += 1
        print("%i; Predicted: %.3f" % (label[0].item(), prediction))

    print("Accuracy: %.3f%%" % (n_correct / len(dataloader) * 100))
