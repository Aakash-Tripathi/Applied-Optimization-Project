import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model import CIFAR10Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def sgd(params, lr):
    with torch.no_grad():  # Update parameters without tracking gradients
        for param in params:
            if param.grad is not None:
                param -= lr * param.grad
                param.grad.zero_()


def train(net, train_loader, criterion, lr):
    net.train()
    correct = 0
    total = 0
    total_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        sgd(net.parameters(), lr)
        total_loss += loss.item()
        average_loss = total_loss / len(train_loader)

        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
    return average_loss, accuracy


def test(net, test_loader, criterion):
    net.eval
    correct = 0
    total = 0
    total_loss = 0
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item()
        test_loss = total_loss / len(test_loader)

        with torch.no_grad():
            _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    test_accuracy = 100 * correct / total
    return test_loss, test_accuracy


def simulated_annealing_optimize(
    net, train_loader, test_loader, criterion, initial_lr, epochs, T_initial, alpha
):
    lrs = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    best_lr = initial_lr
    best_test_loss = float("inf")  # Initialize to a very high value
    T = T_initial
    for epoch in range(epochs):
        lr = best_lr + np.random.normal(0, best_lr * 0.2)  # Perturb around best_lr
        lr = max(lr, 1e-8)  # Avoid non-positive learning rate
        train_loss, train_accuracy = train(net, train_loader, criterion, lr)
        test_loss, test_accuracy = test(net, test_loader, criterion)

        lrs.append(lr)
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        # Use the test loss for the acceptance criterion
        if test_loss < best_test_loss or np.random.rand() < np.exp(
            (best_test_loss - test_loss) / T
        ):
            best_lr, best_test_loss = lr, test_loss

        print(
            f"Epoch: {epoch}, Learning Rate: {lr:.6f}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}"
        )

        T *= alpha  # Cooling down

    return (
        lrs,
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
    )


def train_with_static_lr(net, train_loader, test_loader, criterion, lr, epochs):
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    for epoch in range(epochs):
        train_loss, train_accuracy = train(net, train_loader, criterion, lr)
        test_loss, test_accuracy = test(net, test_loader, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch: {epoch}, Learning Rate: {lr:.6f}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}"
        )

    return train_losses, test_losses, train_accuracies, test_accuracies


def save_to_csv(
    filename, lrs, train_losses, test_losses, train_accuracies, test_accuracies
):
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Epoch",
                "Learning Rate",
                "Train Loss",
                "Test Loss",
                "Train Accuracy",
                "Test Accuracy",
            ]
        )
        for i in range(len(lrs)):
            writer.writerow(
                [
                    i,
                    lrs[i],
                    train_losses[i],
                    test_losses[i],
                    train_accuracies[i],
                    test_accuracies[i],
                ]
            )


def load_from_csv(filename):
    with open(filename, "r") as file:
        reader = csv.reader(file)
        next(reader, None)  # skip the headers
        data = list(reader)
    # Convert string data to float and unpack
    epochs, lrs, train_losses, test_losses, train_accuracies, test_accuracies = zip(
        *data
    )
    return (
        [int(epoch) for epoch in epochs],
        [float(lr) for lr in lrs],
        [float(loss) for loss in train_losses],
        [float(loss) for loss in test_losses],
        [float(acc) for acc in train_accuracies],
        [float(acc) for acc in test_accuracies],
    )


def plot_from_csv(sa_filename, static_filename):
    (
        sa_epochs,
        sa_lrs,
        sa_train_losses,
        sa_test_losses,
        sa_train_accuracies,
        sa_test_accuracies,
    ) = load_from_csv(sa_filename)
    (
        static_epochs,
        static_lrs,
        static_train_losses,
        static_test_losses,
        static_train_accuracies,
        static_test_accuracies,
    ) = load_from_csv(static_filename)

    plot_combined_results(
        sa_lrs,
        sa_train_losses,
        sa_test_losses,
        sa_train_accuracies,
        sa_test_accuracies,
        static_train_losses,
        static_test_losses,
        static_train_accuracies,
        static_test_accuracies,
        static_lrs[0],
    )


def plot_combined_results(
    sa_lrs,
    sa_train_losses,
    sa_test_losses,
    sa_train_accuracies,
    sa_test_accuracies,
    static_train_losses,
    static_test_losses,
    static_train_accuracies,
    static_test_accuracies,
    static_lr,
):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("CIFAR-10 CNN Hyperparameter Optimization", fontsize=16)

    # Plot losses
    axs[0].plot(sa_train_losses, label="SA Train", color="r")
    axs[0].plot(sa_test_losses, label="SA Test", color="b")
    axs[0].plot(static_train_losses, label="Static Train", linestyle="--", color="r")
    axs[0].plot(static_test_losses, label="Static Test", linestyle="--", color="b")
    axs[0].set_title("Losses")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot accuracies
    axs[1].plot(sa_train_accuracies, label="SA Train", color="r")
    axs[1].plot(sa_test_accuracies, label="SA Test", color="b")
    axs[1].plot(
        static_train_accuracies, label="Static Train", linestyle="--", color="r"
    )
    axs[1].plot(static_test_accuracies, label="Static Test", linestyle="--", color="b")
    axs[1].set_title("Accuracies")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()

    # Plot learning rates
    axs[2].plot(sa_lrs, label="SA", color="b")
    axs[2].hlines(
        static_lr,
        0,
        len(sa_lrs) - 1,
        colors="r",
        linestyles="dashed",
        label="Static",
        color="r",
    )
    axs[2].set_title("Learning Rate")
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Learning Rate")
    axs[2].legend()

    plt.tight_layout()
    plt.savefig("./results/plot/cifar10-cnn.png")


def main():
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(
        root="./data", download=True, train=True, transform=transform
    )
    testset = datasets.CIFAR10(
        root="./data", download=True, train=False, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    initial_lr = 0.01
    epochs = 100
    T_initial = 1.0
    alpha = 0.95
    criterion = nn.CrossEntropyLoss()

    # Simulated Annealing Optimization
    net_sa = CIFAR10Model().to(device)
    (
        sa_lrs,
        sa_train_losses,
        sa_test_losses,
        sa_train_accuracies,
        sa_test_accuracies,
    ) = simulated_annealing_optimize(
        net_sa, trainloader, testloader, criterion, initial_lr, epochs, T_initial, alpha
    )

    # Static Learning Rate Optimization
    net_static = CIFAR10Model().to(device)
    (
        static_train_losses,
        static_test_losses,
        static_train_accuracies,
        static_test_accuracies,
    ) = train_with_static_lr(
        net_static, trainloader, testloader, criterion, initial_lr, epochs
    )

    save_to_csv(
        "./results/csv/cifar10-cnn-sa_results.csv",
        sa_lrs,
        sa_train_losses,
        sa_test_losses,
        sa_train_accuracies,
        sa_test_accuracies,
    )
    save_to_csv(
        "./results/csv/cifar10-cnn-static_lr_results.csv",
        [initial_lr] * epochs,
        static_train_losses,
        static_test_losses,
        static_train_accuracies,
        static_test_accuracies,
    )


if __name__ == "__main__":
    main()
    plot_from_csv(
        "./results/csv/cifar10-cnn-sa_results.csv",
        "./results/csv/cifar10-cnn-static_lr_results.csv",
    )
