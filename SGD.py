import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model import SimpleMLP

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


def main():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    trainset = datasets.MNIST(
        root="./data", download=True, train=True, transform=transform
    )
    testset = datasets.MNIST(
        root="./data", download=True, train=False, transform=transform
    )

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

    epochs = 10  # 100
    lr = 0.01

    net = SimpleMLP(28 * 28, 100, 10).to(device)
    criterion = nn.CrossEntropyLoss()

    accs = []
    test_accs = []
    losses = []
    test_losses = []
    for epoch in range(epochs):
        print("epoch: ", epoch, "=" * 30)

        loss, accuracy = train(net, trainloader, criterion, lr)
        test_loss, test_accuracy = test(net, testloader, criterion)

        print(
            "train loss",
            loss,
            "test loss ",
            test_loss,
            "|, train acc ",
            accuracy,
            " test acc ",
            test_accuracy,
        )
        accs.append(accuracy)
        test_accs.append(test_accuracy)
        losses.append(loss)
        test_losses.append(test_loss)

    # Start plotting
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label="Loss")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accs, label="Accuracy")
    plt.plot(test_accs, label="Test Accuracy", color="orange")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Show plot
    plt.savefig("./results/sgd-results.png")


if __name__ == "__main__":
    main()
