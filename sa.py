import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torchvision import datasets, transforms

from model import SimpleMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimulatedAnnealing(Optimizer):
    """
    Simulated Annealing optimizer for optimization problems.

    Args:
        params (iterable): Iterable of parameters to optimize.
        initial_temp (float, optional): Initial temperature for the annealing process. Default is 1.0.
        alpha (float, optional): Temperature decay rate. Default is 0.95.
        min_temp (float, optional): Minimum temperature to stop the annealing process. Default is 1e-5.
    """

    def __init__(self, params, initial_temp=1.0, alpha=0.95, min_temp=1e-5):
        self.params = list(params)  # Convert generator to a list
        if self.params:
            self.device = self.params[0].device
        else:
            raise ValueError("No parameters provided to optimizer")
        self.current_temp = initial_temp
        self.alpha = alpha
        self.min_temp = min_temp

    def step(self, evaluate_loss):
        """
        Perform a single optimization step using the Simulated Annealing algorithm.

        Args:
            evaluate_loss (callable): A function that computes the loss to be optimized.

        """
        original_params = [p.clone().detach() for p in self.params]
        original_loss = evaluate_loss()

        for param in self.params:
            noise = torch.randn_like(
                param.data, device=self.device
            )  # Ensure noise is on the same device
            param.data.add_(noise * self.current_temp)

        perturbed_loss = evaluate_loss()

        if perturbed_loss > original_loss:
            acceptance_probability = torch.exp(
                (original_loss - perturbed_loss) / self.current_temp
            )
            if acceptance_probability < torch.rand(1, device=self.device):
                for param, original in zip(self.params, original_params):
                    param.data.copy_(original)
        self.current_temp = max(self.current_temp * self.alpha, self.min_temp)

    def zero_grad(self):
        pass  # No gradients to reset, method overridden to do nothing


def train(model, train_loader, optimizer):
    """
    Trains the model using the provided training data and optimizer.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): The data loader for the training data.
        optimizer (torch.optim.Optimizer): The optimizer used for training.

    Returns:
        tuple: A tuple containing the average training loss and the training accuracy.
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        def evaluate_loss():
            output = model(data)
            loss = F.cross_entropy(output, target)
            return loss

        optimizer.step(evaluate_loss)
        output = model(data)  # Evaluate to get accuracy stats after updates

        with torch.no_grad():
            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = 100 * correct / total
    return train_loss, train_accuracy


def test(model, test_loader):
    """
    Evaluate the performance of a model on a test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.

    Returns:
        tuple: A tuple containing the test loss and test accuracy.
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        total_loss += loss.item()
        test_loss = total_loss / len(test_loader)

        with torch.no_grad():
            _, pred = torch.max(output.data, 1)

        total += target.size(0)
        correct += pred.eq(target.data).cpu().sum()

    test_accuracy = 100.0 * correct / total

    return test_loss, test_accuracy


def main():
    batch_size = 128
    epochs = 100

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )

    trainset = datasets.MNIST(
        root="./data",
        download=True,
        train=True,
        transform=transform,
    )
    testset = datasets.MNIST(
        root="./data",
        download=True,
        train=False,
        transform=transform,
    )

    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = SimpleMLP(input_size=784, hidden_size=128, output_size=10).to(device)
    optimizer = SimulatedAnnealing(model.parameters())

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = train(model, train_loader, optimizer)
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        test_loss, test_accuracy = test(model, test_loader)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        print(
            f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}"
        )

    # Start plotting
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Loss")
    plt.plot(test_losses, label="Test Loss", color="orange")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy", color="orange")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Show plot
    plt.savefig("./results/sa-results.png")


if __name__ == "__main__":
    main()
