import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from IPython.display import clear_output
import torchvision.transforms.functional as TF
from SGD import sgd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def Setup_model(net,criterion,optimizer):
    net = net
    # Loss function
    criterion =  criterion
    # Optimizer (Stochastic Gradient Descent)
    optimizer = optimizer

    return net, criterion, optimizer

def train(net, train_loader, criterion, lr):
    net.train()
    correct = 0
    total = 0   
    total_loss=0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
    
        # optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
     
        loss.backward()
        sgd(net.parameters(),lr)
        total_loss += loss.item()
        average_loss = total_loss / len(train_loader)

        with torch.no_grad():
             _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    accuracy = 100 * correct / total
        
    
   
    return average_loss, accuracy

def test(net ,test_loader, criterion):
    net.eval
    correct = 0
    total = 0   
    total_loss=0
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
