from model import SimpleMLP

from Train_test import train, test, Setup_model
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def main():
    # Define a transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Lambda(lambda x: torch.flatten(x)),
          # Normalize the tensor with mean and standard deviation
    ])

    trainset = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Download and load the testing data
    testset = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    print("loaded MNIST")


    epochs=100
    lr=0.01
    net = SimpleMLP(28*28,100,10)
    net=net.to(device)
    


    criterion=nn.CrossEntropyLoss()
    

    accs=[]
    test_accs=[]
    losses=[]
    test_losses=[]
    for epoch in range(epochs):
        print('epoch: ', epoch, "="*30)

        loss, accuracy = train(net, trainloader, criterion, lr)
        test_loss,test_accuracy = test(net,testloader,criterion)
       

    
        
        print('train loss',loss, 'test loss ',test_loss, "|, train acc ",accuracy," test acc ", test_accuracy)
        accs.append(accuracy)
        test_accs.append(test_accuracy)
        losses.append(loss)
        test_losses.append(test_loss)

        
    
# Start plotting
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Loss')
    plt.plot(test_losses, label='Test Loss',color='orange')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accs, label='Accuracy')
    plt.plot(test_accs, label='Test Accuracy', color='orange')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Show plot
    plt.savefig('results.png')
main()