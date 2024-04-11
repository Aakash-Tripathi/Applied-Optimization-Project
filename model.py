import torch
import torch.nn as nn

# Define the MLP model
class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
 
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
        