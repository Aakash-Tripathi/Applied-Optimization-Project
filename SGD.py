import torch
import torch.nn as nn

def sgd(params, lr):
    with torch.no_grad():  # Update parameters without tracking gradients
        for param in params:
            if param.grad is not None:
                param -= lr * param.grad
                param.grad.zero_() 