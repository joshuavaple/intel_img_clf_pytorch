import torch
import torch.nn as nn
from torchsummary import summary
from config.data_config import INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNEL
import torch.optim as optim



def default_loss():
    return nn.CrossEntropyLoss()


def default_optimizer(model, learning_rate = 0.001):
    return optim.Adam(model.parameters(), lr = learning_rate)


def get_default_device():
    """Picking GPU if available or else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def model_prep_and_summary(model, device):
    """
    Move model to GPU and print model summary
    """
    # Define the model and move it to GPU:
    model = model
    model = model.to(device)
    print('Current device: ' + str(device))
    print('Is Model on CUDA: ' + str(next(model.parameters()).is_cuda))
    # Display model summary:
    summary(model, (INPUT_CHANNEL, INPUT_WIDTH, INPUT_HEIGHT))