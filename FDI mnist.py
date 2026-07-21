import os
import argparse
import numpy as np
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data

from models import *
from data_loader import *
from scipy.stats import wasserstein_distance  # <-- EMD/W2

# ---------------------------
# Parser
# ---------------------------
parser = argparse.ArgumentParser(description='NAS Training (Wasserstein)')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=128, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=50, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------
# Utils
# ---------------------------
class Flatten(torch.nn.Module):
    __constants__ = ['start_dim', 'end_dim']
    def __init__(self, start_dim=1, end_dim=-1):
        super(Flatten, self).__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
    def forward(self, input):
        return input.flatten(self.start_dim, self.end_dim)

def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


# ---------------------------
# Small adaptors / heads
# ---------------------------
class PreNN(nn.Module):
    """Adapts 1-channel MNIST to 3-channel for ImageNet-style backbones."""
    def __init__(self):
        super(PreNN, self).__init__()
        self.encoder = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
    def forward(self, X):
        return self.encoder(X)

class BinaryNN(nn.Module):
    """Binary head: logits of size 2."""
    def __init__(self):
        super(BinaryNN, self).__init__()
        self.classifier = nn.Linear(10, 2)
    def forward(self, X):
        return self.classifier(X)

class Binary4NN(nn.Module):
    """10-way head for full MNIST."""
    def __init__(self):
        super(Binary4NN, self).__init__()
        self.classifier = nn.Linear(10, 10)
    def forward(self, X):
        return self.classifier(X)


# ---------------------------
# Fisher (diagonal approximation)
# ---------------------------
@torch.no_grad()
def _zero_like_params(model):
    precision_matrices = {}
    for n, p in model.named_parameters():
        if p.requires_grad:
            precision_matrices[n] = torch.zeros_like(p)
    return precision_matrices

def diag_fisher(model, data_loader):
    """Diagonal Fisher: E[(∂ log p / ∂θ)^2] estimated via CE grads."""
    precision_matrices = _zero_like_params(model)
    model.train(False)
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # we need grads, so temporarily enable grad
        for n, p in model.named_parameters():
            if p.requires_grad:
                p.grad = None

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # accumulate squared grads mean over batch
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                precision_matrices[n] += (p.grad.detach() ** 2).mean(dim=tuple(range(p.grad.dim())) if p.grad.dim() > 0 else ())

    # turn into Variables on device
    for n in precision_matrices:
        precision_matrices[n] = variable(precision_matrices[n])

    return precision_matrices

def diag_fisher_binary(model, data_loader):
    """
    Fisher when the last head is binary but the backbone expects 10-dim logits.
    We pad logits to shape [B, 10] where first 2 columns are real logits.
    """
    precision_matrices = _zero_like_params(model)
    model.train(False)
    criterion = nn.CrossEntropyLoss()

    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        for n, p in model.named_parameters():
            if p.requires_grad:
                p.grad = None

        outputs = model(inputs)                 # shape [B, 2]
        B = outputs.shape[0]
        padded = torch.zeros(B, 10, device=outputs.device)
        padded[:, :2] = outputs                 # put binary logits in first 2 cols

        loss = criterion(padded, labels)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                precision_matrices[n] += (p.grad.detach() ** 2).mean(dim=tuple(range(p.grad.dim())) if p.grad.dim() > 0 else ())

    for n in precision_matrices:
        precision_matrices[n] = variable(precision_matrices[n])

    return precision_matrices


# ---------------------------
# Convert Fisher tensor -> probability distribution on index line
# ---------------------------
def fisher_to_distribution(t: torch.Tensor, eps: float = 1e-12):
    """
    Map a Fisher tensor to a 1D probability distribution over index positions.
