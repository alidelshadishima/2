import os
import argparse
import numpy as np
from copy import deepcopy

import torch
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data

from models import *
from data_loader import *
from scipy.stats import wasserstein_distance   # اضافه شد

# Parser
parser = argparse.ArgumentParser(description='NAS Training with Wasserstein Distance')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=64, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=10, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()
device = 'cuda'


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


# Simple classifier
class BinaryNN(nn.Module):
    def __init__(self):
        super(BinaryNN, self).__init__()
        self.classifier = nn.Linear(10, 4)
        
    def forward(self, X):
        X = self.classifier(X)
        return X


class Binary4NN(nn.Module):
    def __init__(self):
        super(Binary4NN, self).__init__()
        self.classifier = nn.Linear(10, 10)
        
    def forward(self, X):
        X = self.classifier(X)
        return X


def diag_fisher(model, data):
    precision_matrices = {}
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    for n, p in deepcopy(params).items():
        p.data.zero_()
        precision_matrices[n] = variable(p.data)

    model.eval()
    error = nn.CrossEntropyLoss()
    for inputs, labels in data:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs)
        loss = error(output, labels)
        loss.backward()

        for n, p in model.named_parameters():
            precision_matrices[n].data += (p.grad.data ** 2).mean(0)

    precision_matrices = {n: p for n, p in precision_matrices.items()}
    return precision_matrices


# Load dataset
dataset = 'CIFAR10'
base_task_list = np.array([[1,3,8], [3,8,9], [2,6,7], [10]], dtype=object)
source = 2
target = 2

trainloader, testloader, train_label, test_label = CIFAR_multi_indicator_dataset(dataset, base_task_list[target], args)

# Pretrained net
net = DenseNet121().cuda()
binary = BinaryNN().cuda()
binary4 = Binary4NN().cuda()


# --------
# Source
# --------
net_source = torch.nn.Sequential(net, binary).to(device)
checkpoint = torch.load('./checkpoint/cifar10_task'+str(source+30)+'b.t0')
net_source.load_state_dict(checkpoint['net'])
fisher_matrix_source = diag_fisher(net_source, testloader)

# Normalize
total_source = sum(np.sum(f.cpu().numpy()) for f in fisher_matrix_source.values())
for n in fisher_matrix_source:
    fisher_matrix_source[n] = fisher_matrix_source[n] / total_source


# --------
# Target
# --------
net_target = torch.nn.Sequential(net, binary).to(device)
checkpoint = torch.load('./checkpoint/cifar100_densenet121/cifar100_task'+str(target+20)+'.t0')
net_target.load_state_dict(checkpoint['net'])
fisher_matrix_target = diag_fisher(net_target, testloader)

total_target = sum(np.sum(f.cpu().numpy()) for f in fisher_matrix_target.values())
for n in fisher_matrix_target:
    fisher_matrix_target[n] = fisher_matrix_target[n] / total_target


# --------
# Wasserstein distance between source and target
# --------
distance = 0
for n in fisher_matrix_source:
    f_source = fisher_matrix_source[n].detach().cpu().numpy().flatten()
    f_target = fisher_matrix_target[n].detach().cpu().numpy().flatten()
    
    # Normalization
    f_source = f_source / (np.sum(f_source) + 1e-8)
    f_target = f_target / (np.sum(f_target) + 1e-8)

    # Wasserstein distance
    distance += wasserstein_distance(f_source, f_target)

print("2-Wasserstein Distance between source and target = ", distance)
