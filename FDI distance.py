import os
import argparse
import numpy as np
from copy import deepcopy
from scipy.linalg import sqrtm

import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from data_loader import indicator_dataset   # تابع لود داده‌ها

# -------------------------------
# Parser
# -------------------------------
parser = argparse.ArgumentParser(description='CNN Fisher Comparison')
parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
parser.add_argument('--batch-size-train', default=200, type=int, help='batch size train')
parser.add_argument('--batch-size-test', default=1000, type=int, help='batch size test')
parser.add_argument('--num-epoch', default=10, type=int, help='number of epochs')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------------
# Helper Functions
# -------------------------------
class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.flatten(1)


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


# -------------------------------
# Simple CNN Architecture
# -------------------------------
class CNN(nn.Module):
    def __init__(self, c):
        super(CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            Flatten(),
            nn.Linear(28 * 28 * 32, 1024),
            nn.ReLU(True)
        )
        self.classifier = nn.Linear(1024, c)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x


# -------------------------------
# Fisher Information (Diagonal Approximation)
# -------------------------------
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


# -------------------------------
# Wasserstein-2 Distance
# -------------------------------
def wasserstein2_distance(F1, F2):
    # ensure symmetric
    F1 = (F1 + F1.T) / 2
    F2 = (F2 + F2.T) / 2
    # sqrt(F1)
    F1_sqrt = sqrtm(F1)
    middle = sqrtm(F1_sqrt @ F2 @ F1_sqrt)
    W2 = np.trace(F1 + F2 - 2 * middle).real
    return W2


# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    # dataset
    dataset = 'MNIST'   # or fMNIST / CIFAR10
    source = 8
    target = 9
    indicator = 0
    total_class = 10
    out_dim = 2

    # load train/test data
    trainloader, testloader, train_label, test_label = indicator_dataset(
        dataset, indicator, total_class, None, args
    )

    # load pretrained CNN1
    net_source = CNN(out_dim).to(device)
    checkpoint = torch.load(f'./checkpoint/task{source}.t1', map_location=device)
    net_source.load_state_dict(checkpoint['net'])

    fisher_source = diag_fisher(net_source, trainloader)
    F1 = np.zeros([6, 6])
    F1[0, 0] = np.mean(fisher_source['encoder.0.weight'].cpu().numpy())
    F1[1, 1] = np.mean(fisher_source['encoder.0.bias'].cpu().numpy())
    F1[2, 2] = np.mean(fisher_source['encoder.3.weight'].cpu().numpy())
    F1[3, 3] = np.mean(fisher_source['encoder.3.bias'].cpu().numpy())
    F1[4, 4] = np.mean(fisher_source['classifier.weight'].cpu().numpy())
    F1[5, 5] = np.mean(fisher_source['classifier.bias'].cpu().numpy())

    # load pretrained CNN2
    net_target = CNN(out_dim).to(device)
    checkpoint = torch.load(f'./checkpoint/task{target}.t1', map_location=device)
    net_target.load_state_dict(checkpoint['net'])

    fisher_target = diag_fisher(net_target, trainloader)
    F2 = np.zeros([6, 6])
    F2[0, 0] = np.mean(fisher_target['encoder.0.weight'].cpu().numpy())
    F2[1, 1] = np.mean(fisher_target['encoder.0.bias'].cpu().numpy())
    F2[2, 2] = np.mean(fisher_target['encoder.3.weight'].cpu().numpy())
    F2[3, 3] = np.mean(fisher_target['encoder.3.bias'].cpu().numpy())
    F2[4, 4] = np.mean(fisher_target['classifier.weight'].cpu().numpy())
    F2[5, 5] = np.mean(fisher_target['classifier.bias'].cpu().numpy())

    # compute Wasserstein distance
    distance = wasserstein2_distance(F1, F2)
    print("2-Wasserstein Distance between source and target:", distance)
