# fisher_guided_cell.py
# Build a non-random NAS cell using Fisher information to pick (or combine) the best operations per edge.

import os
import argparse
import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.utils.data import DataLoader

from operation import OPS              # dict: name -> OpClass(in_channels, stride, affine)
from data_loader import *              # your existing dataset loaders

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------------------
# Arg parsing
# ---------------------------
parser = argparse.ArgumentParser(description='Fisher-guided NAS Cell')
parser.add_argument('--lr', default=5e-3, type=float)
parser.add_argument('--batch-size-train', default=64, type=int)
parser.add_argument('--batch-size-test', default=1000, type=int)
parser.add_argument('--num-epoch', default=20, type=int)
parser.add_argument('--calib-batches', default=5, type=int, help='mini-batches used to rank ops by Fisher')
parser.add_argument('--topk', default=2, type=int, help='how many ops to combine per edge (1=pick best)')
parser.add_argument('--dataset', default='quickdraw', choices=['MNIST','fMNIST','quickdraw'])
parser.add_argument('--indicator-idx', default=1, type=int)
parser.add_argument('--num-classes', default=10, type=int)
parser.add_argument('--out-dim', default=2, type=int, help='2 for binary, 10 for full-class, etc.')
args = parser.parse_args()

# ---------------------------
# Utility: Flatten
# ---------------------------
class Flatten(nn.Module):
    def forward(self, x): return torch.flatten(x, 1)

# ---------------------------
# A tiny probe head to turn feature maps into logits for Fisher scoring
# op_out -> GAP -> Linear(out_dim)
# ---------------------------
class ProbeHead(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, x):
        x = self.gap(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# ---------------------------
# Fisher score for an operator on a few calibration batches
# We backprop CE loss through the operator parameters (probe is part of the graph)
# ---------------------------
@torch.no_grad()
def _zeros_like_params(module):
    z = {}
    for n, p in module.named_parameters():
        if p.requires_grad:
            z[n] = torch.zeros_like(p, device=p.device)
    return z

def fisher_score_for_op(op_module, calib_loader, out_dim, max_batches):
    op_module.train()  # we need gradients
    # Probe head is throw-away; we DO NOT use its params after scoring
    # Determine op output channels by a single dry run on first batch
    first_x, _ = next(iter(calib_loader))
    first_x = first_x.to(device)
    with torch.no_grad():
        y = op_module(first_x)
    c_out = y.shape[1]

    probe = ProbeHead(c_out, out_dim).to(device)
    criterion = nn.CrossEntropyLoss()

    # Accumulate E[(grad theta)^2]
    # We'll compute grads batch-by-batch (no optimizer step!), then square & add
    fisher_sums = _zeros_like_params(op_module)
    batches = 0

    for x, labels in calib_loader:
        x = x.to(device)
        labels = labels.to(device).long()

        # forward
        logits = probe(op_module(x))
        loss = criterion(logits, labels if out_dim > 2 else (labels > 0).long())
        # backward: we need grads for op_module only; keep graph False (no step)
        for p in op_module.parameters():
            if p.grad is not None: p.grad.zero_()
        for p in probe.parameters():
            if p.grad is not None: p.grad.zero_()

        loss.backward()

        # accumulate squared grads of op parameters
        with torch.no_grad():
            for n, p in op_module.named_parameters():
                if p.grad is not None:
                    fisher_sums[n] += p.grad.detach()**2

        batches += 1
        if batches >= max_batches:
            break

    # Average over batches, then reduce to a scalar score (sum of means)
    score = 0.0
    with torch.no_grad():
        for n, acc in fisher_sums.items():
            acc /= max(1, batches)
            score += acc.mean().item()
    return score

# ---------------------------
# Combine multiple ops: weighted sum of outputs
# Weights are softmax of Fisher scores (deterministic)
# ---------------------------
class CombinedOp(nn.Module):
    def __init__(self, in_channels, stride, op_specs):
        """
        op_specs: list of (op_name, weight)
        """
        super().__init__()
        self.branches = nn.ModuleList([OPS[name](in_channels, stride, affine=False) for name, _ in op_specs])
        weights = torch.tensor([w for _, w in op_specs], dtype=torch.float32)
        self.register_buffer('weights', torch.softmax(weights, dim=0))

    def forward(self, x):
        outs = [branch(x) for branch in self.branches]
        # all outs must have same shape
        out = 0
        for w, y in zip(self.weights, outs):
            out = out + w * y
        return out

# ---------------------------
# Fisher-guided Cell (non-random)
# Architecture: 3 edges: 0->1, 0->2, 1->2 (like your original Cell)
# For each edge we choose top-k ops by Fisher score on calibration data.
# ---------------------------
class FisherGuidedCell(nn.Module):
    def __init__(self, dim, calib_loader, topk=2, reduction=0, in_channels=1, out_dim=2, calib_batches=5):
        super().__init__()
        stride = 2 if reduction == 1 else 1
        self.dim = dim

        # Rank operations by Fisher per edge
        op_names = list(OPS.keys())

        def select_ops(edge_name, inch, stride):
            # score each op
            scores = []
            for name in op_names:
                op = OPS[name](inch, stride, affine=False).to(device)
                s = fisher_score_for_op(op, calib_loader, out_dim, max_batches=calib_batches)
                scores.append((name, s))
            # sort desc by score and keep topk
            scores.sort(key=lambda z: z[1], reverse=True)
            top = scores[:topk]
            # normalize weights via softmax when building CombinedOp
            return top

        # For simplicity we keep all edges with same in/out channels as original code
        # Edge 0->1
        top01 = select_ops("0_1", in_channels, stride)
        self.op_0_1 = CombinedOp(in_channels, stride, top01)

        # Edge 0->2
        top02 = select_ops("0_2", in_channels, stride)
        self.op_0_2 = CombinedOp(in_channels, stride, top02)

        # Edge 1->2 takes as input the channel count of op_0_1 output.
        # We run a dry-forward on one batch to infer channels.
        with torch.no_grad():
            x0, _ = next(iter(calib_loader))
            x0 = x0.to(device)
            c_after_01 = self.op_0_1(x0).shape[1]
        top12 = select_ops("1_2", c_after_01, stride)
        self.op_1_2 = CombinedOp(c_after_01, stride, top12)

        # Head: concat(s02, s12) -> flatten -> Linear to dim
        self.classifier = nn.Sequential(
            # We need to infer spatial dims; use one dry pass
            nn.Identity()
        )

        # Infer flattened dimension
        with torch.no_grad():
            s01 = self.op_0_1(x0)
            s02 = self.op_0_2(x0)
            s12 = self.op_1_2(s01)
            cat = torch.cat([s02, s12], dim=1)
            flat_dim = int(np.prod(cat.shape[1:]))
        self.head = nn.Linear(flat_dim, dim)

    def forward(self, X):
        s01 = self.op_0_1(X)
        s02 = self.op_0_2(X)
        s12 = self.op_1_2(s01)
        out = torch.cat([s02, s12], dim=1)
        out = torch.flatten(out, 1)
        return self.head(out)

# ---------------------------
# Train / Evaluate (same as before)
# ---------------------------
def fit(model, train_loader, epochs, lr):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    for ep in range(epochs):
        correct = 0
        count = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.long().to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            count += yb.numel()
        print(f'Epoch {ep+1:02d}: train acc = {100*correct/count:.2f}%')

@torch.no_grad()
def evaluate(model, test_loader, total_labels):
    model.eval()
    correct = 0
    count = 0
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.long().to(device)
        logits = model(xb)
        pred = logits.argmax(1)
        correct += (pred == yb).sum().item()
        count += yb.numel()
    print(f'Test accuracy: {100*correct/count:.2f}%')

# ---------------------------
# Main
# ---------------------------
def main():
    # Load data (same API you used before)
    indicator_idx = args.indicator_idx
    out_dim = args.out_dim

    if args.dataset == 'MNIST':
        trainloader, testloader, train_label, test_label = indicator_dataset('MNIST', indicator_idx, args.num_classes, [], args)
    elif args.dataset == 'fMNIST':
        trainloader, testloader, train_label, test_label = indicator_dataset('fMNIST', indicator_idx, args.num_classes, [], args)
    else:
        # QuickDraw example with 10 classes predefined in your project
        class_object = ['apple','baseball-bat','bear','envelope','guitar','lollipop','moon','mouse','mushroom','rabbit']
        trainloader, testloader, train_label, test_label = indicator_dataset('quickdraw', indicator_idx, args.num_classes, class_object, args)

    # Build Fisher-guided cell (no randomness)
    print('Calibrating operations with Fisher information...')
    cell = FisherGuidedCell(dim=out_dim,
                            calib_loader=trainloader,
                            topk=args.topk,
                            reduction=0,
                            in_channels=1,
                            out_dim=out_dim,
                            calib_batches=args.calib_batches).to(device)

    print(cell)  # shows chosen ops and their weights
    fit(cell, trainloader, epochs=args.num_epoch, lr=args.lr)
    evaluate(cell, testloader, test_label)

if __name__ == '__main__':
    main()
