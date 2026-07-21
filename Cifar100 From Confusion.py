import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100

from collections import defaultdict

# -------------------
# Grouping utilities
# -------------------
def build_groups_from_confusion(conf_mat, threshold=None, top_k_pairs=None):
    num_classes = conf_mat.shape[0]
    adjacency = np.zeros((num_classes, num_classes), dtype=bool)

    if threshold is not None:
        # rule: if normalized confusion(i,j) > threshold, connect them
        norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)
        for i in range(num_classes):
            for j in range(num_classes):
                if i != j and norm[i, j] >= threshold:
                    adjacency[i, j] = True
                    adjacency[j, i] = True
    elif top_k_pairs is not None:
        flat = []
        norm = conf_mat / conf_mat.sum(axis=1, keepdims=True)
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                score = (norm[i, j] + norm[j, i]) / 2
                flat.append((score, i, j))
        flat.sort(reverse=True)
        for _, i, j in flat[:top_k_pairs]:
            adjacency[i, j] = adjacency[j, i] = True
    else:
        raise ValueError("Either threshold or top_k_pairs must be set")

    # find connected components
    visited = np.zeros(num_classes, dtype=bool)
    groups = []
    for i in range(num_classes):
        if not visited[i]:
            comp = []
            stack = [i]
            visited[i] = True
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in range(num_classes):
                    if adjacency[u, v] and not visited[v]:
                        visited[v] = True
                        stack.append(v)
            groups.append(comp)
    return groups

# -------------------
# Dataset builder
# -------------------
def make_cifar100_group_dataset(conf_mat, group_idx, args, threshold=None, top_k_pairs=None, balance=True):
    groups = build_groups_from_confusion(conf_mat, threshold=threshold, top_k_pairs=top_k_pairs)
    
    chosen_group = groups[group_idx]
    print(f"[INFO] Selected group {group_idx}: {chosen_group}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = CIFAR100(root='./data', train=False, download=True, transform=transform)

    def relabel_dataset(dataset):
        labels = np.array(dataset.targets)
        new_labels = np.zeros_like(labels)
        new_labels[np.isin(labels, chosen_group)] = 1

        if balance:
            idx1 = np.where(new_labels == 1)[0]
            idx0 = np.where(new_labels == 0)[0]
            np.random.shuffle(idx0)
            idx0 = idx0[:len(idx1)]
            keep_idx = np.concatenate([idx0, idx1])
            dataset.targets = new_labels[keep_idx].tolist()
            dataset.data = dataset.data[keep_idx]
        else:
            dataset.targets = new_labels.tolist()

    relabel_dataset(trainset)
    relabel_dataset(testset)

    trainloader = DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
    testloader = DataLoader(testset, batch_size=args.batch_size_test, shuffle=False)
    return trainloader, testloader, groups

# -------------------
# Main
# -------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR100 grouping from confusion matrix")
    parser.add_argument('--confusion-file', type=str, required=True, help='path to confusion matrix .npy file (100x100)')
    parser.add_argument('--threshold', type=float, default=None, help='threshold for confusion linking')
    parser.add_argument('--top-k-pairs', type=int, default=None, help='use top-K most confused pairs instead of threshold')
    parser.add_argument('--group-index', type=int, default=0, help='which group to pick')
    parser.add_argument('--batch-size-train', type=int, default=200)
    parser.add_argument('--batch-size-test', type=int, default=1000)
    parser.add_argument('--no-balance', action='store_true')
    args = parser.parse_args()

    conf_mat = np.load(args.confusion_file)
    trainloader, testloader, groups = make_cifar100_group_dataset(
        conf_mat, args.group_index, args,
        threshold=args.threshold,
        top_k_pairs=args.top_k_pairs,
        balance=not args.no_balance
    )

    print("[INFO] Number of groups found:", len(groups))
    for gi, g in enumerate(groups):
        print(f"  Group {gi}: {g}")
