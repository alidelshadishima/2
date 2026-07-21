import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms


def build_groups_from_confusion(conf_matrix, threshold=None, top_k=None):
    """
    Build class groups based on confusion matrix.

    Parameters:
        conf_matrix (ndarray): 10x10 confusion matrix (rows = true class, cols = predicted class)
        threshold (float): minimum normalized confusion ratio to consider two classes "confused"
        top_k (int): alternatively, keep only the top-k most confused pairs

    Returns:
        groups (list of sets): each set contains class indices grouped together
    """
    num_classes = conf_matrix.shape[0]
    conf_matrix = conf_matrix.astype(float)

    # Normalize by row (per true class)
    row_sums = conf_matrix.sum(axis=1, keepdims=True) + 1e-8
    norm_conf = conf_matrix / row_sums

    pairs = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j:
                pairs.append(((i, j), norm_conf[i, j]))

    # Sort by confusion strength
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

    # Select pairs by threshold or top_k
    if threshold is not None:
        selected = [p for p in pairs if p[1] >= threshold]
    elif top_k is not None:
        selected = pairs[:top_k]
    else:
        raise ValueError("You must provide either threshold or top_k.")

    # Build groups (Union-Find style clustering)
    groups = []
    for (i, j), score in selected:
        added = False
        for g in groups:
            if i in g or j in g:
                g.update([i, j])
                added = True
                break
        if not added:
            groups.append(set([i, j]))

    # Ensure all classes appear at least once
    all_classes = set(range(num_classes))
    for c in all_classes:
        if not any(c in g for g in groups):
            groups.append(set([c]))

    return groups


def build_mnist_confusion_dataset(group_index, groups, args):
    """
    Create MNIST dataset with new binary labels:
      group[group_index] = 1
      all other classes = 0
    """

    transform = transforms.ToTensor()

    train_data = MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = MNIST(root="./data", train=False, download=True, transform=transform)

    selected_group = groups[group_index]

    # Train labels
    train_labels = np.array(train_data.targets)
    new_train_labels = np.isin(train_labels, list(selected_group)).astype(int)
    train_data.targets = torch.tensor(new_train_labels)

    # Test labels
    test_labels = np.array(test_data.targets)
    new_test_labels = np.isin(test_labels, list(selected_group)).astype(int)
    test_data.targets = torch.tensor(new_test_labels)

    # DataLoaders
    trainloader = DataLoader(train_data, batch_size=args.batch_size_train, shuffle=True)
    testloader = DataLoader(test_data, batch_size=args.batch_size_test, shuffle=False)

    return trainloader, testloader, new_train_labels, new_test_labels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST Confusion-based Dataset Builder")
    parser.add_argument("--confusion-file", type=str, required=True,
                        help="Path to saved confusion matrix .npy file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold for confusion ratio")
    parser.add_argument("--top-k", type=int, default=None,
                        help="Alternatively, select top-k confused pairs")
    parser.add_argument("--group-index", type=int, default=0,
                        help="Which group to use for binary classification")
    parser.add_argument("--batch-size-train", type=int, default=200,
                        help="Training batch size")
    parser.add_argument("--batch-size-test", type=int, default=1000,
                        help="Testing batch size")
    args = parser.parse_args()

    # Load confusion matrix
    conf_matrix = np.load(args.confusion_file)

    # Build confusion-based groups
    groups = build_groups_from_confusion(conf_matrix,
                                         threshold=args.threshold,
                                         top_k=args.top_k)

    print("Confusion-based groups:", groups)

    # Build dataset for one group vs. rest
    trainloader, testloader, train_labels, test_labels = build_mnist_confusion_dataset(
        args.group_index, groups, args
    )

    print(f"Selected group {args.group_index}: {groups[args.group_index]}")
    print(f"Train samples: {len(train_labels)}, Test samples: {len(test_labels)}")
