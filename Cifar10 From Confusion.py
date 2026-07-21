"""
Build CIFAR-10 binary datasets (group-vs-rest) from a confusion matrix.

Usage examples:

1) Threshold-based grouping (auto-pick largest connected group):
   python cifar10_from_confusion.py \
       --confusion-file ./conf_mat.npy \
       --threshold 0.08 \
       --auto-group largest \
       --batch-size-train 200 --batch-size-test 1000

2) Inspect groups only (no loaders will be saved/printed for brevity):
   python cifar10_from_confusion.py --confusion-file ./conf_mat.npy --threshold 0.1 --dry-run

3) Choose a specific group index after seeing printed groups:
   python cifar10_from_confusion.py --confusion-file ./conf_mat.npy --threshold 0.1 --group-index 1

Notes:
- The confusion matrix can come from any model (e.g., ResNet, VGG). Shape must be (K, K) with K classes.
- We symmetrize and normalize the confusion matrix to measure mutual confusion between classes,
  then create a graph with edges above a threshold and extract connected components as groups.
- If thresholding yields no edges (or you prefer), you can use --top-k-pairs to form groups from the top K strongest
  confusion pairs via a simple union-find.
- By default, the code constructs ONE binary dataset: selected group (positive=1) vs all other classes (negative=0).
  You can set --dry-run to only print inferred groups without creating loaders.
"""

from __future__ import annotations
import argparse
import os
from typing import List, Tuple

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

# ------------------------------
# Dataset & transforms
# ------------------------------
CIFAR10_MEAN = (0.5, 0.5, 0.5)
CIFAR10_STD = (0.5, 0.5, 0.5)

TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
])

# Default CIFAR-10 class names for convenience
CIFAR10_CLASSES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

# ------------------------------
# Confusion utilities
# ------------------------------

def load_confusion_matrix(path: str) -> np.ndarray:
    """Load a confusion matrix from a .npy or .npz file.
    Expects a square array of shape (K, K).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Confusion file not found: {path}")
    arr = np.load(path)
    if isinstance(arr, np.lib.npyio.NpzFile):
        # Try common keys
        for key in ('confusion', 'cm', 'array'):
            if key in arr:
                arr = arr[key]
                break
        else:
            # take first array if unknown key
            first_key = list(arr.keys())[0]
            arr = arr[first_key]
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"Confusion matrix must be square; got shape {arr.shape}")
    return arr.astype(np.float64)


def mutual_confusion(cm: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Compute a symmetric mutual-confusion matrix.
    Steps:
      1) Row-normalize CM to get conditional errors (prob of predicting j given true i).
      2) Symmetrize: M = 0.5*(P + P^T).
      3) Zero out diagonal (we only care about cross-class confusion).
    Returns M in [0, 1] approximately.
    """
    cm = cm.copy()
    # Avoid zero-division; normalize rows
    row_sums = cm.sum(axis=1, keepdims=True) + eps
    P = cm / row_sums
    M = 0.5 * (P + P.T)
    np.fill_diagonal(M, 0.0)
    return M


def groups_from_threshold(M: np.ndarray, threshold: float) -> List[List[int]]:
    """Build groups by connecting classes whose mutual confusion > threshold,
    then taking connected components.
    """
    K = M.shape[0]
    visited = [False] * K
    adj = [[] for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            if M[i, j] > threshold:
                adj[i].append(j)
                adj[j].append(i)

    def dfs(s: int) -> List[int]:
        stack = [s]
        comp = []
        visited[s] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        return sorted(comp)

    groups = []
    for i in range(K):
        if not visited[i]:
            comp = dfs(i)
            groups.append(comp)
    # groups include singleton classes too (no strong confusions)
    return groups


def groups_from_topk_pairs(M: np.ndarray, top_k_pairs: int) -> List[List[int]]:
    """Build groups by unioning the top-K most-confused pairs.
    Remaining classes not in any pair become singletons.
    """
    K = M.shape[0]
    # Extract upper-triangular pairs
    pairs = []
    for i in range(K):
        for j in range(i + 1, K):
            pairs.append(((i, j), M[i, j]))
    pairs.sort(key=lambda x: x[1], reverse=True)
    pairs = pairs[:max(0, top_k_pairs)]

    # Union-find
    parent = list(range(K))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j), _ in pairs:
        union(i, j)

    # Build components
    comps = {}
    for i in range(K):
        r = find(i)
        comps.setdefault(r, []).append(i)
    return [sorted(v) for v in comps.values()]


# ------------------------------
# Dataset construction
# ------------------------------

def make_cifar10_loader_from_group(
    positive_classes: List[int],
    batch_size_train: int,
    batch_size_test: int,
    balance: bool = True,
) -> Tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Create dataloaders where positive_classes are labeled 1 and others 0.
    If balance=True, downsample the majority class in each split to match the minority count.
    """
    data_train = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=TRANSFORM)
    data_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=TRANSFORM)

    # Train labels -> {0,1}
    y_tr = np.array(data_train.targets)
    pos_mask_tr = np.isin(y_tr, positive_classes)
    y_tr_bin = np.zeros_like(y_tr)
    y_tr_bin[pos_mask_tr] = 1

    # Test labels -> {0,1}
    y_te = np.array(data_test.targets)
    pos_mask_te = np.isin(y_te, positive_classes)
    y_te_bin = np.zeros_like(y_te)
    y_te_bin[pos_mask_te] = 1

    def apply_balance(data_obj, y_bin):
        if not balance:
            return np.ones_like(y_bin, dtype=bool)
        idx1 = (y_bin == 1)
        idx0 = (y_bin == 0)
        n1 = int(idx1.sum())
        n0 = int(idx0.sum())
        if n1 == 0 or n0 == 0:
            return np.ones_like(y_bin, dtype=bool)  # nothing to balance
        # Downsample majority to minority size
        if n0 > n1:
            keep0 = np.flatnonzero(idx0)[:n1]
            keep1 = np.flatnonzero(idx1)
        else:
            keep1 = np.flatnonzero(idx1)[:n0]
            keep0 = np.flatnonzero(idx0)
        keep_idx = np.sort(np.concatenate([keep0, keep1]))
        mask = np.zeros_like(y_bin, dtype=bool)
        mask[keep_idx] = True
        return mask

    tr_mask = apply_balance(data_train, y_tr_bin)
    te_mask = apply_balance(data_test, y_te_bin)

    # Apply masks
    data_train.data = data_train.data[tr_mask]
    data_train.targets = y_tr_bin[tr_mask].tolist()
    data_test.data = data_test.data[te_mask]
    data_test.targets = y_te_bin[te_mask].tolist()

    trainloader = DataLoader(data_train, batch_size=batch_size_train, shuffle=True)
    testloader = DataLoader(data_test, batch_size=batch_size_test, shuffle=False)

    return trainloader, testloader, np.array(data_train.targets), np.array(data_test.targets)


# ------------------------------
# CLI
# ------------------------------

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 grouping from confusion matrix')
    parser.add_argument('--confusion-file', type=str, required=True,
                        help='Path to confusion matrix (.npy or .npz)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Edge threshold for mutual confusion (e.g., 0.08). If set, uses threshold grouping.')
    parser.add_argument('--top-k-pairs', type=int, default=None,
                        help='Alternative grouping using top-K most confused pairs (e.g., 8). Used if --threshold is not set.')
    parser.add_argument('--auto-group', type=str, default='largest', choices=['largest', 'smallest', 'none'],
                        help='If not specifying --group-index, auto-pick which group to use as positive.')
    parser.add_argument('--group-index', type=int, default=None,
                        help='Explicit index of the group to mark as positive (printed order).')
    parser.add_argument('--batch-size-train', type=int, default=200)
    parser.add_argument('--batch-size-test', type=int, default=1000)
    parser.add_argument('--no-balance', action='store_true', help='Disable class balancing')
    parser.add_argument('--dry-run', action='store_true', help='Only print groups; do not build loaders')
    parser.add_argument('--print-class-names', action='store_true', help='Print CIFAR-10 names next to indices')

    args = parser.parse_args()

    cm = load_confusion_matrix(args.confusion_file)
    M = mutual_confusion(cm)

    if args.threshold is not None:
        groups = groups_from_threshold(M, args.threshold)
    else:
        if args.top_k_pairs is None:
            raise ValueError('Provide either --threshold or --top-k-pairs for grouping.')
        groups = groups_from_topk_pairs(M, args.top_k_pairs)

    # Pretty print groups
    def fmt_group(g):
        if args.print_class_names and len(CIFAR10_CLASSES) >= max(g)+1:
            names = [CIFAR10_CLASSES[i] for i in g]
            return f"{g} => {names}"
        return str(g)

    print("Inferred groups from confusion matrix:")
    for idx, g in enumerate(groups):
        print(f"  Group {idx}: {fmt_group(g)}")

    if args.dry_run:
        return

    # Decide which group to use as positive
    if args.group_index is not None:
        pos_group = groups[args.group_index]
    else:
        if args.auto_group == 'largest':
            pos_group = max(groups, key=len)
        elif args.auto_group == 'smallest':
            pos_group = min(groups, key=len)
        else:
            # 'none' -> default to largest
            pos_group = max(groups, key=len)

    print(f"\nUsing positive group: {fmt_group(pos_group)}")

    trainloader, testloader, y_tr, y_te = make_cifar10_loader_from_group(
        positive_classes=pos_group,
        batch_size_train=args.batch_size_train,
        batch_size_test=args.batch_size_test,
        balance=(not args.no_balance),
    )

    # Summaries
    n_tr_pos = int((y_tr == 1).sum()); n_tr_neg = int((y_tr == 0).sum())
    n_te_pos = int((y_te == 1).sum()); n_te_neg = int((y_te == 0).sum())
    print(f"Train set: {n_tr_pos} positive, {n_tr_neg} negative (total {len(y_tr)})")
    print(f"Test set : {n_te_pos} positive, {n_te_neg} negative (total {len(y_te)})")

    # The loaders are returned only when imported as a module.

if __name__ == '__main__':
    main()
