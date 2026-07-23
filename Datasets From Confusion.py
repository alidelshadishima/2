"""
SISR Dataset Builder with Grouping Support
-------------------------------------------


for Single Image Super-Resolution (SISR).

Supported datasets:
    - DIV2K
    - Flickr2K
    - Set5
    - Set14
    - BSD100
    - Urban100

Main functionality:
    1. Load HR images.
    2. Generate LR images using bicubic downsampling.
    3. Extract aligned LR-HR patches.
    4. Optionally group images based on a grouping matrix.
    5. Build PyTorch DataLoaders for SISR training and testing.

Expected directory structure:

data/
    DIV2K/
        train/
            HR/
                0001.png
                0002.png
                ...
        valid/
            HR/
                0801.png
                0802.png
                ...

    Flickr2K/
        HR/
            000001.png
            ...

    Set5/
        HR/
            baby.png
            bird.png
            butterfly.png
            head.png
            woman.png

    Set14/
        HR/
            ...

    BSD100/
        HR/
            ...

    Urban100/
        HR/
            ...

Example:

python sisr_dataset.py \
    --dataset DIV2K \
    --data-root ./data \
    --scale 4 \
    --patch-size 96 \
    --batch-size 16 \
    --num-workers 4

"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Supported image extensions
# ============================================================

IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
)


# ============================================================
# Utility functions
# ============================================================

def is_image_file(filename: str) -> bool:
    """
    Check whether a file is a supported image file.
    """
    return filename.lower().endswith(IMAGE_EXTENSIONS)


def find_images(folder: str) -> List[str]:
    """
    Recursively find all image files inside a directory.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(
            f"Directory does not exist: {folder}"
        )

    files = []

    for path in folder.rglob("*"):
        if path.is_file() and is_image_file(str(path)):
            files.append(str(path))

    files.sort()

    if len(files) == 0:
        raise RuntimeError(
            f"No image files found in: {folder}"
        )

    return files


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to Tensor.

    Output:
        Tensor with shape [C, H, W]
        Values normalized to [0, 1]
    """

    img = np.array(img).astype(np.float32) / 255.0

    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)

    tensor = torch.from_numpy(img)

    tensor = tensor.permute(2, 0, 1)

    return tensor


# ============================================================
# Bicubic LR generation
# ============================================================

def create_lr_image(
    hr: Image.Image,
    scale: int,
) -> Image.Image:
    """
    Generate Low Resolution (LR) image
    from High Resolution (HR) image.

    Example:

        HR: 384 x 384
        scale = 4
        LR: 96 x 96
    """

    width, height = hr.size

    lr_width = width // scale
    lr_height = height // scale

    lr = hr.resize(
        (lr_width, lr_height),
        Image.Resampling.BICUBIC
    )

    return lr


# ============================================================
# SISR Dataset
# ============================================================

class SISRDataset(Dataset):

    def __init__(
        self,
        hr_files: List[str],
        scale: int = 4,
        patch_size: Optional[int] = 96,
        training: bool = True,
        augment: bool = True,
    ):

        self.hr_files = hr_files

        self.scale = scale

        self.patch_size = patch_size

        self.training = training

        self.augment = augment

    def __len__(self):

        return len(self.hr_files)

    def random_crop(
        self,
        hr: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:

        hr_width, hr_height = hr.size

        lr_patch_size = self.patch_size

        hr_patch_size = (
            lr_patch_size * self.scale
        )

        if (
            hr_width < hr_patch_size
            or
            hr_height < hr_patch_size
        ):

            raise ValueError(
                f"Image too small for patch size. "
                f"Image size: {hr.size}, "
                f"Required HR patch: "
                f"{hr_patch_size}x{hr_patch_size}"
            )

        # Random top-left corner in HR space

        x = random.randint(
            0,
            hr_width - hr_patch_size
        )

        y = random.randint(
            0,
            hr_height - hr_patch_size
        )

        # Crop HR

        hr_crop = hr.crop(
            (
                x,
                y,
                x + hr_patch_size,
                y + hr_patch_size,
            )
        )

        # Generate LR

        lr_crop = create_lr_image(
            hr_crop,
            self.scale
        )

        return lr_crop, hr_crop

    def center_crop(
        self,
        hr: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:

        hr_width, hr_height = hr.size

        if self.patch_size is None:

            # Full image evaluation

            lr = create_lr_image(
                hr,
                self.scale
            )

            return lr, hr

        hr_patch_size = (
            self.patch_size * self.scale
        )

        x = max(
            0,
            (hr_width - hr_patch_size) // 2
        )

        y = max(
            0,
            (hr_height - hr_patch_size) // 2
        )

        hr_crop = hr.crop(
            (
                x,
                y,
                x + hr_patch_size,
                y + hr_patch_size,
            )
        )

        lr_crop = create_lr_image(
            hr_crop,
            self.scale
        )

        return lr_crop, hr_crop

    def apply_augmentation(
        self,
        lr: Image.Image,
        hr: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:

        # Horizontal flip

        if random.random() < 0.5:

            lr = lr.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

            hr = hr.transpose(
                Image.Transpose.FLIP_LEFT_RIGHT
            )

        # Vertical flip

        if random.random() < 0.5:

            lr = lr.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

            hr = hr.transpose(
                Image.Transpose.FLIP_TOP_BOTTOM
            )

        # 90-degree rotation

        if random.random() < 0.5:

            lr = lr.transpose(
                Image.Transpose.ROTATE_90
            )

            hr = hr.transpose(
                Image.Transpose.ROTATE_90
            )

        return lr, hr

    def __getitem__(self, index):

        hr_path = self.hr_files[index]

        hr = Image.open(
            hr_path
        ).convert("RGB")

        if self.training:

            lr, hr = self.random_crop(
                hr
            )

            if self.augment:

                lr, hr = self.apply_augmentation(
                    lr,
                    hr
                )

        else:

            lr, hr = self.center_crop(
                hr
            )

        lr = pil_to_tensor(
            lr
        )

        hr = pil_to_tensor(
            hr
        )

        return {
            "lr": lr,
            "hr": hr,
            "path": hr_path,
        }


# ============================================================
# Dataset path manager
# ============================================================

def get_dataset_paths(
    dataset_name: str,
    data_root: str,
) -> Tuple[str, str]:

    dataset_name = dataset_name.lower()

    if dataset_name == "div2k":

        train_dir = os.path.join(
            data_root,
            "DIV2K",
            "train",
            "HR"
        )

        valid_dir = os.path.join(
            data_root,
            "DIV2K",
            "valid",
            "HR"
        )

    elif dataset_name == "flickr2k":

        train_dir = os.path.join(
            data_root,
            "Flickr2K",
            "HR"
        )

        valid_dir = train_dir

    elif dataset_name == "set5":

        train_dir = os.path.join(
            data_root,
            "Set5",
            "HR"
        )

        valid_dir = train_dir

    elif dataset_name == "set14":

        train_dir = os.path.join(
            data_root,
            "Set14",
            "HR"
        )

        valid_dir = train_dir

    elif dataset_name == "bsd100":

        train_dir = os.path.join(
            data_root,
            "BSD100",
            "HR"
        )

        valid_dir = train_dir

    elif dataset_name == "urban100":

        train_dir = os.path.join(
            data_root,
            "Urban100",
            "HR"
        )

        valid_dir = train_dir

    else:

        raise ValueError(
            f"Unsupported dataset: {dataset_name}"
        )

    return train_dir, valid_dir


# ============================================================
# Dataset builder
# ============================================================

def build_sisr_dataloaders(
    dataset_name: str,
    data_root: str,
    scale: int = 4,
    patch_size: int = 96,
    batch_size: int = 16,
    num_workers: int = 4,
):

    train_dir, valid_dir = get_dataset_paths(
        dataset_name,
        data_root
    )

    train_files = find_images(
        train_dir
    )

    valid_files = find_images(
        valid_dir
    )

    print(
        f"\nDataset: {dataset_name}"
    )

    print(
        f"Training images: {len(train_files)}"
    )

    print(
        f"Validation images: {len(valid_files)}"
    )

    print(
        f"Scale factor: x{scale}"
    )

    print(
        f"Patch size LR: {patch_size}"
    )

    print(
        f"Patch size HR: {patch_size * scale}"
    )

    train_dataset = SISRDataset(
        hr_files=train_files,
        scale=scale,
        patch_size=patch_size,
        training=True,
        augment=True,
    )

    valid_dataset = SISRDataset(
        hr_files=valid_files,
        scale=scale,
        patch_size=patch_size,
        training=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return (
        train_loader,
        valid_loader,
        train_dataset,
        valid_dataset,
    )


# ============================================================
# Grouping utilities
# ============================================================

def load_grouping_matrix(
    path: str,
) -> np.ndarray:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Grouping matrix not found: {path}"
        )

    arr = np.load(path)

    if isinstance(
        arr,
        np.lib.npyio.NpzFile
    ):

        for key in (
            "confusion",
            "cm",
            "array",
            "grouping",
        ):

            if key in arr:

                arr = arr[key]

                break

        else:

            first_key = list(
                arr.keys()
            )[0]

            arr = arr[
                first_key
            ]

    if arr.ndim != 2:

        raise ValueError(
            "Grouping matrix must be 2D."
        )

    return arr.astype(
        np.float64
    )


def normalize_grouping_matrix(
    matrix: np.ndarray,
) -> np.ndarray:

    row_sums = (
        matrix.sum(
            axis=1,
            keepdims=True
        )
        + 1e-9
    )

    normalized = (
        matrix
        /
        row_sums
    )

    return normalized


def groups_from_threshold(
    matrix: np.ndarray,
    threshold: float,
) -> List[List[int]]:

    K = matrix.shape[0]

    visited = [
        False
        for _ in range(K)
    ]

    adjacency = [
        []
        for _ in range(K)
    ]

    for i in range(K):

        for j in range(
            i + 1,
            K
        ):

            value = (
                matrix[i, j]
                +
                matrix[j, i]
            ) / 2.0

            if value > threshold:

                adjacency[i].append(
                    j
                )

                adjacency[j].append(
                    i
                )

    def dfs(
        start
    ):

        stack = [
            start
        ]

        component = []

        visited[start] = True

        while stack:

            u = stack.pop()

            component.append(
                u
            )

            for v in adjacency[u]:

                if not visited[v]:

                    visited[v] = True

                    stack.append(
                        v
                    )

        return sorted(
            component
        )

    groups = []

    for i in range(K):

        if not visited[i]:

            groups.append(
                dfs(i)
            )

    return groups


# ============================================================
# Select image group
# ============================================================

def select_image_group(
    image_files: List[str],
    group_indices: List[int],
) -> List[str]:

    selected = []

    for idx in group_indices:

        if (
            0 <= idx
            <
            len(image_files)
        ):

            selected.append(
                image_files[idx]
            )

    return selected


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "SISR Dataset Builder"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="DIV2K",
        choices=[
            "DIV2K",
            "Flickr2K",
            "Set5",
            "Set14",
            "BSD100",
            "Urban100",
        ],
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[
            2,
            3,
            4,
        ],
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--grouping-file",
        type=str,
        default=None,
        help=
        "Optional grouping matrix (.npy/.npz)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--group-index",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load image paths
    # --------------------------------------------------------

    train_dir, valid_dir = get_dataset_paths(
        args.dataset,
        args.data_root
    )

    train_files = find_images(
        train_dir
    )

    valid_files = find_images(
        valid_dir
    )

    # --------------------------------------------------------
    # Optional grouping
    # --------------------------------------------------------

    if (
        args.grouping_file
        is not None
    ):

        matrix = load_grouping_matrix(
            args.grouping_file
        )

        matrix = normalize_grouping_matrix(
            matrix
        )

        if args.threshold is None:

            raise ValueError(
                "When using grouping-file, "
                "threshold must be provided."
            )

        groups = groups_from_threshold(
            matrix,
            args.threshold
        )

        print(
            "\nDetected groups:"
        )

        for i, group in enumerate(
            groups
        ):

            print(
                f"Group {i}: "
                f"{group}"
            )

        if args.group_index is not None:

            selected_indices = groups[
                args.group_index
            ]

            train_files = select_image_group(
                train_files,
                selected_indices
            )

            print(
                "\nSelected group:"
            )

            print(
                selected_indices
            )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = SISRDataset(
        hr_files=train_files,
        scale=args.scale,
        patch_size=args.patch_size,
        training=True,
        augment=True,
    )

    valid_dataset = SISRDataset(
        hr_files=valid_files,
        scale=args.scale,
        patch_size=args.patch_size,
        training=False,
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print(
        "\n================================"
    )

    print(
        "SISR Dataset Summary"
    )

    print(
        "================================"
    )

    print(
        f"Dataset: {args.dataset}"
    )

    print(
        f"Scale: x{args.scale}"
    )

    print(
        f"Training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation images: "
        f"{len(valid_dataset)}"
    )

    print(
        f"LR patch size: "
        f"{args.patch_size} x "
        f"{args.patch_size}"
    )

    print(
        f"HR patch size: "
        f"{args.patch_size * args.scale} x "
        f"{args.patch_size * args.scale}"
    )

    print(
        f"Train batches: "
        f"{len(train_loader)}"
    )

    print(
        f"Validation batches: "
        f"{len(valid_loader)}"
    )

    print(
        "================================"
    )

    # --------------------------------------------------------
    # Test one batch
    # --------------------------------------------------------

    lr, hr = next(
        iter(train_loader)
    )["lr"], next(
        iter(train_loader)
    )["hr"]

    print(
        "\nExample batch:"
    )

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "HR shape:",
        hr.shape
    )


if __name__ == "__main__":

    main()
