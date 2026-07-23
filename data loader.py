import os
import argparse
from pathlib import Path

import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image


# ============================================================
# Supported datasets
# ============================================================

DATASET_CONFIG = {

    "DIV2K": {
        "train": "DIV2K_train_HR",
        "test": "DIV2K_valid_HR",
    },

    "Flickr2K": {
        "train": "Flickr2K_HR",
        "test": "Flickr2K_HR",
    },

    "Set5": {
        "train": "Set5",
        "test": "Set5",
    },

    "Set14": {
        "train": "Set14",
        "test": "Set14",
    },

    "BSD100": {
        "train": "BSD100",
        "test": "BSD100",
    },

    "Urban100": {
        "train": "Urban100",
        "test": "Urban100",
    },
}


# ============================================================
# Image extensions
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
# Find all images
# ============================================================

def find_images(folder):

    folder = Path(folder)

    if not folder.exists():

        raise FileNotFoundError(
            f"Directory not found: {folder}"
        )

    images = []

    for file in folder.rglob("*"):

        if (
            file.is_file()
            and
            file.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            images.append(
                str(file)
            )

    images.sort()

    if len(images) == 0:

        raise RuntimeError(
            f"No images found in {folder}"
        )

    return images


# ============================================================
# Convert PIL image to Tensor
# ============================================================

def pil_to_tensor(image):

    image = np.array(
        image
    ).astype(
        np.float32
    ) / 255.0

    if image.ndim == 2:

        image = np.expand_dims(
            image,
            axis=-1
        )

    tensor = torch.from_numpy(
        image
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    return tensor


# ============================================================
# Generate LR image from HR
# ============================================================

def generate_lr(
    hr_image,
    scale
):

    width, height = hr_image.size

    lr_width = (
        width // scale
    )

    lr_height = (
        height // scale
    )

    # Make HR divisible by scale

    hr_width = (
        lr_width * scale
    )

    hr_height = (
        lr_height * scale
    )

    hr_image = hr_image.crop(
        (
            0,
            0,
            hr_width,
            hr_height
        )
    )

    lr_image = hr_image.resize(
        (
            lr_width,
            lr_height
        ),
        Image.Resampling.BICUBIC
    )

    return lr_image, hr_image


# ============================================================
# Random crop for SISR
# ============================================================

def random_crop_pair(
    hr_image,
    patch_size,
    scale
):

    hr_patch_size = (
        patch_size * scale
    )

    width, height = hr_image.size

    if (
        width < hr_patch_size
        or
        height < hr_patch_size
    ):

        return None, None

    x = np.random.randint(
        0,
        width - hr_patch_size + 1
    )

    y = np.random.randint(
        0,
        height - hr_patch_size + 1
    )

    hr_crop = hr_image.crop(
        (
            x,
            y,
            x + hr_patch_size,
            y + hr_patch_size
        )
    )

    lr_crop = hr_crop.resize(
        (
            patch_size,
            patch_size
        ),
        Image.Resampling.BICUBIC
    )

    return (
        lr_crop,
        hr_crop
    )


# ============================================================
# Data augmentation
# ============================================================

def augment_pair(
    lr,
    hr
):

    # Horizontal flip

    if np.random.rand() < 0.5:

        lr = lr.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

        hr = hr.transpose(
            Image.Transpose.FLIP_LEFT_RIGHT
        )

    # Vertical flip

    if np.random.rand() < 0.5:

        lr = lr.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

        hr = hr.transpose(
            Image.Transpose.FLIP_TOP_BOTTOM
        )

    # Rotation 90

    if np.random.rand() < 0.5:

        lr = lr.transpose(
            Image.Transpose.ROTATE_90
        )

        hr = hr.transpose(
            Image.Transpose.ROTATE_90
        )

    return (
        lr,
        hr
    )


# ============================================================
# SISR Dataset
# ============================================================

class SISR_Dataset(
    Dataset
):

    def __init__(
        self,
        image_files,
        scale=4,
        patch_size=48,
        training=True,
        augment=True
    ):

        self.image_files = (
            image_files
        )

        self.scale = scale

        self.patch_size = (
            patch_size
        )

        self.training = (
            training
        )

        self.augment = (
            augment
        )

    def __len__(
        self
    ):

        return len(
            self.image_files
        )

    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_files[
                index
            ]
        )

        hr = Image.open(
            image_path
        ).convert(
            "RGB"
        )

        if self.training:

            lr, hr = random_crop_pair(
                hr,
                self.patch_size,
                self.scale
            )

            # If image is too small,
            # select another image

            if lr is None:

                new_index = np.random.randint(
                    0,
                    len(
                        self.image_files
                    )
                )

                return self.__getitem__(
                    new_index
                )

            if self.augment:

                lr, hr = augment_pair(
                    lr,
                    hr
                )

        else:

            lr, hr = generate_lr(
                hr,
                self.scale
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
            "path": image_path
        }


# ============================================================
# Load SISR dataset
# ============================================================

def load_sisr_dataset(
    dataset,
    data_root,
    scale,
    patch_size,
    args
):

    if dataset not in DATASET_CONFIG:

        raise ValueError(
            f"Unknown dataset: {dataset}"
        )

    config = (
        DATASET_CONFIG[
            dataset
        ]
    )

    train_dir = os.path.join(
        data_root,
        config["train"]
    )

    test_dir = os.path.join(
        data_root,
        config["test"]
    )

    print(
        "\nLoading dataset:"
    )

    print(
        dataset
    )

    print(
        "Training directory:",
        train_dir
    )

    print(
        "Testing directory:",
        test_dir
    )

    # --------------------------------------------------------
    # Find HR images
    # --------------------------------------------------------

    train_files = find_images(
        train_dir
    )

    test_files = find_images(
        test_dir
    )

    print(
        "Number of training images:",
        len(train_files)
    )

    print(
        "Number of testing images:",
        len(test_files)
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    trainset = SISR_Dataset(
        image_files=train_files,
        scale=scale,
        patch_size=patch_size,
        training=True,
        augment=True
    )

    testset = SISR_Dataset(
        image_files=test_files,
        scale=scale,
        patch_size=patch_size,
        training=False,
        augment=False
    )

    # --------------------------------------------------------
    # Create DataLoaders
    # --------------------------------------------------------

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    testloader = DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    return (
        trainloader,
        testloader,
        train_files,
        test_files
    )


# ============================================================
# Dataset subset / group
# ============================================================

def select_dataset_group(
    image_files,
    group_indices
):

    selected_files = []

    for idx in group_indices:

        if (
            idx >= 0
            and
            idx < len(
                image_files
            )
        ):

            selected_files.append(
                image_files[
                    idx
                ]
            )

    return selected_files


# ============================================================
# Create groups
# ============================================================

def groups_from_matrix(
    matrix,
    threshold
):

    matrix = np.asarray(
        matrix,
        dtype=np.float64
    )

    # Normalize rows

    row_sum = (
        matrix.sum(
            axis=1,
            keepdims=True
        )
        + 1e-9
    )

    matrix = (
        matrix
        /
        row_sum
    )

    # Symmetric matrix

    matrix = (
        matrix
        +
        matrix.T
    ) / 2.0

    np.fill_diagonal(
        matrix,
        0
    )

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

            if (
                matrix[i, j]
                >
                threshold
            ):

                adjacency[i].append(
                    j
                )

                adjacency[j].append(
                    i
                )

    def dfs(start):

        stack = [
            start
        ]

        group = []

        visited[start] = True

        while stack:

            node = stack.pop()

            group.append(
                node
            )

            for neighbor in adjacency[
                node
            ]:

                if not visited[
                    neighbor
                ]:

                    visited[
                        neighbor
                    ] = True

                    stack.append(
                        neighbor
                    )

        return sorted(
            group
        )

    groups = []

    for i in range(K):

        if not visited[i]:

            groups.append(
                dfs(i)
            )

    return groups


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=
        "SISR Dataset Loader"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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
            "Urban100"
        ]
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="./data"
    )

    # --------------------------------------------------------
    # Super Resolution Scale
    # --------------------------------------------------------

    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        choices=[
            2,
            3,
            4
        ]
    )

    # --------------------------------------------------------
    # Patch size
    # --------------------------------------------------------

    parser.add_argument(
        "--patch-size",
        type=int,
        default=48
    )

    # --------------------------------------------------------
    # Batch sizes
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size-train",
        type=int,
        default=16
    )

    parser.add_argument(
        "--batch-size-test",
        type=int,
        default=1
    )

    # --------------------------------------------------------
    # Workers
    # --------------------------------------------------------

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    # --------------------------------------------------------
    # Optional grouping matrix
    # --------------------------------------------------------

    parser.add_argument(
        "--grouping-file",
        type=str,
        default=None
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None
    )

    parser.add_argument(
        "--group-index",
        type=int,
        default=None
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        trainloader,
        testloader,
        train_files,
        test_files
    ) = load_sisr_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        scale=args.scale,
        patch_size=args.patch_size,
        args=args
    )

    # --------------------------------------------------------
    # Optional grouping
    # --------------------------------------------------------

    if (
        args.grouping_file
        is not None
    ):

        matrix = np.load(
            args.grouping_file
        )

        groups = groups_from_matrix(
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
                f"Group {i}:",
                group
            )

        if (
            args.group_index
            is not None
        ):

            selected_group = groups[
                args.group_index
            ]

            train_files = (
                select_dataset_group(
                    train_files,
                    selected_group
                )
            )

            print(
                "\nSelected group:",
                selected_group
            )

    # --------------------------------------------------------
    # Print batch information
    # --------------------------------------------------------

    batch = next(
        iter(
            trainloader
        )
    )

    lr = batch[
        "lr"
    ]

    hr = batch[
        "hr"
    ]

    print(
        "\n================================"
    )

    print(
        "SISR Dataset Information"
    )

    print(
        "================================"
    )

    print(
        "Dataset:",
        args.dataset
    )

    print(
        "Scale:",
        f"x{args.scale}"
    )

    print(
        "LR shape:",
        lr.shape
    )

    print(
        "HR shape:",
        hr.shape
    )

    print(
        "Training images:",
        len(train_files)
    )

    print(
        "Testing images:",
        len(test_files)
    )

    print(
        "================================"
    )


if __name__ == "__main__":

    main()
