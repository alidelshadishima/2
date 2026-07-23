"""
div2k_feature_transform.py

Fisher-Guided Feature Transformation / Weight Transformation
for Single Image Super-Resolution (SISR)

Dataset:
    DIV2K

Input:
    Low-Resolution (LR) image

Target:
    High-Resolution (HR) image

Pipeline:
    DIV2K
       |
       v
    Source SR Network
       |
       v
    Target SR Network
       |
       v
    Fisher / Feature Extraction
       |
       v
    SR Feature Transformation Network
       |
       v
    Pruning (Optional)
       |
       v
    Fine-Tuning
       |
       v
    PSNR / SSIM Evaluation
"""

import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset

from skimage.metrics import structural_similarity as compare_ssim

# Optional pruning utilities
from utils.masked_layer import MaskedConv2d
from utils.pruning import weight_prune, prune_rate, finetune

# DIV2K data loader
from data_loader import DIV2KDataset


# ============================================================
# Device
# ============================================================

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Parser
# ============================================================

parser = argparse.ArgumentParser(

    description=
    "Fisher-Guided Feature Transformation "
    "for DIV2K Super-Resolution"

)

parser.add_argument(

    "--lr",
    default=1e-4,
    type=float

)

parser.add_argument(

    "--batch-size-train",
    default=16,
    type=int

)

parser.add_argument(

    "--batch-size-test",
    default=1,
    type=int

)

parser.add_argument(

    "--num-epoch",
    default=50,
    type=int

)

parser.add_argument(

    "--scale",
    default=2,
    type=int,
    choices=[2, 3, 4]

)

parser.add_argument(

    "--patch-size",
    default=48,
    type=int

)

parser.add_argument(

    "--source-checkpoint",
    type=str,
    required=True

)

parser.add_argument(

    "--target-checkpoint",
    type=str,
    required=True

)

parser.add_argument(

    "--div2k-root",
    type=str,
    default="./DIV2K"

)

args = parser.parse_args()


# ============================================================
# Training Parameters
# ============================================================

param = {

    "pruning_perc": 90.0,

    "batch_size": args.batch_size_train,

    "test_batch_size": args.batch_size_test,

    "num_epochs": args.num_epoch,

    "learning_rate": args.lr,

    "weight_decay": 5e-4,

    "scale": args.scale

}


# ============================================================
# Charbonnier Loss
# ============================================================

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):

        super().__init__()

        self.eps = eps

    def forward(self, prediction, target):

        diff = prediction - target

        loss = torch.sqrt(

            diff * diff + self.eps * self.eps

        )

        return loss.mean()


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(

    prediction,
    target,
    max_val=1.0

):

    prediction = torch.clamp(

        prediction,

        0.0,

        1.0

    )

    target = torch.clamp(

        target,

        0.0,

        1.0

    )

    mse = F.mse_loss(

        prediction,

        target

    )

    if mse.item() == 0:

        return float("inf")

    psnr = (

        10 *

        torch.log10(

            max_val ** 2 / mse

        )

    )

    return psnr.item()


# ============================================================
# Basic SR Residual Block
# ============================================================

class ResidualBlock(nn.Module):

    def __init__(

        self,

        channels=64

    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                3,

                padding=1

            ),

            nn.ReLU(

                inplace=True

            ),

            nn.Conv2d(

                channels,

                channels,

                3,

                padding=1

            )

        )

    def forward(self, x):

        return (

            x +

            self.block(x)

        )


# ============================================================
# SR Feature Transformation Network
# ============================================================

class SRFeatureTransform(nn.Module):

    """
    Transform source SR features
    toward target SR features.

    Input:
        LR image

    Output:
        HR reconstruction
    """

    def __init__(

        self,

        in_channels=3,

        out_channels=3,

        features=64,

        num_blocks=8,

        scale=2

    ):

        super().__init__()

        self.scale = scale

        self.head = nn.Conv2d(

            in_channels,

            features,

            3,

            padding=1

        )

        self.body = nn.Sequential(

            *[

                ResidualBlock(

                    features

                )

                for _ in range(

                    num_blocks

                )

            ]

        )

        self.body_conv = nn.Conv2d(

            features,

            features,

            3,

            padding=1

        )

        self.upsample = nn.Sequential(

            nn.Conv2d(

                features,

                features *

                scale *

                scale,

                3,

                padding=1

            ),

            nn.PixelShuffle(

                scale

            ),

            nn.ReLU(

                inplace=True

            )

        )

        self.tail = nn.Conv2d(

            features,

            out_channels,

            3,

            padding=1

        )

    def forward(self, x):

        x = self.head(x)

        residual = x

        x = self.body(x)

        x = self.body_conv(x)

        x = x + residual

        x = self.upsample(x)

        x = self.tail(x)

        return x


# ============================================================
# DIV2K Data Loader
# ============================================================

def load_div2k(

    root,

    scale,

    patch_size,

    batch_size_train,

    batch_size_test

):

    """
    Expected DIV2KDataset interface:

        DIV2KDataset(
            root=...,
            train=True,
            scale=...,
            patch_size=...
        )

    Each sample:

        LR, HR
    """

    train_dataset = DIV2KDataset(

        root=root,

        train=True,

        scale=scale,

        patch_size=patch_size

    )

    test_dataset = DIV2KDataset(

        root=root,

        train=False,

        scale=scale,

        patch_size=None

    )

    trainloader = DataLoader(

        train_dataset,

        batch_size=batch_size_train,

        shuffle=True,

        num_workers=4,

        pin_memory=True

    )

    testloader = DataLoader(

        test_dataset,

        batch_size=batch_size_test,

        shuffle=False,

        num_workers=4,

        pin_memory=True

    )

    return (

        trainloader,

        testloader

    )


# ============================================================
# Load Source / Target SR Networks
# ============================================================

def load_sr_network(

    checkpoint_path,

    scale

):

    """
    Load pretrained SR network.

    The checkpoint must contain:

        checkpoint["net"]

    or directly contain state_dict.
    """

    model = SRFeatureTransform(

        in_channels=3,

        out_channels=3,

        features=64,

        num_blocks=8,

        scale=scale

    ).to(device)

    checkpoint = torch.load(

        checkpoint_path,

        map_location=device

    )

    if "net" in checkpoint:

        model.load_state_dict(

            checkpoint["net"]

        )

    else:

        model.load_state_dict(

            checkpoint

        )

    return model


# ============================================================
# Extract SR Features
# ============================================================

@torch.no_grad()

def extract_sr_features(

    model,

    dataloader

):

    """

    Extract intermediate SR features.

    Returns:

        feature_list

    """

    model.eval()

    features = []

    for lr, hr in dataloader:

        lr = lr.to(device)

        feature = model.head(

            lr

        )

        features.append(

            feature.cpu()

        )

    return features


# ============================================================
# Extract Last SR Output
# ============================================================

@torch.no_grad()

def extract_sr_output(

    model,

    dataloader

):

    model.eval()

    outputs = []

    targets = []

    for lr, hr in dataloader:

        lr = lr.to(device)

        hr = hr.to(device)

        output = model(

            lr

        )

        outputs.append(

            output.cpu()

        )

        targets.append(

            hr.cpu()

        )

    return (

        outputs,

        targets

    )


# ============================================================
# Train Transformation Network
# ============================================================

def SR_fit(

    model,

    train_loader,

    epochs,

    learning_rate

):

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=learning_rate,

        weight_decay=5e-4

    )

    criterion = CharbonnierLoss()

    model.train()

    for epoch in range(

        epochs

    ):

        total_loss = 0.0

        for lr, hr in train_loader:

            lr = lr.to(device)

            hr = hr.to(device)

            optimizer.zero_grad()

            output = model(

                lr

            )

            loss = criterion(

                output,

                hr

            )

            loss.backward()

            optimizer.step()

            total_loss += (

                loss.item()

            )

        mean_loss = (

            total_loss /

            len(train_loader)

        )

        print(

            "Epoch [{}/{}] "

            "SR Loss: {:.6f}".format(

                epoch + 1,

                epochs,

                mean_loss

            )

        )


# ============================================================
# Evaluate SR Network
# ============================================================

@torch.no_grad()

def SR_evaluate(

    model,

    test_loader

):

    model.eval()

    total_psnr = 0.0

    total_mse = 0.0

    count = 0

    for lr, hr in test_loader:

        lr = lr.to(device)

        hr = hr.to(device)

        output = model(

            lr

        )

        output = torch.clamp(

            output,

            0.0,

            1.0

        )

        mse = F.mse_loss(

            output,

            hr

        )

        psnr = calculate_psnr(

            output,

            hr

        )

        total_mse += mse.item()

        total_psnr += psnr

        count += 1

    mean_mse = (

        total_mse /

        max(count, 1)

    )

    mean_psnr = (

        total_psnr /

        max(count, 1)

    )

    print(

        "Test MSE: {:.6f}".format(

            mean_mse

        )

    )

    print(

        "Test PSNR: {:.4f} dB".format(

            mean_psnr

        )

    )

    return (

        mean_mse,

        mean_psnr

    )


# ============================================================
# Count Non-Zero Parameters
# ============================================================

def countNonZeroWeights(

    model

):

    nonzeros = 0

    allpara = 0

    for param in model.parameters():

        allpara += param.numel()

        nonzeros += (

            param != 0

        ).sum().item()

    return (

        allpara,

        nonzeros

    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print(

        "Device:",

        device

    )

    print(

        "Dataset: DIV2K"

    )

    print(

        "Scale:",

        args.scale

    )


    # --------------------------------------------------------
    # Load DIV2K
    # --------------------------------------------------------

    trainloader, testloader = load_div2k(

        root=args.div2k_root,

        scale=args.scale,

        patch_size=args.patch_size,

        batch_size_train=args.batch_size_train,

        batch_size_test=args.batch_size_test

    )


    # --------------------------------------------------------
    # Load source SR network
    # --------------------------------------------------------

    print(

        "\nLoading source SR network..."

    )

    net_source = load_sr_network(

        args.source_checkpoint,

        args.scale

    )


    # --------------------------------------------------------
    # Load target SR network
    # --------------------------------------------------------

    print(

        "Loading target SR network..."

    )

    net_target = load_sr_network(

        args.target_checkpoint,

        args.scale

    )


    # --------------------------------------------------------
    # Extract source features
    # --------------------------------------------------------

    print(

        "\nExtracting source SR features..."

    )

    source_features = extract_sr_features(

        net_source,

        trainloader

    )


    # --------------------------------------------------------
    # Extract target features
    # --------------------------------------------------------

    print(

        "Extracting target SR features..."

    )

    target_features = extract_sr_features(

        net_target,

        trainloader

    )


    # --------------------------------------------------------
    # Create SR transformation network
    # --------------------------------------------------------

    print(

        "\nCreating SR Feature Transformation Network..."

    )

    torch.manual_seed(

        0

    )

    transform_sr = SRFeatureTransform(

        in_channels=3,

        out_channels=3,

        features=64,

        num_blocks=8,

        scale=args.scale

    ).to(device)

    print(

        transform_sr

    )


    # --------------------------------------------------------
    # Train SR transformation network
    # --------------------------------------------------------

    SR_fit(

        transform_sr,

        trainloader,

        epochs=args.num_epoch,

        learning_rate=args.lr

    )


    # --------------------------------------------------------
    # Evaluate before pruning
    # --------------------------------------------------------

    print(

        "\nEvaluation before pruning:"

    )

    mse, psnr = SR_evaluate(

        transform_sr,

        testloader

    )


    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    all_param, non_zero = (

        countNonZeroWeights(

            transform_sr

        )

    )

    print(

        "All parameters: {}".format(

            all_param

        )

    )

    print(

        "Non-zero parameters: {}".format(

            non_zero

        )

    )


    # ========================================================
    # Optional Pruning
    # ========================================================

    """
    masks = weight_prune(

        transform_sr,

        5

    )

    transform_sr.set_masks(

        masks

    )

    print(

        "Begin SR pruning..."

    )

    # Fine-tuning after pruning

    finetune(

        transform_sr,

        param,

        trainloader

    )

    prune_rate(

        transform_sr

    )

    # Evaluate after pruning

    print(

        "Evaluation after pruning:"

    )

    SR_evaluate(

        transform_sr,

        testloader

    )
    """
