"""
div2k_sr_train.py

DIV2K Single Image Super-Resolution (SISR)

Input:
    Low-Resolution (LR) image

Output:
    Super-Resolved image with the same spatial size as HR target

Dataset:
    DIV2K

This code replaces the original CIFAR-10 classification pipeline.

Main differences:
    - Classification -> Super-Resolution
    - CrossEntropyLoss -> L1Loss
    - Accuracy -> PSNR / SSIM
    - CIFAR10 -> DIV2K
    - Binary classifier -> Reconstruction network
    - ResNet18 -> Residual SR Network

Designed as a baseline for:
    - NAS-SR
    - Fisher-guided NAS
    - NSGA-NAS-SR
    - Multi-objective NAS

Objectives can later be:
    1. Maximize PSNR
    2. Maximize SSIM
    3. Minimize Parameters
    4. Minimize FLOPs
"""

import os
import argparse
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from PIL import Image

import numpy as np


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser(
    description="DIV2K Single Image Super-Resolution"
)

parser.add_argument(
    "--div2k-root",
    default="./DIV2K",
    type=str,
    help="Root directory of DIV2K dataset"
)

parser.add_argument(
    "--scale",
    default=4,
    type=int,
    choices=[2, 3, 4, 8],
    help="Super-resolution scale factor"
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
    default=100,
    type=int
)

parser.add_argument(
    "--lr",
    default=1e-4,
    type=float
)

parser.add_argument(
    "--num-workers",
    default=4,
    type=int
)

parser.add_argument(
    "--patch-size",
    default=96,
    type=int,
    help="HR patch size"
)

parser.add_argument(
    "--num-features",
    default=64,
    type=int
)

parser.add_argument(
    "--num-blocks",
    default=16,
    type=int
)

parser.add_argument(
    "--save-dir",
    default="./checkpoint_sr",
    type=str
)

args = parser.parse_args()


# ============================================================
# Reproducibility
# ============================================================

SEED = 0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DIV2K Dataset
# ============================================================

class DIV2KDataset(Dataset):

    def __init__(
        self,
        root,
        scale=4,
        patch_size=96,
        train=True
    ):

        self.root = root
        self.scale = scale
        self.patch_size = patch_size
        self.train = train

        if train:

            self.hr_dir = os.path.join(
                root,
                "DIV2K_train_HR"
            )

        else:

            self.hr_dir = os.path.join(
                root,
                "DIV2K_valid_HR"
            )

        if not os.path.exists(self.hr_dir):

            raise FileNotFoundError(
                f"HR directory not found:\n{self.hr_dir}"
            )

        self.images = sorted([
            f for f in os.listdir(self.hr_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

        if len(self.images) == 0:

            raise RuntimeError(
                f"No images found in {self.hr_dir}"
            )

        print(
            "Loaded {} DIV2K images from {}".format(
                len(self.images),
                self.hr_dir
            )
        )


    def __len__(self):

        return len(self.images)


    def __getitem__(self, index):

        image_name = self.images[index]

        hr_path = os.path.join(
            self.hr_dir,
            image_name
        )

        hr = Image.open(
            hr_path
        ).convert("RGB")


        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        if self.train:

            # Random crop
            hr_width, hr_height = hr.size

            if (
                hr_width < self.patch_size
                or
                hr_height < self.patch_size
            ):

                hr = TF.resize(
                    hr,
                    (
                        max(
                            hr_height,
                            self.patch_size
                        ),
                        max(
                            hr_width,
                            self.patch_size
                        )
                    )
                )

                hr_width, hr_height = hr.size


            top = random.randint(
                0,
                hr_height - self.patch_size
            )

            left = random.randint(
                0,
                hr_width - self.patch_size
            )


            hr = TF.crop(
                hr,
                top,
                left,
                self.patch_size,
                self.patch_size
            )


            # Random horizontal flip
            if random.random() > 0.5:

                hr = TF.hflip(hr)


            # Random vertical flip
            if random.random() > 0.5:

                hr = TF.vflip(hr)


        # ----------------------------------------------------
        # Convert HR to tensor
        # ----------------------------------------------------

        hr = TF.to_tensor(hr)


        # ----------------------------------------------------
        # Create LR image
        # ----------------------------------------------------

        if self.train:

            lr_size = (
                self.patch_size // self.scale,
                self.patch_size // self.scale
            )

        else:

            hr_h = hr.shape[1]
            hr_w = hr.shape[2]

            lr_size = (
                hr_h // self.scale,
                hr_w // self.scale
            )

            # Make HR divisible by scale
            hr = hr[
                :lr_size[0] * self.scale,
                :lr_size[1] * self.scale
            ]


        lr = TF.resize(
            hr,
            lr_size,
            interpolation=TF.InterpolationMode.BICUBIC
        )


        return lr, hr


# ============================================================
# Residual Block
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
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=3,
                padding=1
            )

        )


    def forward(self, x):

        residual = self.block(x)

        return x + residual * 0.1


# ============================================================
# Residual Super-Resolution Network
# ============================================================

class SRResNetNAS(nn.Module):

    def __init__(
        self,
        scale=4,
        num_features=64,
        num_blocks=16
    ):

        super().__init__()

        self.scale = scale


        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        self.head = nn.Conv2d(
            3,
            num_features,
            kernel_size=3,
            padding=1
        )


        # ----------------------------------------------------
        # Residual body
        # ----------------------------------------------------

        body = []

        for _ in range(num_blocks):

            body.append(
                ResidualBlock(
                    num_features
                )
            )

        self.body = nn.Sequential(
            *body
        )


        self.body_conv = nn.Conv2d(
            num_features,
            num_features,
            kernel_size=3,
            padding=1
        )


        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        upsampling = []

        if scale in [2, 4, 8]:

            num_upsample = int(
                math.log2(scale)
            )

            for _ in range(num_upsample):

                upsampling.append(
                    nn.Conv2d(
                        num_features,
                        num_features * 4,
                        kernel_size=3,
                        padding=1
                    )
                )

                upsampling.append(
                    nn.PixelShuffle(2)
                )

                upsampling.append(
                    nn.ReLU(
                        inplace=True
                    )
                )

        elif scale == 3:

            upsampling.append(
                nn.Conv2d(
                    num_features,
                    num_features * 9,
                    kernel_size=3,
                    padding=1
                )
            )

            upsampling.append(
                nn.PixelShuffle(3)
            )

            upsampling.append(
                nn.ReLU(
                    inplace=True
                )


        self.upsampling = nn.Sequential(
            *upsampling
        )


        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        self.tail = nn.Conv2d(
            num_features,
            3,
            kernel_size=3,
            padding=1
        )


    def forward(self, x):

        # Bicubic baseline
        bicubic = F.interpolate(
            x,
            scale_factor=self.scale,
            mode="bicubic",
            align_corners=False
        )


        # Feature extraction
        x = self.head(x)

        residual = x


        # Residual body
        x = self.body(x)

        x = self.body_conv(x)

        x = x + residual


        # Upsampling
        x = self.upsampling(x)

        # Reconstruction
        x = self.tail(x)


        # Global residual learning
        x = x + bicubic


        # Keep RGB range
        x = torch.clamp(
            x,
            0.0,
            1.0
        )

        return x


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    sr,
    hr,
    max_val=1.0
):

    mse = F.mse_loss(
        sr,
        hr
    )

    if mse.item() == 0:

        return float("inf")

    psnr = 10 * torch.log10(
        max_val ** 2 / mse
    )

    return psnr.item()


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    img1,
    img2,
    window_size=11
):

    # Simple global SSIM approximation
    # For research-grade experiments, use:
    # torchmetrics / skimage / pytorch-msssim

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = img1.mean(
        dim=(2, 3),
        keepdim=True
    )

    mu2 = img2.mean(
        dim=(2, 3),
        keepdim=True
    )

    sigma1 = (
        (img1 - mu1) ** 2
    ).mean(
        dim=(2, 3),
        keepdim=True
    )

    sigma2 = (
        (img2 - mu2) ** 2
    ).mean(
        dim=(2, 3),
        keepdim=True
    )

    sigma12 = (
        (img1 - mu1)
        *
        (img2 - mu2)
    ).mean(
        dim=(2, 3),
        keepdim=True
    )


    ssim = (

        (2 * mu1 * mu2 + C1)
        *
        (2 * sigma12 + C2)

    ) / (

        (mu1 ** 2 + mu2 ** 2 + C1)
        *
        (sigma1 + sigma2 + C2)

    )


    return ssim.mean().item()


# ============================================================
# Training
# ============================================================

def fit(
    model,
    train_loader,
    optimizer,
    epoch
):

    model.train()

    total_loss = 0.0


    for batch_idx, (
        lr,
        hr
    ) in enumerate(train_loader):

        lr = lr.to(
            device,
            non_blocking=True
        )

        hr = hr.to(
            device,
            non_blocking=True
        )


        optimizer.zero_grad()


        sr = model(lr)


        # L1 loss is commonly used for SR
        loss = F.l1_loss(
            sr,
            hr
        )


        loss.backward()


        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += loss.item()


        if batch_idx % 100 == 0:

            print(
                "Epoch [{}/{}] "
                "Batch [{}/{}] "
                "L1 Loss: {:.6f}".format(

                    epoch,
                    args.num_epoch,

                    batch_idx,
                    len(train_loader),

                    loss.item()
                )
            )


    avg_loss = (
        total_loss
        /
        len(train_loader)
    )


    return avg_loss


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    test_loader
):

    model.eval()


    total_psnr = 0.0
    total_ssim = 0.0
    total_loss = 0.0


    for lr, hr in test_loader:

        lr = lr.to(device)
        hr = hr.to(device)


        sr = model(lr)


        loss = F.l1_loss(
            sr,
            hr
        )


        psnr = calculate_psnr(
            sr,
            hr
        )


        ssim = calculate_ssim(
            sr,
            hr
        )


        total_loss += loss.item()

        total_psnr += psnr

        total_ssim += ssim


    avg_loss = (
        total_loss
        /
        len(test_loader)
    )

    avg_psnr = (
        total_psnr
        /
        len(test_loader)
    )

    avg_ssim = (
        total_ssim
        /
        len(test_loader)
    )


    print(
        "\nValidation Results:"
    )

    print(
        "L1 Loss : {:.6f}".format(
            avg_loss
        )
    )

    print(
        "PSNR    : {:.4f} dB".format(
            avg_psnr
        )
    )

    print(
        "SSIM    : {:.4f}".format(
            avg_ssim
        )
    )


    return (
        avg_psnr,
        avg_ssim
    )


# ============================================================
# Model Complexity
# ============================================================

def count_parameters(
    model
):

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":


    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    trainset = DIV2KDataset(
        root=args.div2k_root,
        scale=args.scale,
        patch_size=args.patch_size,
        train=True
    )


    testset = DIV2KDataset(
        root=args.div2k_root,
        scale=args.scale,
        patch_size=args.patch_size,
        train=False
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
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )


    # --------------------------------------------------------
    # Create SR network
    # --------------------------------------------------------

    torch.manual_seed(0)


    model = SRResNetNAS(

        scale=args.scale,

        num_features=args.num_features,

        num_blocks=args.num_blocks

    ).to(device)


    print(
        "\nModel:"
    )

    print(model)


    # --------------------------------------------------------
    # Count parameters
    # --------------------------------------------------------

    total_params, trainable_params = \
        count_parameters(model)


    print(
        "\nTotal parameters: {:,}".format(
            total_params
        )
    )

    print(
        "Trainable parameters: {:,}".format(
            trainable_params
        )
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=args.lr,

        betas=(0.9, 0.999),

        weight_decay=0

    )


    # --------------------------------------------------------
    # Learning rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.StepLR(

        optimizer,

        step_size=50,

        gamma=0.5

    )


    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        args.save_dir,
        exist_ok=True
    )


    best_psnr = -float("inf")


    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(
        1,
        args.num_epoch + 1
    ):


        print(
            "\n================================="
        )

        print(
            "Epoch {}/{}".format(
                epoch,
                args.num_epoch
            )
        )

        print(
            "================================="
        )


        train_loss = fit(

            model,

            trainloader,

            optimizer,

            epoch

        )


        print(
            "Average Train L1 Loss: {:.6f}".format(
                train_loss
            )
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        psnr, ssim = evaluate(

            model,

            testloader

        )


        scheduler.step()


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if psnr > best_psnr:

            best_psnr = psnr


            checkpoint = {

                "epoch":
                    epoch,

                "model":
                    model.state_dict(),

                "optimizer":
                    optimizer.state_dict(),

                "psnr":
                    psnr,

                "ssim":
                    ssim,

                "scale":
                    args.scale,

                "num_parameters":
                    total_params

            }


            save_path = os.path.join(

                args.save_dir,

                "best_x{}.pth".format(
                    args.scale
                )

            )


            torch.save(

                checkpoint,

                save_path

            )


            print(
                "Best model saved:"
            )

            print(
                save_path
            )


    print(
        "\nTraining completed."
    )

    print(
        "Best PSNR: {:.4f} dB".format(
            best_psnr
        )
    )
