import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms.functional as TF


# ============================================================
# Configuration
# ============================================================

parser = argparse.ArgumentParser(
    description="NSGA-II NAS for Single Image Super-Resolution on DIV2K"
)

parser.add_argument("--data-root", type=str, default="./DIV2K")
parser.add_argument("--scale", type=int, default=4)

parser.add_argument("--patch-size", type=int, default=96)
parser.add_argument("--batch-size", type=int, default=16)

parser.add_argument("--num-epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-4)

parser.add_argument("--num-workers", type=int, default=4)

args = parser.parse_args()


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DIV2K Dataset
# ============================================================

class DIV2KSRDataset(Dataset):

    def __init__(
        self,
        root,
        split="train",
        scale=4,
        patch_size=96
    ):

        self.root = root
        self.split = split
        self.scale = scale
        self.patch_size = patch_size

        if split == "train":

            self.hr_dir = os.path.join(
                root,
                "DIV2K_train_HR"
            )

        else:

            self.hr_dir = os.path.join(
                root,
                "DIV2K_valid_HR"
            )

        self.files = sorted([
            os.path.join(
                self.hr_dir,
                f
            )
            for f in os.listdir(self.hr_dir)
            if f.lower().endswith(
                (".png", ".jpg", ".jpeg")
            )
        ])

    def __len__(self):

        return len(self.files)

    def __getitem__(self, idx):

        hr = Image.open(
            self.files[idx]
        ).convert("RGB")

        hr = TF.to_tensor(hr)

        if self.split == "train":

            _, h, w = hr.shape

            ps = self.patch_size
            scale = self.scale

            if h < ps * scale or w < ps * scale:

                hr = F.interpolate(
                    hr.unsqueeze(0),
                    size=(
                        max(h, ps * scale),
                        max(w, ps * scale)
                    ),
                    mode="bicubic",
                    align_corners=False
                ).squeeze(0)

                _, h, w = hr.shape

            top = random.randint(
                0,
                h - ps * scale
            )

            left = random.randint(
                0,
                w - ps * scale
            )

            hr = hr[
                :,
                top:top + ps * scale,
                left:left + ps * scale
            ]

            # Random augmentation

            if random.random() < 0.5:
                hr = torch.flip(
                    hr,
                    dims=[2]
                )

            if random.random() < 0.5:
                hr = torch.flip(
                    hr,
                    dims=[1]
                )

        else:

            # Validation:
            # crop to dimensions divisible by scale

            _, h, w = hr.shape

            h = h - (h % self.scale)
            w = w - (w % self.scale)

            hr = hr[:, :h, :w]

        # Generate LR image using bicubic degradation

        _, H, W = hr.shape

        lr = F.interpolate(
            hr.unsqueeze(0),
            size=(
                H // self.scale,
                W // self.scale
            ),
            mode="bicubic",
            align_corners=False
        ).squeeze(0)

        return lr, hr


# ============================================================
# SR Operations
# ============================================================

class Identity(nn.Module):

    def forward(self, x):

        return x


class Zero(nn.Module):

    def forward(self, x):

        return torch.zeros_like(x)


class ConvBlock(nn.Module):

    def __init__(
        self,
        channels,
        kernel_size=3,
        dilation=1
    ):

        super().__init__()

        padding = (
            kernel_size // 2
        ) * dilation

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),

            nn.ReLU(
                inplace=True
            )

        )

    def forward(self, x):

        return self.block(x)


class DepthwiseSeparableConv(nn.Module):

    def __init__(
        self,
        channels
    ):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                channels,
                channels,
                3,
                padding=1,
                groups=channels
            ),

            nn.Conv2d(
                channels,
                channels,
                1
            ),

            nn.ReLU(
                inplace=True
            )

        )

    def forward(self, x):

        return self.block(x)


# ============================================================
# Search Space
# ============================================================

OPS = {

    "conv3x3":
        lambda C:
        ConvBlock(
            C,
            3,
            1
        ),

    "conv5x5":
        lambda C:
        ConvBlock(
            C,
            5,
            1
        ),

    "dil_conv3x3":
        lambda C:
        ConvBlock(
            C,
            3,
            2
        ),

    "sep_conv3x3":
        lambda C:
        DepthwiseSeparableConv(
            C
        ),

    "skip":
        lambda C:
        Identity(),

    "none":
        lambda C:
        Zero()
}


# ============================================================
# Level-1 Search Cell
# ============================================================

class SearchCell(nn.Module):

    """
    Level-1 architecture search.

    DAG:

        node0
        / | \
       /  |  \
      v   v   v

    Each edge contains one candidate operation.

    NSGA-II optimizes the selected operations.
    """

    def __init__(
        self,
        channels,
        genotype
    ):

        super().__init__()

        self.channels = channels

        self.genotype = genotype

        self.op_0_1 = OPS[
            genotype[0]
        ](channels)

        self.op_0_2 = OPS[
            genotype[1]
        ](channels)

        self.op_0_3 = OPS[
            genotype[2]
        ](channels)

        self.op_1_2 = OPS[
            genotype[3]
        ](channels)

        self.op_1_3 = OPS[
            genotype[4]
        ](channels)

        self.op_2_3 = OPS[
            genotype[5]
        ](channels)

    def forward(
        self,
        x
    ):

        node0 = x

        node1 = self.op_0_1(
            node0
        )

        node2 = (

            self.op_0_2(
                node0
            )

            +

            self.op_1_2(
                node1
            )

        )

        node3 = (

            self.op_0_3(
                node0
            )

            +

            self.op_1_3(
                node1
            )

            +

            self.op_2_3(
                node2
            )

        )

        return node3


# ============================================================
# Level-2 Macro Architecture
# ============================================================

class MacroSRNetwork(nn.Module):

    """
    Level-2 architecture.

    Searches:

    - number of cells
    - feature channels
    - upsampling scale

    Cell structure is determined by Level-1 genotype.
    """

    def __init__(
        self,
        genotype,
        num_cells=8,
        channels=64,
        scale=4
    ):

        super().__init__()

        self.scale = scale

        self.head = nn.Conv2d(
            3,
            channels,
            3,
            padding=1
        )

        self.cells = nn.ModuleList([

            SearchCell(
                channels,
                genotype
            )

            for _ in range(
                num_cells
            )

        ])

        self.body = nn.Conv2d(
            channels,
            channels,
            3,
            padding=1
        )

        self.upsample = nn.Sequential(

            nn.Conv2d(
                channels,
                channels * scale * scale,
                3,
                padding=1
            ),

            nn.PixelShuffle(
                scale
            ),

            nn.Conv2d(
                channels,
                3,
                3,
                padding=1
            )

        )

    def forward(
        self,
        x
    ):

        shallow = self.head(
            x
        )

        out = shallow

        for cell in self.cells:

            out = cell(
                out
            )

        out = self.body(
            out
        )

        out = out + shallow

        out = self.upsample(
            out
        )

        return out


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    sr,
    hr
):

    sr = torch.clamp(
        sr,
        0,
        1
    )

    mse = F.mse_loss(
        sr,
        hr
    )

    if mse == 0:

        return 100

    return (
        10 *
        torch.log10(
            1.0 / mse
        )
    ).item()


# ============================================================
# Train
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer
):

    model.train()

    total_loss = 0

    criterion = nn.L1Loss()

    for lr, hr in loader:

        lr = lr.to(
            device
        )

        hr = hr.to(
            device
        )

        optimizer.zero_grad()

        sr = model(
            lr
        )

        loss = criterion(
            sr,
            hr
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return (
        total_loss /
        len(loader)
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()

def evaluate(
    model,
    loader
):

    model.eval()

    psnr_sum = 0

    for lr, hr in loader:

        lr = lr.to(
            device
        )

        hr = hr.to(
            device
        )

        sr = model(
            lr
        )

        psnr = calculate_psnr(
            sr,
            hr
        )

        psnr_sum += psnr

    return (
        psnr_sum /
        len(loader)
    )


# ============================================================
# Model Complexity
# ============================================================

def count_parameters(
    model
):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad

    )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    train_dataset = DIV2KSRDataset(

        root=args.data_root,

        split="train",

        scale=args.scale,

        patch_size=args.patch_size

    )

    valid_dataset = DIV2KSRDataset(

        root=args.data_root,

        split="valid",

        scale=args.scale,

        patch_size=args.patch_size

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=True

    )

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=1,

        shuffle=False,

        num_workers=args.num_workers

    )


    # Example Level-1 genotype

    genotype = [

        "conv3x3",

        "sep_conv3x3",

        "conv5x5",

        "skip",

        "dil_conv3x3",

        "conv3x3"

    ]


    # Example Level-2 architecture

    model = MacroSRNetwork(

        genotype=genotype,

        num_cells=8,

        channels=64,

        scale=args.scale

    ).to(device)


    print(model)

    print(

        "Number of parameters:",

        count_parameters(
            model
        )

    )


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=args.lr

    )


    best_psnr = 0


    for epoch in range(

        args.num_epochs

    ):

        loss = train_one_epoch(

            model,

            train_loader,

            optimizer

        )

        psnr = evaluate(

            model,

            valid_loader

        )

        print(

            f"Epoch {epoch+1}: "

            f"Loss={loss:.6f}, "

            f"PSNR={psnr:.4f} dB"

        )

        if psnr > best_psnr:

            best_psnr = psnr

            torch.save(

                {

                    "model":

                    model.state_dict(),

                    "genotype":

                    genotype,

                    "psnr":

                    psnr

                },

                "best_nsga2_nas_sr.pth"

            )
