"""
This code is used to train the default Super-Resolution network
for the baseline/incoming tasks in NAS-SR.

Task:
    DIV2K Single Image Super-Resolution (SISR)

Input:
    Low-Resolution (LR) image

Target:
    High-Resolution (HR) image

The original classification pipeline:
    Image -> CNN -> Class label

is replaced by:

    LR Image -> SR Network -> Super-Resolved Image
                         |
                         +----> Compare with HR Image

Loss:
    L1 Loss

Evaluation:
    PSNR
    SSIM


"""

import os
import argparse
import random
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
from torchvision.transforms import functional as TF


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# Parser
# ============================================================

parser = argparse.ArgumentParser(
    description='DIV2K Super-Resolution Training'
)

parser.add_argument(
    '--div2k-root',
    default='./DIV2K',
    type=str,
    help='Path to DIV2K dataset'
)

parser.add_argument(
    '--scale',
    default=4,
    type=int,
    choices=[2, 3, 4, 8],
    help='Super-resolution scale factor'
)

parser.add_argument(
    '--lr',
    default=1e-4,
    type=float,
    help='Learning rate'
)

parser.add_argument(
    '--batch-size-train',
    default=16,
    type=int,
    help='Training batch size'
)

parser.add_argument(
    '--batch-size-test',
    default=1,
    type=int,
    help='Testing batch size'
)

parser.add_argument(
    '--num-epoch',
    default=100,
    type=int,
    help='Number of training epochs'
)

parser.add_argument(
    '--patch-size',
    default=96,
    type=int,
    help='HR training patch size'
)

parser.add_argument(
    '--num-features',
    default=64,
    type=int,
    help='Number of feature channels'
)

parser.add_argument(
    '--num-blocks',
    default=16,
    type=int,
    help='Number of residual blocks'
)

parser.add_argument(
    '--num-workers',
    default=4,
    type=int,
    help='Number of dataloader workers'
)

parser.add_argument(
    '--save-dir',
    default='./checkpoint_sr',
    type=str,
    help='Checkpoint directory'
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
# Create checkpoint directory
# ============================================================

if not os.path.exists(args.save_dir):

    os.makedirs(
        args.save_dir
    )


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


        # ----------------------------------------------------
        # Select train / validation directory
        # ----------------------------------------------------

        if train:

            self.hr_dir = os.path.join(
                root,
                'DIV2K_train_HR'
            )

        else:

            self.hr_dir = os.path.join(
                root,
                'DIV2K_valid_HR'
            )


        if not os.path.exists(
            self.hr_dir
        ):

            raise FileNotFoundError(
                "DIV2K directory not found:\n{}".format(
                    self.hr_dir
                )
            )


        # ----------------------------------------------------
        # Load image names
        # ----------------------------------------------------

        self.images = sorted([

            f for f in os.listdir(
                self.hr_dir
            )

            if f.lower().endswith(
                (
                    '.png',
                    '.jpg',
                    '.jpeg'
                )
            )

        ])


        if len(self.images) == 0:

            raise RuntimeError(
                "No images found in {}".format(
                    self.hr_dir
                )
            )


        print(
            "Loaded {} images from {}".format(
                len(self.images),
                self.hr_dir
            )
        )


    def __len__(
        self
    ):

        return len(
            self.images
        )


    def __getitem__(
        self,
        index
    ):

        # ----------------------------------------------------
        # Load HR image
        # ----------------------------------------------------

        image_name = self.images[index]

        hr_path = os.path.join(
            self.hr_dir,
            image_name
        )

        hr = Image.open(
            hr_path
        ).convert('RGB')


        # ====================================================
        # Training
        # ====================================================

        if self.train:

            hr_width, hr_height = hr.size


            # ------------------------------------------------
            # Random crop
            # ------------------------------------------------

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

                hr_height
                -
                self.patch_size

            )


            left = random.randint(

                0,

                hr_width
                -
                self.patch_size

            )


            hr = TF.crop(

                hr,

                top,

                left,

                self.patch_size,

                self.patch_size

            )


            # ------------------------------------------------
            # Data augmentation
            # ------------------------------------------------

            if random.random() > 0.5:

                hr = TF.hflip(
                    hr
                )


            if random.random() > 0.5:

                hr = TF.vflip(
                    hr
                )


        # ----------------------------------------------------
        # Convert HR image to tensor
        # ----------------------------------------------------

        hr = TF.to_tensor(
            hr
        )


        # ====================================================
        # Generate LR image
        # ====================================================

        if self.train:

            lr_height = (
                self.patch_size
                //
                self.scale
            )

            lr_width = (
                self.patch_size
                //
                self.scale
            )

        else:

            hr_height = hr.shape[1]

            hr_width = hr.shape[2]


            # Ensure divisibility by scale
            hr_height = (
                hr_height
                //
                self.scale
            ) * self.scale


            hr_width = (
                hr_width
                //
                self.scale
            ) * self.scale


            hr = hr[
                :hr_height,
                :hr_width
            ]


            lr_height = (
                hr_height
                //
                self.scale
            )

            lr_width = (
                hr_width
                //
                self.scale
            )


        # ----------------------------------------------------
        # Bicubic downsampling
        # ----------------------------------------------------

        lr = TF.resize(

            hr,

            (
                lr_height,
                lr_width
            ),

            interpolation=
            TF.InterpolationMode.BICUBIC

        )


        return lr, hr


# ============================================================
# Residual Block
# ============================================================

class ResidualBlock(
    nn.Module
):

    def __init__(
        self,
        channels=64
    ):

        super(
            ResidualBlock,
            self
        ).__init__()


        self.conv1 = nn.Conv2d(

            channels,

            channels,

            kernel_size=3,

            stride=1,

            padding=1

        )


        self.relu = nn.ReLU(

            inplace=True

        )


        self.conv2 = nn.Conv2d(

            channels,

            channels,

            kernel_size=3,

            stride=1,

            padding=1

        )


    def forward(
        self,
        x
    ):

        residual = x


        out = self.conv1(
            x
        )


        out = self.relu(
            out
        )


        out = self.conv2(
            out
        )


        # Residual connection
        out = out + residual


        return out


# ============================================================
# Super-Resolution Network
# ============================================================

class SRNetwork(
    nn.Module
):

    def __init__(
        self,
        scale=4,
        num_features=64,
        num_blocks=16
    ):

        super(
            SRNetwork,
            self
        ).__init__()


        self.scale = scale


        # ----------------------------------------------------
        # Initial feature extraction
        # ----------------------------------------------------

        self.head = nn.Conv2d(

            3,

            num_features,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ----------------------------------------------------
        # Residual body
        # ----------------------------------------------------

        residual_blocks = []


        for _ in range(
            num_blocks
        ):

            residual_blocks.append(

                ResidualBlock(

                    channels=num_features

                )

            )


        self.body = nn.Sequential(

            *residual_blocks

        )


        self.body_conv = nn.Conv2d(

            num_features,

            num_features,

            kernel_size=3,

            stride=1,

            padding=1

        )


        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        upsampling_layers = []


        if scale in [2, 4, 8]:

            num_upsampling = int(

                math.log2(
                    scale
                )

            )


            for _ in range(
                num_upsampling
            ):


                upsampling_layers.append(

                    nn.Conv2d(

                        num_features,

                        num_features * 4,

                        kernel_size=3,

                        stride=1,

                        padding=1

                    )

                )


                upsampling_layers.append(

                    nn.PixelShuffle(
                        2
                    )

                )


                upsampling_layers.append(

                    nn.ReLU(
                        inplace=True
                    )

                )


        elif scale == 3:


            upsampling_layers.append(

                nn.Conv2d(

                    num_features,

                    num_features * 9,

                    kernel_size=3,

                    stride=1,

                    padding=1

                )

            )


            upsampling_layers.append(

                nn.PixelShuffle(
                    3
                )

            )


            upsampling_layers.append(

                nn.ReLU(
                    inplace=True
                )

            )


        self.upsampling = nn.Sequential(

            *upsampling_layers

        )


        # ----------------------------------------------------
        # Reconstruction layer
        # ----------------------------------------------------

        self.tail = nn.Conv2d(

            num_features,

            3,

            kernel_size=3,

            stride=1,

            padding=1

        )


    def forward(
        self,
        x
    ):

        # ----------------------------------------------------
        # Global residual learning
        # ----------------------------------------------------

        bicubic = F.interpolate(

            x,

            scale_factor=self.scale,

            mode='bicubic',

            align_corners=False

        )


        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        x = self.head(
            x
        )


        residual = x


        # ----------------------------------------------------
        # Residual blocks
        # ----------------------------------------------------

        x = self.body(
            x
        )


        x = self.body_conv(
            x
        )


        # Global feature residual
        x = x + residual


        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        x = self.upsampling(
            x
        )


        # ----------------------------------------------------
        # Image reconstruction
        # ----------------------------------------------------

        x = self.tail(
            x
        )


        # ----------------------------------------------------
        # Global image residual
        # ----------------------------------------------------

        x = x + bicubic


        # ----------------------------------------------------
        # Keep image range [0,1]
        # ----------------------------------------------------

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
    hr
):

    mse = F.mse_loss(

        sr,

        hr

    )


    if mse.item() == 0:

        return float(
            'inf'
        )


    psnr = (

        10.0

        *

        torch.log10(

            1.0

            /

            mse

        )

    )


    return psnr.item()


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
    img1,
    img2
):

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
# Train the Super-Resolution Network
# ============================================================

def fit(
    model,
    train_loader
):

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=args.lr,

        betas=(0.9, 0.999)

    )


    # --------------------------------------------------------
    # L1 loss
    # --------------------------------------------------------

    error = nn.L1Loss()


    EPOCHS = args.num_epoch


    model.train()


    for epoch in range(
        EPOCHS
    ):


        total_loss = 0.0


        for batch_idx, (
            lr_images,
            hr_images
        ) in enumerate(
            train_loader
        ):


            lr_images = lr_images.to(

                device,

                non_blocking=True

            )


            hr_images = hr_images.to(

                device,

                non_blocking=True

            )


            # ------------------------------------------------
            # Clear gradients
            # ------------------------------------------------

            optimizer.zero_grad()


            # ------------------------------------------------
            # Forward
            # ------------------------------------------------

            sr_images = model(

                lr_images

            )


            # ------------------------------------------------
            # Reconstruction loss
            # ------------------------------------------------

            loss = error(

                sr_images,

                hr_images

            )


            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            loss.backward()


            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0

            )


            optimizer.step()


            total_loss += loss.item()


            if batch_idx % 50 == 0:

                print(

                    'Epoch : {} ({:.0f}%) '
                    'L1 Loss: {:.6f}'.format(

                        epoch + 1,

                        100.0
                        *
                        batch_idx
                        /
                        len(train_loader),

                        loss.item()

                    )

                )


        avg_loss = (

            total_loss

            /

            len(train_loader)

        )


        print(

            'Epoch {} Average L1 Loss: {:.6f}'.format(

                epoch + 1,

                avg_loss

            )

        )


# ============================================================
# Evaluate Super-Resolution Network
# ============================================================

best_psnr = -float(
    'inf'
)


@torch.no_grad()
def evaluate(
    model,
    test_loader,
    save_flag=True,
    index=101
):

    global best_psnr


    model.eval()


    total_psnr = 0.0

    total_ssim = 0.0

    total_loss = 0.0


    for (
        lr_images,
        hr_images
    ) in test_loader:


        lr_images = lr_images.to(
            device
        )


        hr_images = hr_images.to(
            device
        )


        # ----------------------------------------------------
        # Generate SR image
        # ----------------------------------------------------

        sr_images = model(

            lr_images

        )


        # ----------------------------------------------------
        # Calculate reconstruction loss
        # ----------------------------------------------------

        loss = F.l1_loss(

            sr_images,

            hr_images

        )


        # ----------------------------------------------------
        # Calculate PSNR
        # ----------------------------------------------------

        psnr = calculate_psnr(

            sr_images,

            hr_images

        )


        # ----------------------------------------------------
        # Calculate SSIM
        # ----------------------------------------------------

        ssim = calculate_ssim(

            sr_images,

            hr_images

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

        '\nSuper-Resolution Evaluation:'

    )


    print(

        'L1 Loss: {:.6f}'.format(

            avg_loss

        )

    )


    print(

        'PSNR: {:.4f} dB'.format(

            avg_psnr

        )

    )


    print(

        'SSIM: {:.4f}'.format(

            avg_ssim

        )

    )


    # --------------------------------------------------------
    # Save best network
    # --------------------------------------------------------

    if save_flag and avg_psnr > best_psnr:


        best_psnr = avg_psnr


        print(
            'Saving best Super-Resolution network...'
        )


        state = {

            'net':
                model.state_dict(),

            'psnr':
                avg_psnr,

            'ssim':
                avg_ssim,

            'l1_loss':
                avg_loss,

            'scale':
                args.scale,

            'epoch':
                args.num_epoch

        }


        save_path = os.path.join(

            args.save_dir,

            'duty{}_x{}.pth'.format(

                index,

                args.scale

            )

        )


        torch.save(

            state,

            save_path

        )


        print(

            'Model saved to:',

            save_path

        )


    return (

        avg_psnr,

        avg_ssim

    )


# ============================================================
# Main Code
# ============================================================

if __name__ == "__main__":


    # ========================================================
    # Super-Resolution Task ID
    # ========================================================

    dutyID = 101


    # ========================================================
    # Dataset
    # ========================================================

    dataset = 'DIV2K'


    print(
        '\nLoading DIV2K Super-Resolution dataset...'
    )


    print(

        'Scale factor: x{}'.format(

            args.scale

        )

    )


    # ========================================================
    # Training Dataset
    # ========================================================

    trainset = DIV2KDataset(

        root=args.div2k_root,

        scale=args.scale,

        patch_size=args.patch_size,

        train=True

    )


    # ========================================================
    # Validation Dataset
    # ========================================================

    testset = DIV2KDataset(

        root=args.div2k_root,

        scale=args.scale,

        patch_size=args.patch_size,

        train=False

    )


    # ========================================================
    # DataLoaders
    # ========================================================

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


    # ========================================================
    # Initialize Super-Resolution Network
    # ========================================================

    torch.manual_seed(0)


    sr_net = SRNetwork(

        scale=args.scale,

        num_features=args.num_features,

        num_blocks=args.num_blocks

    ).to(device)


    print(

        '\nSuper-Resolution Network:'

    )


    print(
        sr_net
    )


    # ========================================================
    # Number of parameters
    # ========================================================

    total_params = sum(

        p.numel()

        for p in sr_net.parameters()

    )


    trainable_params = sum(

        p.numel()

        for p in sr_net.parameters()

        if p.requires_grad

    )


    print(

        '\nTotal parameters: {:,}'.format(

            total_params

        )

    )


    print(

        'Trainable parameters: {:,}'.format(

            trainable_params

        )

    )


    # ========================================================
    # Train
    # ========================================================

    fit(

        sr_net,

        trainloader

    )


    # ========================================================
    # Evaluate
    # ========================================================

    evaluate(

        sr_net,

        testloader,

        save_flag=True,

        index=dutyID

    )
