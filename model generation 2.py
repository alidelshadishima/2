```python
# ============================================================
# fisher_guided_cell_sr.py
#
# Fisher-Guided NAS for Single Image Super-Resolution
#
# Dataset:
#       DIV2K
#
# Task:
#       Single Image Super-Resolution (SISR)
#
# Scale:
#       x4
#
# Objectives:
#       - Minimize Reconstruction Loss
#       - Maximize PSNR
#       - Maximize SSIM
#
# Fisher Information is used to:
#       1. Rank candidate operations
#       2. Select top-k operations
#       3. Build a Fisher-guided cell
#
# ============================================================


import os
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from PIL import Image


# ============================================================
# Device
# ============================================================

device = torch.device(

    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser(

    description=
    "Fisher-Guided NAS for DIV2K Super-Resolution"
)


parser.add_argument(

    "--train-dir",

    type=str,

    default=
    "./data/DIV2K_train_HR",

    help=
    "DIV2K HR training directory"
)


parser.add_argument(

    "--valid-dir",

    type=str,

    default=
    "./data/DIV2K_valid_HR",

    help=
    "DIV2K HR validation directory"
)


parser.add_argument(

    "--scale",

    type=int,

    default=4,

    choices=[2, 3, 4],

    help=
    "Super-resolution scale factor"
)


parser.add_argument(

    "--patch-size",

    type=int,

    default=48,

    help=
    "LR patch size"
)


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


parser.add_argument(

    "--num-epoch",

    type=int,

    default=100
)


parser.add_argument(

    "--lr",

    type=float,

    default=1e-4
)


parser.add_argument(

    "--calib-batches",

    type=int,

    default=5,

    help=
    "Number of calibration batches for Fisher scoring"
)


parser.add_argument(

    "--topk",

    type=int,

    default=2,

    help=
    "Number of Fisher-selected operations per edge"
)


parser.add_argument(

    "--num-workers",

    type=int,

    default=4
)


args = parser.parse_args()


# ============================================================
# DIV2K Dataset
# ============================================================

class DIV2KSRDataset(
    torch.utils.data.Dataset
):


    def __init__(

        self,

        hr_dir,

        scale=4,

        patch_size=48,

        training=True

    ):

        self.hr_dir = hr_dir

        self.scale = scale

        self.patch_size = patch_size

        self.hr_patch_size = (

            patch_size
            *
            scale
        )

        self.training = training


        extensions = (

            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"
        )


        self.files = [

            os.path.join(

                hr_dir,

                f
            )

            for f in sorted(

                os.listdir(
                    hr_dir
                )

            )

            if f.lower().endswith(
                extensions
            )
        ]


        if len(self.files) == 0:

            raise RuntimeError(

                "No DIV2K images found in "
                +
                hr_dir
            )


    def __len__(self):

        return len(
            self.files
        )


    def __getitem__(

        self,

        index

    ):

        image = Image.open(

            self.files[
                index
            ]

        ).convert(
            "RGB"
        )


        hr = (

            torch
            .from_numpy(

                np.array(
                    image
                )
            )

            .permute(
                2,
                0,
                1
            )

            .float()

            / 255.0
        )


        if self.training:

            lr, hr = (

                self.random_crop(
                    hr
                )
            )

        else:

            lr, hr = (

                self.validation_pair(
                    hr
                )
            )


        return lr, hr


    def random_crop(

        self,

        hr

    ):

        _, h, w = hr.shape


        if (

            h
            <
            self.hr_patch_size

            or

            w
            <
            self.hr_patch_size

        ):

            hr = F.interpolate(

                hr.unsqueeze(0),

                size=(

                    max(
                        h,
                        self.hr_patch_size
                    ),

                    max(
                        w,
                        self.hr_patch_size
                    )

                ),

                mode="bicubic",

                align_corners=False

            ).squeeze(0)


            _, h, w = hr.shape


        top = np.random.randint(

            0,

            h
            -
            self.hr_patch_size
            +
            1
        )


        left = np.random.randint(

            0,

            w
            -
            self.hr_patch_size
            +
            1
        )


        hr = hr[

            :,

            top:
            top
            +
            self.hr_patch_size,

            left:
            left
            +
            self.hr_patch_size
        ]


        lr = F.interpolate(

            hr.unsqueeze(0),

            size=(

                self.patch_size,

                self.patch_size
            ),

            mode="bicubic",

            align_corners=False

        ).squeeze(0)


        # Random horizontal flip

        if np.random.rand() < 0.5:

            lr = torch.flip(

                lr,

                [2]
            )

            hr = torch.flip(

                hr,

                [2]
            )


        # Random vertical flip

        if np.random.rand() < 0.5:

            lr = torch.flip(

                lr,

                [1]
            )

            hr = torch.flip(

                hr,

                [1]
            )


        return lr, hr


    def validation_pair(

        self,

        hr

    ):

        _, h, w = hr.shape


        h = h - h % self.scale

        w = w - w % self.scale


        hr = hr[

            :,

            :h,

            :w
        ]


        lr = F.interpolate(

            hr.unsqueeze(0),

            size=(

                h // self.scale,

                w // self.scale
            ),

            mode="bicubic",

            align_corners=False

        ).squeeze(0)


        return lr, hr


# ============================================================
# Build Data Loaders
# ============================================================

def build_dataloaders():

    train_dataset = DIV2KSRDataset(

        args.train_dir,

        scale=args.scale,

        patch_size=args.patch_size,

        training=True
    )


    valid_dataset = DIV2KSRDataset(

        args.valid_dir,

        scale=args.scale,

        patch_size=args.patch_size,

        training=False
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=args.batch_size_train,

        shuffle=True,

        num_workers=args.num_workers,

        pin_memory=True
    )


    valid_loader = DataLoader(

        valid_dataset,

        batch_size=args.batch_size_test,

        shuffle=False,

        num_workers=args.num_workers,

        pin_memory=True
    )


    return (

        train_loader,

        valid_loader
    )


# ============================================================
# Candidate SR Operations
# ============================================================

class Conv3x3(nn.Module):

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

                1,

                1
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class Conv5x5(nn.Module):

    def __init__(

        self,

        channels

    ):

        super().__init__()


        self.block = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                5,

                1,

                2
            ),

            nn.ReLU(
                inplace=True
            )
        )


    def forward(self, x):

        return self.block(x)


class DilatedConv3x3(nn.Module):

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

                1,

                2,

                dilation=2
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

                1,

                1,

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


class ResidualBlock(nn.Module):

    def __init__(

        self,

        channels

    ):

        super().__init__()


        self.conv1 = nn.Conv2d(

            channels,

            channels,

            3,

            1,

            1
        )


        self.conv2 = nn.Conv2d(

            channels,

            channels,

            3,

            1,

            1
        )


    def forward(self, x):

        residual = x


        x = F.relu(

            self.conv1(x),

            inplace=True
        )


        x = self.conv2(x)


        return (

            x
            +
            residual
        )


class SkipConnection(nn.Module):

    def __init__(

        self,

        channels

    ):

        super().__init__()


    def forward(self, x):

        return x


# ============================================================
# Operation Dictionary
# ============================================================

SR_OPS = {

    "conv3x3":
    Conv3x3,

    "conv5x5":
    Conv5x5,

    "dilated_conv3x3":
    DilatedConv3x3,

    "depthwise_separable":
    DepthwiseSeparableConv,

    "residual_block":
    ResidualBlock,

    "skip_connection":
    SkipConnection

}


# ============================================================
# Fisher Information Utilities
# ============================================================

def zero_like_params(

    module

):

    fisher = {}


    for name, param in module.named_parameters():

        if param.requires_grad:

            fisher[name] = torch.zeros_like(

                param,

                device=param.device
            )


    return fisher


# ============================================================
# Fisher Score for SR Operation
# ============================================================

def fisher_score_for_op(

    op_module,

    calib_loader,

    max_batches

):


    op_module = op_module.to(
        device
    )


    op_module.train()


    # --------------------------------------------------------
    # Reconstruction Head
    # --------------------------------------------------------

    head = nn.Sequential(

        nn.Conv2d(

            64,

            64,

            3,

            1,

            1
        ),

        nn.ReLU(
            inplace=True
        ),

        nn.Conv2d(

            64,

            3,

            3,

            1,

            1
        )
    ).to(
        device
    )


    criterion = nn.L1Loss()


    fisher_sums = (

        zero_like_params(
            op_module
        )
    )


    batches = 0


    for lr, hr in calib_loader:

        lr = lr.to(
            device
        )


        hr = hr.to(
            device
        )


        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = op_module(

            lr
        )


        # ----------------------------------------------------
        # Convert 3-channel input to 64-channel representation
        # if necessary
        # ----------------------------------------------------

        if features.shape[1] != 64:

            adapter = nn.Conv2d(

                features.shape[1],

                64,

                1
            ).to(
                device
            )


            features = adapter(

                features
            )


        # ----------------------------------------------------
        # Upsampling
        # ----------------------------------------------------

        features = F.interpolate(

            features,

            size=hr.shape[-2:],

            mode="bilinear",

            align_corners=False
        )


        sr = head(

            features
        )


        loss = criterion(

            sr,

            hr
        )


        op_module.zero_grad()

        head.zero_grad()


        loss.backward()


        with torch.no_grad():

            for name, param in op_module.named_parameters():

                if (

                    param.grad
                    is not None

                ):

                    fisher_sums[name] += (

                        param.grad.detach()
                        **
                        2
                    )


        batches += 1


        if batches >= max_batches:

            break


    # --------------------------------------------------------
    # Aggregate Fisher Information
    # --------------------------------------------------------

    score = 0.0


    with torch.no_grad():

        for name, value in fisher_sums.items():

            value /= max(

                1,

                batches
            )


            score += (

                value.mean().item()
            )


    return score


# ============================================================
# Fisher-Weighted Combined Operation
# ============================================================

class CombinedOp(

    nn.Module

):


    def __init__(

        self,

        channels,

        op_specs

    ):

        super().__init__()


        self.branches = nn.ModuleList(

            [

                SR_OPS[name](

                    channels
                )

                for name, _ in op_specs

            ]
        )


        scores = torch.tensor(

            [

                score

                for _, score

                in op_specs

            ],

            dtype=torch.float32
        )


        self.register_buffer(

            "weights",

            torch.softmax(

                scores,

                dim=0
            )
        )


    def forward(

        self,

        x

    ):

        outputs = [

            branch(x)

            for branch

            in self.branches

        ]


        output = 0.0


        for weight, value in zip(

            self.weights,

            outputs

        ):

            output = (

                output
                +
                weight
                *
                value
            )


        return output


# ============================================================
# Fisher-Guided SR Cell
# ============================================================

class FisherGuidedSRCell(

    nn.Module

):


    def __init__(

        self,

        channels,

        calib_loader,

        topk=2,

        calib_batches=5,

        scale=4

    ):

        super().__init__()


        self.channels = channels

        self.scale = scale


        # ----------------------------------------------------
        # Rank operations
        # ----------------------------------------------------

        scores = []


        print(

            "\nCalculating Fisher scores..."
        )


        for name, op_class in SR_OPS.items():


            print(

                "Evaluating:",

                name
            )


            op = nn.Sequential(

                nn.Conv2d(

                    3,

                    channels,

                    3,

                    1,

                    1
                ),

                op_class(

                    channels
                )
            )


            score = fisher_score_for_op(

                op,

                calib_loader,

                calib_batches
            )


            scores.append(

                (

                    name,

                    score
                )
            )


            print(

                name,

                "Fisher Score:",

                score
            )


        # ----------------------------------------------------
        # Sort operations
        # ----------------------------------------------------

        scores.sort(

            key=lambda x: x[1],

            reverse=True
        )


        top_ops = (

            scores[
                :topk
            ]
        )


        print(

            "\nSelected Fisher Operations:"
        )


        for name, score in top_ops:

            print(

                name,

                score
            )


        # ----------------------------------------------------
        # Stem
        # ----------------------------------------------------

        self.stem = nn.Conv2d(

            3,

            channels,

            3,

            1,

            1
        )


        # ----------------------------------------------------
        # Fisher-Guided Cell
        # ----------------------------------------------------

        self.cell = CombinedOp(

            channels,

            top_ops
        )


        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        self.reconstruction = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                3,

                1,

                1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(

                channels,

                3
                *
                scale
                *
                scale,

                3,

                1,

                1
            ),

            nn.PixelShuffle(

                scale
            )
        )


    def forward(

        self,

        x

    ):

        # ----------------------------------------------------
        # Bicubic baseline
        # ----------------------------------------------------

        bicubic = F.interpolate(

            x,

            scale_factor=self.scale,

            mode="bicubic",

            align_corners=False
        )


        # ----------------------------------------------------
        # Feature extraction
        # ----------------------------------------------------

        features = self.stem(

            x
        )


        # ----------------------------------------------------
        # Fisher-guided operations
        # ----------------------------------------------------

        features = self.cell(

            features
        )


        # ----------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------

        residual = self.reconstruction(

            features
        )


        # ----------------------------------------------------
        # Global residual learning
        # ----------------------------------------------------

        return (

            bicubic
            +
            residual
        )


# ============================================================
# Charbonnier Loss
# ============================================================

class CharbonnierLoss(

    nn.Module

):


    def __init__(

        self,

        eps=1e-3

    ):

        super().__init__()


        self.eps = eps


    def forward(

        self,

        prediction,

        target

    ):

        diff = (

            prediction
            -
            target
        )


        loss = torch.sqrt(

            diff
            *
            diff
            +
            self.eps
            *
            self.eps
        )


        return loss.mean()


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

        return 100.0


    return (

        10
        *
        torch.log10(

            1.0
            /
            mse
        )
    ).item()


# ============================================================
# Training
# ============================================================

def fit(

    model,

    train_loader,

    epochs,

    lr

):


    model.train()


    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=lr
    )


    criterion = CharbonnierLoss()


    for epoch in range(

        epochs
    ):


        epoch_loss = 0.0


        for lr_image, hr_image in train_loader:


            lr_image = lr_image.to(

                device
            )


            hr_image = hr_image.to(

                device
            )


            optimizer.zero_grad()


            sr_image = model(

                lr_image
            )


            loss = criterion(

                sr_image,

                hr_image
            )


            loss.backward()


            optimizer.step()


            epoch_loss += (

                loss.item()
            )


        epoch_loss /= len(

            train_loader
        )


        print(

            "Epoch",

            epoch + 1,

            "/",

            epochs,

            "Loss:",

            epoch_loss
        )


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


    count = 0


    for lr_image, hr_image in test_loader:


        lr_image = lr_image.to(

            device
        )


        hr_image = hr_image.to(

            device
        )


        sr_image = model(

            lr_image
        )


        psnr = calculate_psnr(

            sr_image,

            hr_image
        )


        total_psnr += psnr


        count += 1


    mean_psnr = (

        total_psnr
        /
        max(
            1,
            count
        )
    )


    print(

        "\nValidation PSNR:",

        mean_psnr,

        "dB"
    )


    return mean_psnr


# ============================================================
# Main
# ============================================================

def main():


    # --------------------------------------------------------
    # Load DIV2K
    # --------------------------------------------------------

    print(

        "Loading DIV2K..."
    )


    train_loader, test_loader = (

        build_dataloaders()
    )


    print(

        "Training samples:",

        len(
            train_loader.dataset
        )
    )


    print(

        "Validation samples:",

        len(
            test_loader.dataset
        )
    )


    # --------------------------------------------------------
    # Build Fisher-Guided Cell
    # --------------------------------------------------------

    print(

        "\nBuilding Fisher-Guided SR Cell..."
    )


    model = FisherGuidedSRCell(

        channels=64,

        calib_loader=train_loader,

        topk=args.topk,

        calib_batches=args.calib_batches,

        scale=args.scale

    ).to(

        device
    )


    print(

        "\nFinal Fisher-Guided Model:"
    )


    print(

        model
    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    fit(

        model,

        train_loader,

        epochs=args.num_epoch,

        lr=args.lr
    )


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate(

        model,

        test_loader
    )


    # --------------------------------------------------------
    # Save Model
    # --------------------------------------------------------

    os.makedirs(

        "./checkpoints",

        exist_ok=True
    )


    torch.save(

        {

            "model_state_dict":
            model.state_dict(),

            "scale":
            args.scale,

            "topk":
            args.topk

        },

        "./checkpoints/fisher_guided_div2k_sr.pth"
    )


    print(

        "\nModel saved successfully."
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":

    main()
```
