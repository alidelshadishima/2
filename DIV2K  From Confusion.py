```python
import os
import argparse
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image


# ============================================================
# DIV2K Single Image Super-Resolution Dataset
#
# Dataset structure:
#
# data/
# ├── DIV2K_train_HR/
# │   ├── 0001.png
# │   ├── 0002.png
# │   └── ...
# │
# └── DIV2K_valid_HR/
#     ├── 0801.png
#     ├── 0802.png
#     └── ...
#
# This loader:
# 1. Loads HR images
# 2. Generates LR images using bicubic degradation
# 3. Extracts LR/HR patches
# 4. Returns (LR, HR)
#
# No classification labels are used.
# ============================================================


# ============================================================
# DIV2K SR Dataset
# ============================================================

class DIV2KSRDataset(Dataset):

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


        # ----------------------------------------------------
        # Find HR images
        # ----------------------------------------------------

        valid_extensions = (

            ".png",
            ".jpg",
            ".jpeg",
            ".bmp"
        )


        self.image_files = [

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
                valid_extensions
            )
        ]


        if len(
            self.image_files
        ) == 0:

            raise RuntimeError(

                "No images found in: "
                +
                str(
                    hr_dir
                )
            )


    # ========================================================
    # Dataset Length
    # ========================================================

    def __len__(self):

        return len(
            self.image_files
        )


    # ========================================================
    # Get Image
    # ========================================================

    def __getitem__(
        self,
        index
    ):

        image_path = (

            self.image_files[
                index
            ]
        )


        # ----------------------------------------------------
        # Load HR image
        # ----------------------------------------------------

        hr_image = Image.open(

            image_path
        ).convert(
            "RGB"
        )


        # ----------------------------------------------------
        # Convert image to Tensor
        # ----------------------------------------------------

        hr = (

            torch
            .from_numpy(

                np.array(
                    hr_image
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


        # ----------------------------------------------------
        # Training Mode
        # ----------------------------------------------------

        if self.training:

            lr, hr = (

                self.random_crop(
                    hr
                )
            )


        # ----------------------------------------------------
        # Validation Mode
        # ----------------------------------------------------

        else:

            lr, hr = (

                self.prepare_validation_image(
                    hr
                )
            )


        return lr, hr


    # ========================================================
    # Random Crop for Training
    # ========================================================

    def random_crop(
        self,
        hr
    ):

        _, h, w = hr.shape


        # ----------------------------------------------------
        # Ensure HR image is large enough
        # ----------------------------------------------------

        if (

            h
            <
            self.hr_patch_size

            or

            w
            <
            self.hr_patch_size

        ):

            new_h = max(

                h,

                self.hr_patch_size
            )

            new_w = max(

                w,

                self.hr_patch_size
            )


            hr = (

                torch
                .nn
                .functional
                .interpolate(

                    hr.unsqueeze(
                        0
                    ),

                    size=(

                        new_h,

                        new_w
                    ),

                    mode="bicubic",

                    align_corners=False
                )

                .squeeze(
                    0
                )
            )


            _, h, w = hr.shape


        # ----------------------------------------------------
        # Random HR crop
        # ----------------------------------------------------

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


        hr_patch = (

            hr[

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
        )


        # ----------------------------------------------------
        # Generate LR image
        # ----------------------------------------------------

        lr_patch = (

            torch
            .nn
            .functional
            .interpolate(

                hr_patch.unsqueeze(
                    0
                ),

                size=(

                    self.patch_size,

                    self.patch_size
                ),

                mode="bicubic",

                align_corners=False
            )

            .squeeze(
                0
            )
        )


        # ----------------------------------------------------
        # Random Horizontal Flip
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            lr_patch = torch.flip(

                lr_patch,

                dims=[2]
            )


            hr_patch = torch.flip(

                hr_patch,

                dims=[2]
            )


        # ----------------------------------------------------
        # Random Vertical Flip
        # ----------------------------------------------------

        if np.random.rand() < 0.5:

            lr_patch = torch.flip(

                lr_patch,

                dims=[1]
            )


            hr_patch = torch.flip(

                hr_patch,

                dims=[1]
            )


        return (

            lr_patch,

            hr_patch
        )


    # ========================================================
    # Validation Preparation
    # ========================================================

    def prepare_validation_image(
        self,
        hr
    ):

        _, h, w = hr.shape


        # ----------------------------------------------------
        # Make dimensions divisible by scale
        # ----------------------------------------------------

        h = (

            h
            -
            h % self.scale
        )


        w = (

            w
            -
            w % self.scale
        )


        hr = hr[

            :,

            :h,

            :w
        ]


        # ----------------------------------------------------
        # Generate LR
        # ----------------------------------------------------

        lr = (

            torch
            .nn
            .functional
            .interpolate(

                hr.unsqueeze(
                    0
                ),

                size=(

                    h // self.scale,

                    w // self.scale
                ),

                mode="bicubic",

                align_corners=False
            )

            .squeeze(
                0
            )
        )


        return (

            lr,

            hr
        )


# ============================================================
# Build DIV2K Data Loaders
# ============================================================

def build_div2k_dataset(
    args
):

    # --------------------------------------------------------
    # Training Dataset
    # --------------------------------------------------------

    train_dataset = DIV2KSRDataset(

        hr_dir=args.train_dir,

        scale=args.scale,

        patch_size=args.patch_size,

        training=True
    )


    # --------------------------------------------------------
    # Validation Dataset
    # --------------------------------------------------------

    valid_dataset = DIV2KSRDataset(

        hr_dir=args.valid_dir,

        scale=args.scale,

        patch_size=args.patch_size,

        training=False
    )


    # --------------------------------------------------------
    # Data Loaders
    # --------------------------------------------------------

    trainloader = DataLoader(

        train_dataset,

        batch_size=
        args.batch_size_train,

        shuffle=True,

        num_workers=
        args.num_workers,

        pin_memory=True
    )


    testloader = DataLoader(

        valid_dataset,

        batch_size=
        args.batch_size_test,

        shuffle=False,

        num_workers=
        args.num_workers,

        pin_memory=True
    )


    return (

        trainloader,

        testloader
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(

        description=
        "DIV2K Dataset Builder for Single Image Super-Resolution"
    )


    # --------------------------------------------------------
    # Dataset paths
    # --------------------------------------------------------

    parser.add_argument(

        "--train-dir",

        type=str,

        default=
        "./data/DIV2K_train_HR",

        help=
        "Path to DIV2K training HR images"
    )


    parser.add_argument(

        "--valid-dir",

        type=str,

        default=
        "./data/DIV2K_valid_HR",

        help=
        "Path to DIV2K validation HR images"
    )


    # --------------------------------------------------------
    # Super-resolution scale
    # --------------------------------------------------------

    parser.add_argument(

        "--scale",

        type=int,

        default=4,

        choices=[

            2,

            3,

            4

        ],

        help=
        "Super-resolution scale factor"
    )


    # --------------------------------------------------------
    # LR Patch Size
    # --------------------------------------------------------

    parser.add_argument(

        "--patch-size",

        type=int,

        default=48,

        help=
        "Low-resolution training patch size"
    )


    # --------------------------------------------------------
    # Batch Size
    # --------------------------------------------------------

    parser.add_argument(

        "--batch-size-train",

        type=int,

        default=16,

        help=
        "Training batch size"
    )


    parser.add_argument(

        "--batch-size-test",

        type=int,

        default=1,

        help=
        "Validation batch size"
    )


    # --------------------------------------------------------
    # Number of Workers
    # --------------------------------------------------------

    parser.add_argument(

        "--num-workers",

        type=int,

        default=4,

        help=
        "Number of DataLoader workers"
    )


    args = parser.parse_args()


    # ========================================================
    # Build DIV2K Data Loaders
    # ========================================================

    trainloader, testloader = (

        build_div2k_dataset(
            args
        )
    )


    # ========================================================
    # Print Dataset Information
    # ========================================================

    print(
        "\nDIV2K Dataset Loaded"
    )


    print(

        "Training images:",

        len(
            trainloader.dataset
        )
    )


    print(

        "Validation images:",

        len(
            testloader.dataset
        )
    )


    print(

        "Scale: x",

        args.scale
    )


    print(

        "LR patch size:",

        args.patch_size
    )


    print(

        "HR patch size:",

        args.patch_size
        *
        args.scale
    )


    # ========================================================
    # Test One Batch
    # ========================================================

    lr_batch, hr_batch = next(

        iter(
            trainloader
        )
    )


    print(

        "\nLR batch shape:",

        lr_batch.shape
    )


    print(

        "HR batch shape:",

        hr_batch.shape
    )
```
