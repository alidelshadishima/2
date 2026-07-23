```python
import os
import argparse
from copy import deepcopy

import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scipy.stats import wasserstein_distance

from PIL import Image


# ============================================================
# Argument Parser
# ============================================================

parser = argparse.ArgumentParser(
    description='Fisher Information and Wasserstein Distance '
                'for Single Image Super-Resolution'
)

parser.add_argument(
    '--lr',
    default=1e-4,
    type=float,
    help='learning rate'
)

parser.add_argument(
    '--batch-size-train',
    default=16,
    type=int,
    help='training batch size'
)

parser.add_argument(
    '--batch-size-test',
    default=1,
    type=int,
    help='testing batch size'
)

parser.add_argument(
    '--num-epoch',
    default=10,
    type=int,
    help='number of epochs'
)

parser.add_argument(
    '--scale',
    default=4,
    type=int,
    choices=[2, 3, 4],
    help='super-resolution scale factor'
)

parser.add_argument(
    '--patch-size',
    default=48,
    type=int,
    help='LR patch size'
)

parser.add_argument(
    '--data-root',
    default='./data',
    type=str,
    help='dataset root directory'
)

parser.add_argument(
    '--source-checkpoint',
    default='./checkpoint/source_sr.pth',
    type=str,
    help='source SR model checkpoint'
)

parser.add_argument(
    '--target-checkpoint',
    default='./checkpoint/target_sr.pth',
    type=str,
    help='target SR model checkpoint'
)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

device = torch.device(
    'cuda' if torch.cuda.is_available() else 'cpu'
)

print(
    'Using device:',
    device
)


# ============================================================
# Image utilities
# ============================================================

IMAGE_EXTENSIONS = (
    '.png',
    '.jpg',
    '.jpeg',
    '.bmp',
    '.tif',
    '.tiff'
)


def find_images(folder):

    if not os.path.exists(folder):

        raise FileNotFoundError(
            f'Directory not found: {folder}'
        )

    files = []

    for root, _, filenames in os.walk(folder):

        for filename in filenames:

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                files.append(
                    os.path.join(
                        root,
                        filename
                    )
                )

    files.sort()

    return files


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
# SISR Dataset
# ============================================================

class SISRDataset(Dataset):

    def __init__(
        self,
        hr_files,
        scale=4,
        patch_size=48,
        training=True
    ):

        self.hr_files = hr_files

        self.scale = scale

        self.patch_size = patch_size

        self.training = training

    def __len__(
        self
    ):

        return len(
            self.hr_files
        )

    def __getitem__(
        self,
        index
    ):

        hr_path = (
            self.hr_files[
                index
            ]
        )

        hr = Image.open(
            hr_path
        ).convert(
            'RGB'
        )

        width, height = hr.size

        # ====================================================
        # Training
        # ====================================================

        if self.training:

            hr_patch_size = (
                self.patch_size
                *
                self.scale
            )

            # If image is too small,
            # choose another image

            if (
                width < hr_patch_size
                or
                height < hr_patch_size
            ):

                new_index = np.random.randint(
                    0,
                    len(
                        self.hr_files
                    )
                )

                return self.__getitem__(
                    new_index
                )

            # Random crop

            x = np.random.randint(
                0,
                width - hr_patch_size + 1
            )

            y = np.random.randint(
                0,
                height - hr_patch_size + 1
            )

            hr = hr.crop(
                (
                    x,
                    y,
                    x + hr_patch_size,
                    y + hr_patch_size
                )
            )

        # ====================================================
        # Generate LR image
        # ====================================================

        lr_width = (
            hr.width
            //
            self.scale
        )

        lr_height = (
            hr.height
            //
            self.scale
        )

        # Make HR divisible by scale

        hr = hr.crop(
            (
                0,
                0,
                lr_width * self.scale,
                lr_height * self.scale
            )
        )

        lr = hr.resize(
            (
                lr_width,
                lr_height
            ),
            Image.Resampling.BICUBIC
        )

        # ====================================================
        # Data augmentation
        # ====================================================

        if self.training:

            if np.random.rand() < 0.5:

                lr = lr.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                hr = hr.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

            if np.random.rand() < 0.5:

                lr = lr.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

                hr = hr.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

        # ====================================================
        # Convert to Tensor
        # ====================================================

        lr = pil_to_tensor(
            lr
        )

        hr = pil_to_tensor(
            hr
        )

        return lr, hr


# ============================================================
# Load DIV2K Dataset
# ============================================================

def load_div2k_dataset(
    data_root,
    scale,
    patch_size,
    args
):

    train_dir = os.path.join(
        data_root,
        'DIV2K_train_HR'
    )

    valid_dir = os.path.join(
        data_root,
        'DIV2K_valid_HR'
    )

    train_files = find_images(
        train_dir
    )

    valid_files = find_images(
        valid_dir
    )

    print(
        'Number of training images:',
        len(train_files)
    )

    print(
        'Number of validation images:',
        len(valid_files)
    )

    train_dataset = SISRDataset(
        hr_files=train_files,
        scale=scale,
        patch_size=patch_size,
        training=True
    )

    valid_dataset = SISRDataset(
        hr_files=valid_files,
        scale=scale,
        patch_size=patch_size,
        training=False
    )

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    testloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size_test,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return (
        trainloader,
        testloader
    )


# ============================================================
# Example SR Model
# ============================================================
#
# IMPORTANT:
# Replace this model with your own NAS-SR model,
# EDSR, RDN, RCAN, SwinIR, etc.
#
# The model must satisfy:
#
# input : [B, 3, H, W]
# output: [B, 3, H*scale, W*scale]
#
# ============================================================

class SimpleSR(nn.Module):

    def __init__(
        self,
        scale=4
    ):

        super(
            SimpleSR,
            self
        ).__init__()

        self.scale = scale

        self.feature = nn.Sequential(

            nn.Conv2d(
                3,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                64,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                64,
                3 * scale * scale,
                kernel_size=3,
                padding=1
            )
        )

        self.upsample = nn.PixelShuffle(
            scale
        )

    def forward(
        self,
        x
    ):

        x = self.feature(
            x
        )

        x = self.upsample(
            x
        )

        return x


# ============================================================
# Fisher Information Matrix for SISR
# ============================================================

def diag_fisher_sr(
    model,
    data,
    device
):

    precision_matrices = {}

    params = {
        n: p
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    # Initialize Fisher matrices

    for n, p in params.items():

        precision_matrices[n] = torch.zeros_like(
            p,
            device=device
        )

    model.eval()

    # ========================================================
    # Reconstruction loss
    # ========================================================

    criterion = nn.MSELoss()

    # ========================================================
    # Calculate Fisher Information
    # ========================================================

    for batch_idx, batch in enumerate(
        data
    ):

        lr_image, hr_image = batch

        lr_image = lr_image.to(
            device,
            non_blocking=True
        )

        hr_image = hr_image.to(
            device,
            non_blocking=True
        )

        model.zero_grad(
            set_to_none=True
        )

        # SR prediction

        sr_output = model(
            lr_image
        )

        # Reconstruction loss

        loss = criterion(
            sr_output,
            hr_image
        )

        loss.backward()

        # ====================================================
        # Accumulate squared gradients
        # ====================================================

        for n, p in model.named_parameters():

            if (
                p.requires_grad
                and
                p.grad is not None
            ):

                precision_matrices[n] += (
                    p.grad.detach()
                    ** 2
                )

    # ========================================================
    # Average Fisher over batches
    # ========================================================

    num_batches = max(
        len(data),
        1
    )

    for n in precision_matrices:

        precision_matrices[n] /= (
            num_batches
        )

    return precision_matrices


# ============================================================
# Normalize Fisher Matrix
# ============================================================

def normalize_fisher(
    fisher_matrix
):

    total = sum(

        torch.sum(
            f
        ).item()

        for f
        in fisher_matrix.values()

    )

    total = max(
        total,
        1e-12
    )

    normalized = {}

    for n, f in fisher_matrix.items():

        normalized[n] = (
            f
            /
            total
        )

    return normalized


# ============================================================
# Wasserstein Distance
# ============================================================

def fisher_wasserstein_distance(
    fisher_source,
    fisher_target
):

    distance = 0.0

    common_parameters = (

        set(
            fisher_source.keys()
        )

        &

        set(
            fisher_target.keys()
        )
    )

    for n in common_parameters:

        f_source = (
            fisher_source[n]
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        f_target = (
            fisher_target[n]
            .detach()
            .cpu()
            .numpy()
            .flatten()
        )

        # ----------------------------------------------------
        # Convert Fisher values into probability distributions
        # ----------------------------------------------------

        f_source = np.abs(
            f_source
        )

        f_target = np.abs(
            f_target
        )

        f_source = (
            f_source
            /
            (
                np.sum(
                    f_source
                )
                +
                1e-12
            )
        )

        f_target = (
            f_target
            /
            (
                np.sum(
                    f_target
                )
                +
                1e-12
            )
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # scipy.stats.wasserstein_distance(x, y)
        # compares distributions whose values are x and y.
        #
        # Here the Fisher values are used as samples.
        #
        # For parameter-wise Fisher comparison:
        # ----------------------------------------------------

        parameter_distance = (
            wasserstein_distance(
                f_source,
                f_target
            )
        )

        distance += (
            parameter_distance
        )

    return distance


# ============================================================
# Main
# ============================================================

def main():

    # ========================================================
    # Load SISR Dataset
    # ========================================================

    trainloader, testloader = (
        load_div2k_dataset(
            data_root=args.data_root,
            scale=args.scale,
            patch_size=args.patch_size,
            args=args
        )
    )

    # ========================================================
    # SOURCE MODEL
    # ========================================================

    print(
        '\nLoading source SR model...'
    )

    net_source = SimpleSR(
        scale=args.scale
    ).to(
        device
    )

    if os.path.exists(
        args.source_checkpoint
    ):

        checkpoint_source = torch.load(
            args.source_checkpoint,
            map_location=device
        )

        if 'net' in checkpoint_source:

            net_source.load_state_dict(
                checkpoint_source[
                    'net'
                ]
            )

        else:

            net_source.load_state_dict(
                checkpoint_source
            )

    else:

        print(
            'WARNING: Source checkpoint '
            'not found.'
        )

    # ========================================================
    # SOURCE FISHER
    # ========================================================

    print(
        '\nComputing Fisher Matrix '
        'for source SR model...'
    )

    fisher_matrix_source = (
        diag_fisher_sr(
            net_source,
            testloader,
            device
        )
    )

    fisher_matrix_source = (
        normalize_fisher(
            fisher_matrix_source
        )
    )

    # ========================================================
    # TARGET MODEL
    # ========================================================

    print(
        '\nLoading target SR model...'
    )

    net_target = SimpleSR(
        scale=args.scale
    ).to(
        device
    )

    if os.path.exists(
        args.target_checkpoint
    ):

        checkpoint_target = torch.load(
            args.target_checkpoint,
            map_location=device
        )

        if 'net' in checkpoint_target:

            net_target.load_state_dict(
                checkpoint_target[
                    'net'
                ]
            )

        else:

            net_target.load_state_dict(
                checkpoint_target
            )

    else:

        print(
            'WARNING: Target checkpoint '
            'not found.'
        )

    # ========================================================
    # TARGET FISHER
    # ========================================================

    print(
        '\nComputing Fisher Matrix '
        'for target SR model...'
    )

    fisher_matrix_target = (
        diag_fisher_sr(
            net_target,
            testloader,
            device
        )
    )

    fisher_matrix_target = (
        normalize_fisher(
            fisher_matrix_target
        )
    )

    # ========================================================
    # Wasserstein Distance
    # ========================================================

    print(
        '\nComputing Wasserstein Distance...'
    )

    distance = (
        fisher_wasserstein_distance(
            fisher_matrix_source,
            fisher_matrix_target
        )
    )

    print(
        '\n======================================'
    )

    print(
        'Fisher-Wasserstein Distance'
    )

    print(
        '======================================'
    )

    print(
        'Source Model:',
        args.source_checkpoint
    )

    print(
        'Target Model:',
        args.target_checkpoint
    )

    print(
        'Scale:',
        f'x{args.scale}'
    )

    print(
        'Wasserstein Distance:',
        distance
    )

    print(
        '======================================'
    )


if __name__ == '__main__':

    main()
```
