```python
import os
import argparse
import numpy as np

from scipy.linalg import sqrtm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from PIL import Image


# ============================================================
# Parser
# ============================================================

parser = argparse.ArgumentParser(
    description='SISR Fisher Information Comparison '
                'using Wasserstein-2 Distance'
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
    help='root directory of dataset'
)

parser.add_argument(
    '--source-checkpoint',
    default='./checkpoint/source_sr.pth',
    type=str,
    help='source SR checkpoint'
)

parser.add_argument(
    '--target-checkpoint',
    default='./checkpoint/target_sr.pth',
    type=str,
    help='target SR checkpoint'
)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

device = torch.device(
    'cuda'
    if torch.cuda.is_available()
    else 'cpu'
)

print(
    'Device:',
    device
)


# ============================================================
# Image Utilities
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

    image_files = []

    for root, _, files in os.walk(folder):

        for file in files:

            if file.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                image_files.append(
                    os.path.join(
                        root,
                        file
                    )
                )

    image_files.sort()

    if len(image_files) == 0:

        raise RuntimeError(
            f'No images found in {folder}'
        )

    return image_files


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

        image_path = (
            self.hr_files[
                index
            ]
        )

        hr = Image.open(
            image_path
        ).convert(
            'RGB'
        )

        width, height = hr.size

        # ====================================================
        # Training mode
        # ====================================================

        if self.training:

            hr_patch_size = (
                self.patch_size
                *
                self.scale
            )

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
        # Ensure dimensions divisible by scale
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

        hr = hr.crop(
            (
                0,
                0,
                lr_width * self.scale,
                lr_height * self.scale
            )
        )

        # ====================================================
        # Generate LR using Bicubic
        # ====================================================

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
# Load DIV2K
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
        'Training images:',
        len(train_files)
    )

    print(
        'Validation images:',
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
# Simple SISR Model
#
# This model is only a placeholder.
# Replace it with:
#
# EDSR
# RDN
# RCAN
# SwinIR
# NAS-SR
# Your Super-Network
#
# ============================================================

class SimpleSR(
    nn.Module
):

    def __init__(
        self,
        scale=4
    ):

        super(
            SimpleSR,
            self
        ).__init__()

        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            64,
            64,
            kernel_size=3,
            padding=1
        )

        self.reconstruction = nn.Conv2d(
            64,
            3 * scale * scale,
            kernel_size=3,
            padding=1
        )

        self.pixel_shuffle = nn.PixelShuffle(
            scale
        )

        self.relu = nn.ReLU(
            inplace=True
        )

    def forward(
        self,
        x
    ):

        x = self.relu(
            self.conv1(x)
        )

        x = self.relu(
            self.conv2(x)
        )

        x = self.relu(
            self.conv3(x)
        )

        x = self.reconstruction(
            x
        )

        x = self.pixel_shuffle(
            x
        )

        return x


# ============================================================
# Fisher Information Matrix
#
# Diagonal approximation
#
# For SISR:
#
#     LR -> SR Network -> SR Image
#                       |
#                       | MSE
#                       v
#                      HR
#
# ============================================================

def diag_fisher_sr(
    model,
    data,
    device
):

    precision_matrices = {}

    # --------------------------------------------------------
    # Initialize Fisher matrices
    # --------------------------------------------------------

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            precision_matrices[
                name
            ] = torch.zeros_like(
                parameter,
                device=device
            )

    # Evaluation mode

    model.eval()

    # Reconstruction loss

    criterion = nn.MSELoss()

    # --------------------------------------------------------
    # Calculate Fisher
    # --------------------------------------------------------

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

        # SR output

        sr_output = model(
            lr_image
        )

        # Reconstruction loss

        loss = criterion(
            sr_output,
            hr_image
        )

        loss.backward()

        # Accumulate squared gradients

        for name, parameter in model.named_parameters():

            if (
                parameter.requires_grad
                and
                parameter.grad is not None
            ):

                precision_matrices[
                    name
                ] += (
                    parameter.grad.detach()
                    ** 2
                )

    # --------------------------------------------------------
    # Average over batches
    # --------------------------------------------------------

    num_batches = max(
        len(data),
        1
    )

    for name in precision_matrices:

        precision_matrices[
            name
        ] /= num_batches

    return precision_matrices


# ============================================================
# Build compact Fisher matrix
#
# Each layer becomes one diagonal element.
#
# This preserves the main idea of your original code:
#
# F1[i,i] = mean(Fisher of layer i)
#
# ============================================================

def build_layer_fisher_matrix(
    fisher,
    model
):

    layer_values = []

    visited_layers = set()

    # --------------------------------------------------------
    # Group parameters by module
    # --------------------------------------------------------

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:

            continue

        # Extract module name
        #
        # Example:
        #
        # conv1.weight
        # conv1.bias
        #
        # -> conv1

        if '.' in name:

            layer_name = (
                name.rsplit(
                    '.',
                    1
                )[0]
            )

        else:

            layer_name = name

        if layer_name in visited_layers:

            continue

        visited_layers.add(
            layer_name
        )

        # ----------------------------------------------------
        # Collect all parameters belonging to layer
        # ----------------------------------------------------

        values = []

        for param_name in fisher:

            if (
                param_name.startswith(
                    layer_name + '.'
                )
            ):

                values.append(
                    fisher[
                        param_name
                    ]
                )

        if len(values) == 0:

            continue

        # Average Fisher of layer

        layer_fisher = torch.cat(
            [
                value.reshape(
                    -1
                )
                for value in values
            ]
        )

        layer_values.append(
            torch.mean(
                layer_fisher
            ).item()
        )

    # --------------------------------------------------------
    # Create diagonal matrix
    # --------------------------------------------------------

    n = len(
        layer_values
    )

    F = np.zeros(
        (
            n,
            n
        ),
        dtype=np.float64
    )

    for i in range(n):

        F[
            i,
            i
        ] = layer_values[
            i
        ]

    return F


# ============================================================
# Normalize Fisher Matrix
# ============================================================

def normalize_fisher_matrix(
    F
):

    trace = np.trace(
        F
    )

    if trace < 1e-12:

        return F

    return (
        F
        /
        trace
    )


# ============================================================
# Wasserstein-2 Distance
#
# Bures-Wasserstein distance between
# positive semi-definite Fisher matrices.
#
# ============================================================

def wasserstein2_distance(
    F1,
    F2
):

    # --------------------------------------------------------
    # Symmetrize
    # --------------------------------------------------------

    F1 = (
        F1
        +
        F1.T
    ) / 2.0

    F2 = (
        F2
        +
        F2.T
    ) / 2.0

    # --------------------------------------------------------
    # Handle numerical errors
    # --------------------------------------------------------

    eps = 1e-8

    F1 = (
        F1
        +
        eps * np.eye(
            F1.shape[0]
        )
    )

    F2 = (
        F2
        +
        eps * np.eye(
            F2.shape[0]
        )
    )

    # --------------------------------------------------------
    # Matrix square root of F1
    # --------------------------------------------------------

    F1_sqrt = sqrtm(
        F1
    ).real

    # --------------------------------------------------------
    # Middle matrix
    # --------------------------------------------------------

    middle = (
        F1_sqrt
        @
        F2
        @
        F1_sqrt
    )

    middle_sqrt = sqrtm(
        middle
    ).real

    # --------------------------------------------------------
    # W2 squared
    # --------------------------------------------------------

    W2_squared = np.trace(

        F1
        +
        F2
        -
        2.0
        *
        middle_sqrt

    ).real

    # Numerical stability

    W2_squared = max(
        W2_squared,
        0.0
    )

    return W2_squared


# ============================================================
# Load checkpoint
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device
):

    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            f'Checkpoint not found: '
            f'{checkpoint_path}'
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if isinstance(
        checkpoint,
        dict
    ) and 'net' in checkpoint:

        model.load_state_dict(
            checkpoint[
                'net'
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    return model


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':

    # ========================================================
    # Dataset
    # ========================================================

    dataset = 'DIV2K'

    # ========================================================
    # Load data
    # ========================================================

    (
        trainloader,
        testloader
    ) = load_div2k_dataset(
        data_root=args.data_root,
        scale=args.scale,
        patch_size=args.patch_size,
        args=args
    )

    # ========================================================
    # SOURCE SR MODEL
    # ========================================================

    print(
        '\nLoading Source SR Model...'
    )

    net_source = SimpleSR(
        scale=args.scale
    ).to(
        device
    )

    net_source = load_checkpoint(
        net_source,
        args.source_checkpoint,
        device
    )

    # ========================================================
    # SOURCE FISHER
    # ========================================================

    print(
        'Computing Source Fisher...'
    )

    fisher_source = diag_fisher_sr(
        net_source,
        trainloader,
        device
    )

    # ========================================================
    # SOURCE COMPACT FISHER MATRIX
    # ========================================================

    F1 = build_layer_fisher_matrix(
        fisher_source,
        net_source
    )

    F1 = normalize_fisher_matrix(
        F1
    )

    print(
        'Source Fisher Matrix Shape:',
        F1.shape
    )

    # ========================================================
    # TARGET SR MODEL
    # ========================================================

    print(
        '\nLoading Target SR Model...'
    )

    net_target = SimpleSR(
        scale=args.scale
    ).to(
        device
    )

    net_target = load_checkpoint(
        net_target,
        args.target_checkpoint,
        device
    )

    # ========================================================
    # TARGET FISHER
    # ========================================================

    print(
        'Computing Target Fisher...'
    )

    fisher_target = diag_fisher_sr(
        net_target,
        trainloader,
        device
    )

    # ========================================================
    # TARGET COMPACT FISHER MATRIX
    # ========================================================

    F2 = build_layer_fisher_matrix(
        fisher_target,
        net_target
    )

    F2 = normalize_fisher_matrix(
        F2
    )

    print(
        'Target Fisher Matrix Shape:',
        F2.shape
    )

    # ========================================================
    # Check dimensions
    # ========================================================

    if F1.shape != F2.shape:

        raise ValueError(
            'Source and Target Fisher matrices '
            'must have the same dimensions. '
            f'Got {F1.shape} and {F2.shape}. '
            'Source and Target models must have '
            'the same layer structure.'
        )

    # ========================================================
    # Wasserstein-2 Distance
    # ========================================================

    distance = wasserstein2_distance(
        F1,
        F2
    )

    # ========================================================
    # Results
    # ========================================================

    print(
        '\n=========================================='
    )

    print(
        'SISR Fisher-Wasserstein Comparison'
    )

    print(
        '=========================================='
    )

    print(
        'Dataset:',
        dataset
    )

    print(
        'Scale:',
        f'x{args.scale}'
    )

    print(
        'Source Checkpoint:',
        args.source_checkpoint
    )

    print(
        'Target Checkpoint:',
        args.target_checkpoint
    )

    print(
        'Fisher Matrix Shape:',
        F1.shape
    )

    print(
        '2-Wasserstein Distance:',
        distance
    )

    print(
        '=========================================='
    )
```
