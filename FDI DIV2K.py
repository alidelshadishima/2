```python
import os
import argparse
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from scipy.stats import wasserstein_distance


# ============================================================
# Parser
# ============================================================

parser = argparse.ArgumentParser(
    description='DIV2K SISR Fisher-Wasserstein Analysis'
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
    help='super-resolution scale'
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
    help='DIV2K root directory'
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

parser.add_argument(
    '--fisher-batches',
    default=100,
    type=int,
    help='number of batches for Fisher estimation'
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
    'Using device:',
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
# DIV2K SISR Dataset
# ============================================================

class DIV2KSISRDataset(
    Dataset
):

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

        # ----------------------------------------------------
        # Load HR
        # ----------------------------------------------------

        hr = Image.open(
            image_path
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

            # If image is too small

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


            # ------------------------------------------------
            # Random crop
            # ------------------------------------------------

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
        # Generate LR
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


        # ----------------------------------------------------
        # Crop HR to be divisible by scale
        # ----------------------------------------------------

        hr = hr.crop(
            (
                0,
                0,
                lr_width * self.scale,
                lr_height * self.scale
            )
        )


        # ----------------------------------------------------
        # Bicubic downsampling
        # ----------------------------------------------------

        lr = hr.resize(
            (
                lr_width,
                lr_height
            ),
            Image.Resampling.BICUBIC
        )


        # ====================================================
        # Data Augmentation
        # ====================================================

        if self.training:

            # Horizontal Flip

            if np.random.rand() < 0.5:

                lr = lr.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )

                hr = hr.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )


            # Vertical Flip

            if np.random.rand() < 0.5:

                lr = lr.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )

                hr = hr.transpose(
                    Image.Transpose.FLIP_TOP_BOTTOM
                )


            # Rotate 90

            if np.random.rand() < 0.5:

                lr = lr.transpose(
                    Image.Transpose.ROTATE_90
                )

                hr = hr.transpose(
                    Image.Transpose.ROTATE_90
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


        return (
            lr,
            hr
        )


# ============================================================
# Load DIV2K
# ============================================================

def load_div2k(
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
        '\nDIV2K Dataset'
    )

    print(
        'Training images:',
        len(train_files)
    )

    print(
        'Validation images:',
        len(valid_files)
    )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = DIV2KSISRDataset(
        hr_files=train_files,
        scale=scale,
        patch_size=patch_size,
        training=True
    )

    valid_dataset = DIV2KSISRDataset(
        hr_files=valid_files,
        scale=scale,
        patch_size=patch_size,
        training=False
    )


    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
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
# Example SISR Model
#
# .
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
# Fisher Information
#
# Diagonal Fisher approximation
#
# For SISR:
#
# LR -> SR Model -> SR
#                 |
#                 | MSE
#                 v
#                 HR
#
# ============================================================

def diag_fisher_sr(
    model,
    data_loader,
    device,
    max_batches=None
):

    precision_matrices = {}


    # --------------------------------------------------------
    # Initialize Fisher
    # --------------------------------------------------------

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            precision_matrices[
                name
            ] = torch.zeros_like(
                parameter,
                device=device
            )


    model.eval()


    # --------------------------------------------------------
    # SR Reconstruction Loss
    # --------------------------------------------------------

    criterion = nn.MSELoss()


    processed_batches = 0


    # ========================================================
    # Fisher Estimation
    # ========================================================

    for lr_image, hr_image in data_loader:


        if (
            max_batches is not None
            and
            processed_batches >= max_batches
        ):

            break


        lr_image = lr_image.to(
            device,
            non_blocking=True
        )

        hr_image = hr_image.to(
            device,
            non_blocking=True
        )


        # ----------------------------------------------------
        # Clear gradients
        # ----------------------------------------------------

        model.zero_grad(
            set_to_none=True
        )


        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        sr_output = model(
            lr_image
        )


        # ----------------------------------------------------
        # Reconstruction Loss
        # ----------------------------------------------------

        loss = criterion(
            sr_output,
            hr_image
        )


        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()


        # ----------------------------------------------------
        # Fisher = E[(gradient)^2]
        # ----------------------------------------------------

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


        processed_batches += 1


    # ========================================================
    # Average Fisher
    # ========================================================

    processed_batches = max(
        processed_batches,
        1
    )


    for name in precision_matrices:

        precision_matrices[
            name
        ] /= processed_batches


    return precision_matrices


# ============================================================
# Fisher Tensor -> Probability Distribution
# ============================================================

def fisher_to_distribution(
    tensor,
    eps=1e-12
):

    # --------------------------------------------------------
    # Flatten
    # --------------------------------------------------------

    values = (
        tensor
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float64
        )
        .flatten()
    )


    # --------------------------------------------------------
    # Fisher values are non-negative
    # --------------------------------------------------------

    values = np.abs(
        values
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    total = np.sum(
        values
    )


    if total < eps:

        values = np.ones_like(
            values
        )

        total = np.sum(
            values
        )


    probabilities = (
        values
        /
        (
            total
            +
            eps
        )
    )


    # --------------------------------------------------------
    # Positions
    # --------------------------------------------------------

    positions = np.arange(
        len(
            probabilities
        ),
        dtype=np.float64
    )


    return (
        positions,
        probabilities
    )


# ============================================================
# Wasserstein Distance between Fisher Tensors
# ============================================================

def fisher_wasserstein_distance(
    fisher_source,
    fisher_target
):

    total_distance = 0.0

    parameter_names = (

        set(
            fisher_source.keys()
        )

        &

        set(
            fisher_target.keys()
        )
    )


    # --------------------------------------------------------
    # Compare corresponding parameters
    # --------------------------------------------------------

    for name in parameter_names:


        source_positions, source_prob = (
            fisher_to_distribution(
                fisher_source[
                    name
                ]
            )
        )


        target_positions, target_prob = (
            fisher_to_distribution(
                fisher_target[
                    name
                ]
            )
        )


        # ----------------------------------------------------
        # Same architecture:
        # same parameter tensor size
        # ----------------------------------------------------

        if len(
            source_prob
        ) != len(
            target_prob
        ):

            print(
                f'Warning: Shape mismatch '
                f'for parameter {name}. '
                f'Skipping.'
            )

            continue


        # ----------------------------------------------------
        # Wasserstein distance
        #
        # Fisher values are used as weights
        # over parameter positions.
        # ----------------------------------------------------

        distance = (
            wasserstein_distance(
                source_positions,
                target_positions,
                u_weights=source_prob,
                v_weights=target_prob
            )
        )


        total_distance += (
            distance
        )


    return total_distance


# ============================================================
# Load Checkpoint
# ============================================================

def load_model_checkpoint(
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


    # --------------------------------------------------------
    # Checkpoint format:
    #
    # {'net': state_dict}
    #
    # or directly:
    #
    # state_dict
    # --------------------------------------------------------

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        'net' in checkpoint
    ):

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
    # Load DIV2K
    # ========================================================

    (
        trainloader,
        testloader
    ) = load_div2k(
        data_root=args.data_root,
        scale=args.scale,
        patch_size=args.patch_size,
        args=args
    )


    # ========================================================
    # SOURCE MODEL
    # ========================================================

    print(
        '\nLoading Source SR Model...'
    )


    net_source = SimpleSR(
        scale=args.scale
    ).to(
        device
    )


    net_source = load_model_checkpoint(
        net_source,
        args.source_checkpoint,
        device
    )


    # ========================================================
    # TARGET MODEL
    # ========================================================

    print(
        'Loading Target SR Model...'
    )


    net_target = SimpleSR(
        scale=args.scale
    ).to(
        device
    )


    net_target = load_model_checkpoint(
        net_target,
        args.target_checkpoint,
        device
    )


    # ========================================================
    # SOURCE FISHER
    # ========================================================

    print(
        '\nComputing Source Fisher...'
    )


    fisher_source = diag_fisher_sr(
        model=net_source,
        data_loader=trainloader,
        device=device,
        max_batches=args.fisher_batches
    )


    # ========================================================
    # TARGET FISHER
    # ========================================================

    print(
        'Computing Target Fisher...'
    )


    fisher_target = diag_fisher_sr(
        model=net_target,
        data_loader=trainloader,
        device=device,
        max_batches=args.fisher_batches
    )


    # ========================================================
    # Fisher-Wasserstein Distance
    # ========================================================

    print(
        '\nComputing Fisher-Wasserstein Distance...'
    )


    distance = (
        fisher_wasserstein_distance(
            fisher_source,
            fisher_target
        )
    )


    # ========================================================
    # Results
    # ========================================================

    print(
        '\n=========================================='
    )

    print(
        'DIV2K SISR Fisher-Wasserstein Analysis'
    )

    print(
        '=========================================='
    )

    print(
        'Dataset: DIV2K'
    )

    print(
        'Scale:',
        f'x{args.scale}'
    )

    print(
        'LR Patch Size:',
        args.patch_size
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
        'Fisher Batches:',
        args.fisher_batches
    )

    print(
        'Wasserstein Distance:',
        distance
    )

    print(
        '=========================================='
    )
```
