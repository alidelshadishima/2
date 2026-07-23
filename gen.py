"""
fisher_guided_nsga2_nas_sr.py

Fisher-Guided Two-Level NSGA-II-NAS-SR
Dataset: DIV2K

Level 1:
    Search for the macro architecture:
        - number of cells
        - number of channels
        - cell topology

Level 2:
    Search for operations inside each DAG cell:
        - conv_3x3
        - conv_5x5
        - dil_conv_3x3
        - sep_conv_3x3
        - skip_connect

Fisher:
    Used to initialize the population from architectures
    with similar Fisher Information fingerprints.

Objectives:
    1. Maximize PSNR
    2. Minimize model complexity (# parameters)

The code is designed as a building block for a
two-level NSGA-II NAS framework for Single Image Super-Resolution.
"""

import os
import random
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import wasserstein_distance


# ============================================================
# 1. SR OPERATIONS
# ============================================================

class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.zeros_like(x)


class Conv3x3(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                C, C,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


class Conv5x5(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                C, C,
                kernel_size=5,
                padding=2,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


class DilConv3x3(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.op = nn.Sequential(
            nn.Conv2d(
                C, C,
                kernel_size=3,
                padding=2,
                dilation=2,
                bias=False
            ),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


class SepConv3x3(nn.Module):
    def __init__(self, C):
        super().__init__()

        self.op = nn.Sequential(

            # Depthwise convolution
            nn.Conv2d(
                C, C,
                kernel_size=3,
                padding=1,
                groups=C,
                bias=False
            ),

            # Pointwise convolution
            nn.Conv2d(
                C, C,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.op(x)


OPS = {

    "conv_3x3":
        lambda C: Conv3x3(C),

    "conv_5x5":
        lambda C: Conv5x5(C),

    "dil_conv_3x3":
        lambda C: DilConv3x3(C),

    "sep_conv_3x3":
        lambda C: SepConv3x3(C),

    "skip_connect":
        lambda C: Identity(),

    "none":
        lambda C: Zero()
}


# ============================================================
# 2. SEARCH SPACE
# ============================================================

SR_SEARCH_OPS = [

    "conv_3x3",
    "conv_5x5",
    "dil_conv_3x3",
    "sep_conv_3x3",
    "skip_connect"
]


# ============================================================
# 3. TWO-LEVEL ARCHITECTURE REPRESENTATION
# ============================================================

class TwoLevelArchitecture:

    def __init__(
        self,
        num_cells=8,
        channels=64,
        num_nodes=4,
        scale=2
    ):

        # -----------------------------
        # Level 1: Macro Architecture
        # -----------------------------

        self.num_cells = num_cells
        self.channels = channels
        self.num_nodes = num_nodes
        self.scale = scale

        # -----------------------------
        # Level 2: Micro Architecture
        # -----------------------------

        self.cell_ops = []

        for cell_id in range(num_cells):

            cell = []

            for node in range(num_nodes):

                op = random.choice(SR_SEARCH_OPS)

                cell.append(op)

            self.cell_ops.append(cell)

    def clone(self):

        return deepcopy(self)

    def __repr__(self):

        return (
            f"TwoLevelArchitecture("
            f"cells={self.num_cells}, "
            f"channels={self.channels}, "
            f"nodes={self.num_nodes}, "
            f"scale={self.scale})"
        )


# ============================================================
# 4. SR CELL
# ============================================================

class SRCell(nn.Module):

    def __init__(
        self,
        channels,
        operations
    ):

        super().__init__()

        self.operations = nn.ModuleList(

            [
                OPS[op](channels)
                for op in operations
            ]

        )

    def forward(self, x):

        states = [x]

        for op in self.operations:

            out = op(states[-1])

            states.append(out)

        # Residual aggregation

        result = x

        for state in states[1:]:

            result = result + state

        return result / len(states)


# ============================================================
# 5. SR NETWORK
# ============================================================

class NAS_SR_Network(nn.Module):

    def __init__(
        self,
        architecture,
        in_channels=3,
        out_channels=3
    ):

        super().__init__()

        self.architecture = architecture

        C = architecture.channels

        scale = architecture.scale

        # --------------------------------
        # Shallow Feature Extraction
        # --------------------------------

        self.head = nn.Conv2d(
            in_channels,
            C,
            kernel_size=3,
            padding=1
        )

        # --------------------------------
        # Searchable Cells
        # --------------------------------

        self.cells = nn.ModuleList()

        for cell_ops in architecture.cell_ops:

            self.cells.append(

                SRCell(
                    C,
                    cell_ops
                )

            )

        # --------------------------------
        # Reconstruction
        # --------------------------------

        self.body = nn.Conv2d(
            C,
            C,
            kernel_size=3,
            padding=1
        )

        # --------------------------------
        # Upsampling
        # --------------------------------

        self.upsample = nn.Sequential(

            nn.Conv2d(
                C,
                C * scale * scale,
                kernel_size=3,
                padding=1
            ),

            nn.PixelShuffle(scale),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                C,
                out_channels,
                kernel_size=3,
                padding=1
            )

        )

    def forward(self, x):

        shallow = self.head(x)

        out = shallow

        for cell in self.cells:

            residual = cell(out)

            out = out + residual

        out = self.body(out)

        out = out + shallow

        out = self.upsample(out)

        return out


# ============================================================
# 6. FISHER INFORMATION
# ============================================================

def compute_fisher_information(
    model,
    data_loader,
    device="cuda",
    max_batches=5
):

    model.train()

    fisher = {}

    for name, param in model.named_parameters():

        if param.requires_grad:

            fisher[name] = torch.zeros_like(
                param,
                device=device
            )

    criterion = nn.L1Loss()

    for batch_idx, batch in enumerate(data_loader):

        if batch_idx >= max_batches:

            break

        # DIV2K loader should return:
        # LR image, HR image

        lr, hr = batch

        lr = lr.to(device)

        hr = hr.to(device)

        model.zero_grad()

        sr = model(lr)

        loss = criterion(
            sr,
            hr
        )

        loss.backward()

        for name, param in model.named_parameters():

            if (
                param.requires_grad
                and param.grad is not None
            ):

                fisher[name] += (
                    param.grad.detach() ** 2
                )

    # Average

    for name in fisher:

        fisher[name] /= max_batches

    return fisher


# ============================================================
# 7. FISHER FINGERPRINT
# ============================================================

def fisher_fingerprint(
    fisher_dict,
    eps=1e-12
):

    values = []

    for name in sorted(
        fisher_dict.keys()
    ):

        value = fisher_dict[name]

        value = value.detach().cpu().numpy()

        values.append(
            value.flatten()
        )

    if len(values) == 0:

        return np.array([])

    vector = np.concatenate(values)

    vector = np.abs(vector)

    vector = vector / (
        vector.sum() + eps
    )

    return vector


# ============================================================
# 8. WASSERSTEIN FISHER DISTANCE
# ============================================================

def fisher_wasserstein_distance(
    f1,
    f2
):

    if len(f1) == 0 or len(f2) == 0:

        return np.inf

    length = min(
        len(f1),
        len(f2)
    )

    f1 = f1[:length]

    f2 = f2[:length]

    f1 = f1 / (
        f1.sum() + 1e-12
    )

    f2 = f2 / (
        f2.sum() + 1e-12
    )

    positions = np.arange(
        length
    )

    distance = wasserstein_distance(

        positions,

        positions,

        u_weights=f1,

        v_weights=f2

    )

    return float(distance)


# ============================================================
# 9. FISHER-GUIDED ARCHITECTURE INITIALIZATION
# ============================================================

class FisherGuidedSRArchitecture:

    def __init__(
        self,
        fisher_bank,
        task_fisher,
        k_nearest=3,
        scale=2
    ):

        self.fisher_bank = fisher_bank

        self.task_fisher = task_fisher

        self.k_nearest = k_nearest

        self.scale = scale

        self.task_vector = fisher_fingerprint(
            task_fisher
        )

    def find_nearest_architectures(self):

        distances = []

        for name, fisher in self.fisher_bank.items():

            arch_vector = fisher_fingerprint(
                fisher
            )

            distance = fisher_wasserstein_distance(

                self.task_vector,

                arch_vector

            )

            distances.append(

                (
                    name,
                    distance
                )

            )

        distances.sort(
            key=lambda x: x[1]
        )

        return distances[
            :self.k_nearest
        ]

    def initialize_population(
        self,
        population_size
    ):

        nearest = (

            self.find_nearest_architectures()

        )

        population = []

        for i in range(
            population_size
        ):

            if len(nearest) > 0:

                selected = random.choice(
                    nearest
                )

                # In a real implementation,
                # retrieve architecture template
                # from architecture bank.

                architecture = (
                    TwoLevelArchitecture(
                        num_cells=random.randint(
                            4,
                            16
                        ),
                        channels=random.choice(
                            [32, 48, 64, 96]
                        ),
                        num_nodes=4,
                        scale=self.scale
                    )
                )

            else:

                architecture = (
                    TwoLevelArchitecture(
                        scale=self.scale
                    )
                )

            population.append(
                architecture
            )

        return population


# ============================================================
# 10. TWO-LEVEL MUTATION
# ============================================================

def mutate_level_1(
    architecture
):

    """
    Macro-level mutation

    Changes:
        - number of cells
        - number of channels
    """

    child = architecture.clone()

    mutation = random.choice(

        [
            "num_cells",
            "channels"
        ]

    )

    if mutation == "num_cells":

        child.num_cells = random.randint(
            4,
            16
        )

        current_cells = len(
            child.cell_ops
        )

        if child.num_cells > current_cells:

            for _ in range(
                child.num_cells
                - current_cells
            ):

                child.cell_ops.append(

                    [
                        random.choice(
                            SR_SEARCH_OPS
                        )

                        for _ in range(
                            child.num_nodes
                        )
                    ]

                )

        else:

            child.cell_ops = (
                child.cell_ops[
                    :child.num_cells
                ]
            )

    elif mutation == "channels":

        child.channels = random.choice(

            [
                32,
                48,
                64,
                96,
                128
            ]

        )

    return child


def mutate_level_2(
    architecture
):

    """
    Micro-level mutation

    Changes operation inside DAG cells.
    """

    child = architecture.clone()

    cell_id = random.randrange(

        len(
            child.cell_ops
        )

    )

    node_id = random.randrange(

        len(
            child.cell_ops[
                cell_id
            ]
        )

    )

    child.cell_ops[
        cell_id
    ][
        node_id
    ] = random.choice(

        SR_SEARCH_OPS

    )

    return child


# ============================================================
# 11. TWO-LEVEL MUTATION
# ============================================================

def two_level_mutation(
    architecture,
    p_level1=0.3,
    p_level2=0.7
):

    child = architecture.clone()

    if random.random() < p_level1:

        child = mutate_level_1(
            child
        )

    if random.random() < p_level2:

        child = mutate_level_2(
            child
        )

    return child


# ============================================================
# 12. CROSSOVER
# ============================================================

def two_level_crossover(
    parent1,
    parent2
):

    child = parent1.clone()

    # -------------------------
    # Level 1 crossover
    # -------------------------

    if random.random() < 0.5:

        child.num_cells = (
            parent2.num_cells
        )

    if random.random() < 0.5:

        child.channels = (
            parent2.channels
        )

    # -------------------------
    # Level 2 crossover
    # -------------------------

    min_cells = min(

        len(
            parent1.cell_ops
        ),

        len(
            parent2.cell_ops
        )

    )

    child.cell_ops = []

    for i in range(
        min_cells
    ):

        if random.random() < 0.5:

            cell = deepcopy(

                parent1.cell_ops[
                    i
                ]

            )

        else:

            cell = deepcopy(

                parent2.cell_ops[
                    i
                ]

            )

        child.cell_ops.append(
            cell
        )

    child.num_cells = len(
        child.cell_ops
    )

    return child


# ============================================================
# 13. MODEL COMPLEXITY
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
# 14. NSGA-II OBJECTIVES
# ============================================================

def evaluate_architecture(
    architecture,
    data_loader,
    device="cuda"
):

    model = NAS_SR_Network(
        architecture
    ).to(device)

    n_params = count_parameters(
        model
    )

    # -----------------------------
    # Quick evaluation
    # -----------------------------

    model.eval()

    total_psnr = 0.0

    count = 0

    with torch.no_grad():

        for lr, hr in data_loader:

            lr = lr.to(device)

            hr = hr.to(device)

            sr = model(lr)

            mse = F.mse_loss(
                sr,
                hr
            )

            psnr = (
                -10.0
                * torch.log10(
                    mse + 1e-10
                )
            )

            total_psnr += (
                psnr.item()
            )

            count += 1

    mean_psnr = (

        total_psnr
        / max(count, 1)

    )

    return {

        "psnr": mean_psnr,

        "params": n_params

    }


# ============================================================
# 15. NSGA-II FITNESS
# ============================================================

def objective_vector(
    metrics
):

    """

    NSGA-II objective:

    Objective 1:
        maximize PSNR

    Objective 2:
        minimize parameters

    """

    return [

        -metrics["psnr"],

        metrics["params"]

    ]


# ============================================================
# 16. EXAMPLE
# ============================================================

if __name__ == "__main__":

    device = (

        "cuda"

        if torch.cuda.is_available()

        else "cpu"

    )

    print(
        "Device:",
        device
    )

    # --------------------------------
    # DIV2K configuration
    # --------------------------------

    scale = 2

    population_size = 20

    # --------------------------------
    # Example task architecture
    # --------------------------------

    task_architecture = (

        TwoLevelArchitecture(

            num_cells=8,

            channels=64,

            num_nodes=4,

            scale=scale

        )

    )

    print(
        "Initial architecture:"
    )

    print(
        task_architecture
    )

    # --------------------------------
    # Build SR model
    # --------------------------------

    model = NAS_SR_Network(

        task_architecture,

        in_channels=3,

        out_channels=3

    )

    print(
        "Number of parameters:",
        count_parameters(
            model
        )
    )

    # --------------------------------
    # Example mutation
    # --------------------------------

    child = two_level_mutation(

        task_architecture

    )

    print(
        "Mutated architecture:"
    )

    print(
        child
    )

    # --------------------------------
    # Example crossover
    # --------------------------------

    parent2 = (

        TwoLevelArchitecture(

            num_cells=12,

            channels=48,

            num_nodes=4,

            scale=scale

        )

    )

    child2 = two_level_crossover(

        task_architecture,

        parent2

    )

    print(
        "Crossover architecture:"
    )

    print(
        child2
    )
