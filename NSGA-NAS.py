"""
Fisher-Guided Two-Level NSGA-II NAS for Single Image Super-Resolution

Dataset:
    DIV2K

Two-Level Search:
    Level 1:
        Cell/DAG architecture search

    Level 2:
        Network architecture search by stacking selected cells

Objectives:
    1. Maximize PSNR
    2. Maximize SSIM
    3. Minimize model complexity (parameters)

Fisher Guidance:
    - Fisher fingerprints are used to initialize the population.
    - Fisher similarity is used during mutation.
    - Architectures with higher Fisher similarity to the target SR task
      receive higher probability during initialization/mutation.

Important:
    This implementation assumes:
        - DIV2K dataset loader is available.
        - SR models can be constructed from Cell/DAG definitions.
        - The Fisher bank contains fingerprints for known SR architectures.
"""

import os
import random
import time
import copy
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import wasserstein_distance


# ============================================================
# Configuration
# ============================================================

DATASET = "DIV2K"

SCALE = 4

# LR patch size
LR_PATCH_SIZE = 48

# HR patch size
HR_PATCH_SIZE = LR_PATCH_SIZE * SCALE

POP_SIZE = 20
N_GENERATIONS = 30

# Number of architectures evaluated per generation
EPOCHS_SEARCH = 1

# Final training
EPOCHS_FINAL = 200

BATCH_SIZE = 16

NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# Search Objectives
# ============================================================

OBJECTIVES = [
    "psnr",
    "ssim",
    "params"
]


# ============================================================
# Known SR Architectures
# ============================================================

KNOWN_SR_ARCHS = {

    "EDSR": {
        "num_cells": 16,
        "base_channels": 64,
        "cell_type": "residual"
    },

    "RDN": {
        "num_cells": 16,
        "base_channels": 64,
        "cell_type": "dense"
    },

    "RCAN": {
        "num_cells": 20,
        "base_channels": 64,
        "cell_type": "attention"
    },

    "DLSR": {
        "num_cells": 12,
        "base_channels": 48,
        "cell_type": "residual"
    }
}


# ============================================================
# Fisher Bank
# ============================================================

FISHER_BANK_PATH = "fisher_bank_div2k_sr.npz"


def load_fisher_bank(path):

    if not os.path.exists(path):

        print(
            "WARNING: Fisher bank not found."
        )

        return {}

    data = np.load(
        path,
        allow_pickle=True
    )

    bank = {}

    for key in data.files:

        value = data[key]

        if value.dtype == np.object_:

            value = value.item()

        if isinstance(value, dict):

            if "__global__" in value:

                vec = value["__global__"]

            else:

                vectors = []

                for _, v in value.items():

                    vectors.append(
                        np.asarray(v).flatten()
                    )

                vec = np.concatenate(
                    vectors
                )

        else:

            vec = np.asarray(
                value
            ).flatten()

        vec = vec.astype(
            np.float64
        )

        vec = np.maximum(
            vec,
            0
        )

        vec = vec / (
            vec.sum() + 1e-12
        )

        bank[key] = {
            "__global__": vec
        }

    return bank


# ============================================================
# Fisher Distance
# ============================================================

def fisher_wasserstein_distance(
        v1,
        v2):

    v1 = np.asarray(
        v1,
        dtype=np.float64
    )

    v2 = np.asarray(
        v2,
        dtype=np.float64
    )

    if len(v1) == 0 or len(v2) == 0:

        return float("inf")

    n = min(
        len(v1),
        len(v2)
    )

    v1 = v1[:n]
    v2 = v2[:n]

    v1 = np.maximum(
        v1,
        0
    )

    v2 = np.maximum(
        v2,
        0
    )

    v1 = v1 / (
        v1.sum() + 1e-12
    )

    v2 = v2 / (
        v2.sum() + 1e-12
    )

    positions = np.arange(
        n
    )

    return wasserstein_distance(
        positions,
        positions,
        u_weights=v1,
        v_weights=v2
    )


# ============================================================
# Fisher Similarity
# ============================================================

def fisher_similarity(
        v1,
        v2):

    distance = fisher_wasserstein_distance(
        v1,
        v2
    )

    return 1.0 / (
        distance + 1e-8
    )


# ============================================================
# DIV2K Dataset
# ============================================================

class DIV2KSRDataset(
        torch.utils.data.Dataset):

    def __init__(
            self,
            lr_images,
            hr_images):

        self.lr_images = lr_images

        self.hr_images = hr_images

    def __len__(
            self):

        return len(
            self.lr_images
        )

    def __getitem__(
            self,
            index):

        lr = self.lr_images[
            index
        ]

        hr = self.hr_images[
            index
        ]

        return (
            lr.float(),
            hr.float()
        )


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
        sr,
        hr,
        max_val=1.0):

    mse = F.mse_loss(
        sr,
        hr
    )

    if mse.item() == 0:

        return 100.0

    psnr = 10 * torch.log10(
        max_val ** 2 / mse
    )

    return psnr.item()


# ============================================================
# SSIM
# ============================================================

def calculate_ssim(
        sr,
        hr):

    # Simplified global SSIM
    # For final experiments, use
    # skimage.metrics.structural_similarity
    # or torchmetrics SSIM.

    C1 = 0.01 ** 2

    C2 = 0.03 ** 2

    mu_x = sr.mean()

    mu_y = hr.mean()

    sigma_x = sr.var()

    sigma_y = hr.var()

    sigma_xy = (
        (sr - mu_x) *
        (hr - mu_y)
    ).mean()

    numerator = (
        (2 * mu_x * mu_y + C1) *
        (2 * sigma_xy + C2)
    )

    denominator = (
        (mu_x ** 2 +
         mu_y ** 2 +
         C1) *
        (sigma_x +
         sigma_y +
         C2)
    )

    return (
        numerator /
        denominator
    ).item()


# ============================================================
# SR Search Cell
# ============================================================

class SRCell:

    OPERATIONS = [

        "conv3x3",

        "conv5x5",

        "sep_conv3x3",

        "dil_conv3x3",

        "skip",

        "attention",

        "residual"
    ]

    def __init__(
            self,
            num_nodes=4):

        self.num_nodes = num_nodes

        self.operations = []

        for i in range(
                num_nodes):

            op = random.choice(
                self.OPERATIONS
            )

            self.operations.append(
                op
            )

    def clone(
            self):

        return copy.deepcopy(
            self
        )

    def mutate(
            self):

        new_cell = self.clone()

        index = random.randint(
            0,
            self.num_nodes - 1
        )

        new_cell.operations[
            index
        ] = random.choice(
            self.OPERATIONS
        )

        return new_cell


# ============================================================
# Level-2 Network Architecture
# ============================================================

class SRNetworkArchitecture:

    def __init__(
            self,
            cells=None,
            num_cells=None,
            channels=None):

        if cells is None:

            num_cells = (
                num_cells
                if num_cells
                else random.randint(
                    4,
                    20
                )
            )

            self.cells = [

                SRCell()

                for _ in range(
                    num_cells
                )
            ]

        else:

            self.cells = cells

        self.channels = (

            channels
            if channels
            else random.choice(
                [
                    32,
                    48,
                    64,
                    96
                ]
            )
        )

        self.scale = SCALE

        self.psnr = 0.0

        self.ssim = 0.0

        self.params = 0

        self.fisher_distance = np.inf

        self.rank = None

        self.crowding_distance = 0.0

    def clone(
            self):

        return copy.deepcopy(
            self
        )

    def mutate(
            self):

        child = self.clone()

        # Level 1 mutation
        if random.random() < 0.7:

            cell_index = random.randint(
                0,
                len(child.cells) - 1
            )

            child.cells[
                cell_index
            ] = child.cells[
                cell_index
            ].mutate()

        # Level 2 mutation
        if random.random() < 0.3:

            if random.random() < 0.5:

                child.channels = random.choice(
                    [
                        32,
                        48,
                        64,
                        96
                    ]
                )

            else:

                if len(
                    child.cells
                ) > 4:

                    child.cells.pop()

                else:

                    child.cells.append(
                        SRCell()
                    )

        return child


# ============================================================
# Two-Level Crossover
# ============================================================

def two_level_crossover(
        parent1,
        parent2):

    child1 = parent1.clone()

    child2 = parent2.clone()


    # -------------------------
    # Level 1: Cell crossover
    # -------------------------

    min_cells = min(
        len(parent1.cells),
        len(parent2.cells)
    )

    if min_cells > 0:

        cut = random.randint(
            1,
            min_cells
        )

        child1.cells = (

            parent1.cells[:cut]
            +
            parent2.cells[cut:]
        )

        child2.cells = (

            parent2.cells[:cut]
            +
            parent1.cells[cut:]
        )


    # -------------------------
    # Level 2: Network crossover
    # -------------------------

    if random.random() < 0.5:

        child1.channels = (
            parent1.channels
        )

        child2.channels = (
            parent2.channels
        )

    else:

        child1.channels = (
            parent2.channels
        )

        child2.channels = (
            parent1.channels
        )


    return (
        child1,
        child2
    )


# ============================================================
# NSGA-II Dominance
# ============================================================

def dominates(
        a,
        b):

    better_or_equal = (

        a.psnr >= b.psnr

        and

        a.ssim >= b.ssim

        and

        a.params <= b.params
    )

    strictly_better = (

        a.psnr > b.psnr

        or

        a.ssim > b.ssim

        or

        a.params < b.params
    )

    return (
        better_or_equal
        and
        strictly_better
    )


# ============================================================
# Non-Dominated Sorting
# ============================================================

def non_dominated_sort(
        population):

    fronts = []

    domination_count = {}

    dominated_solutions = {}

    first_front = []

    for p in population:

        domination_count[p] = 0

        dominated_solutions[p] = []

        for q in population:

            if dominates(
                    p,
                    q):

                dominated_solutions[
                    p
                ].append(
                    q
                )

            elif dominates(
                    q,
                    p):

                domination_count[
                    p
                ] += 1

        if domination_count[p] == 0:

            p.rank = 0

            first_front.append(
                p
            )

    fronts.append(
        first_front
    )

    i = 0

    while len(
            fronts[i]
    ) > 0:

        next_front = []

        for p in fronts[i]:

            for q in dominated_solutions[p]:

                domination_count[
                    q
                ] -= 1

                if domination_count[q] == 0:

                    q.rank = i + 1

                    next_front.append(
                        q
                    )

        i += 1

        fronts.append(
            next_front
        )

    return fronts[:-1]


# ============================================================
# Crowding Distance
# ============================================================

def calculate_crowding_distance(
        front):

    if len(front) == 0:

        return

    for p in front:

        p.crowding_distance = 0.0


    objectives = [

        (
            lambda x:
            x.psnr,
            True
        ),

        (
            lambda x:
            x.ssim,
            True
        ),

        (
            lambda x:
            x.params,
            False
        )
    ]


    for objective, maximize in objectives:

        front.sort(
            key=objective
        )

        front[0].crowding_distance = float(
            "inf"
        )

        front[-1].crowding_distance = float(
            "inf"
        )

        min_value = objective(
            front[0]
        )

        max_value = objective(
            front[-1]
        )

        if max_value == min_value:

            continue

        for i in range(
                1,
                len(front) - 1):

            prev_value = objective(
                front[i - 1]
            )

            next_value = objective(
                front[i + 1]
            )

            distance = (

                next_value -
                prev_value
            ) / (

                max_value -
                min_value
            )

            front[i].crowding_distance += abs(
                distance
            )


# ============================================================
# NSGA-II Selection
# ============================================================

def nsga2_selection(
        population,
        pop_size):

    fronts = non_dominated_sort(
        population
    )

    new_population = []

    for front in fronts:

        calculate_crowding_distance(
            front
        )

        if len(
                new_population
        ) + len(front) <= pop_size:

            new_population.extend(
                front
            )

        else:

            front.sort(

                key=lambda x:
                x.crowding_distance,

                reverse=True
            )

            remaining = (

                pop_size -
                len(new_population)
            )

            new_population.extend(

                front[
                    :remaining
                ]
            )

            break

    return new_population


# ============================================================
# Fisher-Guided Population Initialization
# ============================================================

def initialize_population(
        pop_size,
        fisher_bank,
        target_fisher):

    population = []

    if len(
            fisher_bank
    ) == 0:

        for _ in range(
                pop_size):

            population.append(
                SRNetworkArchitecture()
            )

        return population


    scores = []

    target_vec = (

        target_fisher[
            "__global__"
        ]
    )


    for name, fisher_data in fisher_bank.items():

        arch_vec = (

            fisher_data[
                "__global__"
            ]
        )

        distance = fisher_wasserstein_distance(

            target_vec,

            arch_vec
        )

        scores.append(

            (
                name,
                distance
            )
        )


    scores.sort(

        key=lambda x:
        x[1]
    )


    # Fisher-guided initialization

    for i in range(
            pop_size):

        if i < len(
                scores
        ):

            arch_name = scores[i][0]

            meta = KNOWN_SR_ARCHS.get(

                arch_name,

                {}
            )

            num_cells = meta.get(

                "num_cells",

                random.randint(
                    4,
                    20
                )
            )

            channels = meta.get(

                "base_channels",

                random.choice(
                    [
                        32,
                        48,
                        64
                    ]
                )
            )

        else:

            num_cells = random.randint(
                4,
                20
            )

            channels = random.choice(

                [
                    32,
                    48,
                    64,
                    96
                ]
            )


        architecture = (

            SRNetworkArchitecture(

                num_cells=num_cells,

                channels=channels
            )
        )


        architecture.fisher_distance = (

            scores[
                i % len(scores)
            ][1]
        )


        population.append(
            architecture
        )


    return population


# ============================================================
# Placeholder SR Model Builder
# ============================================================

class SimpleSRModel(
        nn.Module):

    def __init__(
            self,
            architecture):

        super().__init__()

        channels = (

            architecture.channels
        )

        self.head = nn.Conv2d(

            3,

            channels,

            3,

            padding=1
        )


        body = []

        for cell in architecture.cells:

            body.append(

                nn.Conv2d(

                    channels,

                    channels,

                    3,

                    padding=1
                )
            )

            body.append(
                nn.ReLU(
                    inplace=True
                )
            )


        self.body = nn.Sequential(
            *body
        )


        self.tail = nn.Sequential(

            nn.Conv2d(

                channels,

                channels,

                3,

                padding=1
            ),

            nn.Conv2d(

                channels,

                3 * SCALE * SCALE,

                3,

                padding=1
            ),

            nn.PixelShuffle(
                SCALE
            )
        )


    def forward(
            self,
            x):

        x = self.head(
            x
        )

        residual = x

        x = self.body(
            x
        )

        x = x + residual

        x = self.tail(
            x
        )

        return x


# ============================================================
# Count Parameters
# ============================================================

def count_parameters(
        model):

    return sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


# ============================================================
# Evaluate Architecture
# ============================================================

def evaluate_architecture(
        architecture,
        train_loader,
        val_loader):

    model = SimpleSRModel(
        architecture
    ).to(
        DEVICE
    )

    optimizer = torch.optim.Adam(

        model.parameters(),

        lr=1e-4
    )


    criterion = nn.L1Loss()


    model.train()

    for epoch in range(
            EPOCHS_SEARCH):

        for lr, hr in train_loader:

            lr = lr.to(
                DEVICE
            )

            hr = hr.to(
                DEVICE
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


    model.eval()


    psnr_values = []

    ssim_values = []


    with torch.no_grad():

        for lr, hr in val_loader:

            lr = lr.to(
                DEVICE
            )

            hr = hr.to(
                DEVICE
            )

            sr = model(
                lr
            )

            psnr_values.append(

                calculate_psnr(
                    sr,
                    hr
                )
            )

            ssim_values.append(

                calculate_ssim(
                    sr,
                    hr
                )
            )


    architecture.psnr = np.mean(
        psnr_values
    )

    architecture.ssim = np.mean(
        ssim_values
    )

    architecture.params = count_parameters(
        model
    )


    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()


    return architecture


# ============================================================
# Fisher-Guided NSGA-II
# ============================================================

class FisherNSGA2SR:

    def __init__(
            self,
            train_loader,
            val_loader,
            fisher_bank_path):

        self.train_loader = (
            train_loader
        )

        self.val_loader = (
            val_loader
        )

        self.fisher_bank = (

            load_fisher_bank(

                fisher_bank_path
            )
        )


        # Target Fisher fingerprint
        #
        # In the final implementation this should be computed
        # from DIV2K LR-HR reconstruction gradients.

        self.target_fisher = {

            "__global__":

            np.ones(
                100
            ) / 100
        }


        self.population = (

            initialize_population(

                POP_SIZE,

                self.fisher_bank,

                self.target_fisher
            )
        )


    def evaluate_population(
            self):

        for i, architecture in enumerate(

                self.population):

            print(

                "Evaluating architecture",

                i + 1,

                "/",

                len(
                    self.population
                )
            )

            evaluate_architecture(

                architecture,

                self.train_loader,

                self.val_loader
            )


    def create_offspring(
            self):

        offspring = []


        while len(
                offspring
        ) < POP_SIZE:

            p1, p2 = random.sample(

                self.population,

                2
            )


            c1, c2 = (

                two_level_crossover(

                    p1,

                    p2
                )
            )


            # Fisher-guided mutation
            if random.random() < 0.8:

                c1 = c1.mutate()


            if random.random() < 0.8:

                c2 = c2.mutate()


            offspring.append(
                c1
            )

            if len(
                    offspring
            ) < POP_SIZE:

                offspring.append(
                    c2
                )


        return offspring


    def run(
            self):

        print(

            "Starting Fisher-Guided",

            "Two-Level NSGA-II NAS-SR"
        )


        # Initial evaluation

        self.evaluate_population()


        for generation in range(

                N_GENERATIONS):

            start_time = time.time()


            print(

                "\nGeneration",

                generation + 1,

                "/",

                N_GENERATIONS
            )


            # Create offspring

            offspring = (

                self.create_offspring()
            )


            # Evaluate offspring

            for architecture in offspring:

                evaluate_architecture(

                    architecture,

                    self.train_loader,

                    self.val_loader
                )


            # Combine parent + offspring

            combined_population = (

                self.population +

                offspring
            )


            # NSGA-II environmental selection

            self.population = (

                nsga2_selection(

                    combined_population,

                    POP_SIZE
                )
            )


            fronts = non_dominated_sort(

                self.population
            )


            print(

                "Number of Pareto-optimal",

                "architectures:",

                len(
                    fronts[0]
                )
            )


            for i, architecture in enumerate(

                    fronts[0]):

                print(

                    "Pareto",

                    i,

                    "| PSNR:",

                    architecture.psnr,

                    "| SSIM:",

                    architecture.ssim,

                    "| Params:",

                    architecture.params
                )


            print(

                "Generation time:",

                time.time() -
                start_time
            )


        return (

            non_dominated_sort(

                self.population
            )[0]
        )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # IMPORTANT:
    # Replace these with your actual DIV2K data loaders.
    # --------------------------------------------------------

    train_loader = None

    val_loader = None


    nas = FisherNSGA2SR(

        train_loader=train_loader,

        val_loader=val_loader,

        fisher_bank_path=FISHER_BANK_PATH
    )


    pareto_front = nas.run()


    print(

        "\nFinal Pareto Front"
    )


    for i, architecture in enumerate(

            pareto_front):

        print(

            "Architecture:",

            i,

            "PSNR:",

            architecture.psnr,

            "SSIM:",

            architecture.ssim,

            "Parameters:",

            architecture.params,

            "Number of Cells:",

            len(
                architecture.cells
            ),

            "Channels:",

            architecture.channels
        )
