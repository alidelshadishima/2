"""
population.py

Fisher-Guided Population Initialization
for Two-Level NSGA-II-NAS-SR

Dataset:
    DIV2K

Task:
    Single Image Super-Resolution (SISR)

Search:
    Level 1 -> Macro Architecture
    Level 2 -> Micro / DAG Cell Operations

Initialization:
    Fisher Information Similarity

Optimization:
    NSGA-II

Objectives:
    1. Maximize PSNR
    2. Minimize Number of Parameters
    3. Optionally Minimize FLOPs
"""

import os
import random
import numpy as np

from copy import deepcopy


# ============================================================
# Import SR architecture and Fisher utilities
# ============================================================

from fisher_guided_particle import (
    TwoLevelArchitecture,
    NAS_SR_Network,
    fisher_fingerprint,
    fisher_wasserstein_distance
)


# ============================================================
# DIV2K SR Architecture Bank
# ============================================================

KNOWN_SR_ARCHS = {

    "sr_residual_small":
        "graphs/sr_residual_small_dag.json",

    "sr_residual_medium":
        "graphs/sr_residual_medium_dag.json",

    "sr_residual_large":
        "graphs/sr_residual_large_dag.json",

    "sr_dense":
        "graphs/sr_dense_dag.json",

    "sr_attention":
        "graphs/sr_attention_dag.json",

    "sr_depthwise":
        "graphs/sr_depthwise_dag.json"
}


# ============================================================
# Architecture Individual
# ============================================================

class SRIndividual:

    def __init__(
        self,
        architecture,
        device="cuda"
    ):

        self.architecture = architecture

        self.device = device

        self.model = None

        # NSGA-II objective values

        self.psnr = None

        self.ssim = None

        self.num_parameters = None

        self.flops = None

        # NSGA-II fields

        self.rank = None

        self.crowding_distance = 0.0

        self.dominated_solutions = []

        self.domination_count = 0

        self.fitness = None

    # --------------------------------------------------------
    # Build PyTorch SR model
    # --------------------------------------------------------

    def build_model(self):

        self.model = NAS_SR_Network(

            architecture=self.architecture,

            in_channels=3,

            out_channels=3

        ).to(self.device)

        return self.model

    # --------------------------------------------------------
    # Clone individual
    # --------------------------------------------------------

    def clone(self):

        new_individual = SRIndividual(

            architecture=deepcopy(
                self.architecture
            ),

            device=self.device

        )

        return new_individual

    # --------------------------------------------------------
    # Architecture description
    # --------------------------------------------------------

    def get_architecture(self):

        return self.architecture

    def __repr__(self):

        return (

            "SRIndividual("

            f"cells={self.architecture.num_cells}, "

            f"channels={self.architecture.channels}, "

            f"scale={self.architecture.scale}, "

            f"PSNR={self.psnr}, "

            f"Params={self.num_parameters}"

            ")"

        )


# ============================================================
# Population
# ============================================================

class Population:

    def __init__(
        self,
        pop_size,
        fisher_matrix,
        fisher_bank,
        input_width=48,
        input_height=48,
        input_channels=3,
        output_channels=3,
        scale=2,
        device="cuda",
        k_nearest=3
    ):

        """
        Fisher-Guided Initial Population
        for Two-Level NSGA-II-NAS-SR.

        Parameters
        ----------
        pop_size:
            Number of individuals.

        fisher_matrix:
            Fisher Information of target DIV2K task.

        fisher_bank:
            Fisher fingerprints of known SR architectures.

        input_width:
            LR input width.

        input_height:
            LR input height.

        input_channels:
            Number of LR channels.

        output_channels:
            Number of HR output channels.

        scale:
            Super-resolution scale factor.

        device:
            cuda or cpu.

        k_nearest:
            Number of nearest architectures selected
            using Fisher similarity.
        """

        self.pop_size = pop_size

        self.fisher_matrix = fisher_matrix

        self.fisher_bank = fisher_bank

        self.input_width = input_width

        self.input_height = input_height

        self.input_channels = input_channels

        self.output_channels = output_channels

        self.scale = scale

        self.device = device

        self.k_nearest = k_nearest

        self.individuals = []

        # ----------------------------------------------------
        # Step 1
        # Compute Fisher fingerprint of target task
        # ----------------------------------------------------

        self.target_fisher_vector = (

            fisher_fingerprint(

                self.fisher_matrix

            )

        )

        # ----------------------------------------------------
        # Step 2
        # Find Fisher-similar architectures
        # ----------------------------------------------------

        self.architecture_scores = (

            self.compute_fisher_similarity()

        )

        # ----------------------------------------------------
        # Step 3
        # Select nearest architectures
        # ----------------------------------------------------

        self.nearest_architectures = (

            self.select_nearest_architectures()

        )

        # ----------------------------------------------------
        # Step 4
        # Build initial NSGA-II population
        # ----------------------------------------------------

        self.initialize_population()

    # ========================================================
    # Fisher Similarity
    # ========================================================

    def compute_fisher_similarity(self):

        scores = []

        for (
            architecture_name,
            architecture_fisher
        ) in self.fisher_bank.items():

            # -----------------------------------------------
            # Convert architecture Fisher to fingerprint
            # -----------------------------------------------

            arch_vector = (

                fisher_fingerprint(

                    architecture_fisher

                )

            )

            # -----------------------------------------------
            # Wasserstein Fisher distance
            # -----------------------------------------------

            distance = (

                fisher_wasserstein_distance(

                    self.target_fisher_vector,

                    arch_vector

                )

            )

            scores.append(

                (
                    architecture_name,
                    distance
                )

            )

        # Smaller distance = higher similarity

        scores.sort(

            key=lambda x: x[1]

        )

        return scores

    # ========================================================
    # Select Fisher Nearest Architectures
    # ========================================================

    def select_nearest_architectures(self):

        nearest = (

            self.architecture_scores[

                :self.k_nearest

            ]

        )

        print(
            "\nFisher-guided SR architectures:"
        )

        for name, distance in nearest:

            print(

                f"{name} "
                f"-> Fisher distance = "
                f"{distance:.6f}"

            )

        return nearest

    # ========================================================
    # Create Two-Level Architecture
    # ========================================================

    def create_architecture_from_template(
        self,
        architecture_name
    ):

        """
        Convert known SR architecture
        into a two-level searchable architecture.

        Level 1:
            Macro:
                - number of cells
                - number of channels

        Level 2:
            Micro:
                - operations in each DAG cell
        """

        # ----------------------------------------------------
        # Default SR architecture parameters
        # ----------------------------------------------------

        if architecture_name == "sr_residual_small":

            num_cells = 6

            channels = 32

            num_nodes = 4

        elif architecture_name == "sr_residual_medium":

            num_cells = 12

            channels = 64

            num_nodes = 4

        elif architecture_name == "sr_residual_large":

            num_cells = 20

            channels = 96

            num_nodes = 5

        elif architecture_name == "sr_dense":

            num_cells = 12

            channels = 64

            num_nodes = 5

        elif architecture_name == "sr_attention":

            num_cells = 16

            channels = 64

            num_nodes = 5

        elif architecture_name == "sr_depthwise":

            num_cells = 10

            channels = 48

            num_nodes = 4

        else:

            # Random fallback

            num_cells = random.randint(
                6,
                16
            )

            channels = random.choice(

                [
                    32,
                    48,
                    64,
                    96
                ]

            )

            num_nodes = 4

        # ----------------------------------------------------
        # Build two-level architecture
        # ----------------------------------------------------

        architecture = (

            TwoLevelArchitecture(

                num_cells=num_cells,

                channels=channels,

                num_nodes=num_nodes,

                scale=self.scale

            )

        )

        return architecture

    # ========================================================
    # Fisher-Guided Initial Population
    # ========================================================

    def initialize_population(self):

        """
        Initialize population around
        Fisher-similar SR architectures.

        The population is not purely random.

        Fisher-similar architectures have higher
        probability of being selected.
        """

        if len(
            self.nearest_architectures
        ) == 0:

            raise RuntimeError(

                "No Fisher-similar "
                "architectures found."

            )

        # ----------------------------------------------------
        # Generate population
        # ----------------------------------------------------

        for i in range(
            self.pop_size
        ):

            # -----------------------------------------------
            # Select one Fisher-nearest architecture
            # -----------------------------------------------

            architecture_name, distance = (

                random.choice(

                    self.nearest_architectures

                )

            )

            # -----------------------------------------------
            # Build two-level architecture
            # -----------------------------------------------

            architecture = (

                self.create_architecture_from_template(

                    architecture_name

                )

            )

            # -----------------------------------------------
            # Create NSGA-II individual
            # -----------------------------------------------

            individual = (

                SRIndividual(

                    architecture=architecture,

                    device=self.device

                )

            )

            # -----------------------------------------------
            # Build SR model
            # -----------------------------------------------

            individual.build_model()

            # -----------------------------------------------
            # Save Fisher initialization information
            # -----------------------------------------------

            individual.fisher_source = (

                architecture_name

            )

            individual.fisher_distance = (

                distance

            )

            # -----------------------------------------------
            # Add to population
            # -----------------------------------------------

            self.individuals.append(

                individual

            )

        print(

            f"\nInitial population size: "
            f"{len(self.individuals)}"

        )

    # ========================================================
    # Get Population
    # ========================================================

    def get_population(self):

        return self.individuals

    # ========================================================
    # Get Best Fisher Initialization
    # ========================================================

    def get_best_fisher_individual(self):

        if len(
            self.individuals
        ) == 0:

            return None

        return min(

            self.individuals,

            key=lambda x:
                x.fisher_distance

        )


# ============================================================
# Example
# ============================================================

if __name__ == "__main__":

    device = (

        "cuda"

        if __import__(
            "torch"
        ).cuda.is_available()

        else "cpu"

    )

    # --------------------------------------------------------
    # Load target Fisher matrix
    # --------------------------------------------------------

    target_fisher = np.load(

        "fisher/div2k_target_fisher.npy",

        allow_pickle=True

    ).item()

    # --------------------------------------------------------
    # Load Fisher bank
    # --------------------------------------------------------

    fisher_bank = np.load(

        "fisher/div2k_sr_fisher_bank.npy",

        allow_pickle=True

    ).item()

    # --------------------------------------------------------
    # Create Fisher-Guided Population
    # --------------------------------------------------------

    population = Population(

        pop_size=20,

        fisher_matrix=target_fisher,

        fisher_bank=fisher_bank,

        input_width=48,

        input_height=48,

        input_channels=3,

        output_channels=3,

        scale=2,

        device=device,

        k_nearest=3

    )

    # --------------------------------------------------------
    # Print population
    # --------------------------------------------------------

    print(
        "\nInitial NSGA-II Population:"
    )

    for i, individual in enumerate(

        population.get_population()

    ):

        print(

            f"Individual {i}:",

            individual

        )

        print(

            "  Fisher source:",

            individual.fisher_source

        )

        print(

            "  Fisher distance:",

            individual.fisher_distance

        )
