```python
import os
import time
import numpy as np

import matplotlib

# Use non-interactive backend
matplotlib.use('Agg')

import matplotlib.pyplot as plt


# ============================================================
# TWO-LEVEL NSGA-II-NAS-SR
#
# Single Image Super-Resolution
# Dataset: DIV2K
#
# Level 1:
# Global Architecture Search
#
# Level 2:
# Local Exploitation / Refinement
#
# Objectives:
#   1. Maximize PSNR
#   2. Maximize SSIM
#   3. Minimize Number of Parameters
#
# ============================================================


if __name__ == '__main__':

    # ========================================================
    # General Configuration
    # ========================================================

    dataset = "DIV2K"

    scale = 4


    # ========================================================
    # DIV2K Paths
    # ========================================================

    data_root = "./data"

    train_dir = os.path.join(
        data_root,
        "DIV2K_train_HR"
    )

    valid_dir = os.path.join(
        data_root,
        "DIV2K_valid_HR"
    )


    # ========================================================
    # Results Directory
    # ========================================================

    results_path = (

        "./results/"
        +
        dataset
        +
        "/NSGA2-TwoLevel/x"
        +
        str(scale)
        +
        "/"
    )


    os.makedirs(
        results_path,
        exist_ok=True
    )


    # ========================================================
    # Number of Independent Runs
    # ========================================================

    number_runs = 10


    # ========================================================
    # ========================================================
    # LEVEL 1
    # GLOBAL SEARCH
    # ========================================================
    # ========================================================

    level1_generations = 30

    level1_population_size = 40

    level1_batch_size = 16

    level1_epochs = 1


    # ========================================================
    # LEVEL 2
    # LOCAL SEARCH
    # ========================================================

    level2_generations = 20

    level2_population_size = 20

    level2_batch_size = 8

    level2_epochs = 3


    # ========================================================
    # Training Patch
    # ========================================================

    lr_patch_size = 48

    hr_patch_size = (

        lr_patch_size
        *
        scale
    )


    # ========================================================
    # Architecture Search Space
    # ========================================================

    min_layer = 4

    max_layer = 20


    # Maximum feature channels

    max_conv_output_channels = 256


    # ========================================================
    # SR Operation Search Space
    # ========================================================

    # No Fully Connected layers
    #
    # No Pooling by default
    #
    # because spatial resolution is important in SR.

    operations = [

        "conv3x3",

        "conv5x5",

        "sepconv3x3",

        "dilated_conv3x3",

        "residual_block",

        "skip_connection",

        "channel_attention",

        "pixel_attention"

    ]


    # ========================================================
    # Probability of Operations
    # ========================================================

    operation_probabilities = {

        "conv3x3":
        0.25,

        "conv5x5":
        0.10,

        "sepconv3x3":
        0.15,

        "dilated_conv3x3":
        0.10,

        "residual_block":
        0.15,

        "skip_connection":
        0.10,

        "channel_attention":
        0.10,

        "pixel_attention":
        0.05

    }


    # ========================================================
    # NSGA-II Parameters
    # ========================================================

    crossover_probability = 0.9

    mutation_probability_level1 = 0.20

    mutation_probability_level2 = 0.05


    # ========================================================
    # Objective Weights
    #
    # Actual NSGA-II uses Pareto dominance.
    #
    # These weights can be used only for final selection.
    # ========================================================

    psnr_weight = 0.5

    ssim_weight = 0.3

    complexity_weight = 0.2


    # ========================================================
    # Loss Function
    # ========================================================

    loss_function = "MSE"


    # ========================================================
    # Dropout
    # ========================================================

    dropout = 0.0


    # ========================================================
    # Results Arrays
    # ========================================================

    # Columns:
    #
    # 0 = PSNR
    # 1 = SSIM
    # 2 = Parameters

    all_best_metrics = np.zeros(
        (
            number_runs,
            3
        )
    )


    all_running_times = []

    all_best_parameters = []

    all_pareto_fronts = []


    # ========================================================
    # Global Best
    # ========================================================

    global_best_psnr = -np.inf

    global_best_ssim = -np.inf

    global_best_model = None

    global_best_architecture = None


    # ========================================================
    # Independent Runs
    # ========================================================

    for run in range(
        number_runs
    ):

        print(
            "\n"
            +
            "=" * 70
        )

        print(
            "Two-Level NSGA-II-NAS-SR"
        )

        print(
            "Run:",
            run + 1,
            "/",
            number_runs
        )

        print(
            "=" * 70
        )


        start_time = time.time()


        # ====================================================
        # ====================================================
        # LEVEL 1
        # GLOBAL SEARCH
        # ====================================================
        # ====================================================

        print(
            "\n"
            +
            "=" * 60
        )

        print(
            "LEVEL 1: GLOBAL ARCHITECTURE SEARCH"
        )

        print(
            "=" * 60
        )


        nsga2_level1 = NSGA2NASSR(

            dataset=dataset,

            data_root=data_root,

            train_dir=train_dir,

            valid_dir=valid_dir,


            # ------------------------------------------------
            # SR
            # ------------------------------------------------

            scale=scale,

            lr_patch_size=lr_patch_size,

            hr_patch_size=hr_patch_size,


            # ------------------------------------------------
            # NSGA-II
            # ------------------------------------------------

            population_size=
            level1_population_size,

            generations=
            level1_generations,


            # ------------------------------------------------
            # Architecture
            # ------------------------------------------------

            min_layer=min_layer,

            max_layer=max_layer,

            max_out_ch=
            max_conv_output_channels,


            # ------------------------------------------------
            # Search Space
            # ------------------------------------------------

            operations=
            operations,

            operation_probabilities=
            operation_probabilities,


            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            batch_size=
            level1_batch_size,

            epochs=
            level1_epochs,


            # ------------------------------------------------
            # NSGA-II
            # ------------------------------------------------

            crossover_probability=
            crossover_probability,

            mutation_probability=
            mutation_probability_level1,


            # ------------------------------------------------
            # Objectives
            # ------------------------------------------------

            objectives=[
                "PSNR",
                "SSIM",
                "PARAMETERS"
            ],


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss_function=
            loss_function
        )


        # ====================================================
        # Run Level 1
        # ====================================================

        nsga2_level1.run()


        # ====================================================
        # Level 1 Pareto Front
        # ====================================================

        pareto_level1 = (

            nsga2_level1.get_pareto_front()
        )


        print(
            "\nLevel 1 Pareto Front Size:",
            len(
                pareto_level1
            )
        )


        # ====================================================
        # Save Level 1 Pareto Front
        # ====================================================

        np.save(

            results_path
            +
            "run_"
            +
            str(run)
            +
            "_level1_pareto.npy",

            np.array(
                pareto_level1,
                dtype=object
            ),

            allow_pickle=True
        )


        # ====================================================
        # ====================================================
        # LEVEL 2
        # LOCAL SEARCH
        # ====================================================
        # ====================================================

        print(
            "\n"
            +
            "=" * 60
        )

        print(
            "LEVEL 2: LOCAL PARETO REFINEMENT"
        )

        print(
            "=" * 60
        )


        # ====================================================
        # Select Elite Architectures
        # ====================================================

        elite_architectures = (

            nsga2_level1.select_elite_architectures(

                pareto_front=
                pareto_level1,

                max_elite=
                level2_population_size
            )
        )


        print(
            "Number of Elite Architectures:",
            len(
                elite_architectures
            )
        )


        # ====================================================
        # Create Level 2 NSGA-II
        # ====================================================

        nsga2_level2 = NSGA2NASSR(

            dataset=dataset,

            data_root=data_root,

            train_dir=train_dir,

            valid_dir=valid_dir,


            # ------------------------------------------------
            # SR
            # ------------------------------------------------

            scale=scale,

            lr_patch_size=lr_patch_size,

            hr_patch_size=hr_patch_size,


            # ------------------------------------------------
            # Initial Population
            # ------------------------------------------------

            initial_population=
            elite_architectures,


            # ------------------------------------------------
            # NSGA-II
            # ------------------------------------------------

            population_size=
            level2_population_size,

            generations=
            level2_generations,


            # ------------------------------------------------
            # Architecture
            # ------------------------------------------------

            min_layer=min_layer,

            max_layer=max_layer,

            max_out_ch=
            max_conv_output_channels,


            # ------------------------------------------------
            # Search Space
            # ------------------------------------------------

            operations=
            operations,

            operation_probabilities=
            operation_probabilities,


            # ------------------------------------------------
            # Training
            # ------------------------------------------------

            batch_size=
            level2_batch_size,

            epochs=
            level2_epochs,


            # ------------------------------------------------
            # Local Search
            # ------------------------------------------------

            crossover_probability=
            crossover_probability,

            mutation_probability=
            mutation_probability_level2,


            # ------------------------------------------------
            # Objectives
            # ------------------------------------------------

            objectives=[
                "PSNR",
                "SSIM",
                "PARAMETERS"
            ],


            # ------------------------------------------------
            # Loss
            # ------------------------------------------------

            loss_function=
            loss_function
        )


        # ====================================================
        # Run Level 2
        # ====================================================

        nsga2_level2.run()


        # ====================================================
        # Final Pareto Front
        # ====================================================

        final_pareto_front = (

            nsga2_level2.get_pareto_front()
        )


        print(
            "\nFinal Pareto Front Size:",
            len(
                final_pareto_front
            )
        )


        # ====================================================
        # Save Final Pareto Front
        # ====================================================

        np.save(

            results_path
            +
            "run_"
            +
            str(run)
            +
            "_final_pareto.npy",

            np.array(
                final_pareto_front,
                dtype=object
            ),

            allow_pickle=True
        )


        all_pareto_fronts.append(

            final_pareto_front
        )


        # ====================================================
        # Select Final Architecture
        # ====================================================

        best_candidate = (

            nsga2_level2.select_best_compromise(

                pareto_front=
                final_pareto_front,

                psnr_weight=
                psnr_weight,

                ssim_weight=
                ssim_weight,

                complexity_weight=
                complexity_weight
            )
        )


        # ====================================================
        # Full Training of Final Architecture
        # ====================================================

        print(
            "\n"
            +
            "=" * 60
        )

        print(
            "FINAL FULL TRAINING"
        )

        print(
            "=" * 60
        )


        n_parameters = (

            nsga2_level2.fit_final_architecture(

                architecture=
                best_candidate,

                batch_size=
                level2_batch_size,

                epochs=
                300
            )
        )


        # ====================================================
        # Final Evaluation
        # ====================================================

        final_metrics = (

            nsga2_level2.evaluate_architecture(

                architecture=
                best_candidate,

                batch_size=
                level2_batch_size
            )
        )


        final_psnr = (

            final_metrics[
                "PSNR"
            ]
        )


        final_ssim = (

            final_metrics[
                "SSIM"
            ]
        )


        final_parameters = (

            final_metrics.get(

                "PARAMETERS",

                n_parameters
            )
        )


        # ====================================================
        # Print Results
        # ====================================================

        print(
            "\n"
            +
            "=" * 60
        )

        print(
            "FINAL RESULTS"
        )

        print(
            "=" * 60
        )

        print(
            "PSNR:",
            final_psnr,
            "dB"
        )

        print(
            "SSIM:",
            final_ssim
        )

        print(
            "Parameters:",
            final_parameters
        )


        # ====================================================
        # Store Results
        # ====================================================

        all_best_metrics[
            run,
            0
        ] = final_psnr


        all_best_metrics[
            run,
            1
        ] = final_ssim


        all_best_metrics[
            run,
            2
        ] = final_parameters


        all_best_parameters.append(

            final_parameters
        )


        # ====================================================
        # Update Global Best
        # ====================================================

        if (

            final_psnr
            >
            global_best_psnr

        ):

            global_best_psnr = (

                final_psnr
            )


            global_best_ssim = (

                final_ssim
            )


            global_best_architecture = (

                best_candidate
            )


            # ------------------------------------------------
            # Save Best Architecture
            # ------------------------------------------------

            np.save(

                results_path
                +
                "BEST_ARCHITECTURE.npy",

                np.array(

                    best_candidate,

                    dtype=object
                ),

                allow_pickle=True
            )


            # ------------------------------------------------
            # Save Best Model
            # ------------------------------------------------

            if hasattr(

                nsga2_level2,

                "save_model"
            ):

                nsga2_level2.save_model(

                    results_path
                    +
                    "BEST_NAS_SR_MODEL"
                )


        # ====================================================
        # Runtime
        # ====================================================

        end_time = time.time()


        running_time = (

            end_time
            -
            start_time
        )


        all_running_times.append(

            running_time
        )


        print(
            "\nRun Time:",
            running_time,
            "seconds"
        )


        # ====================================================
        # Save Intermediate Results
        # ====================================================

        np.save(

            results_path
            +
            "all_best_metrics.npy",

            all_best_metrics
        )


        np.save(

            results_path
            +
            "all_running_times.npy",

            np.array(
                all_running_times
            )
        )


        np.save(

            results_path
            +
            "all_parameters.npy",

            np.array(
                all_best_parameters
            )
        )


        # ====================================================
        # Save Current Results
        # ====================================================

        current_mean = (

            np.mean(

                all_best_metrics[
                    :run + 1
                ],

                axis=0
            )
        )


        output_str = (

            "Dataset: DIV2K\n"

            +

            "Scale: x"
            +
            str(
                scale
            )
            +
            "\n"

            +

            "Algorithm: Two-Level NSGA-II-NAS-SR\n"

            +

            "Level 1 Generations: "
            +
            str(
                level1_generations
            )
            +
            "\n"

            +

            "Level 2 Generations: "
            +
            str(
                level2_generations
            )
            +
            "\n"

            +

            "Final PSNR: "
            +
            str(
                final_psnr
            )
            +
            " dB\n"

            +

            "Final SSIM: "
            +
            str(
                final_ssim
            )
            +
            "\n"

            +

            "Final Parameters: "
            +
            str(
                final_parameters
            )
            +
            "\n"

            +

            "Mean PSNR: "
            +
            str(
                current_mean[
                    0
                ]
            )
            +
            " dB\n"

            +

            "Mean SSIM: "
            +
            str(
                current_mean[
                    1
                ]
            )
            +
            "\n"

            +

            "Mean Parameters: "
            +
            str(
                current_mean[
                    2
                ]
            )
            +
            "\n"

            +

            "Running Times: "
            +
            str(
                all_running_times
            )
            +
            "\n"
        )


        print(
            "\n"
            +
            output_str
        )


        with open(

            results_path
            +
            "final_results.txt",

            "w"
        ) as f:

            f.write(
                output_str
            )


    # ========================================================
    # End of All Runs
    # ========================================================

    print(
        "\n"
        +
        "=" * 70
    )

    print(
        "ALL RUNS COMPLETED"
    )

    print(
        "=" * 70
    )


    # ========================================================
    # Final Statistics
    # ========================================================

    mean_psnr = np.mean(

        all_best_metrics[
            :,
            0
        ]
    )


    mean_ssim = np.mean(

        all_best_metrics[
            :,
            1
        ]
    )


    mean_parameters = np.mean(

        all_best_metrics[
            :,
            2
        ]
    )


    std_psnr = np.std(

        all_best_metrics[
            :,
            0
        ]
    )


    std_ssim = np.std(

        all_best_metrics[
            :,
            1
        ]
    )


    print(
        "\nFinal Statistics:"
    )


    print(
        "Mean PSNR:",
        mean_psnr,
        "+/-",
        std_psnr
    )


    print(
        "Mean SSIM:",
        mean_ssim,
        "+/-",
        std_ssim
    )


    print(
        "Mean Parameters:",
        mean_parameters
    )


    # ========================================================
    # Save Final Statistics
    # ========================================================

    final_output = (

        "========================================\n"

        "TWO-LEVEL NSGA-II-NAS-SR RESULTS\n"

        "========================================\n"

        "Dataset: DIV2K\n"

        "Scale: x"
        +
        str(
            scale
        )
        +
        "\n"

        "Mean PSNR: "
        +
        str(
            mean_psnr
        )
        +
        " +/- "
        +
        str(
            std_psnr
        )
        +
        " dB\n"

        "Mean SSIM: "
        +
        str(
            mean_ssim
        )
        +
        " +/- "
        +
        str(
            std_ssim
        )
        +
        "\n"

        "Mean Parameters: "
        +
        str(
            mean_parameters
        )
        +
        "\n"

        "Best PSNR: "
        +
        str(
            global_best_psnr
        )
        +
        " dB\n"

        "Best SSIM: "
        +
        str(
            global_best_ssim
        )
        +
        "\n"

        "========================================\n"
    )


    with open(

        results_path
        +
        "FINAL_STATISTICS.txt",

        "w"
    ) as f:

        f.write(
            final_output
        )


    print(
        final_output
    )
```
