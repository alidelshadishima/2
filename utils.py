import numpy as np
from copy import deepcopy
from itertools import zip_longest


# ============================================================
# SR Search Space
# ============================================================

SR_OPERATIONS = [
    "conv_3x3",
    "conv_5x5",
    "sep_conv_3x3",
    "dilated_conv_3x3",
    "residual",
    "skip",
    "attention"
]


# ============================================================
# Create SR DAG Node
# ============================================================

def add_sr_node(nodes, max_channels=64):

    operation = np.random.choice(SR_OPERATIONS)

    channels = np.random.choice(
        [16, 24, 32, 48, 64]
    )

    node = {
        "type": operation,
        "channels": int(channels),
        "kernel": 3,
    }

    if operation == "conv_5x5":
        node["kernel"] = 5

    if operation == "dilated_conv_3x3":
        node["dilation"] = 2

    nodes.append(node)

    return nodes


# ============================================================
# Create SR Cell
# ============================================================

def create_sr_cell(
        num_nodes=4,
        max_channels=64):

    cell = []

    for _ in range(num_nodes):

        cell = add_sr_node(
            cell,
            max_channels
        )

    return cell


# ============================================================
# Difference between two SR architectures
# ============================================================

def difference_sr_architecture(
        p1,
        p2):

    diff = []

    for a, b in zip_longest(
            p1,
            p2):

        if a is not None and b is not None:

            if a["type"] == b["type"]:

                diff.append({
                    "type": "keep"
                })

            else:

                diff.append(
                    deepcopy(a)
                )

        elif a is not None:

            diff.append(
                deepcopy(a)
            )

        else:

            diff.append({
                "type": "remove"
            })

    return diff


# ============================================================
# Fisher-guided / NSGA architecture mutation
# ============================================================

def mutate_sr_architecture(
        architecture,
        mutation_prob=0.2):

    new_arch = deepcopy(
        architecture
    )

    for node in new_arch:

        if np.random.rand() < mutation_prob:

            node["type"] = np.random.choice(
                SR_OPERATIONS
            )

        if np.random.rand() < mutation_prob:

            node["channels"] = np.random.choice(
                [16, 24, 32, 48, 64]
            )

    return new_arch


# ============================================================
# Crossover
# ============================================================

def crossover_sr_architecture(
        parent1,
        parent2):

    child = []

    max_len = max(
        len(parent1),
        len(parent2)
    )

    for i in range(max_len):

        if i < len(parent1) and \
           i < len(parent2):

            if np.random.rand() < 0.5:

                child.append(
                    deepcopy(parent1[i])
                )

            else:

                child.append(
                    deepcopy(parent2[i])
                )

        elif i < len(parent1):

            child.append(
                deepcopy(parent1[i])
            )

        else:

            child.append(
                deepcopy(parent2[i])
            )

    return child


# ============================================================
# NSGA-II Dominance
# ============================================================

def dominates(obj_a, obj_b):

    """
    Objectives:

    obj[0] = PSNR      maximize
    obj[1] = SSIM      maximize
    obj[2] = Complexity minimize
    """

    psnr_better = obj_a[0] >= obj_b[0]
    ssim_better = obj_a[1] >= obj_b[1]
    complexity_better = obj_a[2] <= obj_b[2]

    strictly_better = (
        obj_a[0] > obj_b[0] or
        obj_a[1] > obj_b[1] or
        obj_a[2] < obj_b[2]
    )

    return (
        psnr_better and
        ssim_better and
        complexity_better and
        strictly_better
    )


# ============================================================
# Fast Non-Dominated Sorting
# ============================================================

def fast_non_dominated_sort(
        population):

    fronts = [[]]

    domination_count = {}

    dominated_solutions = {}

    for p in population:

        domination_count[id(p)] = 0

        dominated_solutions[id(p)] = []

        for q in population:

            if p is q:
                continue

            if dominates(
                    p.objectives,
                    q.objectives):

                dominated_solutions[
                    id(p)
                ].append(q)

            elif dominates(
                    q.objectives,
                    p.objectives):

                domination_count[
                    id(p)
                ] += 1

        if domination_count[id(p)] == 0:

            p.rank = 0

            fronts[0].append(p)

    i = 0

    while len(fronts[i]) > 0:

        next_front = []

        for p in fronts[i]:

            for q in dominated_solutions[
                    id(p)]:

                domination_count[
                    id(q)
                ] -= 1

                if domination_count[
                        id(q)] == 0:

                    q.rank = i + 1

                    next_front.append(q)

        i += 1

        fronts.append(
            next_front
        )

    return fronts[:-1]


# ============================================================
# Crowding Distance
# ============================================================

def crowding_distance(
        front):

    if len(front) == 0:

        return

    for p in front:

        p.crowding_distance = 0.0

    num_objectives = 3

    for m in range(
            num_objectives):

        front.sort(
            key=lambda x:
                x.objectives[m]
        )

        front[0].crowding_distance = float(
            "inf"
        )

        front[-1].crowding_distance = float(
            "inf"
        )

        min_value = front[0].objectives[m]
        max_value = front[-1].objectives[m]

        if max_value == min_value:

            continue

        for i in range(
                1,
                len(front) - 1):

            previous_value = \
                front[i - 1].objectives[m]

            next_value = \
                front[i + 1].objectives[m]

            distance = (
                next_value -
                previous_value
            ) / (
                max_value -
                min_value
            )

            front[i].crowding_distance += \
                abs(distance)


# ============================================================
# NSGA-II Selection
# ============================================================

def nsga_selection(
        population,
        pop_size):

    fronts = fast_non_dominated_sort(
        population
    )

    new_population = []

    for front in fronts:

        crowding_distance(
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
                front[:remaining]
            )

            break

    return new_population


# ============================================================
# Generate offspring
# ============================================================

def generate_offspring(
        population,
        mutation_prob=0.2):

    offspring = []

    while len(
            offspring
    ) < len(population):

        parent1 = np.random.choice(
            population
        )

        parent2 = np.random.choice(
            population
        )

        child_arch = crossover_sr_architecture(
            parent1.layers,
            parent2.layers
        )

        child_arch = mutate_sr_architecture(
            child_arch,
            mutation_prob
        )

        child = deepcopy(
            parent1
        )

        child.layers = child_arch

        offspring.append(
            child
        )

    return offspring
