"""
fisher_pso_population.py

Fisher-guided PSO-NAS:
- Initial population seeded from known-architecture bank using Fisher fingerprint similarity.
- In each iteration compute task Fisher (few calibration batches).
- Use Fisher distances + an annealing schedule to control exploration/exploitation.
- Skip expensive training for particles that are too similar to current gBest.
- Architectures represented as DAG/cell; particles are constructed from DAG layers.
"""

import os
import numpy as np
from copy import deepcopy
from scipy.stats import wasserstein_distance
import random
import math
import time

# Placeholders for your project's modules - adapt as needed
import utils          # must implement: load_fisher_bank, load_dag, build_architecture_from_dag, fisher_distance_vectorized, computeVelocity, updateParticle
from particle import Particle  # must implement Particle.from_layers(...) and regular velocity/update methods
from data_loader import calibration_loader_for_dataset, full_train_test_loaders  # adapt to your project
from torch.utils.data import DataLoader

# ---------------------------
# Configurable parameters
# ---------------------------
KNOWN_ARCHS = {   # map arch name -> dag json path (you must provide files)
    "resnet18": "graphs/resnet18_dag.json",
    "vgg16": "graphs/vgg16_dag.json",
    "densenet121": "graphs/densenet121_dag.json",
    "mobilenet": "graphs/mobilenet_dag.json",
}

FISHER_BANK_PATH = "fisher_bank.npz"   # file with arch_name -> fisher fingerprint
K_NEAREST = 3                          # how many nearest known archs to fuse per particle initialisation

# PSO / search hyper-params
POP_SIZE = 8
N_ITER = 20
EPOCHS = 8               # number of epochs for full training when we decide to train
CALIB_BATCHES = 4        # number of batches used to compute task Fisher each iter (cheap)
TRAIN_IF_NOVEL = True    # whether to train only novel particles
INIT_TEMPERATURE = 1.0   # initial "temperature" (controls exploration)
FINAL_TEMPERATURE = 0.05 # final temperature (more exploitation)
NOVELTY_START = 0.5      # initial novelty threshold (higher -> easier to be considered novel)
NOVELTY_END = 0.05       # final novelty threshold (lower -> require small distance to be novel)
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


# ---------------------------
# Helpers: Fisher operations
# ---------------------------
def load_fisher_bank(path):
    """
    Expect .npz with keys = arch_name, values = dict-like with "__global__" vector or a single vector saved.
    """
    arr = np.load(path, allow_pickle=True)
    bank = {}
    for k in arr.files:
        v = arr[k].item() if arr[k].dtype == np.object_ else arr[k]
        # normalize to sum to 1 (if numeric vector)
        if isinstance(v, dict) and "__global__" in v:
            vec = v["__global__"]
            vec = vec.astype(np.float64)
            vec = vec / (vec.sum() + 1e-12)
            bank[k] = {"__global__": vec}
        elif isinstance(v, np.ndarray):
            vec = v.astype(np.float64); vec = vec / (vec.sum() + 1e-12)
            bank[k] = {"__global__": vec}
        else:
            # fallback: try to convert
            try:
                vec = np.asarray(v, dtype=np.float64)
                vec = vec / (vec.sum() + 1e-12)
                bank[k] = {"__global__": vec}
            except Exception:
                bank[k] = v
    return bank

def compute_task_fisher_profile_from_calib(model_builder, calib_loader, max_batches=4):
    """
    Compute a cheap task Fisher fingerprint using a small probe model (Keras/Torch).
    For clarity we expect model_builder() -> a model that accepts batches from calib_loader.
    We'll compute diagonal fisher as average squared grads per trainable parameter and return a flattened global vector.
    NOTE: This helper is a placeholder. You should implement it consistent with your training framework (PyTorch/TensorFlow).
    """
    # Placeholder: if you have prebuilt function utils.compute_task_fisher use it.
    # Here we try to call utils.compute_task_fisher if exists.
    if hasattr(utils, "compute_task_fisher"):
        return utils.compute_task_fisher(model_builder, calib_loader, max_batches=max_batches)
    # Otherwise implement a simple approximate fingerprint:
    # We'll run a forward/backward using PyTorch if the model is PyTorch.
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_builder().to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    accum = None
    batches = 0
    for xb, yb in calib_loader:
        xb = xb.to(device)
        yb = yb.to(device).long()
        model.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        # collect squared grads
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach().abs().mean().cpu().numpy()
                grads.append(np.array([g]))
        vec = np.concatenate(grads)
        if accum is None:
            accum = vec
        else:
            accum += vec
        batches += 1
        if batches >= max_batches:
            break
    if accum is None:
        accum = np.ones(100) * 1e-6  # fallback tiny fingerprint
    accum = accum / (accum.sum() + 1e-12)
    return {"__global__": accum}


def fisher_wasserstein_distance(v1, v2):
    """
    1D Wasserstein between two probability vectors on indices.
    v1, v2: normalized arrays (sum to 1)
    """
    n1, n2 = len(v1), len(v2)
    m = min(n1, n2)
    if m == 0:
        return float('inf')
    a = v1[:m] / (v1[:m].sum() + 1e-12)
    b = v2[:m] / (v2[:m].sum() + 1e-12)
    return wasserstein_distance(np.arange(m), np.arange(m), u_weights=a, v_weights=b)


# ---------------------------
# Particle initialisation using arch bank + fisher bank
# ---------------------------
def initialize_population_from_bank(pop_size, fisher_bank, arch_bank, task_profile, input_shape, output_dim, k_nearest=3):
    """
    Create pop_size particles by fusing nearest-known-architectures to the task fingerprint.
    Returns list of Particle objects.
    """
    arch_scores = []
    for arch_name, arch_fp in fisher_bank.items():
        # reduce arch fingerprint to single vector
        arch_vec = arch_fp.get("__global__")
        if arch_vec is None:
            # if per-edge, concat
            if isinstance(arch_fp, dict):
                pieces = []
                for k, v in arch_fp.items():
                    pieces.append(v.flatten())
                arch_vec = np.concatenate(pieces)
            else:
                arch_vec = np.asarray(arch_fp).flatten()
        arch_vec = arch_vec.astype(np.float64); arch_vec = arch_vec / (arch_vec.sum() + 1e-12)
        # compare to task_profile global
        tv = task_profile["__global__"]
        d = fisher_wasserstein_distance(arch_vec, tv)
        arch_scores.append((arch_name, float(d)))
    arch_scores.sort(key=lambda x: x[1])
    nearest_names = [a for a,_ in arch_scores[:max(k_nearest, pop_size)]]

    particles = []
    # create particles: fuse different combinations to increase diversity
    for i in range(pop_size):
        # choose k nearest for this particle (circular shift to vary)
        chosen = nearest_names[i % len(nearest_names) : i % len(nearest_names) + k_nearest]
        if len(chosen) < k_nearest:
            chosen = chosen + nearest_names[:(k_nearest - len(chosen))]
        # fetch DAGs and build fused layers using utils.fuse_architectures (must be implemented)
        dags = [utils.load_dag(arch_bank[name]) for name in chosen]
        # fuse DAGs into layers (you need utils.fuse_dags or reuse previous fuse logic)
        if hasattr(utils, "fuse_architecture_templates"):
            fused_layers = utils.fuse_architecture_templates(dags, scores=None, output_dim=output_dim)
        else:
            # fallback: pick one template deterministically (first)
            fused_layers = utils.build_architecture_from_dag(dags[0], input_shape[0], input_shape[1], input_shape[2], output_dim)
        # construct particle from layers (class method)
        p = Particle.from_layers(fused_layers)
        particles.append(p)
    return particles


# ---------------------------
# Main PSO loop with Fisher guidance
# ---------------------------
class FisherPSO:
    def __init__(self, dataset_name, arch_bank, fisher_bank_path, pop_size=POP_SIZE, n_iter=N_ITER,
                 calib_batches=CALIB_BATCHES, k_nearest=K_NEAREST, input_shape=(28,28,1), output_dim=10):
        self.dataset = dataset_name
        self.arch_bank = arch_bank
        self.fisher_bank = load_fisher_bank(fisher_bank_path)
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.calib_batches = calib_batches
        self.k_nearest = k_nearest
        self.input_shape = input_shape
        self.output_dim = output_dim

        # prepare small calibration loader and full train/test loaders (you must implement these functions)
        self.calib_loader = calibration_loader_for_dataset(self.dataset, batch_size=32)  # few batches only
        self.train_loader, self.test_loader, self.x_train, self.y_train, self.x_test, self.y_test = full_train_test_loaders(self.dataset)

        # compute initial task profile
        def probe_builder():
            # this must return a PyTorch model compatible with calib loader
            return utils.make_probe_model(self.input_shape, self.output_dim)
        self.task_profile = compute_task_fisher_profile_from_calib(probe_builder, self.calib_loader, max_batches=self.calib_batches)

        # Initialize population from bank
        self.particles = initialize_population_from_bank(self.pop_size, self.fisher_bank, self.arch_bank,
                                                         self.task_profile, self.input_shape, self.output_dim, k_nearest=self.k_nearest)

        # initialize pBest and evaluate initial particles (but skip duplicates)
        for i, p in enumerate(self.particles):
            print(f"Init Particle {i+1}: {p}")
            # compile and fast-eval (you must provide Particle.model_compile & model_fit that accept dataset arrays)
            p.model_compile(dropout_rate=0.5)
            hist = p.model_fit(self.x_train, self.y_train, batch_size=64, epochs=2)  # short warm-up
            p.model_delete()
            p.acc = hist.history.get('accuracy', [0])[-1] if hasattr(hist, 'history') else 0.0
            p.pBest = deepcopy(p)

        # set gBest
        self.gBest = max(self.particles, key=lambda x: x.acc)
        # store gBest fingerprint (compute Fisher for gBest cheaply)
        self.gBest_fisher = self.compute_fisher_for_particle(self.gBest)

    def compute_fisher_for_particle(self, particle):
        """
        Compute a quick fisher fingerprint for an architecture (particle) using calib_loader.
        Reuse the general compute_task_fisher_profile if possible (wrap model builder to particle.model).
        """
        if hasattr(utils, "compute_model_fisher"):
            return utils.compute_model_fisher(particle, self.calib_loader, max_batches=self.calib_batches)
        # fallback: build a probe wrapper that returns activations and compute grads -> simplified
        def builder():
            return particle.build_probe_model(self.input_shape, self.output_dim)  # you must implement this
        return compute_task_fisher_profile_from_calib(builder, self.calib_loader, max_batches=self.calib_batches)

    def schedule_temperature(self, iter_idx):
        # linear annealing
        t0 = INIT_TEMPERATURE; t1 = FINAL_TEMPERATURE
        alpha = iter_idx / max(1, (self.n_iter - 1))
        return t0 * (1 - alpha) + t1 * alpha

    def schedule_novelty_threshold(self, iter_idx):
        s0 = NOVELTY_START; s1 = NOVELTY_END
        alpha = iter_idx / max(1, (self.n_iter - 1))
        return s0 * (1 - alpha) + s1 * alpha

    def run(self):
        for it in range(1, self.n_iter+1):
            t0 = time.time()
            temp = self.schedule_temperature(it-1)
            novelty_thresh = self.schedule_novelty_threshold(it-1)
            print(f"\n=== PSO Iter {it}/{self.n_iter} : temp={temp:.4f} novelty_thresh={novelty_thresh:.4f} ===")

            # recompute task fisher profile cheaply each iteration (few batches)
            def probe_builder():
                return utils.make_probe_model(self.input_shape, self.output_dim)
            self.task_profile = compute_task_fisher_profile_from_calib(probe_builder, self.calib_loader, max_batches=self.calib_batches)

            # for each particle: compute fisher distance to task and to gBest
            for idx, p in enumerate(self.particles):
                # compute particle fisher quickly (or fetch cached)
                pf = self.compute_fisher_for_particle(p)
                # distance to task
                d_task = fisher_wasserstein_distance(pf["__global__"], self.task_profile["__global__"])
                # distance to gBest
                d_gbest = fisher_wasserstein_distance(pf["__global__"], self.gBest_fisher["__global__"])

                # Decide whether to train this particle:
                # - If particle is too similar to current gBest (d_gbest < novelty_thresh), skip expensive training
                # - Otherwise, train for full EPOCHS
                print(f"Particle {idx+1}: d_task={d_task:.4f} d_gbest={d_gbest:.4f}")

                do_train = (d_gbest >= novelty_thresh)

                # Velocity update: combine PSO velocity with Fisher-driven attraction
                # utils.computeVelocity should accept optional fisher signals
                if hasattr(utils, "computeVelocity"):
                    # we pass fisher distances as extra guidance
                    p.velocity(self.gBest.layers, Cg={"task_distance": d_task, "gbest_distance": d_gbest, "temperature": temp})
                else:
                    # fallback: call original signature with gBest layers and a coefficient derived from temp
                    p.velocity(self.gBest.layers, Cg= (1.0 - temp))  # simple heuristic

                # Update architecture
                p.update()
                print(" -> New architecture:", p)

                # If training is permitted (novel), do it; otherwise perform a cheap eval (e.g., 1 epoch) or skip
                p.model_compile(dropout_rate=0.5)
                if do_train:
                    print(" Training (full) particle ...")
                    hist = p.model_fit(self.x_train, self.y_train, batch_size=64, epochs=EPOCHS)
                else:
                    print(" Skipping full train (too similar to gBest), performing short eval ...")
                    hist = p.model_fit(self.x_train, self.y_train, batch_size=64, epochs=1)

                # evaluate
                p.model_delete()
                p.acc = hist.history.get('accuracy', [0])[-1] if hasattr(hist, 'history') else 0.0

                # check pBest / gBest improvements
                if p.acc >= p.pBest.acc:
                    print(" New pBest for particle", idx+1)
                    p.pBest = deepcopy(p)
                    if p.acc >= self.gBest.acc:
                        print(" New gBest found by particle", idx+1)
                        self.gBest = deepcopy(p)
                        self.gBest_fisher = self.compute_fisher_for_particle(self.gBest)
            # end for each particle

            # log iteration statistics (optional)
            g_acc = self.gBest.acc
            print(f"End of iter {it}: gBest acc = {g_acc:.4f} ; time {time.time()-t0:.1f}s")

        return self.gBest


# ---------------------------
# Example usage (pseudo)
# ---------------------------
if __name__ == "__main__":
    arch_bank = KNOWN_ARCHS  # mapping name->dag json path
    pso = FisherPSO(dataset_name="mnist", arch_bank=arch_bank, fisher_bank_path=FISHER_BANK_PATH,
                    pop_size=POP_SIZE, n_iter=N_ITER, calib_batches=CALIB_BATCHES, k_nearest=K_NEAREST,
                    input_shape=(28,28,1), output_dim=10)
    best_particle = pso.run()
    print("Best architecture:", best_particle)
