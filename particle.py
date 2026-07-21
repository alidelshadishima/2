"""
fisher_guided_particle.py

Replaces the random Particle initialization with a Fisher-guided construction
that uses a bank of known architectures (DAG/cell templates) + Fisher fingerprints.

Assumptions (prepare beforehand):
- arch_bank.py: defines ARCH_BANK, a dict mapping arch_name -> arch_template
    arch_template: {
        "name": "ResNetBlockA",
        "edges": [("0->1", ["conv3","skip","sep3"]), ("0->2", ["conv3","sep5"]), ...],
        "meta": {...}
    }
- fisher_bank.npz: a dict-like .npz or .npy with keys = arch_name, values = {edge_name: fisher_vector(np.array)}
- utils.py: provides add_conv/add_pool/add_fc, computeVelocity, updateParticle, and mapping from op-names to layer specs
- data_loader.py: provides calibration loader and main train/test loaders

This code uses Keras (like original PSO code). It only replaces Particle.initialization().
"""
import os
import numpy as np
from copy import deepcopy
from scipy.stats import wasserstein_distance

# Keras / TensorFlow (same environment as original)
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, BatchNormalization, Activation
from keras.optimizers import Adam
import keras.backend as K

# project utilities (must exist)
import utils               # original project utils: add_conv/add_pool/add_fc/computeVelocity/updateParticle
from data_loader import *  # existing data loaders: calibration loader etc.

# ----------------------------
# Helper functions
# ----------------------------
def load_fisher_bank(path):
    """
    Load a fisher bank .npz or a pickled dict.
    Format expected: dict-like where key = arch_name, value = { edge_name: 1D np.array fisher_vector }
    """
    if path.endswith('.npz'):
        raw = np.load(path, allow_pickle=True)
        # return as dict
        bank = {}
        for k in raw.files:
            bank[k] = raw[k].item() if raw[k].dtype == np.object_ else raw[k]
        return bank
    else:
        # If it's a pickle/npy
        return np.load(path, allow_pickle=True).item()

def compute_task_fisher_profile(model_builder, calib_loader, max_batches=5, device='cpu'):
    """
    Compute a simple diagnostic Fisher fingerprint for the *task* using a small probe model.
    model_builder: function that returns a keras model (or object with named param tensors).
    Return: dict edge_name -> fisher_vector (flattened)
    NOTE: this is a simplified version: we expect model_builder to expose layers named by edge
    or alternatively we compute a global flattened fisher vector.
    """
    # We'll compute a global diagonal fisher across all params as fallback
    # For Keras, easiest is to perform grads via TF backend; here we use a simple approximation:
    #  - For each batch: forward, compute loss, compute grads of weights, accumulate squared grads.
    import tensorflow as tf
    tf_device = '/GPU:0' if device == 'cuda' else '/CPU:0'

    # Build model and compile just for forward/backward
    model = model_builder()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Collect trainable weights
    trainable_vars = model.trainable_weights
    # Initialize accumulator
    accum = [np.zeros(K.get_value(w).shape) for w in trainable_vars]

    batches = 0
    for xb, yb in calib_loader:
        xb = xb.astype('float32')
        with tf.GradientTape() as tape:
            logits = model(xb, training=False)
            loss = loss_fn(yb, logits)
        grads = tape.gradient(loss, trainable_vars)
        # accumulate squared grads
        for i, g in enumerate(grads):
            if g is not None:
                accum[i] += (g.numpy() ** 2).mean(axis=tuple(range(len(g.shape))))  # reduce over axes to scalar per param ? careful
        batches += 1
        if batches >= max_batches:
            break

    # Flatten accum into a single vector (global fingerprint)
    flat = np.concatenate([a.flatten() for a in accum])
    # Normalize
    flat = flat / (flat.sum() + 1e-12)
    # For compatibility with arch-bank keyed by edges, we return as {"__global__": flat}
    return {"__global__": flat}

def fisher_distance_vec(v1, v2):
    """1D wasserstein distance between two normalized vectors (same length or resampled)."""
    # align lengths: if different, resample shorter to longer using simple padding/trunc
    n1, n2 = len(v1), len(v2)
    if n1 == n2:
        return wasserstein_distance(np.arange(n1), np.arange(n2), u_weights=v1, v_weights=v2)
    else:
        m = min(n1, n2)
        v1s = v1[:m] / (v1[:m].sum() + 1e-12)
        v2s = v2[:m] / (v2[:m].sum() + 1e-12)
        return wasserstein_distance(np.arange(m), np.arange(m), u_weights=v1s, v_weights=v2s)

# ----------------------------
# Fisher-guided Particle (PSO) class
# ----------------------------
class FisherParticle:
    """
    Particle that initializes its architecture using Fisher similarity to known architectures.
    After initialization, it still has vel/pBest and can be used in PSO loops (velocity/update).
    """

    def __init__(self,
                 fisher_bank_path,
                 arch_bank,
                 calib_loader,
                 min_layer=3,
                 max_layer=12,
                 max_pool_layers=3,
                 input_width=28,
                 input_height=28,
                 input_channels=1,
                 output_dim=10,
                 k_nearest=3):
        """
        fisher_bank_path: path to numpy .npz or .npy with arch_name -> { '__global__' or edge_name: vector }
        arch_bank: dictionary mapping arch_name -> arch_template (DAG/cell)
        calib_loader: small DataLoader used to compute task fisher profile
        k_nearest: how many nearest archs to consider for combination
        """
        self.input_width = input_width
        self.input_height = input_height
        self.input_channels = input_channels
        self.output_dim = output_dim

        self.min_layer = min_layer
        self.max_layer = max_layer
        self.max_pool_layers = max_pool_layers
        self.k_nearest = k_nearest

        # load external banks
        self.arch_bank = arch_bank          # dict: arch_name -> arch_template
        self.fisher_bank = load_fisher_bank(fisher_bank_path)  # dict: arch_name -> {edge: vec}

        # compute task fisher profile (global)
        # model_builder is a tiny probe; we create a small conv->gap->fc network
        def _probe_builder():
            from keras.models import Sequential
            from keras.layers import Conv2D, GlobalAveragePooling2D, Dense
            m = Sequential()
            m.add(Conv2D(8, 3, padding='same', input_shape=(self.input_width, self.input_height, self.input_channels)))
            m.add(GlobalAveragePooling2D())
            m.add(Dense(self.output_dim))
            return m
        task_profile = compute_task_fisher_profile(_probe_builder, calib_loader, max_batches=5, device='cpu')
        # task_profile is {"__global__": vector}

        # compute distances to each architecture in bank
        arch_scores = []
        for arch_name, arch_fp in self.fisher_bank.items():
            # arch_fp may have per-edge vectors; we reduce to a single global vector by concatenation or averaging
            if "__global__" in arch_fp:
                arch_vec = arch_fp["__global__"]
            else:
                # concat all edges vectors (deterministic)
                allv = []
                for k, v in arch_fp.items():
                    allv.append(v.flatten())
                arch_vec = np.concatenate(allv)
                arch_vec = arch_vec / (arch_vec.sum() + 1e-12)
            # compare arch_vec with task_profile global vec
            tv = task_profile["__global__"]
            # align lengths: resample/truncate
            L = min(len(arch_vec), len(tv))
            arch_vec_s = arch_vec[:L] / (arch_vec[:L].sum() + 1e-12)
            tv_s = tv[:L] / (tv[:L].sum() + 1e-12)
            d = wasserstein_distance(np.arange(L), np.arange(L), u_weights=arch_vec_s, v_weights=tv_s)
            arch_scores.append((arch_name, float(d)))

        # sort by distance (small -> similar)
        arch_scores.sort(key=lambda x: x[1])
        self.nearest_archs = [t[0] for t in arch_scores[:self.k_nearest]]
        self.nearest_scores = [t[1] for t in arch_scores[:self.k_nearest]]

        # Build particle architecture by fusing the DAGs/cells of nearest architectures
        self.layers = self._fuse_architectures(self.nearest_archs, self.nearest_scores)

        # initialize PSO fields
        self.acc = None
        self.vel = [{"type": "keep" if l["type"] != "fc" else "keep_fc"} for l in self.layers]
        self.pBest = deepcopy(self)
        self.model = None

    def __str__(self):
        return "FisherParticle(nearest={}, layers={})".format(self.nearest_archs, [l["type"] for l in self.layers])

    def _fuse_architectures(self, arch_names, scores):
        """
        Fuse architectures by voting/weighted combination of edges.
        arch_bank[arch_name] expected to contain 'edges': list of (edge_name, op_list)
            where op_list is sequence of op names or layer specs.
        We produce a linear list 'layers' compatible with original PSO (conv/pool/fc entries).
        Approach (simple and deterministic):
          - For each position (index) across candidate arch templates, collect layer types and counts.
          - Use weighted voting (weights = 1/(score+eps)) to pick a final layer type and params.
        NOTE: You must ensure arch templates align in a position-wise manner; otherwise we do concatenation/truncation.
        """
        weights = np.array([1.0 / (s + 1e-12) for s in scores])
        weights = weights / weights.sum()

        # collect layered sequences from arch_bank
        seqs = []
        for name in arch_names:
            tpl = self.arch_bank[name]
            # expect tpl['layers'] = list of dicts like {"type": "conv","ou_c":32,"kernel":3} or "max_pool" / "fc"
            seqs.append(tpl['layers'])

        # determine max length
        max_len = max(len(s) for s in seqs)

        fused = []
        for idx in range(max_len):
            # gather proposals from archs that have this index
            proposals = []
            for ai, seq in enumerate(seqs):
                if idx < len(seq):
                    proposals.append((seq[idx], weights[ai]))
            if not proposals:
                continue
            # vote on type
            type_scores = {}
            for prop, w in proposals:
                t = prop['type']
                type_scores.setdefault(t, 0.0)
                type_scores[t] += w
            # pick best type
            best_type = max(type_scores.items(), key=lambda x: x[1])[0]

            # determine parameters for selected type by weighted average of numeric params from proposals of that type
            if best_type == 'conv':
                ou_vals = []
                ker_vals = []
                wsum = 0.0
                for prop, w in proposals:
                    if prop['type'] == 'conv':
                        ou_vals.append(prop.get('ou_c', 16) * w)
                        ker_vals.append(prop.get('kernel', 3) * w)
                        wsum += w
                if wsum == 0:
                    out_c = int(np.round(np.mean([p.get('ou_c', 16) for p,_ in proposals])))
                    kernel = int(np.round(np.mean([p.get('kernel',3) for p,_ in proposals])))
                else:
                    out_c = int(np.round(sum(ou_vals)/ (wsum + 1e-12)))
                    kernel = int(np.round(sum(ker_vals)/ (wsum + 1e-12)))
                fused.append({"type":"conv", "ou_c": max(1, out_c), "kernel": max(1, kernel)})
            elif best_type in ('max_pool', 'avg_pool'):
                fused.append({"type":best_type, "ou_c": -1, "kernel": 2})
            elif best_type == 'fc':
                # choose ou_c as weighted average, fallback to output_dim for final
                ou_vals = []
                wsum = 0.0
                for prop, w in proposals:
                    if prop['type'] == 'fc':
                        ou_vals.append(prop.get('ou_c', self.output_dim) * w)
                        wsum += w
                if wsum == 0:
                    out_c = self.output_dim
                else:
                    out_c = int(np.round(sum(ou_vals)/(wsum + 1e-12)))
                fused.append({"type":"fc", "ou_c": max(self.output_dim, out_c), "kernel": -1})
            else:
                # unknown => skip or keep conv fallback
                fused.append({"type":"conv", "ou_c":16, "kernel":3})

        # enforce validation rules -> ensure last layer is fc with output_dim
        if len(fused) == 0 or fused[-1]['type'] != 'fc':
            fused.append({"type":"fc", "ou_c": self.output_dim, "kernel": -1})
        else:
            fused[-1]['ou_c'] = self.output_dim

        # truncate pool count if exceed
        pool_count = 0
        for l in fused:
            if l['type'] in ('max_pool','avg_pool'):
                pool_count += 1
                if pool_count > self.max_pool_layers:
                    l['type'] = 'remove'

        validated = [l for l in fused if l['type'] != 'remove']
        return validated

    # PSO methods (velocity & update) reuse utils functions
    def velocity(self, gBest, Cg):
        # compute velocity using existing utils (adapt signature as original)
        self.vel = utils.computeVelocity(gBest, self.pBest.layers, self.layers, Cg)

    def update(self):
        new_p = utils.updateParticle(self.layers, self.vel)
        new_p = self.validate(new_p)
        self.layers = new_p
        self.model = None

    def validate(self, list_layers):
        # same validation as original Particle.validate
        list_layers[-1] = {"type": "fc", "ou_c": self.output_dim, "kernel": -1}
        self.num_pool_layers = 0
        for i in range(len(list_layers)):
            if list_layers[i]["type"] in ("max_pool", "avg_pool"):
                self.num_pool_layers += 1
                if self.num_pool_layers >= self.max_pool_layers:
                    list_layers[i]["type"] = "remove"
        updated_list_layers = []
        for l in list_layers:
            if l["type"] == "conv":
                updated_list_layers.append({"type": "conv", "ou_c": l.get("ou_c", 16), "kernel": l.get("kernel",3)})
            elif l["type"] == "fc":
                updated_list_layers.append(l)
            elif l["type"] == "max_pool":
                updated_list_layers.append({"type":"max_pool","ou_c":-1,"kernel":2})
            elif l["type"] == "avg_pool":
                updated_list_layers.append({"type":"avg_pool","ou_c":-1,"kernel":2})
        # ensure last fc is correct
        if len(updated_list_layers) == 0 or updated_list_layers[-1]['type'] != 'fc':
            updated_list_layers.append({"type":"fc","ou_c":self.output_dim,"kernel":-1})
        updated_list_layers[-1]['ou_c'] = self.output_dim
        return updated_list_layers

    # model_compile and model_fit can reuse original Particle.model_compile / model_fit implementations:
    def model_compile(self, dropout_rate=0.5):
        # uses same building routine as original Particle.model_compile
        list_layers = self.layers
        self.model = Sequential()
        for i in range(len(list_layers)):
            if list_layers[i]["type"] == "conv":
                n_out_filters = list_layers[i]["ou_c"]
                kernel_size = list_layers[i]["kernel"]
                if i == 0:
                    in_w = self.input_width; in_h = self.input_height; in_c = self.input_channels
                    self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same",
                                           activation=None, input_shape=(in_w,in_h,in_c)))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("relu"))
                else:
                    self.model.add(Dropout(dropout_rate))
                    self.model.add(Conv2D(n_out_filters, kernel_size, strides=(1,1), padding="same", activation=None))
                    self.model.add(BatchNormalization())
                    self.model.add(Activation("relu"))
            elif list_layers[i]["type"] == "max_pool":
                self.model.add(MaxPooling2D(pool_size=(3,3), strides=2))
            elif list_layers[i]["type"] == "avg_pool":
                self.model.add(AveragePooling2D(pool_size=(3,3), strides=2))
            elif list_layers[i]["type"] == "fc":
                if i==0 or list_layers[i-1]["type"]!="fc":
                    self.model.add(Flatten())
                self.model.add(Dropout(dropout_rate))
                if i==len(list_layers)-1:
                    self.model.add(Dense(list_layers[i]["ou_c"], activation="softmax"))
                else:
                    self.model.add(Dense(list_layers[i]["ou_c"], activation="relu"))
        adam = Adam(lr=0.001)
        self.model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    def model_delete(self):
        del self.model
        K.clear_session()
        self.model = None
