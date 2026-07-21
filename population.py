from particle import Particle
from utils import fisher_similarity, build_architecture_from_dag, load_fisher, load_dag

# Predefined well-known architectures represented as DAGs (JSON files)
KNOWN_ARCHS = {
    "resnet18": "graphs/resnet18_dag.json",
    "vgg16": "graphs/vgg16_dag.json",
    "densenet121": "graphs/densenet121_dag.json",
    "mobilenet": "graphs/mobilenet_dag.json"
}

class Population:
    def __init__(self, pop_size, fisher_matrix, input_width, input_height, input_channels, output_dim):
        """
        Population of particles built using Fisher information similarity 
        and known architectures as DAGs.

        Args:
            pop_size (int): number of particles in the swarm
            fisher_matrix (np.ndarray): Fisher information matrix of the target dataset
            input_width, input_height, input_channels (int): input dimensions
            output_dim (int): number of output classes
        """

        self.particles = []

        # Step 1: compute Fisher similarity between dataset and each known architecture
        scores = {}
        for name, arch_path in KNOWN_ARCHS.items():
            arch_fisher = load_fisher(arch_path.replace("graphs", "fisher"))  # load Fisher matrix for that arch
            scores[name] = fisher_similarity(fisher_matrix, arch_fisher)

        # Step 2: sort architectures by similarity (smaller = more similar)
        sorted_archs = sorted(scores.items(), key=lambda x: x[1])
        best_archs = [arch for arch, _ in sorted_archs[:pop_size]]

        # Step 3: build particles based on DAGs of the best architectures
        for arch in best_archs:
            dag = load_dag(KNOWN_ARCHS[arch])
            layers = build_architecture_from_dag(dag, input_width, input_height, input_channels, output_dim)
            self.particles.append(Particle.from_layers(layers))  # construct particle from DAG
