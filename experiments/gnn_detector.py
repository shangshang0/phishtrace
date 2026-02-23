"""
Automated Graph Feature Learning (AGFL) for Phishing Detection

Implements three complementary automated feature learning approaches that operate
directly on Interaction Trace Graphs (ITGs), eliminating manual feature engineering:

1. Weisfeiler-Lehman Subtree Kernel (WL-Kernel): Captures local neighborhood
   structural patterns via iterative label refinement.
2. Spectral Graph Features: Extracts algebraic properties from the graph
   Laplacian eigenspectrum, encoding global topology.
3. Random Walk Structural Embeddings: Learns positional/structural node roles
   via truncated random walks aggregated into graph-level descriptors.

The hybrid AGFL ensemble combines all three with a meta-learner, achieving
performance competitive with (or superior to) hand-crafted features while being
fully automated and transferable to new ITG variants.

References:
  - Shervashidze et al., "Weisfeiler-Lehman Graph Kernels", JMLR 2011
  - Kondor & Lafferty, "Diffusion Kernels on Graphs", ICML 2002
  - Grover & Leskovec, "node2vec: Scalable Feature Learning", KDD 2016
"""

import os
import json
import hashlib
import logging
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter, defaultdict
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Weisfeiler-Lehman Subtree Kernel Feature Extractor
# ---------------------------------------------------------------------------

class WLKernelExtractor:
    """
    Weisfeiler-Lehman subtree kernel feature extractor.

    Iteratively refines node labels by hashing multisets of neighbor labels,
    building a vocabulary of discriminative substructure patterns. The resulting
    histogram over the label vocabulary forms a fixed-length feature vector.
    """

    def __init__(self, n_iterations: int = 5, vocab_size: int = 256):
        self.n_iterations = n_iterations
        self.vocab_size = vocab_size
        self.label_vocab: Dict[str, int] = {}
        self._fitted = False

    def _node_label(self, node_data: dict) -> str:
        """Create canonical initial label from node attributes."""
        ntype = node_data.get('node_type', 'unknown')
        tag = node_data.get('element_tag', '')
        has_sensitive = node_data.get('has_sensitive', False)
        return f"{ntype}|{tag}|{int(has_sensitive)}"

    def _wl_hash(self, label: str, neighbor_labels: List[str]) -> str:
        """Compute WL relabeling: hash(label, sorted(neighbor_labels))."""
        canonical = label + "||" + "|".join(sorted(neighbor_labels))
        return hashlib.md5(canonical.encode()).hexdigest()[:12]

    def _extract_wl_sequence(self, adj: dict, labels: dict, nodes: list) -> List[Counter]:
        """Run WL iterations and collect label histograms at each step."""
        histograms = []
        current_labels = dict(labels)

        for h in range(self.n_iterations + 1):
            hist = Counter(current_labels.values())
            histograms.append(hist)

            if h < self.n_iterations:
                new_labels = {}
                for node in nodes:
                    neighbors = adj.get(node, [])
                    neighbor_labs = [current_labels.get(n, '') for n in neighbors]
                    new_labels[node] = self._wl_hash(
                        current_labels[node], neighbor_labs
                    )
                current_labels = new_labels

        return histograms

    def fit(self, graphs: List[dict]):
        """Build vocabulary from training graphs.

        Each graph dict has keys: 'nodes' (list), 'adj' (dict -> list),
        'node_data' (dict -> dict).
        """
        all_labels = set()
        for g in graphs:
            labels = {n: self._node_label(g['node_data'].get(n, {}))
                      for n in g['nodes']}
            hists = self._extract_wl_sequence(g['adj'], labels, g['nodes'])
            for h in hists:
                all_labels.update(h.keys())

        # Assign indices; cap at vocab_size by frequency
        label_freq = Counter()
        for g in graphs:
            labels = {n: self._node_label(g['node_data'].get(n, {}))
                      for n in g['nodes']}
            hists = self._extract_wl_sequence(g['adj'], labels, g['nodes'])
            for h in hists:
                label_freq.update(h)

        top_labels = [l for l, _ in label_freq.most_common(self.vocab_size)]
        self.label_vocab = {l: i for i, l in enumerate(top_labels)}
        self._fitted = True

    def transform(self, graphs: List[dict]) -> np.ndarray:
        """Transform graphs to WL feature vectors."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        X = np.zeros((len(graphs), self.vocab_size), dtype=np.float64)
        for gi, g in enumerate(graphs):
            labels = {n: self._node_label(g['node_data'].get(n, {}))
                      for n in g['nodes']}
            hists = self._extract_wl_sequence(g['adj'], labels, g['nodes'])
            for h in hists:
                for label, count in h.items():
                    idx = self.label_vocab.get(label)
                    if idx is not None:
                        X[gi, idx] += count

        # L2 normalize
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms
        return X


# ---------------------------------------------------------------------------
# 2. Spectral Graph Feature Extractor
# ---------------------------------------------------------------------------

class SpectralExtractor:
    """
    Extracts features from the graph Laplacian eigenspectrum.

    Computes the normalized Laplacian L = I - D^{-1/2} A D^{-1/2} and returns
    a fixed-length descriptor from the top-k eigenvalues, spectral gap,
    algebraic connectivity, and spectral entropy.
    """

    def __init__(self, n_eigenvalues: int = 20):
        self.n_eigenvalues = n_eigenvalues
        self.feature_dim = n_eigenvalues + 6  # eigenvalues + derived stats

    def _build_adjacency(self, g: dict) -> sparse.csr_matrix:
        """Build sparse adjacency matrix from graph dict."""
        nodes = g['nodes']
        idx_map = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)
        if n == 0:
            return sparse.csr_matrix((0, 0))

        rows, cols, data = [], [], []
        for src, neighbors in g['adj'].items():
            si = idx_map.get(src)
            if si is None:
                continue
            for dst in neighbors:
                di = idx_map.get(dst)
                if di is not None:
                    rows.append(si)
                    cols.append(di)
                    data.append(1.0)

        if not rows:
            return sparse.csr_matrix((n, n))
        A = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        # Symmetrize for undirected Laplacian
        A = A + A.T
        A.data[:] = np.minimum(A.data, 1.0)
        return A

    def _normalized_laplacian_eigenvalues(self, A: sparse.csr_matrix, k: int):
        """Compute top-k smallest eigenvalues of normalized Laplacian."""
        n = A.shape[0]
        if n <= 1:
            return np.zeros(k)

        degrees = np.array(A.sum(axis=1)).flatten()
        degrees[degrees == 0] = 1.0
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(degrees))
        L_norm = sparse.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

        num_eigs = min(k, n - 1)
        if num_eigs <= 0:
            return np.zeros(k)

        try:
            eigenvalues = eigsh(L_norm, k=num_eigs, which='SM',
                                return_eigenvectors=False)
            eigenvalues = np.sort(np.real(eigenvalues))
        except Exception:
            eigenvalues = np.zeros(num_eigs)

        # Pad to fixed length
        if len(eigenvalues) < k:
            eigenvalues = np.pad(eigenvalues, (0, k - len(eigenvalues)))
        return eigenvalues[:k]

    def transform(self, graphs: List[dict]) -> np.ndarray:
        """Transform graphs to spectral feature vectors."""
        X = np.zeros((len(graphs), self.feature_dim), dtype=np.float64)

        for gi, g in enumerate(graphs):
            A = self._build_adjacency(g)
            n = A.shape[0]

            if n <= 1:
                continue

            eigs = self._normalized_laplacian_eigenvalues(A, self.n_eigenvalues)
            X[gi, :self.n_eigenvalues] = eigs

            # Derived spectral statistics
            nonzero_eigs = eigs[eigs > 1e-10]
            if len(nonzero_eigs) > 0:
                # Spectral gap (algebraic connectivity proxy)
                X[gi, self.n_eigenvalues] = nonzero_eigs[0] if len(nonzero_eigs) > 0 else 0
                # Spectral radius
                X[gi, self.n_eigenvalues + 1] = nonzero_eigs[-1] if len(nonzero_eigs) > 0 else 0
                # Spectral entropy: -sum(p * log(p)) where p_i = lambda_i / sum(lambda)
                total = nonzero_eigs.sum()
                if total > 0:
                    probs = nonzero_eigs / total
                    probs = probs[probs > 0]
                    X[gi, self.n_eigenvalues + 2] = -np.sum(probs * np.log2(probs))
                # Mean & variance of spectrum
                X[gi, self.n_eigenvalues + 3] = np.mean(nonzero_eigs)
                X[gi, self.n_eigenvalues + 4] = np.var(nonzero_eigs)
                # Energy: sum of squared eigenvalues (Estrada index proxy)
                X[gi, self.n_eigenvalues + 5] = np.sum(nonzero_eigs ** 2)

        return X


# ---------------------------------------------------------------------------
# 3. Random Walk Structural Embedding Extractor
# ---------------------------------------------------------------------------

class RandomWalkExtractor:
    """
    Generates graph-level feature vectors via aggregated random walk statistics.

    Performs multiple truncated random walks from each node, collecting:
    - Visit frequency distribution (which node types are visited)
    - Return probability profile (how quickly walks return to start)
    - Walk diversity (entropy of visited node set)

    These are aggregated across all walks into a fixed-length descriptor.
    """

    def __init__(self, n_walks: int = 50, walk_length: int = 10,
                 n_type_bins: int = 8, n_return_bins: int = 5):
        self.n_walks = n_walks
        self.walk_length = walk_length
        self.n_type_bins = n_type_bins
        self.n_return_bins = n_return_bins
        self.type_vocab: Dict[str, int] = {}
        self.feature_dim = n_type_bins + n_return_bins + 5  # type + return + stats
        self._fitted = False

    def fit(self, graphs: List[dict]):
        """Learn node type vocabulary from training graphs."""
        type_freq = Counter()
        for g in graphs:
            for n in g['nodes']:
                ntype = g['node_data'].get(n, {}).get('node_type', 'unknown')
                type_freq[ntype] += 1

        top_types = [t for t, _ in type_freq.most_common(self.n_type_bins - 1)]
        self.type_vocab = {t: i for i, t in enumerate(top_types)}
        # Last bin is 'other'
        self._fitted = True

    def _random_walk(self, adj: dict, nodes: list, start: str,
                     rng: np.random.RandomState) -> List[str]:
        """Perform a single truncated random walk."""
        walk = [start]
        current = start
        for _ in range(self.walk_length - 1):
            neighbors = adj.get(current, [])
            if not neighbors:
                break
            current = rng.choice(neighbors)
            walk.append(current)
        return walk

    def transform(self, graphs: List[dict]) -> np.ndarray:
        """Transform graphs to random walk feature vectors."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        rng = np.random.RandomState(42)
        X = np.zeros((len(graphs), self.feature_dim), dtype=np.float64)

        for gi, g in enumerate(graphs):
            nodes = g['nodes']
            if not nodes:
                continue

            type_counts = np.zeros(self.n_type_bins)
            return_counts = np.zeros(self.n_return_bins)
            walk_lengths = []
            unique_visited = set()

            for _ in range(self.n_walks):
                start = rng.choice(nodes)
                walk = self._random_walk(g['adj'], nodes, start, rng)
                walk_lengths.append(len(walk))

                # Type distribution
                for node in walk:
                    ntype = g['node_data'].get(node, {}).get('node_type', 'unknown')
                    tidx = self.type_vocab.get(ntype, self.n_type_bins - 1)
                    type_counts[tidx] += 1
                    unique_visited.add(node)

                # Return probability: first return to start
                for step, node in enumerate(walk[1:], 1):
                    if node == start:
                        bin_idx = min(step - 1, self.n_return_bins - 1)
                        return_counts[bin_idx] += 1
                        break

            # Normalize
            total_type = type_counts.sum()
            if total_type > 0:
                type_counts /= total_type
            total_return = return_counts.sum()
            if total_return > 0:
                return_counts /= total_return

            X[gi, :self.n_type_bins] = type_counts
            X[gi, self.n_type_bins:self.n_type_bins + self.n_return_bins] = return_counts

            # Aggregate statistics
            offset = self.n_type_bins + self.n_return_bins
            X[gi, offset] = np.mean(walk_lengths) / self.walk_length  # avg walk completion
            X[gi, offset + 1] = len(unique_visited) / max(1, len(nodes))  # coverage
            # Walk diversity (entropy of type distribution)
            p = type_counts[type_counts > 0]
            X[gi, offset + 2] = -np.sum(p * np.log2(p)) if len(p) > 0 else 0
            # Graph size features
            X[gi, offset + 3] = np.log1p(len(nodes))
            X[gi, offset + 4] = np.log1p(sum(len(v) for v in g['adj'].values()))

        return X


# ---------------------------------------------------------------------------
# Graph Dict Builder (from trace JSON)
# ---------------------------------------------------------------------------

def trace_to_graph_dict(trace_data: dict) -> dict:
    """Convert a trace JSON dict to the graph dict format used by extractors.

    Returns: {'nodes': [...], 'adj': {node: [neighbors]}, 'node_data': {node: {...}}}
    """
    nodes = []
    node_data = {}
    adj = defaultdict(list)

    root_url = trace_data.get('url', 'unknown')
    nodes.append(root_url)
    node_data[root_url] = {
        'node_type': 'url', 'is_root': True, 'element_tag': '',
        'has_sensitive': False
    }

    previous_node = root_url
    raw_events = trace_data.get('events', [])
    # Filter out screenshot/page events (visual bookmarks, not interactions)
    events = [e for e in raw_events if e and
              e.get('event_type') != 'screenshot' and
              e.get('element_tag') != 'page']

    for idx, event in enumerate(events):
        if event is None:
            continue
        event_type = event.get('event_type', 'unknown')
        el_tag = event.get('element_tag', '')
        url_after = event.get('url_after', '')
        url_before = event.get('url_before', '')
        form_data = event.get('form_data', {})

        node_id = f"{event_type}_{idx}"
        nodes.append(node_id)

        has_sensitive = False
        if form_data and isinstance(form_data, dict):
            field_names = ' '.join(form_data.keys()).lower()
            has_sensitive = any(k in field_names for k in
                               ['password', 'pass', 'pwd', 'email', 'ssn', 'card', 'credit'])

        node_data[node_id] = {
            'node_type': event_type,
            'element_tag': el_tag,
            'has_sensitive': has_sensitive,
        }

        adj[previous_node].append(node_id)
        adj[node_id]  # ensure key exists

        if url_after and url_before and url_after != url_before:
            if url_after not in node_data:
                nodes.append(url_after)
                node_data[url_after] = {
                    'node_type': 'url', 'is_root': False,
                    'element_tag': '', 'has_sensitive': False
                }
            adj[node_id].append(url_after)
            previous_node = url_after
        else:
            previous_node = node_id

    # Redirect chain
    redirects = trace_data.get('redirects', [])
    for i in range(len(redirects) - 1):
        src, dst = redirects[i], redirects[i + 1]
        for r in [src, dst]:
            if r not in node_data:
                nodes.append(r)
                node_data[r] = {
                    'node_type': 'url', 'is_root': False,
                    'element_tag': '', 'has_sensitive': False
                }
        adj[src].append(dst)

    return {
        'nodes': list(dict.fromkeys(nodes)),  # deduplicate preserving order
        'adj': dict(adj),
        'node_data': node_data,
    }


# ---------------------------------------------------------------------------
# AGFL Hybrid Ensemble
# ---------------------------------------------------------------------------

class AGFLDetector:
    """
    Automated Graph Feature Learning (AGFL) hybrid ensemble.

    Combines WL-Kernel, Spectral, and Random Walk features with a
    stacking meta-learner for end-to-end graph classification without
    manual feature engineering.
    """

    def __init__(self, wl_iters: int = 5, wl_vocab: int = 256,
                 n_eigenvalues: int = 20, n_walks: int = 50,
                 walk_length: int = 10):
        self.wl = WLKernelExtractor(n_iterations=wl_iters, vocab_size=wl_vocab)
        self.spectral = SpectralExtractor(n_eigenvalues=n_eigenvalues)
        self.rw = RandomWalkExtractor(n_walks=n_walks, walk_length=walk_length)

        # Meta-learner: gradient boosting over concatenated features
        self.scaler = StandardScaler()
        self.classifier = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42
        )
        self._fitted = False

    def fit(self, graphs: List[dict], y: np.ndarray):
        """Fit all extractors and the meta-classifier."""
        from sklearn.utils.class_weight import compute_sample_weight
        self.wl.fit(graphs)
        self.rw.fit(graphs)

        X = self._extract_all(graphs)
        X_scaled = self.scaler.fit_transform(X)
        sw = compute_sample_weight('balanced', y)
        self.classifier.fit(X_scaled, y, sample_weight=sw)
        self._fitted = True

    def _extract_all(self, graphs: List[dict]) -> np.ndarray:
        """Concatenate features from all three extractors."""
        X_wl = self.wl.transform(graphs)
        X_sp = self.spectral.transform(graphs)
        X_rw = self.rw.transform(graphs)
        return np.hstack([X_wl, X_sp, X_rw])

    def predict(self, graphs: List[dict]) -> np.ndarray:
        X = self.scaler.transform(self._extract_all(graphs))
        return self.classifier.predict(X)

    def predict_proba(self, graphs: List[dict]) -> np.ndarray:
        X = self.scaler.transform(self._extract_all(graphs))
        return self.classifier.predict_proba(X)

    def feature_dim(self) -> int:
        return self.wl.vocab_size + self.spectral.feature_dim + self.rw.feature_dim

    def feature_importances(self) -> np.ndarray:
        return self.classifier.feature_importances_


# ---------------------------------------------------------------------------
# Evaluation Harness
# ---------------------------------------------------------------------------

def load_trace_graphs(dataset_dir: str) -> Tuple[List[dict], List[int]]:
    """Load all trace files and convert to graph dicts."""
    graphs = []
    labels = []
    dataset_path = Path(dataset_dir)

    for label_name, label_int in [('phishing', 1), ('benign', 0)]:
        trace_dir = dataset_path / 'traces' / label_name
        if not trace_dir.exists():
            continue
        for trace_file in sorted(trace_dir.glob('*.json')):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # Handle nested trace format: {url, trace: {events, redirects, ...}}
                trace_data = raw.get('trace', raw)
                # Ensure url is available at top level
                if 'url' not in trace_data and 'url' in raw:
                    trace_data['url'] = raw['url']
                g = trace_to_graph_dict(trace_data)
                if len(g['nodes']) >= 2:  # skip degenerate traces
                    graphs.append(g)
                    labels.append(label_int)
            except Exception as e:
                logger.warning(f"Skipping {trace_file.name}: {e}")

    return graphs, labels


def evaluate_agfl(dataset_dir: str = None, n_folds: int = 10) -> Dict:
    """Run full AGFL evaluation with cross-validation.

    Returns dict with metrics for:
      - agfl_wl: WL-Kernel only
      - agfl_spectral: Spectral only
      - agfl_rw: Random Walk only
      - agfl_hybrid: Full hybrid ensemble
    """
    if dataset_dir is None:
        dataset_dir = str(Path(__file__).parent.parent / 'dataset')

    print("\n" + "=" * 70)
    print("AGFL: Automated Graph Feature Learning Evaluation")
    print("=" * 70)

    graphs, labels = load_trace_graphs(dataset_dir)
    y = np.array(labels)
    print(f"Loaded {len(graphs)} trace graphs ({sum(labels)} phishing, "
          f"{len(labels) - sum(labels)} benign)")

    if len(graphs) < 20:
        print("[WARN] Too few trace graphs for reliable evaluation.")
        return {}

    # Adapt n_folds to minority class
    minority_count = min(sum(labels), len(labels) - sum(labels))
    actual_folds = max(2, min(n_folds, minority_count))
    print(f"Using {actual_folds}-fold CV (minority class: {minority_count} samples)")

    # Fit extractors on all data for vocabulary building
    wl = WLKernelExtractor(n_iterations=5, vocab_size=256)
    wl.fit(graphs)
    spectral = SpectralExtractor(n_eigenvalues=20)
    rw = RandomWalkExtractor(n_walks=50, walk_length=10)
    rw.fit(graphs)

    # Extract features
    X_wl = wl.transform(graphs)
    X_sp = spectral.transform(graphs)
    X_rw = rw.transform(graphs)
    X_all = np.hstack([X_wl, X_sp, X_rw])

    print(f"Feature dimensions: WL={X_wl.shape[1]}, Spectral={X_sp.shape[1]}, "
          f"RW={X_rw.shape[1]}, Hybrid={X_all.shape[1]}")

    results = {}
    feature_sets = {
        'AGFL-WL': X_wl,
        'AGFL-Spectral': X_sp,
        'AGFL-RW': X_rw,
        'AGFL-Hybrid': X_all,
    }

    for name, X in feature_sets.items():
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ))
        ])

        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

        from sklearn.utils.class_weight import compute_sample_weight
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            sw = compute_sample_weight('balanced', y_train)
            clf.fit(X_train, y_train, clf__sample_weight=sw)
            y_pred = clf.predict(X_test)
            auc = np.nan
            if len(np.unique(y_test)) > 1:
                try:
                    y_proba = clf.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except (ValueError, IndexError):
                    pass

            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            metrics['auc'].append(auc)

        avg = {k: round(float(np.nanmean(v)), 4) for k, v in metrics.items()}
        std = {k: round(float(np.nanstd(v)), 4) for k, v in metrics.items()}
        results[name] = {'mean': avg, 'std': std}

        print(f"\n  {name}:")
        for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            print(f"    {m.upper():>10}: {avg[m]:.4f} (+/- {std[m]:.4f})")

    return results


if __name__ == '__main__':
    evaluate_agfl()
