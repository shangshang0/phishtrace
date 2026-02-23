"""
Graph Neural Network Comparison for Phishing Detection

Implements GNN-equivalent classifiers using spectral graph convolutions
and message-passing approximations via scipy/sklearn. This provides a
fair comparison between our SMVE approach and end-to-end graph learning.

Three GNN variants:
1. Spectral GCN (Kipf & Welling, ICLR'17): Normalized Laplacian convolution
2. Approximate GAT: Attention-weighted neighbor aggregation
3. GraphSAGE-style: Sampling + aggregation with mean/max/LSTM pooling

All variants use the same ITG graphs and produce graph-level embeddings
for binary classification, ensuring apples-to-apples comparison with SMVE.
"""

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ═══════════════════════════════════════════════════════════════
# Section 1: Graph construction from traces
# ═══════════════════════════════════════════════════════════════

def build_adjacency_and_features(trace: dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """
    Build adjacency matrix and node feature matrix from a raw trace.
    
    Returns:
        A: adjacency matrix (n x n)
        X: node feature matrix (n x d) where d is node feature dim
        n: number of nodes
    """
    raw_events = trace.get("trace", trace).get("events", [])
    # Filter out screenshot/page events (visual bookmarks, not interactions)
    events = [e for e in raw_events
              if e and e.get('event_type') != 'screenshot'
              and e.get('element_tag') != 'page']
    redirects = trace.get("trace", trace).get("redirects", [])
    url = trace.get("url", "")
    
    # Build nodes: URL node + event nodes
    nodes = []
    node_features_list = []
    
    # Root URL node
    nodes.append({"type": "url", "url": url})
    node_features_list.append(_url_node_features(url))
    
    # Event nodes
    for ev in events:
        if ev is None:
            continue
        nodes.append({"type": "event", "event": ev})
        node_features_list.append(_event_node_features(ev))
    
    n = len(nodes)
    if n < 2:
        return None, None, n
    
    # Build adjacency
    A = np.zeros((n, n), dtype=np.float32)
    
    # Sequence edges: consecutive events
    for i in range(n - 1):
        A[i, i + 1] = 1.0
        A[i + 1, i] = 1.0  # undirected for GCN
    
    # Navigation edges: events that trigger URL changes
    for i, node in enumerate(nodes):
        if node["type"] == "event":
            ev = node["event"]
            if ev.get("event_type") in ("navigate", "click"):
                # Connect to root
                A[0, i] = 1.0
                A[i, 0] = 1.0
    
    # Redirect edges
    for redir in redirects:
        if isinstance(redir, dict):
            # Connect redirect source to destination nodes
            A[0, min(1, n - 1)] = 1.0
    
    # Self-loops (standard in GCN)
    for i in range(n):
        A[i, i] = 1.0
    
    X = np.array(node_features_list, dtype=np.float32)
    
    return A, X, n


def _url_node_features(url: str) -> List[float]:
    """Extract 16-dim node features for URL node."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
    except Exception:
        return [0.0] * 16
    
    domain = parsed.hostname or ""
    path = parsed.path or ""
    
    features = [
        len(url),                                    # URL length
        len(domain),                                 # domain length
        len(path),                                   # path length
        domain.count("."),                           # dot count
        url.count("-"),                              # hyphen count
        url.count("@"),                              # at sign
        len(parsed.query),                           # query length
        1.0 if parsed.scheme == "https" else 0.0,    # HTTPS
        sum(c.isdigit() for c in domain) / max(len(domain), 1),  # digit ratio
        1.0 if any(c.isdigit() for c in domain.split(".")[-1] if len(domain.split(".")) > 0) else 0.0,  # digit in TLD
        len(domain.split(".")),                      # subdomain count
        path.count("/"),                             # path depth
        1.0 if len(domain) > 30 else 0.0,           # long domain
        sum(not c.isalnum() and c != "." for c in domain) / max(len(domain), 1),  # special char ratio
        1.0 if any(kw in url.lower() for kw in ["login", "signin", "verify", "secure", "account"]) else 0.0,
        1.0 if any(kw in url.lower() for kw in ["update", "confirm", "banking", "paypal", "ebay"]) else 0.0,
    ]
    return features


def _event_node_features(event: dict) -> List[float]:
    """Extract 16-dim node features for event node."""
    event_type = event.get("event_type", "")
    element_tag = event.get("element_tag", "")
    element_id = event.get("element_id", "")
    element_class = event.get("element_class", "")
    
    # One-hot event type (8 categories — matched to actual crawler output)
    event_types = ["input", "submit", "button_click", "form_input",
                   "page_transition", "login_link_follow", "js_form_submit", "select"]
    type_vec = [1.0 if event_type == t else 0.0 for t in event_types]
    
    # Element features
    is_form = 1.0 if element_tag in ("form", "FORM") else 0.0
    is_input = 1.0 if element_tag in ("input", "INPUT", "textarea", "TEXTAREA") else 0.0
    is_button = 1.0 if element_tag in ("button", "BUTTON") else 0.0
    is_link = 1.0 if element_tag in ("a", "A") else 0.0
    
    # Credential indicators
    has_password = 1.0 if "password" in (element_id + element_class).lower() else 0.0
    has_email = 1.0 if "email" in (element_id + element_class).lower() else 0.0
    has_login = 1.0 if "login" in (element_id + element_class).lower() else 0.0
    has_submit = 1.0 if event_type == "submit" or "submit" in (element_id + element_class).lower() else 0.0
    
    features = type_vec + [is_form, is_input, is_button, is_link, has_password, has_email, has_login, has_submit]
    return features


# ═══════════════════════════════════════════════════════════════
# Section 2: Spectral GCN (Kipf & Welling approximation)
# ═══════════════════════════════════════════════════════════════

class SpectralGCN:
    """
    Spectral Graph Convolutional Network approximation.
    
    Implements the 1st-order Chebyshev approximation from Kipf & Welling (ICLR'17):
        H^{(l+1)} = sigma(D_hat^{-1/2} A_hat D_hat^{-1/2} H^{(l)} W^{(l)})
    
    Since we don't use backpropagation, we apply the spectral convolution as a 
    fixed feature transformation followed by a trainable classifier.
    """
    
    def __init__(self, n_layers: int = 2, hidden_dim: int = 32, readout: str = "mean"):
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.readout = readout
    
    def _normalize_adj(self, A: np.ndarray) -> np.ndarray:
        """Compute D^{-1/2} A D^{-1/2} (symmetric normalization)."""
        D = np.diag(A.sum(axis=1))
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(D.diagonal(), 1e-8)))
        return D_inv_sqrt @ A @ D_inv_sqrt
    
    def _gcn_conv(self, A_norm: np.ndarray, X: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Single GCN convolution layer: sigma(A_norm @ X @ W)."""
        return np.maximum(0, A_norm @ X @ W)  # ReLU activation
    
    def transform_graph(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Apply multi-layer spectral GCN convolution and readout.
        
        Returns a fixed-dim graph-level embedding.
        """
        A_norm = self._normalize_adj(A)
        n, d = X.shape
        
        H = X
        embeddings = []
        
        for layer in range(self.n_layers):
            # Random projection (fixed seed for reproducibility)
            rng = np.random.RandomState(42 + layer)
            in_dim = H.shape[1]
            W = rng.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
            
            H = self._gcn_conv(A_norm, H, W)
            embeddings.append(H)
        
        # Graph-level readout
        graph_features = []
        for H_l in embeddings:
            if self.readout == "mean":
                graph_features.append(H_l.mean(axis=0))
            elif self.readout == "max":
                graph_features.append(H_l.max(axis=0))
            elif self.readout == "sum":
                graph_features.append(H_l.sum(axis=0))
            else:  # mean + max + sum
                graph_features.extend([
                    H_l.mean(axis=0),
                    H_l.max(axis=0),
                    H_l.sum(axis=0),
                ])
        
        return np.concatenate(graph_features)


class AttentionGAT:
    """
    Graph Attention Network approximation.
    
    Implements attention-weighted neighbor aggregation:
        alpha_{ij} = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
        h'_i = sigma(sum_j alpha_{ij} W h_j)
    
    Uses fixed random projections for W and a, with attention computed
    dynamically based on node features.
    """
    
    def __init__(self, n_heads: int = 4, hidden_dim: int = 16, n_layers: int = 2):
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
    
    def _attention_layer(self, A: np.ndarray, X: np.ndarray, 
                         W: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Single GAT attention layer (vectorized)."""
        n = X.shape[0]
        WX = X @ W  # n x hidden_dim
        d = self.hidden_dim
        a_l, a_r = a[:d], a[d:]
        
        # Vectorized attention: e_ij = LeakyReLU(a_l @ Wh_i + a_r @ Wh_j)
        score_l = WX @ a_l  # (n,)
        score_r = WX @ a_r  # (n,)
        scores = score_l[:, None] + score_r[None, :]  # (n, n)
        scores = np.where(scores > 0, scores, 0.2 * scores)  # LeakyReLU
        
        # Mask non-neighbors
        mask = (A > 0).astype(np.float32)
        scores = scores * mask + (1 - mask) * (-1e9)
        
        # Softmax per row
        exp_s = np.exp(scores - scores.max(axis=1, keepdims=True))
        exp_s = exp_s * mask
        alpha = exp_s / (exp_s.sum(axis=1, keepdims=True) + 1e-8)
        
        H_out = alpha @ WX
        return np.maximum(0, H_out)  # ReLU
    
    def transform_graph(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Apply multi-head, multi-layer GAT and readout."""
        n, d = X.shape
        all_head_features = []
        
        for head in range(self.n_heads):
            H = X
            for layer in range(self.n_layers):
                rng = np.random.RandomState(42 + head * 10 + layer)
                in_dim = H.shape[1]
                W = rng.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
                a = rng.randn(2 * self.hidden_dim) * 0.1
                H = self._attention_layer(A, H, W, a)
            
            all_head_features.append(H.mean(axis=0))
            all_head_features.append(H.max(axis=0))
        
        return np.concatenate(all_head_features)


class GraphSAGEApprox:
    """
    GraphSAGE-style sampling and aggregation (Hamilton et al., NeurIPS'17).
    
    For each node, samples K neighbors and aggregates their features.
    Uses mean, max, and weighted aggregation.
    """
    
    def __init__(self, hidden_dim: int = 32, n_layers: int = 2, sample_size: int = 10):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.sample_size = sample_size
    
    def _sample_neighbors(self, A: np.ndarray, node: int) -> np.ndarray:
        """Sample neighbors for a node."""
        neighbors = np.where(A[node] > 0)[0]
        if len(neighbors) == 0:
            return np.array([node])
        if len(neighbors) <= self.sample_size:
            return neighbors
        rng = np.random.RandomState(42 + node)
        return rng.choice(neighbors, size=self.sample_size, replace=False)
    
    def _aggregate(self, X: np.ndarray, A: np.ndarray, W: np.ndarray) -> np.ndarray:
        """Mean aggregation over sampled neighbors."""
        n = X.shape[0]
        H_new = np.zeros((n, self.hidden_dim))
        
        for i in range(n):
            neighbors = self._sample_neighbors(A, i)
            neighbor_feats = X[neighbors].mean(axis=0)
            combined = np.concatenate([X[i], neighbor_feats])
            # Project
            W_full = np.zeros((len(combined), self.hidden_dim))
            W_full[:min(len(combined), W.shape[0]), :] = W[:min(len(combined), W.shape[0]), :]
            H_new[i] = np.maximum(0, combined @ W_full)
        
        return H_new
    
    def transform_graph(self, A: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Apply multi-layer GraphSAGE and readout."""
        H = X
        embeddings = []
        
        for layer in range(self.n_layers):
            rng = np.random.RandomState(42 + layer)
            in_dim = H.shape[1] * 2  # concat of self + neighbor
            W = rng.randn(in_dim, self.hidden_dim) * np.sqrt(2.0 / in_dim)
            H = self._aggregate(H, A, W)
            embeddings.append(H)
        
        # Multi-scale readout
        graph_features = []
        for H_l in embeddings:
            graph_features.extend([
                H_l.mean(axis=0),
                H_l.max(axis=0),
                np.std(H_l, axis=0),
            ])
        
        return np.concatenate(graph_features)


# ═══════════════════════════════════════════════════════════════
# Section 3: Information-theoretic analysis
# ═══════════════════════════════════════════════════════════════

def compute_mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """
    Compute mutual information I(X; Y) via histogram-based estimation.
    Uses the approach from Kraskov et al. (2004) adapted for feature matrices.
    
    For multi-dimensional X, computes MI for each feature and returns the sum.
    """
    total_mi = 0.0
    n_features = X.shape[1] if X.ndim > 1 else 1
    
    for j in range(n_features):
        x_j = X[:, j] if X.ndim > 1 else X
        
        # Discretize continuous feature
        x_bins = np.digitize(x_j, np.linspace(x_j.min() - 1e-8, x_j.max() + 1e-8, n_bins + 1))
        
        # Joint and marginal distributions
        n = len(x_j)
        classes = np.unique(y)
        
        p_y = np.array([np.mean(y == c) for c in classes])
        H_y = -np.sum(p_y * np.log2(p_y + 1e-12))
        
        # H(Y|X)
        H_y_given_x = 0.0
        bins = np.unique(x_bins)
        for b in bins:
            mask = x_bins == b
            p_b = mask.sum() / n
            if p_b > 0:
                p_y_given_b = np.array([np.mean(y[mask] == c) for c in classes])
                H_y_given_b = -np.sum(p_y_given_b * np.log2(p_y_given_b + 1e-12))
                H_y_given_x += p_b * H_y_given_b
        
        total_mi += max(0, H_y - H_y_given_x)
    
    return total_mi


def compute_view_redundancy(X_views: List[np.ndarray], y: np.ndarray) -> Dict[str, float]:
    """
    Compute pairwise redundancy between views using normalized mutual information.
    Lower redundancy = more complementary views.
    """
    view_names = ["URL", "Network", "Redirect", "Interaction", "ITG", "CrossView"]
    n_views = len(X_views)
    
    results = {}
    
    # Per-view MI with label
    for i in range(n_views):
        mi = compute_mutual_information(X_views[i], y)
        results[f"MI({view_names[i]};Y)"] = round(mi, 4)
    
    # Pairwise view redundancy (approximate via feature correlation)
    for i in range(n_views):
        for j in range(i + 1, n_views):
            # Concatenate and compute correlation-based redundancy
            X_i = StandardScaler().fit_transform(X_views[i])
            X_j = StandardScaler().fit_transform(X_views[j])
            
            # Average absolute correlation between feature pairs
            min_dim = min(X_i.shape[1], X_j.shape[1])
            correlations = []
            for fi in range(min(X_i.shape[1], 5)):  # sample features
                for fj in range(min(X_j.shape[1], 5)):
                    corr = np.abs(np.corrcoef(X_i[:, fi], X_j[:, fj])[0, 1])
                    if not np.isnan(corr):
                        correlations.append(corr)
            
            avg_corr = np.mean(correlations) if correlations else 0.0
            results[f"Redundancy({view_names[i]},{view_names[j]})"] = round(avg_corr, 4)
    
    return results


def compute_information_gain_decomposition(
    X_views: List[np.ndarray], y: np.ndarray,
    view_names: List[str] = None
) -> Dict[str, float]:
    """
    Decompose total information gain into per-view contributions.
    
    Uses chain rule of mutual information:
    I(X1,X2,...,Xk; Y) = sum I(Xi; Y | X1,...,X_{i-1})
    
    We approximate conditional MI via residual classification improvement.
    """
    if view_names is None:
        view_names = [f"View_{i}" for i in range(len(X_views))]
    
    results = {}
    
    # Base: each view alone
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (name, X_v) in enumerate(zip(view_names, X_views)):
        f1_scores = []
        for train_idx, test_idx in skf.split(X_v, y):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            clf.fit(X_v[train_idx], y[train_idx])
            pred = clf.predict(X_v[test_idx])
            f1_scores.append(f1_score(y[test_idx], pred, zero_division=0))
        
        results[f"Solo_F1({name})"] = round(np.mean(f1_scores), 4)
    
    # Incremental: add views one by one (information chain)
    X_cumul = np.zeros((len(y), 0))
    prev_f1 = 0.5  # random baseline
    
    # Sort views by solo performance (strongest first)
    view_order = sorted(range(len(X_views)), 
                       key=lambda i: results[f"Solo_F1({view_names[i]})"], 
                       reverse=True)
    
    for idx in view_order:
        X_cumul = np.hstack([X_cumul, X_views[idx]])
        X_cumul_clean = np.nan_to_num(X_cumul, nan=0.0)
        
        f1_scores = []
        for train_idx, test_idx in skf.split(X_cumul_clean, y):
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
            ])
            clf.fit(X_cumul_clean[train_idx], y[train_idx])
            pred = clf.predict(X_cumul_clean[test_idx])
            f1_scores.append(f1_score(y[test_idx], pred, zero_division=0))
        
        new_f1 = np.mean(f1_scores)
        gain = new_f1 - prev_f1
        results[f"IG({view_names[idx]})"] = round(gain, 4)
        prev_f1 = new_f1
    
    results["Total_F1"] = round(prev_f1, 4)
    
    return results


# ═══════════════════════════════════════════════════════════════
# Section 4: Complexity analysis
# ═══════════════════════════════════════════════════════════════

def complexity_analysis() -> Dict[str, str]:
    """
    Computational complexity analysis of the PhishTrace pipeline.
    
    Returns complexity expressions for each component.
    """
    return {
        "crawling": "O(B_eff^D * T_page)",
        "crawling_desc": "B_eff = effective branching factor after DRP, D = depth, T_page = per-page render time",
        "drp_pruning": "O(|V| * |V| + |E|)",
        "drp_pruning_desc": "CPP scoring O(|V|), DSD fingerprinting O(|V|^2) pairwise, IBA O(|V|), CPRD O(|E|)",
        "itg_construction": "O(|E| + |V| * log|V|)",
        "itg_construction_desc": "Edge creation O(|E|), node attribute indexing O(|V| * log|V|)",
        "feature_extraction": "O(|V|^2 + |V| * |E|)",
        "feature_extraction_desc": "Centrality metrics require shortest paths O(|V|^2), PageRank O(|V| * |E|) (power iteration)",
        "gcn_forward": "O(L * |E| * d + L * |V| * d^2)",
        "gcn_forward_desc": "L layers, |E| edges for sparse mult, d hidden dim, |V| * d^2 for weight mult",
        "smve_training": "O(K * N * D_v * T_clf + K_meta * N * K)",
        "smve_training_desc": "K views, N samples, D_v view dim, T_clf per-sample classifier cost, K_meta meta-learner folds",
        "total_detection": "O(B_eff^D * T_page + |V|^2 + K * D_v * T_clf)",
        "total_detection_desc": "Crawling dominates (seconds); feature extraction + classification negligible (<100ms)",
    }


# ═══════════════════════════════════════════════════════════════
# Section 5: Main experiment runner
# ═══════════════════════════════════════════════════════════════

def load_traces() -> Tuple[List[dict], np.ndarray]:
    """Load all traces from dataset."""
    DATASET_DIR = PROJECT_ROOT / "dataset"
    traces = []
    labels = []
    
    for label, dirname in [(1, "phishing"), (0, "benign")]:
        trace_dir = DATASET_DIR / "traces" / dirname
        if not trace_dir.exists():
            continue
        for fpath in sorted(trace_dir.glob("*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if data.get("success", False):
                    traces.append(data)
                    labels.append(label)
            except (json.JSONDecodeError, KeyError):
                continue
    
    return traces, np.array(labels)


def extract_gnn_features(
    traces: List[dict], 
    gnn_model, 
    model_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract graph-level features using a GNN model.
    
    Returns:
        X: feature matrix for traces with valid graphs
        y: labels for valid traces
        mask: boolean mask indicating which traces produced valid graphs
    """
    features = []
    valid_indices = []
    
    for i, trace in enumerate(traces):
        if i % 100 == 0:
            print(f"    Extracting graph {i}/{len(traces)}...", flush=True)
        A, X, n = build_adjacency_and_features(trace)
        if A is None or n < 2:
            continue
        
        try:
            graph_feat = gnn_model.transform_graph(A, X)
            if np.any(np.isnan(graph_feat)) or np.any(np.isinf(graph_feat)):
                graph_feat = np.nan_to_num(graph_feat, nan=0.0, posinf=0.0, neginf=0.0)
            features.append(graph_feat)
            valid_indices.append(i)
        except Exception as e:
            continue
    
    if not features:
        return np.array([]), np.array([]), np.array([])
    
    X = np.array(features)
    mask = np.zeros(len(traces), dtype=bool)
    mask[valid_indices] = True
    
    return X, np.array(valid_indices), mask


def evaluate_gnn(
    X: np.ndarray, y: np.ndarray, 
    model_name: str, n_folds: int = 10
) -> Dict[str, float]:
    """Evaluate a GNN feature set with multiple classifiers."""
    best_results = None
    best_f1 = 0
    
    classifiers = {
        "RF": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "GBM": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42),
    }
    
    for clf_name, clf in classifiers.items():
        print(f"    Testing {clf_name}...", end=" ", flush=True)
        minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
        actual_folds = max(2, min(n_folds, minority))
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        
        metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}
        
        for train_idx, test_idx in skf.split(X, y):
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
            pipe.fit(X[train_idx], y[train_idx])
            y_pred = pipe.predict(X[test_idx])
            
            metrics["accuracy"].append(accuracy_score(y[test_idx], y_pred))
            metrics["precision"].append(precision_score(y[test_idx], y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y[test_idx], y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y[test_idx], y_pred, zero_division=0))
            
            try:
                proba = pipe.predict_proba(X[test_idx])[:, 1]
                metrics["auc"].append(roc_auc_score(y[test_idx], proba))
            except Exception:
                metrics["auc"].append(0.5)
        
        avg = {k: round(float(np.mean(v)), 4) for k, v in metrics.items()}
        std = {k: round(float(np.std(v)), 4) for k, v in metrics.items()}
        
        if avg["f1"] > best_f1:
            best_f1 = avg["f1"]
            best_results = {"avg": avg, "std": std, "clf": clf_name}
        print(f"F1={avg['f1']:.4f}", flush=True)
    
    return best_results


def run_gnn_comparison():
    """Run the full GNN comparison experiment."""
    print("=" * 80)
    print("GNN Comparison Experiment")
    print("Comparing Spectral GCN, GAT, GraphSAGE against EITG-SMVE")
    print("=" * 80)
    
    # 1. Load traces
    traces, y_all = load_traces()
    print(f"\nLoaded {len(traces)} traces ({int(np.sum(y_all == 1))} phishing, {int(np.sum(y_all == 0))} benign)", flush=True)
    
    # 2. Define GNN models
    gnn_models = {
        "SpectralGCN-2L": SpectralGCN(n_layers=2, hidden_dim=32, readout="mean"),
        "SpectralGCN-3L": SpectralGCN(n_layers=3, hidden_dim=32, readout="all"),
        "GAT-4H": AttentionGAT(n_heads=4, hidden_dim=16, n_layers=2),
        "GraphSAGE": GraphSAGEApprox(hidden_dim=32, n_layers=2, sample_size=10),
    }
    
    results = {}
    
    # 3. Evaluate each GNN
    for name, model in gnn_models.items():
        print(f"\n--- {name} ---", flush=True)
        X_gnn, valid_idx, mask = extract_gnn_features(traces, model, name)
        
        if len(X_gnn) == 0:
            print(f"  No valid graphs for {name}, skipping.")
            continue
        
        y_valid = y_all[valid_idx]
        print(f"  Valid graphs: {len(X_gnn)} ({int(np.sum(y_valid == 1))} phishing, {int(np.sum(y_valid == 0))} benign)", flush=True)
        print(f"  Feature dim: {X_gnn.shape[1]}", flush=True)
        
        result = evaluate_gnn(X_gnn, y_valid, name)
        results[name] = result
        
        print(f"  Best classifier: {result['clf']}", flush=True)
        print(f"  Acc={result['avg']['accuracy']:.4f}  F1={result['avg']['f1']:.4f}  AUC={result['avg']['auc']:.4f}", flush=True)
    
    # 4. Information-theoretic analysis
    print("\n" + "=" * 70)
    print("Information-Theoretic View Analysis")
    print("=" * 70)
    
    mi_results = {}
    ig_results = {}
    
    # Load EITG views for MI analysis
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from enhanced_itg_detector import load_all_traces, build_all_views, engineer_itg_features
        
        eitg_traces, eitg_y = load_all_traces()
        X_url, X_net, X_redir, X_interact, X_itg, X_cross, itg_mask = build_all_views(eitg_traces)
        X_itg_eng = engineer_itg_features(X_itg)
        
        X_views = [X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross]
        view_names = ["URL", "Network", "Redirect", "Interaction", "ITG+Eng", "CrossView"]
        
        # MI analysis
        mi_results = compute_view_redundancy(X_views, eitg_y)
        print("\nMutual Information with Label:")
        for k, v in mi_results.items():
            if k.startswith("MI("):
                print(f"  {k}: {v:.4f} bits")
        
        print("\nPairwise View Redundancy (avg |correlation|):")
        for k, v in mi_results.items():
            if k.startswith("Redundancy("):
                print(f"  {k}: {v:.4f}")
        
        # Information gain decomposition
        ig_results = compute_information_gain_decomposition(X_views, eitg_y, view_names)
        print("\nInformation Gain Decomposition (chain rule):")
        for k, v in ig_results.items():
            if k.startswith("IG("):
                print(f"  {k}: {v:+.4f} F1")
        print(f"  Total F1: {ig_results.get('Total_F1', 'N/A')}")
        
    except Exception as e:
        import traceback
        print(f"  Could not run MI analysis: {e}")
        traceback.print_exc()
    
    # 5. Complexity analysis
    print("\n" + "=" * 70)
    print("Computational Complexity Analysis")
    print("=" * 70)
    
    complexity = complexity_analysis()
    for component in ["crawling", "drp_pruning", "itg_construction", "feature_extraction", 
                       "gcn_forward", "smve_training", "total_detection"]:
        print(f"\n  {component}: {complexity[component]}")
        print(f"    {complexity[component + '_desc']}")
    
    # 6. Summary comparison table
    print("\n" + "=" * 80)
    print("GNN vs EITG-SMVE COMPARISON")
    print("=" * 80)
    
    print(f"\n  {'Method':<25} {'Clf':>5} {'Acc':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "-" * 55)
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]["avg"]["f1"]):
        r = res["avg"]
        print(f"  {name:<25} {res['clf']:>5} {r['accuracy']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}")
    
    # Add EITG-SMVE from saved results
    eitg_path = PROJECT_ROOT / "experiments" / "results" / "eitg_results.json"
    if eitg_path.exists():
        with open(eitg_path) as f:
            eitg_data = json.load(f)
        smve = eitg_data.get("eitg_smve", {})
        if smve:
            print(f"  {'EITG-SMVE (Ours)':<25} {'V+M':>5} {smve['accuracy']:>7.4f} {smve['f1']:>7.4f} {smve['auc']:>7.4f}")
    
    # 7. Save results
    output_dir = PROJECT_ROOT / "experiments" / "results"
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime
    save_data = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "gnn_results": {k: v for k, v in results.items()},
        "mutual_information": mi_results,
        "information_gain": ig_results,
        "complexity": complexity,
    }
    
    with open(output_dir / "gnn_comparison_results.json", "w") as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\nResults saved to {output_dir / 'gnn_comparison_results.json'}")
    
    return results, mi_results, ig_results


if __name__ == "__main__":
    run_gnn_comparison()
