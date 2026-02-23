"""
Adversarial Robustness Evaluation for PhishTrace

Tests the resilience of graph-based phishing detection against adversarial
perturbations applied to the Interaction Trace Graph (ITG). We implement three
attack families:

1. **Feature-Space Attacks (FGSM-style):** Gradient-based perturbations on the
   feature vector, approximating FGSM~\cite{goodfellow2015explaining} via finite
   differences for the tree-ensemble classifier.

2. **Graph Structural Attacks:** Topology-level mutations (node injection,
   edge rewiring, graph inflation) that simulate evasion techniques an attacker
   could deploy in practice by modifying page behavior.

3. **Semantic-Preserving Attacks:** Perturbations that maintain phishing
   functionality (e.g., adding benign-looking navigation elements, splitting
   credential forms across iframes) while attempting to shift the feature
   distribution toward the benign class.

Each attack is evaluated at multiple perturbation budgets (epsilon) and the
detection accuracy degradation is reported.
"""

import json
import copy
import logging
import warnings
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
from pathlib import Path
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analyzer.graph_builder import InteractionGraphBuilder

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    'num_nodes', 'num_edges', 'avg_degree', 'max_degree',
    'graph_density', 'clustering_coefficient', 'num_connected_components',
    'avg_path_length', 'diameter',
    'num_forms', 'num_redirects', 'has_password_input', 'has_email_input',
    'external_redirects', 'num_input_fields', 'num_buttons', 'num_links',
    'max_redirect_depth', 'form_to_node_ratio',
    'total_interaction_time', 'avg_time_between_events', 'event_frequency',
    'betweenness_centrality_max', 'closeness_centrality_max', 'pagerank_max',
    'in_degree_max', 'out_degree_max',
    'requests_sensitive_data', 'credential_fields_count', 'financial_fields_count',
]


# ---------------------------------------------------------------------------
# Attack 1: Feature-Space FGSM-style perturbation
# ---------------------------------------------------------------------------

class FeatureSpaceAttack:
    """
    Approximates FGSM by estimating gradients via finite differences on the
    trained classifier's decision function, then perturbing features along
    the estimated gradient direction.
    """

    def __init__(self, epsilon: float = 0.1, delta: float = 1e-4):
        self.epsilon = epsilon
        self.delta = delta

    def attack(self, model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Generate adversarial examples for phishing samples (y=1).

        Perturbs features to minimize P(phishing), i.e., shift toward benign.
        """
        X_adv = X.copy()
        phishing_mask = (y == 1)

        for i in np.where(phishing_mask)[0]:
            x = X[i].copy()
            grad = self._estimate_gradient(model, x)
            # Move opposite to gradient of P(phishing) (gradient descent on class 1 prob)
            perturbation = -self.epsilon * np.sign(grad)
            X_adv[i] = x + perturbation
            # Do NOT clip — features are already scaled (zero-centered),
            # clipping to 0 would destroy the StandardScaler distribution.

        return X_adv

    def _estimate_gradient(self, model, x: np.ndarray) -> np.ndarray:
        """Finite-difference gradient estimation for P(phishing)."""
        grad = np.zeros_like(x)
        x_flat = x.reshape(1, -1)
        try:
            base_prob = model.predict_proba(x_flat)[0, 1]
        except (IndexError, ValueError):
            return grad

        for j in range(len(x)):
            x_plus = x.copy()
            x_plus[j] += self.delta
            try:
                prob_plus = model.predict_proba(x_plus.reshape(1, -1))[0, 1]
            except (IndexError, ValueError):
                continue
            grad[j] = (prob_plus - base_prob) / self.delta

        return grad


# ---------------------------------------------------------------------------
# Attack 2: Graph Structural Attacks
# ---------------------------------------------------------------------------

class GraphStructuralAttack:
    """
    Modifies the trace graph structure to evade detection while preserving
    the phishing payload. Three sub-attacks:

    (a) Node Injection: Adds benign-looking navigation nodes to dilute
        suspicious structural patterns.
    (b) Edge Rewiring: Reorganizes interaction flow to reduce centrality
        concentration at credential-harvesting nodes.
    (c) Graph Inflation: Adds dummy interaction cycles to shift temporal
        and density features toward benign distributions.
    """

    def __init__(self, n_inject: int = 5, rewire_prob: float = 0.2,
                 inflate_factor: int = 3):
        self.n_inject = n_inject
        self.rewire_prob = rewire_prob
        self.inflate_factor = inflate_factor

    def inject_nodes(self, trace: dict) -> dict:
        """Add benign-looking navigation events to the trace."""
        perturbed = copy.deepcopy(trace)
        events = perturbed.get('events', [])
        rng = np.random.RandomState(hash(str(trace.get('url', ''))) % (2**31))

        benign_events = [
            {'event_type': 'link_detected', 'element_tag': 'a',
             'element_id': f'nav_{i}', 'element_class': 'nav-link',
             'url_before': trace.get('url', ''), 'url_after': trace.get('url', ''),
             'timestamp': 0, 'form_data': {}}
            for i in range(self.n_inject)
        ]

        # Insert at random positions
        for ev in benign_events:
            pos = rng.randint(0, max(1, len(events)))
            events.insert(pos, ev)

        perturbed['events'] = events
        return perturbed

    def rewire_edges(self, trace: dict) -> dict:
        """Rewire interaction edges to reduce centrality concentration."""
        perturbed = copy.deepcopy(trace)
        events = perturbed.get('events', [])
        rng = np.random.RandomState(hash(str(trace.get('url', ''))) % (2**31) + 1)

        for i in range(len(events)):
            if events[i] is None:
                continue
            if rng.random() < self.rewire_prob and len(events) > 2:
                # Swap URL transitions to break centrality patterns
                j = rng.randint(0, len(events))
                if j != i and events[j] is not None:
                    events[i]['url_after'], events[j]['url_after'] = (
                        events[j].get('url_after', ''),
                        events[i].get('url_after', '')
                    )

        perturbed['events'] = events
        return perturbed

    def inflate_graph(self, trace: dict) -> dict:
        """Add dummy interaction cycles to shift density/temporal features."""
        perturbed = copy.deepcopy(trace)
        events = perturbed.get('events', [])

        # Create a cycle of dummy events
        base_url = trace.get('url', 'http://example.com')
        for cycle in range(self.inflate_factor):
            dummy_events = [
                {'event_type': 'navigation', 'element_tag': 'div',
                 'element_id': f'dummy_cycle_{cycle}_{j}',
                 'element_class': 'content',
                 'url_before': base_url, 'url_after': base_url,
                 'timestamp': 0, 'form_data': {}}
                for j in range(3)
            ]
            events.extend(dummy_events)

        perturbed['events'] = events
        return perturbed

    def combined_attack(self, trace: dict) -> dict:
        """Apply all three structural attacks."""
        t1 = self.inject_nodes(trace)
        t2 = self.rewire_edges(t1)
        t3 = self.inflate_graph(t2)
        return t3


# ---------------------------------------------------------------------------
# Attack 3: Semantic-Preserving Attacks
# ---------------------------------------------------------------------------

class SemanticAttack:
    """
    Perturbations that preserve phishing functionality but attempt to shift
    feature distributions. Simulates a sophisticated attacker who modifies
    page behavior without breaking credential harvesting.
    """

    def __init__(self, n_decoy_links: int = 10, split_forms: bool = True):
        self.n_decoy_links = n_decoy_links
        self.split_forms = split_forms

    def add_decoy_navigation(self, trace: dict) -> dict:
        """Add realistic-looking navigation elements (footer links, nav bar)."""
        perturbed = copy.deepcopy(trace)
        events = perturbed.get('events', [])
        base_url = trace.get('url', '')

        # Simulate a typical benign page's navigation structure
        decoy_types = [
            ('a', 'footer-link', 'link_detected'),
            ('a', 'nav-item', 'link_detected'),
            ('button', 'menu-toggle', 'button_detected'),
            ('div', 'breadcrumb', 'navigation'),
            ('a', 'logo-link', 'link_detected'),
        ]

        for i in range(self.n_decoy_links):
            tag, cls, etype = decoy_types[i % len(decoy_types)]
            events.append({
                'event_type': etype,
                'element_tag': tag,
                'element_id': f'decoy_{cls}_{i}',
                'element_class': cls,
                'url_before': base_url,
                'url_after': base_url,
                'timestamp': 0,
                'form_data': {}
            })

        perturbed['events'] = events
        return perturbed

    def split_credential_form(self, trace: dict) -> dict:
        """Split credential form into multiple smaller forms (iframe simulation)."""
        perturbed = copy.deepcopy(trace)
        events = perturbed.get('events', [])

        new_events = []
        for ev in events:
            if ev is None:
                new_events.append(ev)
                continue
            form_data = ev.get('form_data', {})
            if isinstance(form_data, dict) and len(form_data) > 1:
                # Split into individual field submissions
                for key, val in form_data.items():
                    split_ev = copy.deepcopy(ev)
                    split_ev['form_data'] = {key: val}
                    split_ev['event_type'] = 'form_input'
                    new_events.append(split_ev)
            else:
                new_events.append(ev)

        perturbed['events'] = new_events
        return perturbed


# ---------------------------------------------------------------------------
# Feature extraction helper
# ---------------------------------------------------------------------------

def extract_features(trace_data: dict) -> np.ndarray:
    """Extract feature vector from trace data."""
    builder = InteractionGraphBuilder()
    graph = builder.build_graph_from_dict(trace_data)
    features = builder.extract_features(graph)
    vec = []
    for name in FEATURE_NAMES:
        val = getattr(features, name, 0)
        if isinstance(val, bool):
            val = int(val)
        vec.append(float(val))
    return np.array(vec)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_adversarial_evaluation(dataset_dir: str = None) -> Dict:
    """Run full adversarial robustness evaluation.

    Returns dict with results for each attack type and epsilon.
    """
    if dataset_dir is None:
        dataset_dir = str(Path(__file__).parent.parent / 'dataset')

    dataset_path = Path(dataset_dir)

    print("\n" + "=" * 70)
    print("ADVERSARIAL ROBUSTNESS EVALUATION")
    print("=" * 70)

    # Load traces (handle nested 'trace' key format)
    traces = []
    labels = []
    for label_name, label_int in [('phishing', 1), ('benign', 0)]:
        trace_dir = dataset_path / 'traces' / label_name
        if not trace_dir.exists():
            continue
        for trace_file in sorted(trace_dir.glob('*.json')):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # Only use successful traces (consistent with EITG methodology)
                if not raw.get('success', False):
                    continue
                # Unwrap nested trace format
                trace_data = raw.get('trace', raw)
                if 'url' not in trace_data and 'url' in raw:
                    trace_data['url'] = raw['url']
                traces.append(trace_data)
                labels.append(label_int)
            except Exception as e:
                logger.warning(f"Skipping {trace_file.name}: {e}")

    y = np.array(labels)
    print(f"Loaded {len(traces)} traces ({sum(labels)} phishing, "
          f"{len(labels) - sum(labels)} benign)")

    if len(traces) < 20:
        print("[WARN] Too few traces for adversarial evaluation.")
        return {}

    # Extract clean features
    print("Extracting clean features...")
    X_clean = np.array([extract_features(t) for t in traces])

    # Train base model
    model = RandomForestClassifier(n_estimators=500, max_depth=None,
                                   min_samples_split=2, random_state=42,
                                   class_weight='balanced')

    # Train-test split for adversarial evaluation
    # NOTE: fit scaler on train only to avoid data leakage
    X_train_raw, X_test_raw, y_train, y_test, traces_train, traces_test = train_test_split(
        X_clean, y, traces, test_size=0.3, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    model.fit(X_train, y_train)

    clean_preds = model.predict(X_test)
    clean_acc = accuracy_score(y_test, clean_preds)
    clean_f1 = f1_score(y_test, clean_preds, zero_division=0)
    print(f"\nClean performance: Acc={clean_acc:.4f}, F1={clean_f1:.4f}")

    results = {'clean': {'accuracy': round(clean_acc, 4), 'f1': round(clean_f1, 4)}}

    # --- Attack 1: Feature-Space FGSM ---
    print("\n--- Feature-Space FGSM Attack ---")
    fgsm_results = {}
    for eps in [0.01, 0.05, 0.10, 0.20, 0.50]:
        attack = FeatureSpaceAttack(epsilon=eps)
        X_adv = attack.attack(model, X_test, y_test)
        adv_preds = model.predict(X_adv)
        adv_acc = accuracy_score(y_test, adv_preds)
        adv_f1 = f1_score(y_test, adv_preds, zero_division=0)

        # Evasion rate: fraction of phishing correctly detected before but not after
        phishing_mask = (y_test == 1)
        clean_detected = clean_preds[phishing_mask] == 1
        adv_detected = adv_preds[phishing_mask] == 1
        evasion_rate = float(np.sum(clean_detected & ~adv_detected)) / max(1, np.sum(clean_detected))

        fgsm_results[eps] = {
            'accuracy': round(adv_acc, 4),
            'f1': round(adv_f1, 4),
            'evasion_rate': round(evasion_rate, 4),
        }
        print(f"  eps={eps:.2f}: Acc={adv_acc:.4f}, F1={adv_f1:.4f}, "
              f"Evasion={evasion_rate:.4f}")

    results['fgsm'] = fgsm_results

    # --- Attack 2: Graph Structural Attacks ---
    print("\n--- Graph Structural Attacks ---")
    struct_attack = GraphStructuralAttack(n_inject=5, rewire_prob=0.2, inflate_factor=3)

    for attack_name, attack_fn in [
        ('node_injection', struct_attack.inject_nodes),
        ('edge_rewiring', struct_attack.rewire_edges),
        ('graph_inflation', struct_attack.inflate_graph),
        ('combined_structural', struct_attack.combined_attack),
    ]:
        # Apply structural attack to phishing traces
        adv_features = []
        for i, trace in enumerate(traces_test):
            if y_test[i] == 1:
                try:
                    perturbed = attack_fn(trace)
                    feats = extract_features(perturbed)
                    adv_features.append(scaler.transform(feats.reshape(1, -1))[0])
                except Exception:
                    adv_features.append(X_test[i])
            else:
                adv_features.append(X_test[i])

        X_adv = np.array(adv_features)
        adv_preds = model.predict(X_adv)
        adv_acc = accuracy_score(y_test, adv_preds)
        adv_f1 = f1_score(y_test, adv_preds, zero_division=0)

        phishing_mask = (y_test == 1)
        clean_detected = clean_preds[phishing_mask] == 1
        adv_detected = adv_preds[phishing_mask] == 1
        evasion_rate = float(np.sum(clean_detected & ~adv_detected)) / max(1, np.sum(clean_detected))

        results[attack_name] = {
            'accuracy': round(adv_acc, 4),
            'f1': round(adv_f1, 4),
            'evasion_rate': round(evasion_rate, 4),
        }
        print(f"  {attack_name}: Acc={adv_acc:.4f}, F1={adv_f1:.4f}, "
              f"Evasion={evasion_rate:.4f}")

    # --- Attack 3: Semantic-Preserving Attacks ---
    print("\n--- Semantic-Preserving Attacks ---")
    semantic = SemanticAttack(n_decoy_links=10, split_forms=True)

    for attack_name, attack_fn in [
        ('decoy_navigation', semantic.add_decoy_navigation),
        ('form_splitting', semantic.split_credential_form),
    ]:
        adv_features = []
        for i, trace in enumerate(traces_test):
            if y_test[i] == 1:
                try:
                    perturbed = attack_fn(trace)
                    feats = extract_features(perturbed)
                    adv_features.append(scaler.transform(feats.reshape(1, -1))[0])
                except Exception:
                    adv_features.append(X_test[i])
            else:
                adv_features.append(X_test[i])

        X_adv = np.array(adv_features)
        adv_preds = model.predict(X_adv)
        adv_acc = accuracy_score(y_test, adv_preds)
        adv_f1 = f1_score(y_test, adv_preds, zero_division=0)

        phishing_mask = (y_test == 1)
        clean_detected = clean_preds[phishing_mask] == 1
        adv_detected = adv_preds[phishing_mask] == 1
        evasion_rate = float(np.sum(clean_detected & ~adv_detected)) / max(1, np.sum(clean_detected))

        results[attack_name] = {
            'accuracy': round(adv_acc, 4),
            'f1': round(adv_f1, 4),
            'evasion_rate': round(evasion_rate, 4),
        }
        print(f"  {attack_name}: Acc={adv_acc:.4f}, F1={adv_f1:.4f}, "
              f"Evasion={evasion_rate:.4f}")

    # Summary
    print("\n" + "=" * 70)
    print("ADVERSARIAL ROBUSTNESS SUMMARY")
    print(f"{'Attack':>25} {'Accuracy':>10} {'F1':>10} {'Evasion%':>10}")
    print("-" * 58)
    print(f"{'Clean (no attack)':>25} {results['clean']['accuracy']:>10.4f} "
          f"{results['clean']['f1']:>10.4f} {'---':>10}")
    for key in ['node_injection', 'edge_rewiring', 'graph_inflation',
                'combined_structural', 'decoy_navigation', 'form_splitting']:
        if key in results:
            r = results[key]
            print(f"{key:>25} {r['accuracy']:>10.4f} {r['f1']:>10.4f} "
                  f"{r['evasion_rate']:>10.4f}")
    if 'fgsm' in results:
        for eps, r in results['fgsm'].items():
            print(f"{'FGSM eps=' + str(eps):>25} {r['accuracy']:>10.4f} "
                  f"{r['f1']:>10.4f} {r['evasion_rate']:>10.4f}")

    return results


if __name__ == '__main__':
    results = run_adversarial_evaluation()
    # Save results
    import json as _json
    out_path = Path(__file__).parent / 'results' / 'adversarial_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        _json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
