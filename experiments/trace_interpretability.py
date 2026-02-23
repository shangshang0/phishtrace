"""
Trace Interpretability Module for PhishTrace

Provides human-readable explanations of phishing detection decisions by
analyzing which interaction patterns in the ITG contributed most to the
classification. Three complementary explanation methods:

1. **Feature Attribution:** SHAP-inspired local feature importance using
   tree-based model structure (no external SHAP dependency).
2. **Critical Path Analysis:** Identifies the most suspicious interaction
   sequence (subgraph) within the ITG that drives detection.
3. **Contrastive Explanation:** Shows which features differ most between
   the analyzed ITG and the closest benign prototype, highlighting the
   discriminative interaction patterns.
"""

import json
import logging
import warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.analyzer.graph_builder import InteractionGraphBuilder, GraphFeatures

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

# Human-readable category labels
FEATURE_CATEGORIES = {
    'topological': [
        'num_nodes', 'num_edges', 'avg_degree', 'max_degree',
        'graph_density', 'clustering_coefficient', 'num_connected_components',
        'avg_path_length', 'diameter',
    ],
    'behavioral': [
        'num_forms', 'num_redirects', 'has_password_input', 'has_email_input',
        'external_redirects', 'num_input_fields', 'num_buttons', 'num_links',
        'max_redirect_depth', 'form_to_node_ratio',
    ],
    'temporal': [
        'total_interaction_time', 'avg_time_between_events', 'event_frequency',
    ],
    'centrality': [
        'betweenness_centrality_max', 'closeness_centrality_max', 'pagerank_max',
        'in_degree_max', 'out_degree_max',
    ],
    'sensitive_data': [
        'requests_sensitive_data', 'credential_fields_count', 'financial_fields_count',
    ],
}

# Natural language templates for feature descriptions
FEATURE_DESCRIPTIONS = {
    'num_nodes': 'interaction graph has {val:.0f} nodes (interaction elements)',
    'num_edges': 'graph contains {val:.0f} edges (interaction transitions)',
    'avg_degree': 'average node connectivity is {val:.2f}',
    'max_degree': 'most connected node has {val:.0f} connections',
    'graph_density': 'graph density is {val:.4f} ({qual})',
    'clustering_coefficient': 'local clustering coefficient is {val:.4f}',
    'num_connected_components': '{val:.0f} disconnected interaction component(s)',
    'avg_path_length': 'average interaction path length is {val:.2f} steps',
    'diameter': 'longest shortest path spans {val:.0f} steps',
    'num_forms': '{val:.0f} form submission(s) detected',
    'num_redirects': '{val:.0f} URL redirect(s) observed',
    'has_password_input': 'password input field {"present" if val else "absent"}',
    'has_email_input': 'email input field {"present" if val else "absent"}',
    'external_redirects': '{val:.0f} external domain redirect(s)',
    'num_input_fields': '{val:.0f} input field(s) in the interaction trace',
    'num_buttons': '{val:.0f} button element(s) detected',
    'num_links': '{val:.0f} navigational link(s)',
    'max_redirect_depth': 'deepest redirect chain is {val:.0f} hops',
    'form_to_node_ratio': 'form-to-node ratio is {val:.4f}',
    'total_interaction_time': 'total interaction session spans {val:.2f} seconds',
    'avg_time_between_events': 'average inter-event interval is {val:.2f} seconds',
    'event_frequency': 'interaction event rate is {val:.2f} events/second',
    'betweenness_centrality_max': 'maximum betweenness centrality is {val:.4f}',
    'closeness_centrality_max': 'maximum closeness centrality is {val:.4f}',
    'pagerank_max': 'maximum PageRank score is {val:.4f}',
    'in_degree_max': 'highest in-degree is {val:.0f}',
    'out_degree_max': 'highest out-degree is {val:.0f}',
    'requests_sensitive_data': 'sensitive data request {"detected" if val else "not detected"}',
    'credential_fields_count': '{val:.0f} credential-related field(s)',
    'financial_fields_count': '{val:.0f} financial data field(s)',
}


@dataclass
class TraceExplanation:
    """Structured explanation of a detection decision."""
    url: str = ''
    prediction: str = ''           # 'phishing' or 'benign'
    confidence: float = 0.0
    risk_score: float = 0.0        # 0-100

    # Feature attribution
    top_contributing_features: List[Tuple[str, float, str]] = field(default_factory=list)
    category_contributions: Dict[str, float] = field(default_factory=dict)

    # Critical path
    critical_path: List[str] = field(default_factory=list)
    critical_path_description: str = ''

    # Contrastive explanation
    contrastive_features: List[Tuple[str, float, float, str]] = field(default_factory=list)

    # Natural language summary
    summary: str = ''

    def to_dict(self) -> dict:
        return {
            'url': self.url,
            'prediction': self.prediction,
            'confidence': round(self.confidence, 4),
            'risk_score': round(self.risk_score, 1),
            'top_features': [
                {'name': n, 'importance': round(v, 4), 'description': d}
                for n, v, d in self.top_contributing_features
            ],
            'category_contributions': {
                k: round(v, 4) for k, v in self.category_contributions.items()
            },
            'critical_path': self.critical_path,
            'critical_path_description': self.critical_path_description,
            'contrastive_features': [
                {'name': n, 'sample_val': round(sv, 4), 'benign_avg': round(ba, 4),
                 'description': d}
                for n, sv, ba, d in self.contrastive_features
            ],
            'summary': self.summary,
        }


class TraceInterpreter:
    """
    Interprets PhishTrace detection decisions with multi-method explanations.
    """

    def __init__(self):
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.benign_prototypes: Optional[np.ndarray] = None  # mean feature vector
        self.benign_std: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray,
            model: RandomForestClassifier = None):
        """Train the interpreter on labeled data.

        Args:
            X: Feature matrix (n_samples, n_features), raw (unscaled).
            y: Labels (0=benign, 1=phishing).
            model: Pre-trained model (if None, trains a new one).
        """
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        if model is not None:
            self.model = model
        else:
            self.model = RandomForestClassifier(
                n_estimators=500, max_depth=None,
                min_samples_split=2, random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X_scaled, y)

        # Compute benign prototypes
        benign_mask = (y == 0)
        self.benign_prototypes = X[benign_mask].mean(axis=0)
        self.benign_std = X[benign_mask].std(axis=0)
        self.benign_std[self.benign_std == 0] = 1.0

        self._fitted = True

    def explain(self, trace_data: dict) -> TraceExplanation:
        """Generate full explanation for a single trace."""
        if not self._fitted:
            raise RuntimeError("Call fit() first")

        builder = InteractionGraphBuilder()
        graph = builder.build_graph_from_dict(trace_data)
        features_obj = builder.extract_features(graph)

        # Convert to vector
        x_raw = np.array([
            float(getattr(features_obj, name, 0) if not isinstance(getattr(features_obj, name, 0), bool)
                  else int(getattr(features_obj, name, 0)))
            for name in FEATURE_NAMES
        ])
        x_scaled = self.scaler.transform(x_raw.reshape(1, -1))

        # Prediction
        pred = self.model.predict(x_scaled)[0]
        proba = self.model.predict_proba(x_scaled)[0]
        confidence = float(max(proba))
        phish_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        explanation = TraceExplanation(
            url=trace_data.get('url', 'unknown'),
            prediction='phishing' if pred == 1 else 'benign',
            confidence=confidence,
            risk_score=phish_prob * 100,
        )

        # 1. Feature Attribution (tree-based local importance)
        explanation.top_contributing_features = self._feature_attribution(
            x_scaled[0], x_raw
        )

        # 2. Category-level contribution
        explanation.category_contributions = self._category_contributions(
            explanation.top_contributing_features
        )

        # 3. Critical path analysis
        explanation.critical_path, explanation.critical_path_description = (
            self._critical_path_analysis(graph, trace_data)
        )

        # 4. Contrastive explanation
        explanation.contrastive_features = self._contrastive_explanation(x_raw)

        # 5. Natural language summary
        explanation.summary = self._generate_summary(explanation)

        return explanation

    def _feature_attribution(self, x_scaled: np.ndarray,
                             x_raw: np.ndarray) -> List[Tuple[str, float, str]]:
        """Compute per-feature importance via mean decrease in impurity
        weighted by how far the sample's value deviates from the split."""
        global_imp = self.model.feature_importances_
        n_features = min(len(FEATURE_NAMES), len(global_imp))

        # Weight global importance by z-score of the feature value
        z_scores = np.abs(x_scaled[:n_features])
        weighted = global_imp[:n_features] * (1.0 + z_scores)
        weighted /= (weighted.sum() + 1e-10)

        # Build attribution list
        attributions = []
        for i in range(n_features):
            name = FEATURE_NAMES[i]
            val = float(x_raw[i])
            imp = float(weighted[i])

            # Generate description
            qual = 'sparse' if val < 0.1 else ('moderate' if val < 0.5 else 'dense')
            desc = f"{name}={val:.3g}"
            attributions.append((name, imp, desc))

        # Sort by importance, return top 10
        attributions.sort(key=lambda t: -t[1])
        return attributions[:10]

    def _category_contributions(
        self, attributions: List[Tuple[str, float, str]]
    ) -> Dict[str, float]:
        """Aggregate feature attributions by category."""
        cat_scores = defaultdict(float)
        for name, imp, _ in attributions:
            for category, members in FEATURE_CATEGORIES.items():
                if name in members:
                    cat_scores[category] += imp
                    break
        # Normalize
        total = sum(cat_scores.values()) + 1e-10
        return {k: v / total for k, v in cat_scores.items()}

    def _critical_path_analysis(
        self, graph, trace_data: dict
    ) -> Tuple[List[str], str]:
        """Identify the most suspicious interaction sequence in the ITG."""
        if graph is None or graph.number_of_nodes() <= 1:
            return [], "Insufficient interaction data for path analysis."

        import networkx as nx

        # Score nodes by suspiciousness
        node_scores = {}
        for node, data in graph.nodes(data=True):
            score = 0.0
            ntype = data.get('node_type', '')
            el_tag = str(data.get('element_tag', '')).lower()
            el_id = str(data.get('element_id', '')).lower()
            el_class = str(data.get('element_class', '')).lower()
            combined = f"{el_tag} {el_id} {el_class}"

            # Credential harvesting indicators
            if any(k in combined for k in ['password', 'pwd', 'pass']):
                score += 3.0
            if any(k in combined for k in ['email', 'mail', 'login']):
                score += 2.0
            if any(k in combined for k in ['ssn', 'social', 'card', 'credit', 'cvv']):
                score += 3.0

            # Form submissions
            if ntype in ('submit', 'form_submit'):
                score += 2.5
            if ntype == 'form_input':
                score += 1.5

            # External navigation (redirect)
            if data.get('node_type') == 'url' and not data.get('is_root', False):
                score += 1.0

            node_scores[node] = score

        # Find path from root to highest-scoring node
        root = None
        for n, d in graph.nodes(data=True):
            if d.get('is_root', False):
                root = n
                break

        if root is None:
            root = list(graph.nodes())[0]

        # Find most suspicious node
        if not node_scores:
            return [], "No scored nodes in the interaction graph."

        target = max(node_scores, key=node_scores.get)

        try:
            path = nx.shortest_path(graph, root, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # Use highest-scoring nodes as the critical path
            sorted_nodes = sorted(node_scores.items(), key=lambda x: -x[1])
            path = [n for n, _ in sorted_nodes[:5]]

        # Generate description
        path_labels = []
        for node in path:
            data = graph.nodes.get(node, {})
            label = data.get('label', str(node)[:30])
            path_labels.append(label)

        desc_parts = []
        for i, label in enumerate(path_labels):
            if i == 0:
                desc_parts.append(f"Entry: {label}")
            elif i == len(path_labels) - 1:
                desc_parts.append(f"Target: {label}")
            else:
                desc_parts.append(f"Step {i}: {label}")

        description = " -> ".join(desc_parts)
        return path_labels, description

    def _contrastive_explanation(
        self, x_raw: np.ndarray
    ) -> List[Tuple[str, float, float, str]]:
        """Compare sample features against benign prototype."""
        n_features = min(len(FEATURE_NAMES), len(x_raw), len(self.benign_prototypes))
        diffs = []

        for i in range(n_features):
            name = FEATURE_NAMES[i]
            sample_val = float(x_raw[i])
            benign_avg = float(self.benign_prototypes[i])
            z = abs(sample_val - benign_avg) / float(self.benign_std[i])

            if z > 1.0:  # only report values >1 std dev from benign mean
                direction = "higher" if sample_val > benign_avg else "lower"
                desc = (f"{name} is {z:.1f}x std {direction} than benign average "
                        f"(sample={sample_val:.3g}, benign_avg={benign_avg:.3g})")
                diffs.append((name, sample_val, benign_avg, desc))

        diffs.sort(key=lambda t: abs(t[1] - t[2]) / max(0.001, abs(t[2])), reverse=True)
        return diffs[:8]

    def _generate_summary(self, expl: TraceExplanation) -> str:
        """Generate natural language summary of the detection decision."""
        parts = []

        if expl.prediction == 'phishing':
            parts.append(
                f"The URL '{expl.url}' is classified as PHISHING "
                f"with {expl.confidence:.1%} confidence (risk score: {expl.risk_score:.0f}/100)."
            )
        else:
            parts.append(
                f"The URL '{expl.url}' is classified as BENIGN "
                f"with {expl.confidence:.1%} confidence (risk score: {expl.risk_score:.0f}/100)."
            )

        # Top contributing factors
        if expl.top_contributing_features:
            top3 = expl.top_contributing_features[:3]
            factor_strs = [f"{name} ({imp:.1%})" for name, imp, _ in top3]
            parts.append(
                f"Primary factors: {', '.join(factor_strs)}."
            )

        # Category breakdown
        if expl.category_contributions:
            top_cat = max(expl.category_contributions, key=expl.category_contributions.get)
            parts.append(
                f"The {top_cat} feature category contributes most "
                f"({expl.category_contributions[top_cat]:.0%}) to this decision."
            )

        # Critical path
        if expl.critical_path_description:
            parts.append(f"Suspicious interaction path: {expl.critical_path_description}")

        # Contrastive
        if expl.contrastive_features:
            top_contrast = expl.contrastive_features[0]
            parts.append(f"Most anomalous feature: {top_contrast[3]}")

        return " ".join(parts)


def run_interpretability_demo(dataset_dir: str = None) -> Dict:
    """Run interpretability analysis on sample traces.

    Returns dict with sample explanations and aggregate statistics.
    """
    if dataset_dir is None:
        dataset_dir = str(Path(__file__).parent.parent / 'dataset')

    dataset_path = Path(dataset_dir)

    print("\n" + "=" * 70)
    print("TRACE INTERPRETABILITY ANALYSIS")
    print("=" * 70)

    # Load traces and extract features
    traces = []
    features_list = []
    labels = []

    for label_name, label_int in [('phishing', 1), ('benign', 0)]:
        trace_dir = dataset_path / 'traces' / label_name
        if not trace_dir.exists():
            continue
        for trace_file in sorted(trace_dir.glob('*.json')):
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)

                # Unwrap nested trace format
                trace_data = raw.get('trace', raw)
                if 'url' not in trace_data and 'url' in raw:
                    trace_data['url'] = raw['url']

                builder = InteractionGraphBuilder()
                graph = builder.build_graph_from_dict(trace_data)
                features = builder.extract_features(graph)

                vec = []
                for name in FEATURE_NAMES:
                    val = getattr(features, name, 0)
                    if isinstance(val, bool):
                        val = int(val)
                    vec.append(float(val))

                traces.append(trace_data)
                features_list.append(vec)
                labels.append(label_int)
            except Exception as e:
                logger.warning(f"Skipping {trace_file.name}: {e}")

    X = np.array(features_list)
    y = np.array(labels)
    print(f"Loaded {len(traces)} traces ({sum(labels)} phishing, "
          f"{len(labels) - sum(labels)} benign)")

    if len(traces) < 10:
        print("[WARN] Too few traces for interpretability analysis.")
        return {}

    # Fit interpreter
    interpreter = TraceInterpreter()
    interpreter.fit(X, y)

    # Generate explanations for sample traces
    results = {'explanations': [], 'category_stats': defaultdict(list)}

    # Explain up to 5 phishing and 3 benign
    phishing_indices = [i for i, l in enumerate(labels) if l == 1][:5]
    benign_indices = [i for i, l in enumerate(labels) if l == 0][:3]

    for idx in phishing_indices + benign_indices:
        expl = interpreter.explain(traces[idx])
        results['explanations'].append(expl.to_dict())

        print(f"\n{'='*50}")
        print(f"  URL: {expl.url[:60]}...")
        print(f"  Prediction: {expl.prediction} ({expl.confidence:.1%})")
        print(f"  Risk Score: {expl.risk_score:.0f}/100")
        print(f"  Top Features:")
        for name, imp, desc in expl.top_contributing_features[:5]:
            print(f"    - {name}: {imp:.3f} ({desc})")
        print(f"  Category Breakdown:")
        for cat, score in sorted(expl.category_contributions.items(),
                                 key=lambda x: -x[1]):
            bar = '█' * int(score * 30)
            print(f"    {cat:>15}: {score:.1%} {bar}")
        if expl.critical_path_description:
            print(f"  Critical Path: {expl.critical_path_description[:80]}")
        if expl.contrastive_features:
            print(f"  Top Anomaly: {expl.contrastive_features[0][3][:80]}")

        # Collect category stats
        for cat, score in expl.category_contributions.items():
            results['category_stats'][cat].append(score)

    # Aggregate category statistics
    print("\n" + "=" * 70)
    print("AGGREGATE CATEGORY CONTRIBUTION (across explained samples)")
    for cat, scores in sorted(results['category_stats'].items()):
        avg = np.mean(scores)
        print(f"  {cat:>15}: {avg:.1%} (avg contribution)")

    # Convert defaultdict for serialization
    results['category_stats'] = dict(results['category_stats'])

    return results


if __name__ == '__main__':
    run_interpretability_demo()
