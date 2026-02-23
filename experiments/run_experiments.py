"""
Main Experiment Runner for PhishTrace
Runs graph-based phishing detection experiments and baseline comparisons.

Data sources (in priority order):
1. Real crawled traces → features via graph builder
2. Pre-extracted features from dataset/features.json
3. Crawled URL metadata → structural features
"""

import sys
import os
import warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Suppress sklearn warnings for small datasets
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', message='.*Only one class.*', category=UserWarning)
warnings.filterwarnings('ignore', message='.*UndefinedMetricWarning.*')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    HAS_VIZ = True
except ImportError:
    HAS_VIZ = False

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / 'dataset'


class PhishTraceExperiment:
    """Graph-based phishing detection experiment runner."""

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

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=500, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1, class_weight='balanced'
        )
        self.rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=15, random_state=42,
            class_weight='balanced'
        )
        self.data_source = "unknown"

    def features_dict_to_vector(self, features: Dict) -> np.ndarray:
        vec = []
        for name in self.FEATURE_NAMES:
            val = features.get(name, 0)
            if isinstance(val, bool):
                val = 1 if val else 0
            vec.append(float(val))
        return np.array(vec)

    def engineer_features(self, X: np.ndarray) -> np.ndarray:
        """Add interaction and derived features that capture multi-granularity
        graph patterns — the core methodological contribution of PhishTrace.
        
        These features encode cross-layer relationships in the Interaction
        Trace Graph (ITG) that single-feature baselines miss:
        - Topology × Behavior: how graph structure correlates with form/redirect activity
        - Centrality × Temporal: how information flow relates to interaction timing
        - Composite risk scores: multi-signal aggregation
        """
        # Feature name → index mapping
        idx = {n: i for i, n in enumerate(self.FEATURE_NAMES)}
        n = X.shape[0]
        derived = []

        # ── Topology × Behavior cross features ──
        # Graph density × form-to-node ratio: dense graphs with many forms are suspicious
        derived.append(X[:, idx['graph_density']] * X[:, idx['form_to_node_ratio']])
        # Clustering × credential count: clustered credential pages
        derived.append(X[:, idx['clustering_coefficient']] * X[:, idx['credential_fields_count']])
        # Redirects × external ratio: external redirect chains
        safe_redirects = np.maximum(X[:, idx['num_redirects']], 1e-8)
        derived.append(X[:, idx['external_redirects']] / safe_redirects)
        # Node count × sensitive data flag: complex pages requesting data
        derived.append(X[:, idx['num_nodes']] * X[:, idx['requests_sensitive_data']])

        # ── Centrality × Temporal cross features ──
        # Betweenness × interaction time: bottleneck nodes with long dwell times
        derived.append(X[:, idx['betweenness_centrality_max']] * X[:, idx['total_interaction_time']])
        # PageRank × event frequency: high-authority nodes with rapid events
        derived.append(X[:, idx['pagerank_max']] * X[:, idx['event_frequency']])
        # Closeness × credential fields: central credential harvesting
        derived.append(X[:, idx['closeness_centrality_max']] * X[:, idx['credential_fields_count']])

        # ── Structural complexity scores ──
        # Edge-to-node ratio (measures graph complexity beyond degree)
        safe_nodes = np.maximum(X[:, idx['num_nodes']], 1e-8)
        derived.append(X[:, idx['num_edges']] / safe_nodes)
        # Diameter × redirect depth: deep chains in deep graphs
        derived.append(X[:, idx['diameter']] * X[:, idx['max_redirect_depth']])
        # In/out degree asymmetry: unbalanced flow patterns
        safe_total = np.maximum(X[:, idx['in_degree_max']] + X[:, idx['out_degree_max']], 1e-8)
        derived.append(np.abs(X[:, idx['in_degree_max']] - X[:, idx['out_degree_max']]) / safe_total)

        # ── Composite risk scores ──
        # Authentication risk: password + email + financial indicators
        derived.append(X[:, idx['has_password_input']] + X[:, idx['has_email_input']] +
                       X[:, idx['financial_fields_count']])
        # Navigation complexity: avg_path_length × connected components
        derived.append(X[:, idx['avg_path_length']] * X[:, idx['num_connected_components']])

        derived_matrix = np.column_stack(derived)
        return np.hstack([X, derived_matrix])

    def load_extracted_features(self) -> Tuple[List[Dict], List[int]]:
        """Load pre-extracted features from dataset/features.json."""
        features_file = DATASET_DIR / 'features.json'
        if not features_file.exists():
            return [], []

        data = json.loads(features_file.read_text(encoding='utf-8'))
        features_list = []
        labels = []
        for entry in data:
            features_list.append(entry['features'])
            labels.append(1 if entry['label'] == 'phishing' else 0)

        logger.info(f"Loaded {len(labels)} pre-extracted features from features.json")
        return features_list, labels

    def load_trace_data(self) -> Tuple[List[Dict], List[int]]:
        """Load features from real crawled traces."""
        traces_dir = DATASET_DIR / 'traces'
        if not traces_dir.exists():
            return [], []

        try:
            from analyzer.graph_builder import InteractionGraphBuilder
            builder = InteractionGraphBuilder()
        except ImportError:
            logger.warning("graph_builder not available, skipping trace loading")
            return [], []

        features_list = []
        labels = []

        for label_int, subdir in [(1, 'phishing'), (0, 'benign')]:
            trace_dir = traces_dir / subdir
            if not trace_dir.exists():
                continue
            for trace_file in sorted(trace_dir.glob('*.json')):
                try:
                    data = json.loads(trace_file.read_text(encoding='utf-8'))
                    if not data.get('success') or not data.get('trace'):
                        continue
                    graph = builder.build_graph_from_dict(data['trace'])
                    feats = builder.extract_features(graph)
                    features_list.append(feats.__dict__)
                    labels.append(label_int)
                except Exception as e:
                    logger.warning(f"Failed to process {trace_file.name}: {e}")

        logger.info(f"Loaded {len(labels)} features from crawled traces")
        return features_list, labels

    def extract_url_features(self, url: str, label: str) -> Dict:
        """Extract structural features from URL string (no crawling needed).
        
        IMPORTANT: Features are derived ONLY from the URL string — the label
        parameter is NOT used to set any feature value. This avoids data leakage.
        
        We construct a lightweight URL-structural graph and extract the same
        30-dimensional feature vector used by the full ITG pipeline. This enables
        experiments on datasets where live crawling is not feasible.
        """
        from urllib.parse import urlparse, parse_qs
        parsed = urlparse(url)
        domain = parsed.netloc or parsed.path
        path = parsed.path or '/'
        query = parsed.query or ''

        # Pure URL structural analysis
        domain_no_port = domain.split(':')[0]
        is_ip = bool(all(c.isdigit() or c == '.' for c in domain_no_port) and domain_no_port.count('.') == 3)
        has_at = '@' in url
        has_dash_in_domain = '-' in domain_no_port
        num_dots = domain_no_port.count('.')
        url_len = len(url)
        path_len = len(path)
        num_slashes = path.count('/')
        has_https = parsed.scheme == 'https'
        num_digits_domain = sum(c.isdigit() for c in domain_no_port)
        num_subdomains = max(0, num_dots - 1)
        has_suspicious_tld = any(domain_no_port.endswith(t) for t in
                                ['.xyz', '.top', '.club', '.buzz', '.tk', '.ml', '.ga', '.cf', '.gq',
                                 '.work', '.life', '.online', '.site', '.icu', '.fun'])
        domain_entropy = self._shannon_entropy(domain_no_port)
        path_entropy = self._shannon_entropy(path)

        # Advanced URL signals
        has_long_domain = len(domain_no_port) > 30
        has_deep_path = num_slashes > 4
        has_query_params = len(query) > 0
        num_query_params = len(parse_qs(query))
        has_encoded_chars = '%' in url
        has_double_slash_in_path = '//' in path[1:] if len(path) > 1 else False
        domain_digit_ratio = num_digits_domain / max(1, len(domain_no_port))
        path_has_extension = any(path.endswith(ext) for ext in ['.php', '.html', '.htm', '.asp', '.jsp', '.cgi'])
        has_suspicious_keywords = any(kw in url.lower() for kw in
                                       ['login', 'signin', 'verify', 'update', 'secure', 'account',
                                        'confirm', 'banking', 'paypal', 'password', 'credential'])
        has_brand_in_path = any(b in path.lower() for b in
                                ['apple', 'microsoft', 'google', 'amazon', 'facebook',
                                 'netflix', 'paypal', 'chase', 'wellsfargo', 'bank'])

        # --- Build a lightweight URL-structural graph ---
        # Nodes: URL components (scheme, subdomain levels, path segments, query params)
        # Edges: hierarchical relationships between components
        path_segments = [s for s in path.split('/') if s]
        subdomain_parts = domain_no_port.split('.')
        
        # Graph node count: root + subdomain parts + path segments + query params
        n_nodes = 1 + len(subdomain_parts) + len(path_segments) + num_query_params
        n_nodes = max(2, n_nodes)
        
        # Edges: sequential connections between components
        n_edges = max(1, n_nodes - 1 + (1 if has_query_params else 0))
        
        avg_degree = 2 * n_edges / n_nodes if n_nodes > 0 else 0
        max_degree = min(n_nodes - 1, len(path_segments) + 2)
        
        # Graph density depends on actual vs. possible edges
        max_possible_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max_possible_edges if max_possible_edges > 0 else 0
        
        # Clustering: tree-like structures have 0, meshes have higher
        # URL graphs are generally tree-like → low clustering
        clustering = 0.0 if n_nodes <= 2 else min(0.5, num_query_params * 0.05)
        
        # Connected components: typically 1, unless URL has disconnected parts
        n_components = 1 + (1 if has_at else 0)
        
        # Path length and diameter: based on URL depth
        url_depth = len(path_segments) + len(subdomain_parts)
        avg_path_len = url_depth / 2.0 if url_depth > 0 else 1.0
        diameter = url_depth

        # --- Behavioral features from URL patterns ---
        # These are URL-structural proxies, NOT actual interaction data
        num_forms = 0  # cannot determine from URL alone
        num_redirects = num_subdomains  # subdomain nesting as redirect proxy
        has_password = 'password' in url.lower() or 'pwd' in url.lower()
        has_email = 'email' in url.lower() or 'mail' in url.lower() or 'login' in url.lower()
        external_redirects = 1 if (num_subdomains > 2 or has_at or has_encoded_chars) else 0
        num_inputs = num_query_params  # query parameters as input proxy
        num_buttons = 1 if path_has_extension else 0
        num_links = max(1, len(path_segments))
        max_redirect_depth = min(4, num_subdomains)
        form_node_ratio = 0.0  # cannot determine from URL alone

        # --- Temporal features (URL-structural proxies) ---
        # Longer/more complex URLs suggest longer interaction sessions
        total_time = url_len / 20.0  # normalized URL length as session time proxy
        avg_interval = total_time / max(1, n_nodes)
        event_freq = n_nodes / max(0.1, total_time)

        # --- Centrality measures from URL graph structure ---
        # For tree-like URL graphs, centrality concentrates at root
        bc_max = 1.0 / max(1, n_nodes - 1) if n_nodes > 2 else 0
        cc_max = 1.0 / max(1, avg_path_len)
        pr_max = 1.0 / max(1, n_nodes) + 0.15 * (1 if num_subdomains > 0 else 0)
        in_degree_max = max(1, min(n_nodes - 1, len(subdomain_parts)))
        out_degree_max = max(1, min(n_nodes - 1, len(path_segments) + 1))

        # --- Sensitive data indicators ---
        requests_sensitive = has_suspicious_keywords or has_brand_in_path
        cred_count = int(has_password) + int(has_email)
        fin_count = int(any(kw in url.lower() for kw in ['bank', 'pay', 'credit', 'card', 'wallet']))

        return {
            'num_nodes': n_nodes,
            'num_edges': n_edges,
            'avg_degree': round(avg_degree, 4),
            'max_degree': max_degree,
            'graph_density': round(density, 6),
            'clustering_coefficient': round(clustering, 6),
            'num_connected_components': n_components,
            'avg_path_length': round(avg_path_len, 4),
            'diameter': diameter,
            'num_forms': num_forms,
            'num_redirects': num_redirects,
            'has_password_input': int(has_password),
            'has_email_input': int(has_email),
            'external_redirects': external_redirects,
            'num_input_fields': num_inputs,
            'num_buttons': num_buttons,
            'num_links': num_links,
            'max_redirect_depth': max_redirect_depth,
            'form_to_node_ratio': form_node_ratio,
            'total_interaction_time': round(total_time, 4),
            'avg_time_between_events': round(avg_interval, 4),
            'event_frequency': round(event_freq, 4),
            'betweenness_centrality_max': round(bc_max, 6),
            'closeness_centrality_max': round(cc_max, 6),
            'pagerank_max': round(pr_max, 6),
            'in_degree_max': in_degree_max,
            'out_degree_max': out_degree_max,
            'requests_sensitive_data': int(requests_sensitive),
            'credential_fields_count': cred_count,
            'financial_fields_count': fin_count,
        }

    def _shannon_entropy(self, s: str) -> float:
        """Calculate Shannon entropy of a string."""
        if not s:
            return 0.0
        freq = {}
        for c in s:
            freq[c] = freq.get(c, 0) + 1
        entropy = 0.0
        for count in freq.values():
            p = count / len(s)
            if p > 0:
                entropy -= p * np.log2(p)
        return entropy

    def load_url_features(self) -> Tuple[List[Dict], List[int]]:
        """Extract features from collected URL lists (no crawling)."""
        features_list = []
        labels = []

        for label_name, label_int in [('phishing', 1), ('benign', 0)]:
            urls_file = DATASET_DIR / f'{label_name}_urls.json'
            if not urls_file.exists():
                continue
            data = json.loads(urls_file.read_text(encoding='utf-8'))
            for entry in data.get('urls', []):
                url = entry['url']
                feats = self.extract_url_features(url, label_name)
                features_list.append(feats)
                labels.append(label_int)

        logger.info(f"Extracted URL-based features for {len(labels)} samples")
        return features_list, labels

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load experiment data from best available source."""
        # Priority 1: Pre-extracted features from crawled traces
        features_list, labels = self.load_extracted_features()
        if len(labels) >= 100:
            self.data_source = "crawled_traces"
            print(f"[DATA] Using {len(labels)} samples from crawled trace features")
            X = np.array([self.features_dict_to_vector(f) for f in features_list])
            return X, np.array(labels)

        # Priority 2: Raw trace files
        features_list, labels = self.load_trace_data()
        if len(labels) >= 100:
            self.data_source = "raw_traces"
            print(f"[DATA] Using {len(labels)} samples from raw trace data")
            X = np.array([self.features_dict_to_vector(f) for f in features_list])
            return X, np.array(labels)

        # Priority 3: URL structural features (from collected URLs)
        features_list, labels = self.load_url_features()
        if len(labels) >= 100:
            self.data_source = "url_structural"
            print(f"[DATA] Using {len(labels)} samples from URL structural features")
            X = np.array([self.features_dict_to_vector(f) for f in features_list])
            return X, np.array(labels)

        raise RuntimeError(
            "No data available! Run the pipeline with 'docker compose up phishtrace-micro' "
            "or use 'python scripts/run_all_intersection_experiments.py' after ensuring traces exist."
        )


def run_experiments():
    """Run full experiment suite."""
    print("=" * 80)
    print("PhishTrace: Graph-Based Phishing Detection Experiments")
    print("=" * 80)

    exp = PhishTraceExperiment()

    # Load data from best available source
    X, y = exp.load_data()
    print(f"\nData source: {exp.data_source}")
    print(f"Total samples: {len(y)}")
    print(f"  Phishing: {sum(y)}")
    print(f"  Benign: {len(y) - sum(y)}")

    # 10-fold cross-validation with engineered features
    print("\nRunning 10-fold cross-validation...")
    X_eng = exp.engineer_features(X)  # PhishTrace uses interaction features
    print(f"  Base features: {X.shape[1]}, Engineered features: {X_eng.shape[1]}")
    # Adapt n_splits to minority class size to avoid empty folds
    minority_count = min(int(sum(y)), int(len(y) - sum(y)))
    n_splits = max(2, min(10, minority_count))
    print(f"  Using {n_splits}-fold CV (minority class: {minority_count} samples)")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_eng, y)):
        X_train, X_test = X_eng[train_idx], X_eng[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        exp.model.fit(X_train, y_train)
        y_pred = exp.model.predict(X_test)
        y_proba = exp.model.predict_proba(X_test)[:, 1]
        
        cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        # AUC requires both classes in y_test
        if len(np.unique(y_test)) > 1:
            cv_metrics['auc'].append(roc_auc_score(y_test, y_proba))
        else:
            cv_metrics['auc'].append(np.nan)

    our_results = {k: round(float(np.nanmean(v)), 4) for k, v in cv_metrics.items()}
    our_std = {k: round(float(np.nanstd(v)), 4) for k, v in cv_metrics.items()}

    print("\nOur Method (PhishTrace) Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {metric.upper():>10}: {our_results[metric]:.4f} (+/- {our_std[metric]:.4f})")

    # Feature importance (including engineered features)
    exp.model.fit(X_eng, y)
    importances = exp.model.feature_importances_
    
    # Extended feature names including derived features
    derived_names = [
        'density×form_ratio', 'cluster×cred_count', 'ext_redirect_ratio',
        'nodes×sensitive', 'between×time', 'pagerank×freq',
        'closeness×cred', 'edge_node_ratio', 'diameter×redirect_depth',
        'degree_asymmetry', 'auth_risk_score', 'nav_complexity'
    ]
    all_feat_names = exp.FEATURE_NAMES + derived_names
    feat_imp = sorted(zip(all_feat_names, importances), key=lambda x: -x[1])
    
    print("\nTop 10 Feature Importances:")
    for name, imp in feat_imp[:10]:
        print(f"  {name:>35}: {imp:.4f}")

    # Baseline comparison
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON")
    print("=" * 80)

    # Run baseline experiments with proper cross-validation
    from baselines.baseline_methods import run_all_baselines
    baseline_results = run_all_baselines(X, y, exp.FEATURE_NAMES)

    # Combine results
    all_results = baseline_results.copy()
    all_results['PhishTrace (Ours)'] = our_results

    # Print comparison table
    print(f"\n{'Method':<30} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 78)
    for method, metrics in all_results.items():
        print(f"{method:<30} {metrics['accuracy']:>8.4f} {metrics['precision']:>8.4f} "
              f"{metrics['recall']:>8.4f} {metrics['f1']:>8.4f} {metrics.get('auc', 0):>8.4f}")

    # Save results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump({
            'data_source': exp.data_source,
            'run_timestamp': datetime.utcnow().isoformat(),
            'our_method': our_results,
            'our_method_std': our_std,
            'baselines': baseline_results,
            'all_results': all_results,
            'feature_importances': dict(feat_imp),
            'dataset_size': int(len(y)),
            'phishing_count': int(sum(y)),
            'benign_count': int(len(y) - sum(y)),
        }, f, indent=2)

    print(f"\nResults saved to {output_dir}/results.json")

    # Generate visualizations
    if HAS_VIZ:
        generate_visualizations(all_results, feat_imp, output_dir)

    return our_results, all_results


def generate_visualizations(results: Dict, feat_imp: List, output_dir: str):
    """Generate publication-quality plots."""
    methods = list(results.keys())
    
    # 1. Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    accuracies = [results[m]['accuracy'] for m in methods]
    colors = ['#e74c3c' if 'Ours' not in m else '#2ecc71' for m in methods]
    bars = ax.bar(range(len(methods)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Phishing Detection Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylim([0.6, 1.02])
    ax.grid(axis='y', alpha=0.3)
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'), dpi=300)
    plt.close()

    # 2. Multi-metric comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
        ax = axes[idx // 2, idx % 2]
        values = [results[m][metric] for m in methods]
        colors = ['#e74c3c' if 'Ours' not in m else '#2ecc71' for m in methods]
        bars = ax.bar(range(len(methods)), values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        ax.set_ylim([0.6, 1.02])
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300)
    plt.close()

    # 3. Feature importance plot
    fig, ax = plt.subplots(figsize=(12, 8))
    top_feats = feat_imp[:15]
    names = [n for n, _ in top_feats]
    values = [v for _, v in top_feats]
    ax.barh(range(len(names)), values, color='#3498db', edgecolor='black', linewidth=0.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
    plt.close()

    print(f"Visualizations saved to {output_dir}/")


def run_extended_experiments():
    """Run all experiments including new reviewer-requested evaluations."""
    # 1. Original experiments
    our_results, all_results = run_experiments()

    # 2. AGFL (Automated Graph Feature Learning) evaluation
    try:
        from gnn_detector import evaluate_agfl
        agfl_results = evaluate_agfl()
    except Exception as e:
        print(f"\n[WARN] AGFL evaluation failed: {e}")
        agfl_results = {}

    # 3. Crawl-depth ablation study
    try:
        from depth_ablation import run_depth_ablation
        depth_results = run_depth_ablation()
    except Exception as e:
        print(f"\n[WARN] Depth ablation failed: {e}")
        depth_results = {}

    # 4. Adversarial robustness evaluation
    try:
        from adversarial_eval import run_adversarial_evaluation
        adv_results = run_adversarial_evaluation()
    except Exception as e:
        print(f"\n[WARN] Adversarial evaluation failed: {e}")
        adv_results = {}

    # 5. Trace interpretability analysis
    try:
        from trace_interpretability import run_interpretability_demo
        interp_results = run_interpretability_demo()
    except Exception as e:
        print(f"\n[WARN] Interpretability analysis failed: {e}")
        interp_results = {}

    # Save extended results
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)

    from datetime import datetime
    extended_results = {
        'run_timestamp': datetime.utcnow().isoformat(),
        'main_results': all_results,
        'agfl_results': agfl_results,
        'depth_ablation': depth_results,
        'adversarial_robustness': adv_results,
        'interpretability': interp_results,
    }

    with open(os.path.join(output_dir, 'extended_results.json'), 'w') as f:
        json.dump(extended_results, f, indent=2, default=str)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Extended results saved to {output_dir}/extended_results.json")
    print("=" * 80)

    return extended_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PhishTrace Experiments')
    parser.add_argument('--extended', action='store_true',
                        help='Run extended experiments (AGFL, adversarial, depth ablation, interpretability)')
    args = parser.parse_args()

    if args.extended:
        run_extended_experiments()
    else:
        run_experiments()
