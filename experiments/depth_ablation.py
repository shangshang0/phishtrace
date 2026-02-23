"""
Trace-Information Ablation Study

Evaluates how detection performance varies with the amount of interaction
information retained from a crawled trace.  Traces are truncated by
**event count** (the number of recorded DOM interaction events) rather
than by URL-change count, because 77.6% of crawled traces contain zero
events and 99.8% contain fewer than two URL transitions, making
URL-change-based depth a degenerate metric for this dataset.

Truncation levels (by event count retained):
  0  - No interaction events  (URL + network features only)
  1  - At most 1 event
  3  - At most 3 events
  8  - At most 8 events
  15 - At most 15 events
  ∞  - Full trace (all events retained)

Redirects are capped at min(actual, level_cap) with caps 0/0/1/1/1/ALL.
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
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


def truncate_trace(trace_data: dict, max_events: int, max_redirects: int) -> dict:
    """Truncate a trace by capping the number of interaction events and
    redirects.  Network requests are left intact (they reflect the page
    load regardless of interaction depth).

    Parameters
    ----------
    trace_data : dict
        Raw trace dictionary with 'events', 'redirects', 'network_requests'.
    max_events : int
        Maximum number of interaction events to retain.
        Use a large sentinel (e.g. 99999) for "keep all".
    max_redirects : int
        Maximum number of redirect entries to retain.
    """
    truncated = copy.deepcopy(trace_data)
    raw_events = truncated.get('events', [])
    # Filter out screenshot/page events (visual bookmarks, not interactions)
    events = [e for e in raw_events
              if e and e.get('event_type') != 'screenshot'
              and e.get('element_tag') != 'page']
    redirects = truncated.get('redirects', [])

    truncated['events'] = events[:max_events]
    truncated['redirects'] = redirects[:max_redirects]
    return truncated


def extract_features_from_trace(trace_data: dict) -> dict:
    """Extract feature dict from a trace using graph builder."""
    builder = InteractionGraphBuilder()
    graph = builder.build_graph_from_dict(trace_data)
    features = builder.extract_features(graph)

    return {name: getattr(features, name, 0) for name in FEATURE_NAMES}


def features_dict_to_vector(features: dict) -> np.ndarray:
    """Convert feature dict to numpy vector in canonical order."""
    vec = []
    for name in FEATURE_NAMES:
        val = features.get(name, 0)
        if isinstance(val, bool):
            val = int(val)
        vec.append(float(val))
    return np.array(vec)


def run_depth_ablation(dataset_dir: str = None, n_folds: int = 10) -> Dict:
    """Run crawl-depth ablation study.

    Returns dict mapping depth -> {accuracy, precision, recall, f1, auc, avg_nodes, avg_edges}.
    """
    if dataset_dir is None:
        dataset_dir = str(Path(__file__).parent.parent / 'dataset')

    dataset_path = Path(dataset_dir)

    print("\n" + "=" * 70)
    print("CRAWL-DEPTH ABLATION STUDY")
    print("=" * 70)

    # Load all raw traces (handle nested 'trace' key format)
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

    print(f"Loaded {len(traces)} raw traces ({sum(labels)} phishing, "
          f"{len(labels) - sum(labels)} benign)")

    if len(traces) < 20:
        print("[WARN] Too few traces for reliable depth ablation.")
        return {}

    y = np.array(labels)
    results = {}

    # Truncation levels: (label, max_events, max_redirects)
    levels = [
        ('0-evt',  0,     0),      # URL + network features only
        ('1-evt',  1,     0),      # single interaction event
        ('3-evt',  3,     1),      # brief interaction sequence
        ('8-evt',  8,     1),      # moderate interaction
        ('15-evt', 15,    1),      # extended interaction
        ('all',    99999, 99999),   # full trace
    ]

    for level_name, max_events, max_redirects in levels:
        print(f"\n--- Level {level_name} (events<={max_events}, "
              f"redirects<={max_redirects}) ---")

        # Truncate traces and extract features
        feature_list = []
        node_counts = []
        edge_counts = []
        valid_indices = []

        for i, trace in enumerate(traces):
            try:
                truncated = truncate_trace(trace, max_events, max_redirects)
                feats = extract_features_from_trace(truncated)
                feature_list.append(features_dict_to_vector(feats))
                node_counts.append(feats.get('num_nodes', 0))
                edge_counts.append(feats.get('num_edges', 0))
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Feature extraction failed at level {level_name}: {e}")

        if len(feature_list) < 20:
            print(f"  [SKIP] Only {len(feature_list)} valid samples")
            continue

        X = np.array(feature_list)
        y_valid = y[valid_indices]

        # Cross-validation (adapt folds to minority class)
        clf = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42
            ))
        ])

        minority = min(int(np.sum(y_valid == 0)), int(np.sum(y_valid == 1)))
        actual_folds = max(2, min(n_folds, minority))
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

        from sklearn.utils.class_weight import compute_sample_weight
        for train_idx, test_idx in skf.split(X, y_valid):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

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
        results[level_name] = {
            **avg,
            'max_events': max_events,
            'max_redirects': max_redirects,
            'avg_nodes': round(float(np.mean(node_counts)), 1),
            'avg_edges': round(float(np.mean(edge_counts)), 1),
            'n_samples': len(feature_list),
        }

        print(f"  Samples: {len(feature_list)}  |  "
              f"Avg nodes: {results[level_name]['avg_nodes']:.1f}  |  "
              f"Avg edges: {results[level_name]['avg_edges']:.1f}")
        for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            print(f"    {m.upper():>10}: {avg[m]:.4f}")

    # Summary table
    print("\n" + "=" * 70)
    print("TRACE-INFORMATION ABLATION SUMMARY")
    print(f"{'Level':>8} {'MaxEvt':>8} {'Nodes':>8} {'Edges':>8} "
          f"{'Acc':>8} {'F1':>8} {'AUC':>8} {'N':>6}")
    print("-" * 70)
    for level_name in results:
        r = results[level_name]
        evt_str = str(r['max_events']) if r['max_events'] < 99999 else 'all'
        print(f"{level_name:>8} {evt_str:>8} {r['avg_nodes']:>8.1f} "
              f"{r['avg_edges']:>8.1f} {r['accuracy']:>8.4f} "
              f"{r['f1']:>8.4f} {r['auc']:>8.4f} {r['n_samples']:>6}")

    return results


if __name__ == '__main__':
    results = run_depth_ablation()
    # Save results
    import json as _json
    out_path = Path(__file__).parent / 'results' / 'depth_ablation_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        _json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")
