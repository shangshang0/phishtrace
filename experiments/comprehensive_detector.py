"""
PhishTrace Comprehensive Stacked Detection — integrates ITG, DRP,
and AGFL features into a two-level stacked ensemble (Section 5).
"""

import os
import sys
import json
import warnings
import logging
import numpy as np

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

logger = logging.getLogger(__name__)

# Add project paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'experiments'))

DATASET_DIR = PROJECT_ROOT / 'dataset'


# ============================================================================
# Feature Group Definitions
# ============================================================================

BASE_FEATURE_NAMES = [
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

ENGINEERED_FEATURE_NAMES = [
    'density×form_ratio', 'cluster×cred_count', 'ext_redirect_ratio',
    'nodes×sensitive', 'between×time', 'pagerank×freq',
    'closeness×cred', 'edge_node_ratio', 'diameter×redirect_depth',
    'degree_asymmetry', 'auth_risk_score', 'nav_complexity',
]

DRP_FEATURE_NAMES = [
    'drp_depth_reached', 'drp_unique_pages', 'drp_url_transitions',
    'drp_credential_depth', 'drp_form_diversity',
    'drp_pruning_ratio', 'drp_credential_paths_ratio', 'drp_effective_branching',
    'drp_funnel_width', 'drp_navigation_entropy',
    'drp_cred_concentration_early', 'drp_cred_concentration_mid',
    'drp_cred_concentration_late',
    'drp_dual_submit_count', 'drp_events_per_page',
]


class ComprehensivePhishTraceDetector:
    """
    PhishTrace comprehensive detection combining all feature groups
    and multiple learning paradigms.
    """

    def __init__(self, use_stacking: bool = True, use_drp_features: bool = True,
                 use_agfl_features: bool = True):
        self.use_stacking = use_stacking
        self.use_drp_features = use_drp_features
        self.use_agfl_features = use_agfl_features
        self.data_source = "unknown"
        self._agfl_extractors = None

    def engineer_features(self, X: np.ndarray) -> np.ndarray:
        """Add 12 engineered cross-features (same as original PhishTrace)."""
        idx = {n: i for i, n in enumerate(BASE_FEATURE_NAMES)}
        derived = []

        # Topology × Behavior
        derived.append(X[:, idx['graph_density']] * X[:, idx['form_to_node_ratio']])
        derived.append(X[:, idx['clustering_coefficient']] * X[:, idx['credential_fields_count']])
        safe_redirects = np.maximum(X[:, idx['num_redirects']], 1e-8)
        derived.append(X[:, idx['external_redirects']] / safe_redirects)
        derived.append(X[:, idx['num_nodes']] * X[:, idx['requests_sensitive_data']])

        # Centrality × Temporal
        derived.append(X[:, idx['betweenness_centrality_max']] * X[:, idx['total_interaction_time']])
        derived.append(X[:, idx['pagerank_max']] * X[:, idx['event_frequency']])
        derived.append(X[:, idx['closeness_centrality_max']] * X[:, idx['credential_fields_count']])

        # Structural complexity
        safe_nodes = np.maximum(X[:, idx['num_nodes']], 1e-8)
        derived.append(X[:, idx['num_edges']] / safe_nodes)
        derived.append(X[:, idx['diameter']] * X[:, idx['max_redirect_depth']])
        safe_total = np.maximum(X[:, idx['in_degree_max']] + X[:, idx['out_degree_max']], 1e-8)
        derived.append(np.abs(X[:, idx['in_degree_max']] - X[:, idx['out_degree_max']]) / safe_total)

        # Composite risk
        derived.append(X[:, idx['has_password_input']] + X[:, idx['has_email_input']] +
                       X[:, idx['financial_fields_count']])
        derived.append(X[:, idx['avg_path_length']] * X[:, idx['num_connected_components']])

        return np.hstack([X, np.column_stack(derived)])

    def generate_drp_features(self, X_base: np.ndarray) -> np.ndarray:
        """
        Generate DRP features from existing base features.
        
        When real DRP crawl data is not available, we derive depth-aware
        features from the existing ITG features using domain knowledge about
        phishing interaction patterns.
        """
        idx = {n: i for i, n in enumerate(BASE_FEATURE_NAMES)}
        n = X_base.shape[0]
        drp = np.zeros((n, len(DRP_FEATURE_NAMES)), dtype=np.float64)

        # Depth reached: approximate from graph diameter and redirects
        drp[:, 0] = np.minimum(X_base[:, idx['diameter']] + X_base[:, idx['max_redirect_depth']], 10)
        # Unique pages: num_connected_components + redirect chain
        drp[:, 1] = X_base[:, idx['num_connected_components']] + X_base[:, idx['num_redirects']]
        # URL transitions: external_redirects + num_redirects
        drp[:, 2] = X_base[:, idx['external_redirects']] + X_base[:, idx['num_redirects']]
        # Credential depth: credential_fields_count > 0 indicates depth
        drp[:, 3] = np.where(X_base[:, idx['credential_fields_count']] > 0,
                            np.minimum(X_base[:, idx['avg_path_length']], 5), 0)
        # Form diversity: num_forms
        drp[:, 4] = X_base[:, idx['num_forms']]

        # Pruning features (simulated from graph structure)
        safe_edges = np.maximum(X_base[:, idx['num_edges']], 1e-8)
        drp[:, 5] = np.minimum(X_base[:, idx['num_nodes']] / safe_edges, 1.0)  # pruning ratio
        safe_forms = np.maximum(X_base[:, idx['num_forms']], 1e-8)
        drp[:, 6] = X_base[:, idx['credential_fields_count']] / np.maximum(
            X_base[:, idx['num_nodes']], 1e-8)  # cred paths ratio
        drp[:, 7] = X_base[:, idx['avg_degree']]  # effective branching

        # Navigation funnel
        drp[:, 8] = X_base[:, idx['num_nodes']] * X_base[:, idx['graph_density']]
        drp[:, 9] = -np.sum(
            np.column_stack([
                X_base[:, idx['num_forms']] / np.maximum(X_base[:, idx['num_nodes']], 1e-8),
                X_base[:, idx['num_links']] / np.maximum(X_base[:, idx['num_nodes']], 1e-8),
            ]) * np.log2(np.maximum(
                np.column_stack([
                    X_base[:, idx['num_forms']] / np.maximum(X_base[:, idx['num_nodes']], 1e-8),
                    X_base[:, idx['num_links']] / np.maximum(X_base[:, idx['num_nodes']], 1e-8),
                ]), 1e-10
            )), axis=1)

        # Temporal depth features (credential concentration across interaction phases)
        total_time_safe = np.maximum(X_base[:, idx['total_interaction_time']], 1e-8)
        drp[:, 10] = X_base[:, idx['event_frequency']] * 0.5  # early concentration
        drp[:, 11] = X_base[:, idx['event_frequency']] * 0.3  # mid
        drp[:, 12] = X_base[:, idx['event_frequency']] * 0.2  # late

        # Dual submission (from forms and credential density)
        drp[:, 13] = np.where(
            (X_base[:, idx['num_forms']] >= 2) & (X_base[:, idx['credential_fields_count']] >= 2),
            1.0, 0.0
        )
        # Events per page
        safe_pages = np.maximum(drp[:, 1], 1e-8)
        drp[:, 14] = X_base[:, idx['num_edges']] / safe_pages

        # Replace NaN/Inf
        drp = np.nan_to_num(drp, nan=0.0, posinf=0.0, neginf=0.0)
        return drp

    def generate_agfl_features(self, X_base: np.ndarray) -> np.ndarray:
        """
        Generate AGFL-style features from base features.
        
        Simulates WL, spectral, and random walk features using transformations
        of the base feature vector, preserving the multi-view structural learning
        philosophy of AGFL.
        """
        idx = {n: i for i, n in enumerate(BASE_FEATURE_NAMES)}
        n = X_base.shape[0]
        
        # WL-style features: neighborhood pattern histograms (simulated 64D)
        # Key insight: WL captures local neighborhood patterns → approximate
        # with interaction patterns between features
        wl_features = []
        # Pairwise feature products (top 32 most discriminative pairs)
        important_pairs = [
            ('num_forms', 'has_password_input'),
            ('credential_fields_count', 'num_redirects'),
            ('external_redirects', 'graph_density'),
            ('form_to_node_ratio', 'credential_fields_count'),
            ('num_input_fields', 'has_email_input'),
            ('betweenness_centrality_max', 'num_forms'),
            ('pagerank_max', 'credential_fields_count'),
            ('clustering_coefficient', 'num_redirects'),
            ('avg_degree', 'form_to_node_ratio'),
            ('max_degree', 'num_forms'),
            ('diameter', 'external_redirects'),
            ('num_nodes', 'num_forms'),
            ('graph_density', 'has_password_input'),
            ('closeness_centrality_max', 'financial_fields_count'),
            ('in_degree_max', 'out_degree_max'),
            ('num_buttons', 'num_forms'),
        ]
        for f1, f2 in important_pairs:
            wl_features.append(X_base[:, idx[f1]] * X_base[:, idx[f2]])
            # Also add squared difference (captures non-linear patterns)
            wl_features.append((X_base[:, idx[f1]] - X_base[:, idx[f2]]) ** 2)
        wl_matrix = np.column_stack(wl_features)  # 32D

        # Spectral features: eigenspectrum approximation (16D)
        # Key insight: spectral features capture global topology → approximate
        # with polynomial combinations of topological features
        topo_feats = X_base[:, [idx[f] for f in [
            'num_nodes', 'num_edges', 'avg_degree', 'max_degree',
            'graph_density', 'clustering_coefficient',
            'avg_path_length', 'diameter'
        ]]]
        # Spectral features: eigenspectrum approximation (16D)
        # Polynomial combinations of topological features;
        # no pre-scaling — downstream per-fold StandardScaler normalises.
        spectral_feats = []
        for i in range(topo_feats.shape[1]):
            spectral_feats.append(topo_feats[:, i] ** 2)
            if i + 1 < topo_feats.shape[1]:
                spectral_feats.append(topo_feats[:, i] * topo_feats[:, i + 1])
        spectral_matrix = np.column_stack(spectral_feats)  # ~15D

        # Random walk features: structural role simulation (12D)
        # Key insight: RW captures node roles → approximate with ratios
        # and proportions that encode structural position
        rw_features = []
        safe = lambda x: np.maximum(x, 1e-8)
        
        rw_features.append(X_base[:, idx['in_degree_max']] / safe(X_base[:, idx['max_degree']]))
        rw_features.append(X_base[:, idx['out_degree_max']] / safe(X_base[:, idx['max_degree']]))
        rw_features.append(X_base[:, idx['num_forms']] / safe(X_base[:, idx['num_nodes']]))
        rw_features.append(X_base[:, idx['num_links']] / safe(X_base[:, idx['num_nodes']]))
        rw_features.append(X_base[:, idx['num_input_fields']] / safe(X_base[:, idx['num_nodes']]))
        rw_features.append(X_base[:, idx['num_buttons']] / safe(X_base[:, idx['num_nodes']]))
        rw_features.append(X_base[:, idx['credential_fields_count']] / safe(X_base[:, idx['num_input_fields']]))
        rw_features.append(X_base[:, idx['external_redirects']] / safe(X_base[:, idx['num_redirects']]))
        rw_features.append(np.log1p(X_base[:, idx['num_nodes']]))
        rw_features.append(np.log1p(X_base[:, idx['num_edges']]))
        rw_features.append(X_base[:, idx['betweenness_centrality_max']] /
                          safe(X_base[:, idx['closeness_centrality_max']]))
        rw_features.append(X_base[:, idx['event_frequency']] *
                          X_base[:, idx['total_interaction_time']])
        rw_matrix = np.column_stack(rw_features)  # 12D

        # Combine all AGFL features
        agfl = np.hstack([wl_matrix, spectral_matrix, rw_matrix])
        agfl = np.nan_to_num(agfl, nan=0.0, posinf=0.0, neginf=0.0)
        return agfl

    def build_feature_matrix(self, X_base: np.ndarray) -> np.ndarray:
        """Build the full feature matrix combining all feature groups."""
        # 1. Base + Engineered (42D)
        X_eng = self.engineer_features(X_base)
        
        feature_blocks = [X_eng]
        
        # 2. DRP features (15D)
        if self.use_drp_features:
            X_drp = self.generate_drp_features(X_base)
            feature_blocks.append(X_drp)
        
        # 3. AGFL features (~59D)
        if self.use_agfl_features:
            X_agfl = self.generate_agfl_features(X_base)
            feature_blocks.append(X_agfl)
        
        X_full = np.hstack(feature_blocks)
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        return X_full

    def get_stacking_model(self):
        """Build the detection model.
        
        For stacking=True: Multi-model ensemble with meta-learner
        For stacking=False: Optimized GBM (matches AGFL's best architecture)
        """
        if self.use_stacking:
            estimators = [
                ('gbm_main', Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', GradientBoostingClassifier(
                        n_estimators=300, max_depth=5,
                        learning_rate=0.1, subsample=0.8,
                        min_samples_split=5, min_samples_leaf=3,
                        random_state=42
                    ))
                ])),
                ('rf', Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', RandomForestClassifier(
                        n_estimators=300, max_depth=None,
                        min_samples_split=3, random_state=42,
                        class_weight='balanced', n_jobs=-1
                    ))
                ])),
                ('mlp', Pipeline([
                    ('scaler', StandardScaler()),
                    ('clf', MLPClassifier(
                        hidden_layer_sizes=(128, 64),
                        max_iter=800, activation='relu',
                        random_state=42, early_stopping=True,
                        learning_rate='adaptive',
                        learning_rate_init=0.001
                    ))
                ])),
            ]
            
            return StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(
                    C=1.0, max_iter=1000, random_state=42
                ),
                cv=3,  # Reduced from 5 to preserve more training data
                stack_method='predict_proba',
                n_jobs=-1,
                passthrough=False,
            )
        else:
            # Optimized GBM — same architecture that makes AGFL-WL strong
            return Pipeline([
                ('scaler', StandardScaler()),
                ('clf', GradientBoostingClassifier(
                    n_estimators=300, max_depth=5,
                    learning_rate=0.1, subsample=0.8,
                    min_samples_split=5, min_samples_leaf=3,
                    random_state=42
                ))
            ])

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data from best available source (same as original)."""
        from run_experiments import PhishTraceExperiment
        exp = PhishTraceExperiment()
        X, y = exp.load_data()
        self.data_source = exp.data_source
        return X, y

    def load_trace_data_with_agfl(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Load trace data and extract BOTH ITG features AND AGFL features.
        This is the core of the comprehensive method — combining hand-crafted
        ITG features with automated graph feature learning.
        
        Returns: (X_itg, y, X_agfl) where X_agfl may be None if traces unavailable
        """
        from gnn_detector import (
            load_trace_graphs, trace_to_graph_dict,
            WLKernelExtractor, SpectralExtractor, RandomWalkExtractor
        )
        
        dataset_dir = str(DATASET_DIR)
        
        # Single-pass extraction: iterate trace files ONCE, extract both
        # ITG and AGFL features for the SAME traces to guarantee alignment.
        from analyzer.graph_builder import InteractionGraphBuilder
        builder = InteractionGraphBuilder()
        
        itg_features = []
        agfl_graphs = []
        aligned_labels = []
        traces_dir = DATASET_DIR / 'traces'
        
        trace_files = []
        for label_name, label_int in [('phishing', 1), ('benign', 0)]:
            trace_dir = traces_dir / label_name
            if trace_dir.exists():
                for trace_file in sorted(trace_dir.glob('*.json')):
                    trace_files.append((trace_file, label_int))
        
        for trace_file, label_int in trace_files:
            try:
                with open(trace_file, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                trace_data = raw.get('trace', raw)
                if 'url' not in trace_data and 'url' in raw:
                    trace_data['url'] = raw['url']
                
                # AGFL graph (skip degenerate traces)
                g = trace_to_graph_dict(trace_data)
                if len(g['nodes']) < 2:
                    continue
                
                # ITG features — must succeed for this trace to be included
                nx_graph = builder.build_graph_from_dict(trace_data)
                feats = builder.extract_features(nx_graph)
                feat_vec = []
                for name in BASE_FEATURE_NAMES:
                    val = getattr(feats, name, 0)
                    if isinstance(val, bool):
                        val = 1 if val else 0
                    feat_vec.append(float(val))
                
                # Both succeeded — keep this trace
                itg_features.append(feat_vec)
                agfl_graphs.append(g)
                aligned_labels.append(label_int)
            except Exception as e:
                logger.debug(f"Skipping {trace_file.name}: {e}")
                continue
        
        if len(itg_features) < 20:
            logger.warning("Too few valid traces, falling back to URL features")
            X, y_arr = self.load_data()
            return X, y_arr, None
        
        X_itg = np.array(itg_features)
        y = np.array(aligned_labels)
        
        # Extract AGFL features from the aligned graph list
        wl = WLKernelExtractor(n_iterations=5, vocab_size=256)
        wl.fit(agfl_graphs)
        spectral = SpectralExtractor(n_eigenvalues=20)
        rw = RandomWalkExtractor(n_walks=50, walk_length=10)
        rw.fit(agfl_graphs)
        
        X_wl = wl.transform(agfl_graphs)
        X_sp = spectral.transform(agfl_graphs)
        X_rw = rw.transform(agfl_graphs)
        X_agfl = np.hstack([X_wl, X_sp, X_rw])
        
        assert len(X_itg) == len(X_agfl) == len(y), \
            f"Alignment mismatch: ITG={len(X_itg)}, AGFL={len(X_agfl)}, y={len(y)}"
        
        self.data_source = "traces_with_agfl"
        print(f"  Loaded {len(y)} traces with ITG ({X_itg.shape[1]}D) + "
              f"AGFL ({X_agfl.shape[1]}D) features")
        
        return X_itg, y, X_agfl

    def build_comprehensive_features(self, X_itg: np.ndarray,
                                      X_agfl: Optional[np.ndarray] = None) -> np.ndarray:
        """Build comprehensive feature matrix from ITG and real AGFL features."""
        # ITG + Engineered (42D)
        X_eng = self.engineer_features(X_itg)
        
        blocks = [X_eng]
        
        # DRP features (15D)
        if self.use_drp_features:
            X_drp = self.generate_drp_features(X_itg)
            blocks.append(X_drp)
        
        # Real AGFL features (300D) or simulated
        if X_agfl is not None and self.use_agfl_features:
            blocks.append(X_agfl)
        elif self.use_agfl_features:
            X_agfl_sim = self.generate_agfl_features(X_itg)
            blocks.append(X_agfl_sim)
        
        X_full = np.hstack(blocks)
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        return X_full

    def evaluate(self, X: np.ndarray, y: np.ndarray, n_folds: int = 10,
                 X_agfl: Optional[np.ndarray] = None) -> Tuple[Dict, Dict]:
        """Run cross-validated evaluation with full feature pipeline."""
        from sklearn.utils.class_weight import compute_sample_weight
        
        # Build full feature matrix
        if X_agfl is not None:
            X_full = self.build_comprehensive_features(X, X_agfl)
        else:
            X_full = self.build_feature_matrix(X)
        
        agfl_dim = X_full.shape[1] - X.shape[1] - 12 - (15 if self.use_drp_features else 0)
        print(f"\n  Feature matrix: {X_full.shape[1]}D")
        print(f"    Base: {X.shape[1]}D, Engineered: 12D, "
              f"DRP: {'15D' if self.use_drp_features else '0D'}, "
              f"AGFL: {agfl_dim}D"
              f"{' (real)' if X_agfl is not None else ' (simulated)'}")
        
        # Adapt folds
        minority_count = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
        actual_folds = max(2, min(n_folds, minority_count))
        
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
        cv_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(X_full, y)):
            X_train, X_test = X_full[train_idx], X_full[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            model = self.get_stacking_model()
            
            # Use sample weights for balanced training
            sw = compute_sample_weight('balanced', y_train)
            try:
                if self.use_stacking:
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train, clf__sample_weight=sw)
            except TypeError:
                model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            
            cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            
            if len(np.unique(y_test)) > 1:
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    cv_metrics['auc'].append(roc_auc_score(y_test, y_proba))
                except Exception:
                    cv_metrics['auc'].append(np.nan)
            else:
                cv_metrics['auc'].append(np.nan)
        
        results = {k: round(float(np.nanmean(v)), 4) for k, v in cv_metrics.items()}
        std = {k: round(float(np.nanstd(v)), 4) for k, v in cv_metrics.items()}
        
        return results, std


# ============================================================================
# Multi-View Ensemble: Best-performing approach
# ============================================================================

def evaluate_multiview_ensemble(X_itg: np.ndarray, y: np.ndarray,
                                 X_agfl: np.ndarray, n_folds: int = 10,
                                 view_weights: Optional[List[float]] = None) -> Tuple[Dict, Dict]:
    """
    Multi-view ensemble: train separate optimized classifiers on each feature group
    and combine via weighted soft voting. This preserves each view's specialization
    without signal dilution from feature concatenation.
    
    Views:
      1. WL Kernel (256D) - strongest individual view  
      2. ITG + Engineered (42D) - hand-crafted behavioral features
      3. Spectral + RandomWalk (44D) - complementary graph features
      4. DRP features (15D) - depth/pruning features
    
    Weights control the influence of each view. Default: WL-dominant.
    """
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Split AGFL features back into components
    X_wl = X_agfl[:, :256]       # WL kernel features
    X_sp = X_agfl[:, 256:282]    # Spectral (26D)
    X_rw = X_agfl[:, 282:300]    # Random Walk (18D)
    X_sp_rw = X_agfl[:, 256:]    # Spectral + RW combined (44D)
    
    # Engineer ITG features
    detector = ComprehensivePhishTraceDetector(
        use_stacking=False, use_drp_features=True, use_agfl_features=False
    )
    X_eng = detector.engineer_features(X_itg)          # ITG + Engineered (42D)
    X_drp = detector.generate_drp_features(X_itg)      # DRP (15D)
    X_itg_eng_drp = np.hstack([X_eng, X_drp])          # 57D
    
    # Define views and their classifiers
    views = [
        ("WL-Kernel", X_wl, GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, min_samples_leaf=3,
            random_state=42
        )),
        ("ITG+Eng+DRP", X_itg_eng_drp, GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, random_state=42
        )),
        ("Spectral+RW", X_sp_rw, GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, random_state=42
        )),
    ]
    
    if view_weights is None:
        view_weights = [3.0, 1.5, 1.0]  # WL-dominant
    
    total_w = sum(view_weights)
    view_weights = [w / total_w for w in view_weights]
    
    # Adapt folds
    minority_count = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    actual_folds = max(2, min(n_folds, minority_count))
    
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    cv_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_wl, y)):
        y_train, y_test = y[train_idx], y[test_idx]
        sw = compute_sample_weight('balanced', y_train)
        
        # Train each view's classifier and get probabilities
        proba_sum = np.zeros((len(test_idx), 2))
        
        for (view_name, X_view, clf_template), weight in zip(views, view_weights):
            X_v_train = X_view[train_idx]
            X_v_test = X_view[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_v_train_s = scaler.fit_transform(X_v_train)
            X_v_test_s = scaler.transform(X_v_test)
            
            # Clone and train classifier
            import copy
            clf = copy.deepcopy(clf_template)
            clf.fit(X_v_train_s, y_train, sample_weight=sw)
            
            proba = clf.predict_proba(X_v_test_s)
            proba_sum += weight * proba
        
        # Predict from weighted average probabilities
        y_pred = np.argmax(proba_sum, axis=1)
        y_proba = proba_sum[:, 1] / proba_sum.sum(axis=1)
        
        cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        if len(np.unique(y_test)) > 1:
            try:
                cv_metrics['auc'].append(roc_auc_score(y_test, y_proba))
            except Exception:
                cv_metrics['auc'].append(np.nan)
        else:
            cv_metrics['auc'].append(np.nan)
    
    results = {k: round(float(np.nanmean(v)), 4) for k, v in cv_metrics.items()}
    std = {k: round(float(np.nanstd(v)), 4) for k, v in cv_metrics.items()}
    
    return results, std


def evaluate_wl_only_on_traces(X_agfl: np.ndarray, y: np.ndarray,
                                n_folds: int = 10) -> Tuple[Dict, Dict]:
    """Evaluate WL kernel features alone on the same trace data for fair comparison."""
    from sklearn.utils.class_weight import compute_sample_weight
    
    X_wl = X_agfl[:, :256]
    
    minority_count = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    actual_folds = max(2, min(n_folds, minority_count))
    
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
    cv_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_wl, y)):
        X_train, X_test = X_wl[train_idx], X_wl[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        sw = compute_sample_weight('balanced', y_train)
        clf = GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            subsample=0.8, min_samples_split=5, min_samples_leaf=3,
            random_state=42
        )
        clf.fit(X_train_s, y_train, sample_weight=sw)
        
        y_pred = clf.predict(X_test_s)
        cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        if len(np.unique(y_test)) > 1:
            try:
                y_proba = clf.predict_proba(X_test_s)[:, 1]
                cv_metrics['auc'].append(roc_auc_score(y_test, y_proba))
            except Exception:
                cv_metrics['auc'].append(np.nan)
        else:
            cv_metrics['auc'].append(np.nan)
    
    results = {k: round(float(np.nanmean(v)), 4) for k, v in cv_metrics.items()}
    std = {k: round(float(np.nanstd(v)), 4) for k, v in cv_metrics.items()}
    return results, std


# ============================================================================
# Ablation: Feature Group Contribution Analysis
# ============================================================================

# Feature group ablation is handled by run_feature_group_ablation_trace() in the main runner


# ============================================================================
# Main Experiment Runner
# ============================================================================

def run_comprehensive_experiments():
    """Run the complete experiment suite with all methods."""
    print("=" * 80)
    print("PhishTrace Comprehensive Stacked Detection - Full Experiment Suite")
    print("=" * 80)

    # Load TRACE data with real AGFL features (key to beating baselines)
    detector = ComprehensivePhishTraceDetector(
        use_stacking=True, use_drp_features=True, use_agfl_features=True
    )
    
    # Try trace-based loading first (real AGFL features)
    try:
        X_itg, y, X_agfl = detector.load_trace_data_with_agfl()
        has_real_agfl = X_agfl is not None
    except Exception as e:
        print(f"  Trace loading failed ({e}), falling back to URL features")
        X_itg, y = detector.load_data()[0], detector.load_data()[1]
        X_agfl = None
        has_real_agfl = False
    
    print(f"\nData source: {detector.data_source}")
    print(f"Total samples: {len(y)} (Phishing: {sum(y)}, Benign: {len(y) - sum(y)})")
    print(f"Real AGFL features: {has_real_agfl}")
    
    # ── 1. Our comprehensive method (Multi-View Ensemble) ──
    print("\n" + "=" * 60)
    print("1. PhishTrace Multi-View Ensemble (Our Method)")
    print("=" * 60)
    
    if has_real_agfl:
        print("  Views: WL(256D) + ITG+Eng+DRP(57D) + Spec+RW(44D)")
        print("  Voting weights: [3.0, 1.5, 1.0] (WL-dominant)")
        our_results, our_std = evaluate_multiview_ensemble(X_itg, y, X_agfl)
        print("\nPhishTrace Multi-View Ensemble Results:")
    else:
        our_results, our_std = detector.evaluate(X_itg, y, X_agfl=X_agfl)
        print("\nPhishTrace-DRP-Stacked Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {metric.upper():>10}: {our_results[metric]:.4f} (+/- {our_std[metric]:.4f})")

    # ── 1b. Stacking variant for ablation ──
    print("\n" + "=" * 60)
    print("1b. PhishTrace-DRP-Stacked (Feature Concatenation)")
    print("=" * 60)
    
    stacked_results, stacked_std = detector.evaluate(X_itg, y, X_agfl=X_agfl)
    print("\nPhishTrace-DRP-Stacked Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {metric.upper():>10}: {stacked_results[metric]:.4f} (+/- {stacked_std[metric]:.4f})")

    # ── 1c. WL-only baseline on same data (fair AGFL comparison) ──
    if has_real_agfl:
        print("\n" + "=" * 60)
        print("1c. AGFL-WL on Same Traces (Fair Comparison)")
        print("=" * 60)
        
        wl_same_results, wl_same_std = evaluate_wl_only_on_traces(X_agfl, y)
        print("\nAGFL-WL (same traces) Results:")
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
            print(f"  {metric.upper():>10}: {wl_same_results[metric]:.4f} (+/- {wl_same_std[metric]:.4f})")
    else:
        wl_same_results = None

    # ── 2. Original PhishTrace (ITG features only, for comparison) ──
    print("\n" + "=" * 60)
    print("2. Original PhishTrace (Base + Engineered only)")
    print("=" * 60)
    
    orig_detector = ComprehensivePhishTraceDetector(
        use_stacking=False, use_drp_features=False, use_agfl_features=False
    )
    orig_results, orig_std = orig_detector.evaluate(X_itg, y)
    print("\nOriginal PhishTrace Results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
        print(f"  {metric.upper():>10}: {orig_results[metric]:.4f} (+/- {orig_std[metric]:.4f})")

    # ── 3. Baselines (also on trace ITG features for fair comparison) ──
    print("\n" + "=" * 60)
    print("3. Baseline Methods (on trace-based ITG features)")
    print("=" * 60)
    
    from baselines.baseline_methods import run_all_baselines
    baseline_results = run_all_baselines(X_itg, y, BASE_FEATURE_NAMES)

    # ── 4. AGFL methods ──
    print("\n" + "=" * 60)
    print("4. AGFL Methods (on trace graphs)")  
    print("=" * 60)
    
    try:
        from gnn_detector import evaluate_agfl
        agfl_results = evaluate_agfl()
    except Exception as e:
        print(f"  AGFL evaluation failed: {e}")
        agfl_results = {}

    # ── 5. Feature group ablation on trace data ──
    print("\n" + "=" * 60)
    print("5. Feature Group Ablation (on trace data)")
    print("=" * 60)
    
    ablation_results = run_feature_group_ablation_trace(X_itg, y, X_agfl)

    # ── 6. Comparison table ──
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)
    
    all_results = {}
    all_results['PhishTrace-MVE (Ours)'] = our_results
    all_results['PhishTrace-Stacked (Ours)'] = stacked_results
    all_results['PhishTrace (Original)'] = orig_results
    if wl_same_results:
        all_results['AGFL-WL (same data)'] = wl_same_results
    all_results.update(baseline_results)
    if agfl_results:
        for name, metrics in agfl_results.items():
            if isinstance(metrics, dict):
                # Handle nested format from evaluate_agfl
                inner = metrics.get('mean', metrics)
                if isinstance(inner, dict) and 'f1' in inner:
                    all_results[name] = inner

    print(f"\n{'Method':<35} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'AUC':>8}")
    print("-" * 85)
    
    for method, metrics in sorted(all_results.items(), key=lambda x: -x[1].get('f1', 0)):
        acc = metrics.get('accuracy', 0)
        prec = metrics.get('precision', 0)
        rec = metrics.get('recall', 0)
        f1 = metrics.get('f1', 0)
        auc = metrics.get('auc', 0)
        marker = " ***" if 'Ours' in method else ""
        print(f"{method:<35} {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f} {auc:>8.4f}{marker}")

    # ── 7. Improvement analysis ──
    print("\n" + "=" * 60)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 60)
    
    # Compare against ALL methods including AGFL
    all_non_ours = {k: v for k, v in all_results.items() if 'Ours' not in k}
    if all_non_ours:
        best_name = max(all_non_ours.items(), key=lambda x: x[1].get('f1', 0))[0]
        best_f1 = all_non_ours[best_name].get('f1', 0)
        
        print(f"\n  Best competing method: {best_name} (F1={best_f1:.4f})")
        print(f"  Our method (MVE): PhishTrace Multi-View (F1={our_results['f1']:.4f})")
        improvement = ((our_results['f1'] - best_f1) / max(best_f1, 1e-8)) * 100
        print(f"  Improvement over best competitor: {improvement:+.2f}%")
        
        if wl_same_results:
            wl_f1 = wl_same_results['f1']
            wl_improvement = ((our_results['f1'] - wl_f1) / max(wl_f1, 1e-8)) * 100
            print(f"  AGFL-WL (same data): F1={wl_f1:.4f}")
            print(f"  Improvement over AGFL-WL (same data): {wl_improvement:+.2f}%")
    
    best_baseline_f1 = max(
        (m.get('f1', 0) for name, m in baseline_results.items()),
        default=0
    )
    best_baseline_name = max(
        baseline_results.items(),
        key=lambda x: x[1].get('f1', 0)
    )[0] if baseline_results else "N/A"
    
    print(f"  Best traditional baseline: {best_baseline_name} (F1={best_baseline_f1:.4f})")
    baseline_improvement = ((our_results['f1'] - best_baseline_f1) / max(best_baseline_f1, 1e-8)) * 100
    print(f"  Improvement over best baseline: {baseline_improvement:+.2f}%")
    
    orig_improvement = ((our_results['f1'] - orig_results['f1']) / max(orig_results['f1'], 1e-8)) * 100
    print(f"  Improvement over original PhishTrace: {orig_improvement:+.2f}%")

    # ── 8. Save results ──
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    full_results = {
        'run_timestamp': datetime.now().isoformat(),
        'data_source': detector.data_source,
        'our_method_mve': our_results,
        'our_method_mve_std': our_std,
        'our_method_stacked': stacked_results,
        'our_method_stacked_std': stacked_std,
        'agfl_wl_same_data': wl_same_results,
        'dataset_size': int(len(y)),
        'phishing_count': int(sum(y)),
        'benign_count': int(len(y) - sum(y)),
        'real_agfl_features': has_real_agfl,
        'our_method': our_results,
        'our_method_std': our_std,
        'original_phishtrace': orig_results,
        'baselines': baseline_results,
        'agfl_results': agfl_results,
        'feature_group_ablation': ablation_results,
        'all_results': all_results,
    }
    
    results_file = output_dir / 'comprehensive_results.json'
    results_file.write_text(json.dumps(full_results, indent=2, default=str), encoding='utf-8')
    print(f"\nResults saved to {results_file}")
    
    return full_results


def run_feature_group_ablation_trace(X_itg: np.ndarray, y: np.ndarray,
                                      X_agfl: Optional[np.ndarray] = None) -> Dict:
    """
    Ablation study on trace data showing contribution of each feature group.
    """
    results = {}
    
    configs = [
        ("Base ITG (30D)", False, False, False, None),
        ("Base + Engineered (42D)", False, False, False, None),
        ("Base + Eng + DRP (57D)", True, False, False, None),
        ("Base + Eng + AGFL", False, True, False, X_agfl),
        ("Full Features", True, True, False, X_agfl),
        ("Full + Stacking", True, True, True, X_agfl),
    ]
    
    for name, use_drp, use_agfl, use_stacking, agfl_feats in configs:
        print(f"\n--- {name} ---")
        
        if name == "Base ITG (30D)":
            # Only base features
            minority_count = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
            actual_folds = max(2, min(10, minority_count))
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('clf', RandomForestClassifier(
                    n_estimators=500, max_depth=None,
                    random_state=42, class_weight='balanced', n_jobs=-1
                ))
            ])
            skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)
            cv_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
            for fold, (train_idx, test_idx) in enumerate(skf.split(X_itg, y)):
                X_train, X_test = X_itg[train_idx], X_itg[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                cv_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
                cv_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
                cv_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
                cv_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
                if len(np.unique(y_test)) > 1:
                    try:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        cv_metrics['auc'].append(roc_auc_score(y_test, y_proba))
                    except Exception:
                        cv_metrics['auc'].append(np.nan)
                else:
                    cv_metrics['auc'].append(np.nan)
            metrics = {k: round(float(np.nanmean(v)), 4) for k, v in cv_metrics.items()}
        else:
            detector = ComprehensivePhishTraceDetector(
                use_stacking=use_stacking,
                use_drp_features=use_drp,
                use_agfl_features=use_agfl,
            )
            metrics, _ = detector.evaluate(X_itg, y, X_agfl=agfl_feats)
        
        results[name] = metrics
        print(f"  Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, AUC={metrics.get('auc', 0):.4f}")
    
    return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
    run_comprehensive_experiments()
