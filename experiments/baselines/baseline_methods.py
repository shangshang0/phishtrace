"""
Baseline Methods for Phishing Detection - Top Venue Implementations
Implements published phishing detection approaches from USENIX, S&P, CCS, NDSS, etc.
for fair comparison with PhishTrace.

Each baseline uses the feature subset and classifier architecture that most closely
matches the cited paper's core approach, adapted to our ITG feature space.

NO noise injection. Performance differences come from:
1. Different feature subsets (each method sees only what its paradigm provides)
2. Different classifier architectures (matching the paper's approach)
3. Different hyperparameters (matching the paper's reported setup)
"""

import warnings
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               VotingClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, List, Optional, Tuple
import logging

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn')

logger = logging.getLogger(__name__)
class BaselineMethod:
    """Base class for all baseline methods."""
    name = "BaselineMethod"
    venue = "Unknown"
    year = 0
    paper_ref = ""

    def __init__(self):
        self.pipeline = None

    def get_pipeline(self):
        raise NotImplementedError

    def preprocess(self, X, feature_names=None):
        """Override for method-specific feature preprocessing."""
        return X

    def evaluate(self, X, y, n_folds=10, feature_names=None) -> Dict[str, float]:
        """Evaluate with stratified k-fold cross-validation."""
        from sklearn.utils.class_weight import compute_sample_weight
        X_processed = self.preprocess(X, feature_names)
        # Adapt n_splits to minority class size
        minority_count = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
        actual_folds = max(2, min(n_folds, minority_count))
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

        for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_processed, y)):
            X_train, X_test = X_processed[train_idx], X_processed[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = self.get_pipeline()
            # Pass sample_weight for classifiers that need it (via pipeline prefix)
            sw = compute_sample_weight('balanced', y_train)
            try:
                pipeline.fit(X_train, y_train, clf__sample_weight=sw)
            except TypeError:
                # Classifier doesn't accept sample_weight (e.g., MLPClassifier)
                pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            # AUC requires both classes in test fold
            auc = np.nan
            if len(np.unique(y_test)) > 1:
                try:
                    y_proba = pipeline.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                except (AttributeError, ValueError):
                    pass

            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
            metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            metrics['auc'].append(auc)

        return {k: round(float(np.nanmean(v)), 4) for k, v in metrics.items()}


# ─── Baseline 1: Phishpedia (USENIX Security 2021) ───
class PhishpediaBaseline(BaselineMethod):
    """
    Phishpedia: A Hybrid Deep Learning Based Approach to Visually
    Identify Phishing Webpages (USENIX Security 2021).
    
    Core idea: Visual similarity + logo detection + CRP (Credential Requiring Page) detection.
    We approximate the CRP detector component using form/credential features.
    """
    name = "Phishpedia"
    venue = "USENIX Sec"
    year = 2021
    paper_ref = "lin2021phishpedia"

    def preprocess(self, X, feature_names=None):
        """Use credential & visual-proxy features (form ratio, input fields, password presence)."""
        if feature_names:
            idx = [i for i, n in enumerate(feature_names) if n in (
                'has_password_input', 'has_email_input', 'num_forms',
                'num_input_fields', 'credential_fields_count', 'form_to_node_ratio',
                'num_buttons', 'requests_sensitive_data', 'financial_fields_count',
                'num_links', 'num_nodes', 'graph_density'
            )]
            if idx:
                return X[:, idx]
        return X

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                                  activation='relu', random_state=42, early_stopping=True,
                                  learning_rate='adaptive'))
        ])


# ─── Baseline 2: CrawlPhish (USENIX Security 2021) ───
class CrawlPhishBaseline(BaselineMethod):
    """
    CrawlPhish: Large-scale Analysis of Client-side Cloaking
    Techniques in Phishing (USENIX Security 2021).
    
    Core idea: Detect cloaking via redirect chain analysis + content fingerprinting.
    Focus: redirect-centric features (the paper's primary signal is redirect behavior).
    CrawlPhish cannot observe post-submission behavioral patterns; it only
    inspects the redirect chain before the landing page renders.
    """
    name = "CrawlPhish"
    venue = "USENIX Sec"
    year = 2021
    paper_ref = "zhang2021crawlphish"

    def preprocess(self, X, feature_names=None):
        """Focus on redirect and cloaking-related features only.
        CrawlPhish analyses redirect chains pre-render, so interaction
        and form features are unavailable."""
        if feature_names:
            target = {
                'num_redirects', 'external_redirects', 'max_redirect_depth',
                'out_degree_max', 'num_edges', 'num_connected_components',
                'avg_path_length', 'diameter',
            }
            idx = [i for i, n in enumerate(feature_names) if n in target]
            if idx:
                return X[:, idx]
        return X

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                learning_rate=0.1, random_state=42))
        ])


# ─── Baseline 3: PhishFarm (S&P 2019) ───
class PhishFarmBaseline(BaselineMethod):
    """
    PhishFarm: A Scalable Framework for Measuring the Effectiveness
    of Evasion Techniques against Browser Phishing Blacklists (S&P 2019).
    
    Core idea: Evasion effectiveness measurement using URL-level features.
    PhishFarm relies on blacklist coverage and URL lexical features;
    it does not perform deep crawling or behavioral analysis.
    """
    name = "PhishFarm"
    venue = "IEEE S&P"
    year = 2019
    paper_ref = "oest2019phishfarm"

    def preprocess(self, X, feature_names=None):
        """URL structural and link-based features.
        PhishFarm operates at URL-level; interaction-time and deep-crawl
        signals are unavailable."""
        if feature_names:
            idx = [i for i, n in enumerate(feature_names) if n in (
                'num_links', 'num_nodes', 'num_edges',
                'graph_density', 'num_redirects', 'max_redirect_depth',
            )]
            if idx:
                return X[:, idx]
        return X

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=10,
                                           random_state=42, class_weight='balanced'))
        ])


# ─── Baseline 4: VisualPhishNet (CCS 2020) ───
class VisualPhishNetBaseline(BaselineMethod):
    """
    VisualPhishNet: Zero-Day Phishing Website Detection by Visual Similarity (CCS 2020).
    
    Core idea: Visual fingerprinting + triplet network for similarity.
    We proxy visual complexity with topology centrality and page structure features.
    VisualPhishNet is inherently a visual approach; its graph-feature proxy is
    weaker than methods designed for structural analysis.
    """
    name = "VisualPhishNet"
    venue = "ACM CCS"
    year = 2020
    paper_ref = "abdelnabi2020visualphishnet"

    def preprocess(self, X, feature_names=None):
        """Focus on page structural complexity as visual proxy.
        VisualPhishNet analyses screenshots; we proxy visual layout
        complexity with centrality and page structure features."""
        if feature_names:
            idx = [i for i, n in enumerate(feature_names) if n in (
                'num_buttons', 'num_links', 'clustering_coefficient',
                'betweenness_centrality_max', 'closeness_centrality_max',
                'pagerank_max', 'num_input_fields',
            )]
            if idx:
                return X[:, idx]
        return X

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=600,
                                  activation='relu', random_state=42,
                                  early_stopping=True, learning_rate='adaptive'))
        ])


# ─── Baseline 5: TESSERACT-style (USENIX Security 2019) ───
class TesseractBaseline(BaselineMethod):
    """
    TESSERACT: Eliminating Experimental Bias in Malware Classification
    Across Space and Time (USENIX Security 2019).
    
    Core idea: Proper temporal evaluation with concept drift awareness.
    Uses all features but with TESSERACT's recommended experimental setup.
    """
    name = "TESSERACT-RF"
    venue = "USENIX Sec"
    year = 2019
    paper_ref = "pendlebury2019tesseract"

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=100, max_depth=None,
                                           min_samples_split=2, random_state=42,
                                           class_weight='balanced'))
        ])


# ─── Baseline 6: Lateral Phishing Detector (USENIX Security 2019) ───
class LateralPhishingBaseline(BaselineMethod):
    """
    Detecting and Characterizing Lateral Phishing at Scale (USENIX Security 2019).
    
    Core idea: Anomaly-based detection using account behavior and email features.
    We adapt to web features focusing on behavioral anomalies.
    """
    name = "LateralPhish"
    venue = "USENIX Sec"
    year = 2019
    paper_ref = "ho2019detecting"

    def preprocess(self, X, feature_names=None):
        """Behavioral + temporal features."""
        if feature_names:
            idx = [i for i, n in enumerate(feature_names) if n in (
                'total_interaction_time', 'avg_time_between_events', 'event_frequency',
                'requests_sensitive_data', 'credential_fields_count', 'financial_fields_count',
                'num_forms', 'has_password_input', 'has_email_input',
                'num_redirects', 'external_redirects'
            )]
            if idx:
                return X[:, idx]
        return X

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                learning_rate=0.1, random_state=42))
        ])


# ─── Baseline 7: DeltaPhish (ESORICS 2017) ───
class DeltaPhishBaseline(BaselineMethod):
    """
    DeltaPhish: Detecting Phishing Webpages in Compromised Websites (ESORICS 2017).
    
    Core idea: Detect content deltas between original and phishing content.
    DeltaPhish focuses on topology deviation in compromised host pages.
    It uses SVM-RBF which is sensitive to dimensionality; with only
    topology features it captures structural anomalies but misses
    behavioral and temporal signals.
    """
    name = "DeltaPhish"
    venue = "ESORICS"
    year = 2017
    paper_ref = "corona2017deltaphish"

    def preprocess(self, X, feature_names=None):
        """Topology-only features for content delta approximation.
        DeltaPhish compares compromised vs. original page topology;
        it does not observe interaction or temporal signals."""
        if feature_names:
            idx = [i for i, n in enumerate(feature_names) if n in (
                'avg_degree', 'max_degree', 'graph_density',
                'clustering_coefficient', 'betweenness_centrality_max',
                'pagerank_max', 'in_degree_max', 'out_degree_max',
            )]
            if idx:
                return X[:, idx]
        return X

    def get_pipeline(self):
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=10.0, gamma='scale', probability=True,
                        random_state=42, class_weight='balanced'))
        ])


# ─── Baseline 8: Dos & Don'ts ML (USENIX Security 2022) ───
class DosAndDontsBaseline(BaselineMethod):
    """
    Dos and Don'ts of Machine Learning in Computer Security (USENIX Security 2022).
    
    This represents the 'best practices' baseline following Arp et al.'s recommendations:
    proper feature normalization, bias-aware evaluation, calibrated classifier.
    Uses all features with recommended setup.
    """
    name = "Arp-BestPrac"
    venue = "USENIX Sec"
    year = 2022
    paper_ref = "arp2022dos"

    def get_pipeline(self):
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, class_weight='balanced')),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)),
            ('lr', LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight='balanced')),
        ]
        return Pipeline([
            ('scaler', StandardScaler()),
            ('clf', VotingClassifier(estimators=estimators, voting='soft'))
        ])


def get_all_baselines() -> List[BaselineMethod]:
    """Return all baseline methods."""
    return [
        PhishpediaBaseline(),
        CrawlPhishBaseline(),
        PhishFarmBaseline(),
        VisualPhishNetBaseline(),
        TesseractBaseline(),
        LateralPhishingBaseline(),
        DeltaPhishBaseline(),
        DosAndDontsBaseline(),
    ]


def run_all_baselines(X: np.ndarray, y: np.ndarray, feature_names: List[str] = None) -> Dict:
    """Run all baseline methods and return results."""
    baselines = get_all_baselines()
    results = {}

    for baseline in baselines:
        print(f"  Evaluating {baseline.name} ({baseline.venue} {baseline.year})...")
        try:
            metrics = baseline.evaluate(X, y, feature_names=feature_names)
            results[baseline.name] = {
                **metrics,
                'venue': baseline.venue,
                'year': baseline.year,
                'ref': baseline.paper_ref
            }
            print(f"    Acc: {metrics['accuracy']:.4f}, "
                  f"Prec: {metrics['precision']:.4f}, "
                  f"Rec: {metrics['recall']:.4f}, "
                  f"F1: {metrics['f1']:.4f}, "
                  f"AUC: {metrics['auc']:.4f}")
        except Exception as e:
            logger.error(f"  {baseline.name} failed: {e}")
            results[baseline.name] = {
                'accuracy': 0, 'precision': 0, 'recall': 0, 'f1': 0, 'auc': 0,
                'venue': baseline.venue, 'year': baseline.year, 'ref': baseline.paper_ref,
                'error': str(e)
            }

    return results



