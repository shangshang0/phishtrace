#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Baseline URL-Adversarial Robustness Evaluation (v3)

Includes the proper SMVE (per-view voting ensemble) architecture
that demonstrates robustness under URL camouflage attacks.

Key innovation: PhishTrace-SMVE trains separate classifiers per view.
Under URL camouflage, only the URL view classifier is fooled. The 5
behavioral classifiers (Network, Redirect, Interaction, ITG, Cross-view)
maintain their predictions and outvote the compromised URL view.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.enhanced_itg_detector import (
    load_all_traces,
    build_all_views,
    engineer_itg_features,
)
from experiments.baselines.real_baselines import (
    extract_url_features as baseline_extract_url,
    extract_redirect_features as baseline_extract_redirect,
    extract_network_features as baseline_extract_network,
    extract_content_features as baseline_extract_content,
)


# ==================================================================
# Feature-Space Perturbation
# ==================================================================

def benign_replacement_attack(X_test, y_test, url_cols, benign_pool, strength=1.0):
    """Replace URL-feature dims of phishing with benign features."""
    X_atk = X_test.copy()
    rng = np.random.RandomState(42)
    phish_idx = np.where(y_test == 1)[0]
    benign_samples = benign_pool[rng.choice(len(benign_pool), len(phish_idx), replace=True)]
    for i, idx in enumerate(phish_idx):
        X_atk[idx, url_cols] = strength * benign_samples[i] + (1 - strength) * X_atk[idx, url_cols]
    return X_atk


def mean_substitution_attack(X_test, y_test, url_cols, benign_mean):
    """Replace URL features with benign mean."""
    X_atk = X_test.copy()
    for idx in np.where(y_test == 1)[0]:
        X_atk[idx, url_cols] = benign_mean
    return X_atk


# ==================================================================
# Per-View Voting SMVE (Robust Architecture)
# ==================================================================

class SMVEVoting:
    """Per-view voting ensemble that is robust to single-view attacks.

    Architecture:
    - 6 independent view classifiers
    - Weighted soft voting at decision level
    - Under URL camouflage: only URL classifier is fooled,
      5 behavioral classifiers maintain detection
    """

    def __init__(self, view_weights=None):
        self.view_weights = view_weights or [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        self.view_classifiers = None
        self.view_scalers = None

    def _make_classifiers(self):
        return [
            ("url", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42)),
            ("net", RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)),
            ("redir", GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                subsample=0.8, random_state=42)),
            ("interact", RandomForestClassifier(
                n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)),
            ("itg", RandomForestClassifier(
                n_estimators=300, min_samples_split=3,
                class_weight="balanced", random_state=42, n_jobs=-1)),
            ("cross", GradientBoostingClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42)),
        ]

    def fit(self, X_views_train, y_train):
        """Train each view classifier independently."""
        self.view_classifiers = self._make_classifiers()
        self.view_scalers = []
        for v_idx, (name, clf) in enumerate(self.view_classifiers):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_views_train[v_idx])
            clf.fit(X_scaled, y_train)
            self.view_scalers.append(scaler)

    def predict_proba_views(self, X_views_test):
        """Get per-view probabilities."""
        probas = []
        for v_idx, (name, clf) in enumerate(self.view_classifiers):
            X_s = self.view_scalers[v_idx].transform(X_views_test[v_idx])
            try:
                p = clf.predict_proba(X_s)[:, 1]
            except Exception:
                p = clf.predict(X_s).astype(float)
            probas.append(p)
        return probas

    def predict(self, X_views_test):
        """Weighted soft voting prediction."""
        probas = self.predict_proba_views(X_views_test)
        weighted = np.zeros(len(probas[0]))
        total_w = sum(self.view_weights)
        for v_idx, p in enumerate(probas):
            weighted += self.view_weights[v_idx] * p
        weighted /= total_w
        return (weighted >= 0.5).astype(int), weighted


# ==================================================================
# Unified Evaluation
# ==================================================================

def evaluate_simple(name, clf_factory, X_train, y_train, X_test, y_test):
    """Evaluate a standard (non-SMVE) method."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    clf = clf_factory()
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    try:
        y_proba = clf.predict_proba(Xte)[:, 1]
    except Exception:
        y_proba = y_pred.astype(float)
    return y_pred, y_proba


def compute_metrics(y_test, y_pred, y_proba):
    return {
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_proba),
    }


def compute_evasion(y_test, y_pred_clean, y_pred_attacked):
    """Evasion rate: correctly detected phishing that now evade."""
    phish = (y_test == 1)
    detected = (y_pred_clean == 1) & phish
    evaded = detected & (y_pred_attacked == 0)
    return evaded.sum() / max(1, detected.sum())




# ==================================================================
# Main
# ==================================================================

def main():
    """
    Four-phase adversarial robustness evaluation:

    Phase 1: Clean performance (baseline reference)
    Phase 2: Targeted feature replacement (each baseline attacked on ITS OWN
             features) — demonstrates every single-view method is vulnerable
    Phase 3: URL-only camouflage (cross-system comparison) — demonstrates
             SMVE structural defense via per-view voting
    Phase 4: Progressive view corruption (SMVE) — demonstrates graceful
             degradation requiring majority corruption to evade
    """
    print("=" * 72)
    print("CROSS-BASELINE ADVERSARIAL ROBUSTNESS EVALUATION  (v3)")
    print("=" * 72)

    # ──────────────────────────────────────────────────────────────
    # 1. Load & feature extraction
    # ──────────────────────────────────────────────────────────────
    print("\n[1/4] Loading dataset...")
    traces, labels = load_all_traces()
    y = labels.astype(int)
    n_phish, n_benign = int(y.sum()), int((1 - y).sum())
    print(f"  Total: {len(traces)} ({n_phish} phishing, {n_benign} benign)")

    if len(traces) < 20:
        print("[WARN] Too few traces — aborting adversarial eval.")
        out = {"dataset": {"total": len(traces), "phishing": n_phish,
                           "benign": n_benign}, "results": {}}
        out_path = PROJECT_ROOT / "experiments" / "results" / "url_adversarial_v3.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        return

    print("\n[2/4] Extracting features...")

    # Baseline features
    X_url_bl = np.array([baseline_extract_url(t) for t in traces])
    X_redir_bl = np.array([baseline_extract_redirect(t) for t in traces])
    X_net_bl = np.array([baseline_extract_network(t) for t in traces])
    X_content_bl = np.array([baseline_extract_content(t) for t in traces])
    X_stack_bl = np.hstack([X_url_bl, X_redir_bl, X_net_bl, X_content_bl])

    # PhishTrace view features
    X_url_pt, X_net_pt, X_redir_pt, X_interact_pt, X_itg_pt, X_cross_pt, _ = \
        build_all_views(traces)
    X_itg_eng_pt = engineer_itg_features(X_itg_pt)

    view_arrays = [X_url_pt, X_net_pt, X_redir_pt,
                   X_interact_pt, X_itg_eng_pt, X_cross_pt]
    view_names = ["URL", "Network", "Redirect",
                  "Interaction", "ITG+Eng", "Cross"]

    feat_dims = [v.shape[1] for v in view_arrays]
    total_dim = sum(feat_dims)
    print(f"  Baseline: {X_stack_bl.shape[1]}D  "
          f"SMVE: {total_dim}D ({len(view_arrays)} views)")

    # ── train / test split ──
    idx = np.arange(len(traces))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.3, stratify=y, random_state=42)
    y_tr, y_te = y[idx_tr], y[idx_te]
    benign = (y_tr == 0)
    print(f"  Train: {len(idx_tr)}  Test: {len(idx_te)}")

    # Benign feature pools — baseline
    pool_url = X_url_bl[idx_tr][benign]
    pool_redir = X_redir_bl[idx_tr][benign]
    pool_net = X_net_bl[idx_tr][benign]
    pool_content = X_content_bl[idx_tr][benign]
    pool_stack = X_stack_bl[idx_tr][benign]

    # Benign feature pools — SMVE views
    smve_pools = [v[idx_tr][benign] for v in view_arrays]

    # ── classifier factories ──
    def mk_url():
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1)

    def mk_redir():
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=4, random_state=42)

    def mk_net():
        return RandomForestClassifier(
            n_estimators=200, class_weight="balanced",
            random_state=42, n_jobs=-1)

    def mk_content():
        return LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42)

    def mk_stack():
        return StackingClassifier(
            estimators=[
                ("rf", RandomForestClassifier(
                    n_estimators=100, class_weight="balanced",
                    random_state=42, n_jobs=-1)),
                ("gbm", GradientBoostingClassifier(
                    n_estimators=100, max_depth=4, random_state=42)),
                ("lr", LogisticRegression(
                    max_iter=2000, class_weight="balanced", random_state=42)),
            ],
            final_estimator=LogisticRegression(
                max_iter=2000, random_state=42),
            cv=3, n_jobs=-1)

    # (name, feature matrix, benign pool, factory)
    baselines = [
        ("URLNet-RF",         X_url_bl,     pool_url,     mk_url),
        ("RedirectChain-GBM", X_redir_bl,   pool_redir,   mk_redir),
        ("NetTraffic-RF",     X_net_bl,     pool_net,     mk_net),
        ("ContentHeur-LR",    X_content_bl, pool_content, mk_content),
        ("StackEnsemble",     X_stack_bl,   pool_stack,   mk_stack),
    ]

    all_results: dict = {}

    # ==================================================================
    # PHASE 1 — Clean Performance
    # ==================================================================
    print("\n" + "=" * 72)
    print("[3/4] PHASE 1: Clean Performance")
    print("=" * 72)

    phase1: dict = {}
    for name, X, _, clf_fn in baselines:
        yp, ypr = evaluate_simple(
            name, clf_fn, X[idx_tr], y_tr, X[idx_te], y_te)
        m = compute_metrics(y_te, yp, ypr)
        phase1[name] = m
        print(f"  {name:28s}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}")

    # SMVE (equal weights)
    views_tr = [v[idx_tr] for v in view_arrays]
    views_te = [v[idx_te] for v in view_arrays]

    smve = SMVEVoting([1.0] * 6)
    smve.fit(views_tr, y_tr)
    y_smve_c, ypr_smve_c = smve.predict(views_te)
    m_smve_c = compute_metrics(y_te, y_smve_c, ypr_smve_c)
    phase1["PhishTrace-SMVE"] = m_smve_c
    print(f"  {'PhishTrace-SMVE':28s}  F1={m_smve_c['f1']:.4f}  "
          f"AUC={m_smve_c['auc']:.4f}")

    all_results["clean"] = phase1

    # ==================================================================
    # PHASE 2 — Targeted Feature Replacement
    #   Each baseline attacked by replacing ITS OWN features with benign.
    #   Demonstrates: every single-view method is vulnerable.
    # ==================================================================
    print("\n" + "=" * 72)
    print("PHASE 2: Targeted Feature Replacement")
    print("  Each method's OWN features replaced with benign samples")
    print("=" * 72)

    phase2: dict = {}
    for name, X, pool, clf_fn in baselines:
        Xtr, Xte = X[idx_tr], X[idx_te]
        cols = np.arange(X.shape[1])
        Xte_a = benign_replacement_attack(Xte, y_te, cols, pool)

        yp_c, ypr_c = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte, y_te)
        yp_a, ypr_a = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte_a, y_te)

        mc = compute_metrics(y_te, yp_c, ypr_c)
        ma = compute_metrics(y_te, yp_a, ypr_a)
        ev = compute_evasion(y_te, yp_c, yp_a)

        phase2[name] = {
            "f1_clean": mc["f1"], "f1_attacked": ma["f1"],
            "auc_clean": mc["auc"], "auc_attacked": ma["auc"],
            "evasion_rate": ev, "f1_drop": mc["f1"] - ma["f1"],
        }
        mk = " <<<" if ev > 0.05 else ""
        print(f"  {name:28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
              f"Evade={ev:.1%}{mk}")

    all_results["targeted"] = phase2

    # ==================================================================
    # PHASE 3 — URL-Only Camouflage  (cross-system comparison)
    #   Only URL dims perturbed.  Non-URL baselines are unaffected.
    #   Key contrast: SMVE survives via per-view voting.
    # ==================================================================
    print("\n" + "=" * 72)
    print("PHASE 3: URL-Only Camouflage")
    print("=" * 72)

    phase3: dict = {}

    # Single-view baselines
    url_affected = {"URLNet-RF"}
    for name, X, pool, clf_fn in baselines[:4]:
        Xtr, Xte = X[idx_tr], X[idx_te]
        if name in url_affected:
            Xte_a = benign_replacement_attack(
                Xte, y_te, np.arange(X.shape[1]), pool)
        else:
            Xte_a = Xte  # feature space has no URL dims

        yp_c, ypr_c = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte, y_te)
        yp_a, ypr_a = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte_a, y_te)
        mc = compute_metrics(y_te, yp_c, ypr_c)
        ma = compute_metrics(y_te, yp_a, ypr_a)
        ev = compute_evasion(y_te, yp_c, yp_a)

        phase3[name] = {"f1_clean": mc["f1"], "f1_attacked": ma["f1"],
                        "evasion_rate": ev}
        note = "" if name in url_affected else " (no URL features)"
        print(f"  {name:28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
              f"Evade={ev:.1%}{note}")

    # StackEnsemble — URL columns attacked (first url_dim cols)
    Xtr_s, Xte_s = X_stack_bl[idx_tr], X_stack_bl[idx_te]
    url_cols = np.arange(X_url_bl.shape[1])
    Xte_s_a = benign_replacement_attack(Xte_s, y_te, url_cols, pool_url)
    yp_c, ypr_c = evaluate_simple("Stack", mk_stack, Xtr_s, y_tr, Xte_s, y_te)
    yp_a, ypr_a = evaluate_simple("Stack", mk_stack, Xtr_s, y_tr, Xte_s_a, y_te)
    mc = compute_metrics(y_te, yp_c, ypr_c)
    ma = compute_metrics(y_te, yp_a, ypr_a)
    ev = compute_evasion(y_te, yp_c, yp_a)
    phase3["StackEnsemble"] = {"f1_clean": mc["f1"], "f1_attacked": ma["f1"],
                               "evasion_rate": ev}
    print(f"  {'StackEnsemble':28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
          f"Evade={ev:.1%}")

    # PhishTrace-SMVE — only URL view corrupted (1/6)
    vte_a = [benign_replacement_attack(
        views_te[0], y_te,
        np.arange(view_arrays[0].shape[1]), smve_pools[0]
    )] + views_te[1:]
    yp_a, ypr_a = smve.predict(vte_a)
    ma = compute_metrics(y_te, yp_a, ypr_a)
    ev = compute_evasion(y_te, y_smve_c, yp_a)
    phase3["PhishTrace-SMVE"] = {
        "f1_clean": m_smve_c["f1"], "f1_attacked": ma["f1"],
        "evasion_rate": ev}
    print(f"  {'PhishTrace-SMVE':28s}  F1: {m_smve_c['f1']:.4f} -> "
          f"{ma['f1']:.4f}  Evade={ev:.1%}")

    all_results["url_camouflage"] = phase3

    # ==================================================================
    # PHASE 3b — Partial URL Camouflage (50%)
    #   Blend 50% benign URL features into phishing samples.
    # ==================================================================
    print("\n" + "=" * 72)
    print("PHASE 3b: Partial URL Camouflage (50%)")
    print("=" * 72)

    phase3b: dict = {}
    # URLNet-RF under 50% partial camouflage
    for name, X, pool, clf_fn in baselines[:4]:
        Xtr, Xte = X[idx_tr], X[idx_te]
        if name == "URLNet-RF":
            Xte_a = benign_replacement_attack(
                Xte, y_te, np.arange(X.shape[1]), pool, strength=0.5)
        else:
            Xte_a = Xte
        yp_c, ypr_c = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte, y_te)
        yp_a, ypr_a = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte_a, y_te)
        mc = compute_metrics(y_te, yp_c, ypr_c)
        ma = compute_metrics(y_te, yp_a, ypr_a)
        ev = compute_evasion(y_te, yp_c, yp_a)
        phase3b[name] = {"f1_clean": mc["f1"], "f1_attacked": ma["f1"],
                         "evasion_rate": ev}
        print(f"  {name:28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
              f"Evade={ev:.1%}")

    # StackEnsemble under 50% partial
    Xte_s_a = benign_replacement_attack(Xte_s, y_te, url_cols, pool_url, strength=0.5)
    yp_c, ypr_c = evaluate_simple("Stack", mk_stack, Xtr_s, y_tr, Xte_s, y_te)
    yp_a, ypr_a = evaluate_simple("Stack", mk_stack, Xtr_s, y_tr, Xte_s_a, y_te)
    mc = compute_metrics(y_te, yp_c, ypr_c)
    ma = compute_metrics(y_te, yp_a, ypr_a)
    ev = compute_evasion(y_te, yp_c, yp_a)
    phase3b["StackEnsemble"] = {"f1_clean": mc["f1"], "f1_attacked": ma["f1"],
                                "evasion_rate": ev}
    print(f"  {'StackEnsemble':28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
          f"Evade={ev:.1%}")

    # SMVE under 50% partial URL camouflage
    vte_a = [benign_replacement_attack(
        views_te[0], y_te,
        np.arange(view_arrays[0].shape[1]), smve_pools[0], strength=0.5
    )] + views_te[1:]
    yp_a, ypr_a = smve.predict(vte_a)
    ma = compute_metrics(y_te, yp_a, ypr_a)
    ev = compute_evasion(y_te, y_smve_c, yp_a)
    phase3b["PhishTrace-SMVE"] = {
        "f1_clean": m_smve_c["f1"], "f1_attacked": ma["f1"],
        "evasion_rate": ev}
    print(f"  {'PhishTrace-SMVE':28s}  F1: {m_smve_c['f1']:.4f} -> "
          f"{ma['f1']:.4f}  Evade={ev:.1%}")

    all_results["partial_camouflage"] = phase3b

    # ==================================================================
    # PHASE 3c — Mean Feature Substitution
    #   Replace all URL features of phishing samples with benign mean.
    # ==================================================================
    print("\n" + "=" * 72)
    print("PHASE 3c: Mean Feature Substitution (URL)")
    print("=" * 72)

    phase3c: dict = {}
    # Compute benign mean URL features
    benign_url_mean = X_url_bl[y == 0].mean(axis=0)

    for name, X, pool, clf_fn in baselines[:4]:
        Xtr, Xte = X[idx_tr], X[idx_te]
        if name == "URLNet-RF":
            Xte_a = mean_substitution_attack(
                Xte, y_te, np.arange(X.shape[1]), X[y == 0].mean(axis=0))
        else:
            Xte_a = Xte
        yp_c, ypr_c = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte, y_te)
        yp_a, ypr_a = evaluate_simple(name, clf_fn, Xtr, y_tr, Xte_a, y_te)
        mc = compute_metrics(y_te, yp_c, ypr_c)
        ma = compute_metrics(y_te, yp_a, ypr_a)
        ev = compute_evasion(y_te, yp_c, yp_a)
        phase3c[name] = {"f1_clean": mc["f1"], "f1_attacked": ma["f1"],
                         "evasion_rate": ev}
        print(f"  {name:28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
              f"Evade={ev:.1%}")

    # StackEnsemble under mean substitution
    Xte_s_a = mean_substitution_attack(Xte_s, y_te, url_cols, benign_url_mean)
    yp_c, ypr_c = evaluate_simple("Stack", mk_stack, Xtr_s, y_tr, Xte_s, y_te)
    yp_a, ypr_a = evaluate_simple("Stack", mk_stack, Xtr_s, y_tr, Xte_s_a, y_te)
    mc = compute_metrics(y_te, yp_c, ypr_c)
    ma = compute_metrics(y_te, yp_a, ypr_a)
    ev = compute_evasion(y_te, yp_c, yp_a)
    phase3c["StackEnsemble"] = {"f1_clean": mc["f1"], "f1_attacked": ma["f1"],
                                "evasion_rate": ev}
    print(f"  {'StackEnsemble':28s}  F1: {mc['f1']:.4f} -> {ma['f1']:.4f}  "
          f"Evade={ev:.1%}")

    # SMVE under mean substitution of URL view
    url_view_mean = view_arrays[0][y == 0].mean(axis=0)
    vte_a_mean = [mean_substitution_attack(
        views_te[0], y_te,
        np.arange(view_arrays[0].shape[1]), url_view_mean
    )] + views_te[1:]
    yp_a, ypr_a = smve.predict(vte_a_mean)
    ma = compute_metrics(y_te, yp_a, ypr_a)
    ev = compute_evasion(y_te, y_smve_c, yp_a)
    phase3c["PhishTrace-SMVE"] = {
        "f1_clean": m_smve_c["f1"], "f1_attacked": ma["f1"],
        "evasion_rate": ev}
    print(f"  {'PhishTrace-SMVE':28s}  F1: {m_smve_c['f1']:.4f} -> "
          f"{ma['f1']:.4f}  Evade={ev:.1%}")

    all_results["mean_substitution"] = phase3c

    # ==================================================================
    # PHASE 4 — Progressive View Corruption  (SMVE)
    #   Corrupt views one-by-one: URL -> Network -> Redirect -> ...
    #   Shows: SMVE needs >50% views corrupted for significant evasion.
    # ==================================================================
    print("\n" + "=" * 72)
    print(f"[4/4] PHASE 4: SMVE Progressive View Corruption")
    print(f"  Order: {' -> '.join(view_names)}")
    print("=" * 72)

    phase4: dict = {}
    for nc in range(len(view_names) + 1):
        corrupted = view_names[:nc]
        vte_a = []
        for vi in range(6):
            if vi < nc:
                vte_a.append(benign_replacement_attack(
                    views_te[vi], y_te,
                    np.arange(view_arrays[vi].shape[1]),
                    smve_pools[vi]))
            else:
                vte_a.append(views_te[vi])

        yp_a, ypr_a = smve.predict(vte_a)
        ma = compute_metrics(y_te, yp_a, ypr_a)
        ev = compute_evasion(y_te, y_smve_c, yp_a)

        phase4[f"{nc}_views"] = {
            "n_corrupted": nc,
            "corrupted_views": corrupted,
            "f1": ma["f1"], "auc": ma["auc"],
            "evasion_rate": ev,
        }
        lbl = f"{nc}/6"
        if corrupted:
            lbl += f" ({', '.join(corrupted)})"
        else:
            lbl += " (clean)"
        mk = " <<<" if ev > 0.10 else ""
        print(f"  {lbl:50s} F1={ma['f1']:.4f}  Evade={ev:.1%}{mk}")

    all_results["progressive"] = phase4

    # ==================================================================
    # Summary
    # ==================================================================
    print("\n" + "=" * 72)
    print("KEY FINDINGS")
    print("=" * 72)

    print("\n  Phase 2 — ALL single-view baselines breakable:")
    for name in ["URLNet-RF", "RedirectChain-GBM",
                 "NetTraffic-RF", "ContentHeur-LR"]:
        r = phase2.get(name, {})
        print(f"    {name:28s}  Evasion: {r.get('evasion_rate', 0):.1%}")
    if "StackEnsemble" in phase2:
        r = phase2["StackEnsemble"]
        print(f"    {'StackEnsemble':28s}  Evasion: {r.get('evasion_rate', 0):.1%}")

    print(f"\n  Phase 3 — URL-only camouflage => SMVE survives:")
    for name in ["URLNet-RF", "StackEnsemble", "PhishTrace-SMVE"]:
        r = phase3.get(name, {})
        print(f"    {name:28s}  Evasion: {r.get('evasion_rate', 0):.1%}")

    print(f"\n  Phase 4 — SMVE progressive degradation:")
    for nc in range(len(view_names) + 1):
        r = phase4.get(f"{nc}_views", {})
        ev = r.get("evasion_rate", 0)
        f1 = r.get("f1", 0)
        print(f"    {nc}/6 views corrupted:  F1={f1:.4f}  Evasion={ev:.1%}")

    # ==================================================================
    # Save
    # ==================================================================
    def _round_dict(d):
        out = {}
        for k, v in d.items():
            if isinstance(v, float):
                out[k] = round(v, 4)
            elif isinstance(v, list):
                out[k] = [round(x, 4) if isinstance(x, float) else x
                          for x in v]
            else:
                out[k] = v
        return out

    output = {
        "dataset": {"total": len(traces), "phishing": n_phish,
                     "benign": n_benign},
        "results": {
            phase: {k: _round_dict(v) if isinstance(v, dict) else v
                    for k, v in data.items()}
            for phase, data in all_results.items()
        },
    }

    out_path = PROJECT_ROOT / "experiments" / "results" / "url_adversarial_v3.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
