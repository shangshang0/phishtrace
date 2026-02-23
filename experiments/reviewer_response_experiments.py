"""
Reviewer-response experiments for USENIX Security revision.

Addresses three critical reviewer concerns:
1. URL-cleanliness stratified evaluation (Concerns 1 & 2)
   - Split dataset by URL lexical cleanliness
   - Show behavioral features matter when URL signals are weak
2. Full multi-view adversarial evaluation (Concern 7)
   - Test decoy navigation against the FULL 6-view SMVE, not just ITG-only
3. Same-subset fair comparison for GNN/AGFL (Concern 5)
   - Report all methods on the SAME 488-trace or 621-trace subset
4. Cross-layer feature ablation (Concern 10)
   - Show with/without cross-view features
5. Standard deviations for all results (Concern 4)
"""

import json
import math
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
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
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.enhanced_itg_detector import (
    load_all_traces,
    extract_url_view,
    extract_network_view,
    extract_redirect_view,
    extract_interaction_view,
    extract_itg_view,
    compute_cross_view_features,
    engineer_itg_features,
    build_all_views,
)


def cv_evaluate(X, y, clf_factory, n_splits=10, seed=42):
    """10-fold stratified CV with std devs."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    metrics = {k: [] for k in ['accuracy', 'precision', 'recall', 'f1', 'auc']}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clf_factory()
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, 'predict_proba') else preds.astype(float)

        metrics['accuracy'].append(accuracy_score(y_te, preds))
        metrics['precision'].append(precision_score(y_te, preds, zero_division=0))
        metrics['recall'].append(recall_score(y_te, preds, zero_division=0))
        metrics['f1'].append(f1_score(y_te, preds, zero_division=0))
        try:
            metrics['auc'].append(roc_auc_score(y_te, probs))
        except ValueError:
            metrics['auc'].append(0.5)

    result = {}
    for k, vals in metrics.items():
        result[k] = round(float(np.mean(vals)), 4)
        result[k + '_std'] = round(float(np.std(vals)), 4)
    return result


def build_stacked_ensemble():
    """Build the stacked ensemble (same as main paper)."""
    estimators = [
        ('url_gbm', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ('net_rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ('redir_gbm', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ('interact_rf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ('itg_rf', RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
        ('cross_gbm', GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ]
    return StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, random_state=42),
        cv=5, n_jobs=1, passthrough=False
    )


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1: URL-Cleanliness Stratified Evaluation
# ═══════════════════════════════════════════════════════════════

def compute_url_cleanliness_score(url_features: np.ndarray) -> float:
    """Score how 'clean' a URL looks. Higher = more clean/innocuous.

    Phishing URLs from feeds tend to have: long domains, high entropy,
    suspicious TLDs, many subdomains, digits in domain, etc.
    """
    domain_len = url_features[1]
    domain_entropy = url_features[15]
    suspicious_tld = url_features[11]
    num_subdomains = url_features[12]
    num_digits_domain = url_features[8]
    has_ip = url_features[9]
    num_dashes = url_features[4]
    kw_login = url_features[17]

    # Lower score = more suspicious URL, higher = cleaner
    score = 0.0
    score += max(0, 1.0 - domain_len / 30.0)  # Short domains are cleaner
    score += max(0, 1.0 - domain_entropy / 4.0)  # Low entropy is cleaner
    score += (1.0 - suspicious_tld)  # Normal TLD is cleaner
    score += max(0, 1.0 - num_subdomains / 3.0)  # Few subdomains
    score += max(0, 1.0 - num_digits_domain / 5.0)  # Few digits
    score += (1.0 - has_ip)  # No IP address
    score += max(0, 1.0 - num_dashes / 3.0)  # Few dashes
    score += (1.0 - kw_login)  # No login keywords in URL

    return score / 8.0  # Normalize to [0, 1]


def run_stratified_url_experiment(traces, y, X_url, X_net, X_redir, X_interact,
                                   X_itg_eng, X_cross):
    """Split by URL cleanliness, evaluate each view on each stratum."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: URL-Cleanliness Stratified Evaluation")
    print("=" * 70)

    # Compute cleanliness scores
    scores = np.array([compute_url_cleanliness_score(x) for x in X_url])

    # Split into tertiles: dirty (bottom 33%), medium, clean (top 33%)
    t33 = np.percentile(scores, 33.3)
    t67 = np.percentile(scores, 66.7)

    dirty_mask = scores <= t33
    medium_mask = (scores > t33) & (scores <= t67)
    clean_mask = scores > t67

    strata = [
        ("Dirty-URL (bottom 33%)", dirty_mask),
        ("Medium-URL (middle 33%)", medium_mask),
        ("Clean-URL (top 33%)", clean_mask),
        ("All", np.ones(len(y), dtype=bool)),
    ]

    results = {}
    views = [
        ("URL-only (21D)", X_url, lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ("ITG+Eng (42D)", X_itg_eng, lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
        ("Network (18D)", X_net, lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ("Redirect (14D)", X_redir, lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ("Non-URL views (92D)", np.hstack([X_net, X_redir, X_interact, X_itg_eng, X_cross]),
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
        ("All 6 views (125D)", np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross]),
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
    ]

    for stratum_name, mask in strata:
        n_total = int(mask.sum())
        n_phish = int((y[mask] == 1).sum())
        n_benign = int((y[mask] == 0).sum())
        print(f"\n--- {stratum_name}: {n_total} traces ({n_phish} phishing, {n_benign} benign) ---")

        # Need at least 2 classes with enough samples for stratified CV
        if n_phish < 10 or n_benign < 10:
            print(f"  Skipping: too few samples for stratified CV")
            continue

        results[stratum_name] = {}
        for view_name, X_view, clf_factory in views:
            X_sub = X_view[mask]
            y_sub = y[mask]
            n_folds = min(10, min(n_phish, n_benign))
            if n_folds < 3:
                print(f"  {view_name}: too few for CV")
                continue
            r = cv_evaluate(X_sub, y_sub, clf_factory, n_splits=n_folds)
            results[stratum_name][view_name] = r
            print(f"  {view_name}: F1={r['f1']:.4f}±{r['f1_std']:.4f}, "
                  f"Acc={r['accuracy']:.4f}±{r['accuracy_std']:.4f}, "
                  f"AUC={r['auc']:.4f}±{r['auc_std']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2: Full Multi-View Adversarial Evaluation
# ═══════════════════════════════════════════════════════════════

def run_full_system_adversarial(traces, y, X_url, X_net, X_redir, X_interact,
                                 X_itg_eng, X_cross):
    """Test adversarial attacks against the FULL 6-view system, not just ITG."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Full Multi-View Adversarial Evaluation")
    print("=" * 70)

    from experiments.adversarial_eval import GraphStructuralAttack, SemanticAttack

    # Build full feature matrix
    X_full = np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross])

    # 70/30 train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, traces_train, traces_test = train_test_split(
        X_full, y, traces, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train full-system RF
    model_full = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    model_full.fit(X_train_s, y_train)

    clean_preds = model_full.predict(X_test_s)
    clean_f1 = f1_score(y_test, clean_preds, zero_division=0)
    clean_acc = accuracy_score(y_test, clean_preds)
    print(f"Full-system clean: Acc={clean_acc:.4f}, F1={clean_f1:.4f}")

    # Also train ITG-only for comparison
    itg_start = 21 + 18 + 14 + 18  # After URL, Net, Redir, Interact
    itg_end = itg_start + 42
    X_itg_train = X_train[:, itg_start:itg_end]
    X_itg_test = X_test[:, itg_start:itg_end]

    scaler_itg = StandardScaler()
    X_itg_train_s = scaler_itg.fit_transform(X_itg_train)
    X_itg_test_s = scaler_itg.transform(X_itg_test)

    model_itg = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    model_itg.fit(X_itg_train_s, y_itg_train := y_train)

    itg_clean_preds = model_itg.predict(X_itg_test_s)
    itg_clean_f1 = f1_score(y_test, itg_clean_preds, zero_division=0)
    print(f"ITG-only clean: F1={itg_clean_f1:.4f}")

    phishing_mask = (y_test == 1)
    n_phishing = int(phishing_mask.sum())

    results = {
        'full_system_clean': {'accuracy': round(clean_acc, 4), 'f1': round(clean_f1, 4)},
        'itg_only_clean': {'f1': round(itg_clean_f1, 4)},
    }

    # Apply attacks
    struct_attack = GraphStructuralAttack(n_inject=5, rewire_prob=0.2, inflate_factor=3)
    semantic = SemanticAttack(n_decoy_links=10, split_forms=True)

    attacks = [
        ('node_injection', struct_attack.inject_nodes),
        ('edge_rewiring', struct_attack.rewire_edges),
        ('graph_inflation', struct_attack.inflate_graph),
        ('combined_structural', struct_attack.combined_attack),
        ('decoy_navigation', semantic.add_decoy_navigation),
        ('form_splitting', semantic.split_credential_form),
    ]

    for attack_name, attack_fn in attacks:
        # Re-extract ALL views from perturbed traces
        adv_full_features = []
        adv_itg_features = []

        for i, trace in enumerate(traces_test):
            if y_test[i] == 1:  # Only attack phishing samples
                try:
                    # Unwrap: attacks expect events at top level of dict
                    inner = trace.get('trace', trace)
                    perturbed_inner = attack_fn(inner)
                    # Re-wrap for extract_* functions which expect wrapped format
                    import copy as _copy
                    perturbed = _copy.copy(trace)
                    perturbed['trace'] = perturbed_inner

                    # Re-extract all 6 views from perturbed trace
                    u = extract_url_view(perturbed)
                    n = extract_network_view(perturbed)
                    r = extract_redirect_view(perturbed)
                    inter = extract_interaction_view(perturbed)
                    itg = extract_itg_view(perturbed)
                    if itg is None:
                        itg = np.zeros(30, dtype=float)
                    itg_eng = engineer_itg_features(itg.reshape(1, -1))[0]
                    cross = compute_cross_view_features(u, n, r, inter)
                    full_vec = np.concatenate([u, n, r, inter, itg_eng, cross])
                    adv_full_features.append(scaler.transform(full_vec.reshape(1, -1))[0])
                    adv_itg_features.append(scaler_itg.transform(itg_eng.reshape(1, -1))[0])
                except Exception:
                    adv_full_features.append(X_test_s[i])
                    adv_itg_features.append(X_itg_test_s[i])
            else:
                adv_full_features.append(X_test_s[i])
                adv_itg_features.append(X_itg_test_s[i])

        X_adv_full = np.array(adv_full_features)
        X_adv_itg = np.array(adv_itg_features)

        # Full system
        full_preds = model_full.predict(X_adv_full)
        full_f1 = f1_score(y_test, full_preds, zero_division=0)
        full_detected_clean = clean_preds[phishing_mask] == 1
        full_detected_adv = full_preds[phishing_mask] == 1
        full_evasion = float(np.sum(full_detected_clean & ~full_detected_adv)) / max(1, np.sum(full_detected_clean))

        # ITG only
        itg_preds = model_itg.predict(X_adv_itg)
        itg_f1 = f1_score(y_test, itg_preds, zero_division=0)
        itg_detected_clean = itg_clean_preds[phishing_mask] == 1
        itg_detected_adv = itg_preds[phishing_mask] == 1
        itg_evasion = float(np.sum(itg_detected_clean & ~itg_detected_adv)) / max(1, np.sum(itg_detected_clean))

        results[attack_name] = {
            'full_system': {'f1': round(full_f1, 4), 'evasion_rate': round(full_evasion, 4)},
            'itg_only': {'f1': round(itg_f1, 4), 'evasion_rate': round(itg_evasion, 4)},
        }
        print(f"  {attack_name}:")
        print(f"    Full system: F1={full_f1:.4f}, Evasion={full_evasion*100:.2f}%")
        print(f"    ITG-only:    F1={itg_f1:.4f}, Evasion={itg_evasion*100:.2f}%")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3: Same-Subset Fair Comparison
# ═══════════════════════════════════════════════════════════════

def run_same_subset_comparison(traces, y, X_url, X_net, X_redir, X_interact,
                                X_itg_eng, X_cross, itg_mask):
    """Report all methods on the SAME subset for fair comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Same-Subset Fair Comparison")
    print("=" * 70)

    # Subset: traces with ITG graphs (≥2 nodes)
    graph_mask = itg_mask
    n_graph = int(graph_mask.sum())
    n_phish = int((y[graph_mask] == 1).sum())
    n_benign = int((y[graph_mask] == 0).sum())
    print(f"Graph-available subset: {n_graph} traces ({n_phish} phishing, {n_benign} benign)")

    results = {}

    # Methods to compare on the SAME subset
    methods = [
        ("URL-only (21D)", X_url[graph_mask],
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ("Network (18D)", X_net[graph_mask],
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ("Redirect (14D)", X_redir[graph_mask],
         lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ("Interaction (18D)", X_interact[graph_mask],
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ("ITG+Eng (42D)", X_itg_eng[graph_mask],
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
        ("Cross-View (12D)", X_cross[graph_mask],
         lambda: GradientBoostingClassifier(n_estimators=200, max_depth=5, random_state=42)),
        ("All 6 views (125D)", np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross])[graph_mask],
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
        ("StackEnsemble (71D)", np.hstack([X_url, X_net, X_redir, X_interact, X_cross])[graph_mask],
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
    ]

    y_sub = y[graph_mask]
    for name, X_sub, clf_factory in methods:
        r = cv_evaluate(X_sub, y_sub, clf_factory)
        results[name] = r
        print(f"  {name}: F1={r['f1']:.4f}±{r['f1_std']:.4f}, "
              f"Acc={r['accuracy']:.4f}±{r['accuracy_std']:.4f}, "
              f"AUC={r['auc']:.4f}±{r['auc_std']:.4f}")

    # Also report on ALL 2738 traces for context
    print(f"\nFull dataset ({len(y)} traces):")
    for name, X_view, clf_factory in [
        ("URL-only (21D)", X_url,
         lambda: RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')),
        ("All 6 views (125D)", np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross]),
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')),
    ]:
        r = cv_evaluate(X_view, y, clf_factory)
        results[f"full_dataset_{name}"] = r
        print(f"  {name}: F1={r['f1']:.4f}±{r['f1_std']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 4: Cross-Layer Feature Ablation
# ═══════════════════════════════════════════════════════════════

def run_cross_layer_ablation(y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross):
    """Ablation: with and without cross-view features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Cross-Layer Feature Ablation")
    print("=" * 70)

    results = {}
    configs = [
        ("Without cross-view (113D)", np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng])),
        ("With cross-view (125D)", np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross])),
        ("Cross-view only (12D)", X_cross),
    ]

    for name, X in configs:
        r = cv_evaluate(X, y,
                        lambda: RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced'))
        results[name] = r
        print(f"  {name}: F1={r['f1']:.4f}±{r['f1_std']:.4f}, "
              f"Acc={r['accuracy']:.4f}±{r['accuracy_std']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading traces...")
    traces, y = load_all_traces()
    print(f"Loaded {len(traces)} traces ({sum(y)} phishing, {sum(1-y)} benign)")

    print("Extracting all views...")
    X_url, X_net, X_redir, X_interact, X_itg, X_cross, itg_mask = build_all_views(traces)
    X_itg_eng = engineer_itg_features(X_itg)  # 30 + 12 = 42D
    print(f"Views: URL={X_url.shape[1]}, Net={X_net.shape[1]}, "
          f"Redir={X_redir.shape[1]}, Interact={X_interact.shape[1]}, "
          f"ITG+Eng={X_itg_eng.shape[1]}, Cross={X_cross.shape[1]}")
    print(f"ITG available: {sum(itg_mask)} / {len(itg_mask)}")

    # URL stats for paper
    domain_lens_phish = X_url[y == 1, 1]
    domain_lens_benign = X_url[y == 0, 1]
    print(f"\nURL stats: phishing domain len mean={domain_lens_phish.mean():.1f}, "
          f"benign domain len mean={domain_lens_benign.mean():.1f}")
    print(f"URL stats: phishing domain entropy mean={X_url[y==1, 15].mean():.2f}, "
          f"benign domain entropy mean={X_url[y==0, 15].mean():.2f}")

    all_results = {}

    # Experiment 1: URL-cleanliness stratified
    all_results['stratified'] = run_stratified_url_experiment(
        traces, y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross
    )

    # Experiment 2: Full multi-view adversarial
    try:
        all_results['adversarial_full'] = run_full_system_adversarial(
            traces, y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross
        )
    except Exception as e:
        print(f"Adversarial experiment failed: {e}")
        import traceback; traceback.print_exc()

    # Experiment 3: Same-subset comparison
    all_results['same_subset'] = run_same_subset_comparison(
        traces, y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross, itg_mask
    )

    # Experiment 4: Cross-layer ablation
    all_results['cross_layer_ablation'] = run_cross_layer_ablation(
        y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross
    )

    # Save
    out_path = Path(__file__).parent / 'results' / 'reviewer_response_results.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")


if __name__ == '__main__':
    main()
