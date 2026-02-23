"""
Behavioral Invariance Test + Full-System Decoy Navigation Test

Addresses two critical reviewer concerns:
1. INVARIANCE: Evaluate behavioral (non-URL) features specifically on
   the "URL-clean" subset to prove they maintain detection power when
   URL signals are uninformative (i.e., phisher uses clean domains).
2. DECOY NAVIGATION: Test full 6-view SMVE against decoy navigation
   attacks (not just ITG-only as in §7.6.3).

Results are saved to experiments/results/invariance_decoy_results.json
"""

import json
import math
import sys
import copy
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
from sklearn.model_selection import StratifiedKFold, train_test_split
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
    minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    actual_splits = max(2, min(n_splits, minority))
    skf = StratifiedKFold(n_splits=actual_splits, shuffle=True, random_state=seed)
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
        try:
            probs = clf.predict_proba(X_te)[:, 1]
        except (AttributeError, IndexError):
            probs = preds.astype(float)

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


def compute_url_cleanliness_score(url_features: np.ndarray) -> float:
    """Score how 'clean' a URL looks. Higher = more clean/innocuous."""
    domain_len = url_features[1]
    domain_entropy = url_features[15]
    suspicious_tld = url_features[11]
    num_subdomains = url_features[12]
    num_digits_domain = url_features[8]
    has_ip = url_features[9]
    num_dashes = url_features[4]
    kw_login = url_features[17]

    score = 0.0
    score += max(0, 1.0 - domain_len / 30.0)
    score += max(0, 1.0 - domain_entropy / 4.0)
    score += (1.0 - suspicious_tld)
    score += max(0, 1.0 - num_subdomains / 3.0)
    score += max(0, 1.0 - num_digits_domain / 5.0)
    score += (1.0 - has_ip)
    score += max(0, 1.0 - num_dashes / 3.0)
    score += (1.0 - kw_login)
    return score / 8.0


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT A: Behavioral Invariance Across URL-Cleanliness Strata
# ═══════════════════════════════════════════════════════════════

def run_behavioral_invariance(y, X_url, X_net, X_redir, X_interact,
                               X_itg_eng, X_cross):
    """
    Core invariance test: evaluate each INDIVIDUAL behavioral view
    on Dirty/Medium/Clean URL strata. Also evaluate a "truly non-URL"
    bundle (80D: net+redir+interact+ITG, excluding cross-view that
    contains URL-derived signals).

    The key claim to validate: behavioral features maintain detection
    power even when URL is clean (i.e., the phisher used a good domain).
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Behavioral Invariance Across URL-Cleanliness Strata")
    print("=" * 70)

    scores = np.array([compute_url_cleanliness_score(x) for x in X_url])
    t33 = np.percentile(scores, 33.3)
    t67 = np.percentile(scores, 66.7)

    strata = {
        "Dirty-URL": scores <= t33,
        "Medium-URL": (scores > t33) & (scores <= t67),
        "Clean-URL": scores > t67,
        "All": np.ones(len(y), dtype=bool),
    }

    # Truly non-URL: 80D = Network(18) + Redirect(14) + Interaction(18) + ITG+Eng(30)
    # Note: ITG+Eng is 42D but 12D are engineered features from the ITG itself,
    # none depend on URL. Cross-view (12D) is EXCLUDED because it uses URL features.
    X_truly_nonurl = np.hstack([X_net, X_redir, X_interact, X_itg_eng])
    # Also "non-URL 92D" which includes cross-view for comparison
    X_nonurl_92d = np.hstack([X_net, X_redir, X_interact, X_itg_eng, X_cross])

    rf = lambda: RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight='balanced')
    gbm = lambda: GradientBoostingClassifier(
        n_estimators=200, max_depth=5, random_state=42)

    views = {
        "Network (18D)": (X_net, rf),
        "Redirect (14D)": (X_redir, gbm),
        "Interaction (18D)": (X_interact, rf),
        "ITG+Eng (42D)": (X_itg_eng, rf),
        "Truly-non-URL (92D)": (X_truly_nonurl, rf),
        "Non-URL+CrossView (104D)": (X_nonurl_92d, rf),
        "URL-only (21D)": (X_url, rf),
        "All 6 views (125D)": (np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross]), rf),
    }

    results = {}
    for stratum_name, mask in strata.items():
        y_sub = y[mask]
        n_total = int(mask.sum())
        n_phish = int((y_sub == 1).sum())
        n_benign = int((y_sub == 0).sum())
        print(f"\n--- {stratum_name}: {n_total} traces "
              f"({n_phish} phishing, {n_benign} benign) ---")

        if n_phish < 10 or n_benign < 10:
            print(f"  Skipping: insufficient samples")
            continue

        results[stratum_name] = {"n_total": n_total, "n_phish": n_phish,
                                  "n_benign": n_benign}

        for view_name, (X_view, clf_factory) in views.items():
            X_sub = X_view[mask]
            r = cv_evaluate(X_sub, y_sub, clf_factory,
                            n_splits=min(10, min(n_phish, n_benign)))
            results[stratum_name][view_name] = r
            print(f"  {view_name}: "
                  f"F1={r['f1']:.4f}±{r['f1_std']:.4f}  "
                  f"AUC={r['auc']:.4f}±{r['auc_std']:.4f}")

    # Compute invariance metric: ratio of Clean-URL F1 to All F1 for each view
    print("\n--- Invariance Summary (Clean-URL F1 / All F1) ---")
    invariance_ratios = {}
    if "Clean-URL" in results and "All" in results:
        for view_name in views:
            if view_name in results["Clean-URL"] and view_name in results["All"]:
                clean_f1 = results["Clean-URL"][view_name]["f1"]
                all_f1 = results["All"][view_name]["f1"]
                ratio = clean_f1 / max(all_f1, 1e-6)
                invariance_ratios[view_name] = round(ratio, 4)
                print(f"  {view_name}: {ratio:.4f} "
                      f"(Clean={clean_f1:.4f}, All={all_f1:.4f})")

    results["invariance_ratios"] = invariance_ratios
    return results


# ═══════════════════════════════════════════════════════════════
# EXPERIMENT B: Full-System Decoy Navigation Test
# ═══════════════════════════════════════════════════════════════

def _add_decoy_navigation(trace_data: dict, n_decoy=5) -> dict:
    """Add decoy navigation events to a trace to simulate benign-looking
    browsing behavior injected by the phishing page.

    This is a realistic attack where the phishing page loads additional
    benign-looking content (news articles, help pages, etc.) to dilute
    the behavioral signal.
    """
    result = copy.deepcopy(trace_data)
    inner = result.get("trace", result)
    events = inner.get("events", [])
    net_reqs = inner.get("network_requests", [])
    redirects = inner.get("redirects", [])

    # Inject decoy link clicks and navigation events
    decoy_domains = [
        "help.example.com", "support.microsoft.com", "news.google.com",
        "en.wikipedia.org", "docs.github.com", "faq.example.org",
        "blog.cloudflare.com", "status.aws.amazon.com",
    ]

    base_ts = max((e.get("timestamp", 0) for e in events), default=1000)

    for i in range(n_decoy):
        domain = decoy_domains[i % len(decoy_domains)]
        ts = base_ts + (i + 1) * 500

        # Decoy link detection
        events.append({
            "event_type": "link_detected",
            "url": f"https://{domain}/page{i}",
            "timestamp": ts,
            "element_class": "nav-link",
            "element_id": f"decoy-{i}",
        })
        # Decoy navigation (benign-looking)
        events.append({
            "event_type": "navigation",
            "url": f"https://{domain}/page{i}",
            "timestamp": ts + 100,
        })
        # Decoy network requests (typical of benign pages)
        for res_type in ["document", "script", "stylesheet", "image"]:
            net_reqs.append({
                "url": f"https://{domain}/assets/{res_type}{i}.{res_type[:2]}",
                "resource_type": res_type,
                "status": 200,
                "content_type": "text/html" if res_type == "document" else "application/javascript",
                "timestamp": ts + 200,
            })

    inner["events"] = events
    inner["network_requests"] = net_reqs
    return result


def _add_url_camouflage(trace_data: dict) -> dict:
    """Replace phishing URL with a clean-looking domain.

    Simulates hosting phishing on a compromised legitimate domain.
    """
    result = copy.deepcopy(trace_data)

    clean_domains = [
        "secure-update.microsoft-verify.com",
        "login.accounts-service.net",
        "portal.cloud-services.org",
    ]
    import random
    random.seed(hash(result.get("url", "")) % 2**31)
    new_domain = random.choice(clean_domains)
    new_url = f"https://{new_domain}/auth/verify"

    result["url"] = new_url
    if "trace" in result:
        result["trace"]["final_url"] = new_url
    result["final_url"] = new_url
    return result


def _add_form_splitting(trace_data: dict) -> dict:
    """Split credential form into multi-step: email on page 1, password on page 2.

    Defeats detectors that look for email+password on the same page.
    """
    result = copy.deepcopy(trace_data)
    inner = result.get("trace", result)
    events = inner.get("events", [])

    # Find password-related events and add a navigation step before them
    new_events = []
    inserted_step = False
    for e in events:
        etype = e.get("event_type", "")
        eid = str(e.get("element_id", "")) + str(e.get("element_class", ""))
        form_data = str(e.get("form_data", ""))

        if not inserted_step and "password" in (eid + form_data).lower():
            # Insert a navigation event (multi-step form)
            new_events.append({
                "event_type": "navigation",
                "url": result.get("url", "") + "/step2",
                "timestamp": e.get("timestamp", 0) - 50,
            })
            inserted_step = True

        new_events.append(e)

    inner["events"] = new_events
    return result


def run_full_system_decoy_test(traces, y, X_url, X_net, X_redir,
                                X_interact, X_itg_eng, X_cross):
    """
    Test full 6-view SMVE against three adversarial strategies:
    1. Decoy navigation (inject benign browsing)
    2. URL camouflage (clean-looking domain)
    3. Form splitting (multi-step credential harvesting)
    4. Combined: all three simultaneously
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Full-System Decoy Navigation Test")
    print("=" * 70)

    X_full = np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross])

    # 70/30 split
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_full, y, np.arange(len(y)),
        test_size=0.3, random_state=42, stratify=y
    )

    # Train full-system model
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight='balanced')
    model.fit(X_train_s, y_train)

    clean_preds = model.predict(X_test_s)
    clean_probs = model.predict_proba(X_test_s)[:, 1]
    clean_f1 = f1_score(y_test, clean_preds, zero_division=0)
    clean_acc = accuracy_score(y_test, clean_preds)
    print(f"Clean baseline: Acc={clean_acc:.4f}, F1={clean_f1:.4f}")

    # Also train non-URL-only model for comparison
    # Non-URL features start after URL (21D)
    X_nonurl_train = X_train[:, 21:]  # 104D
    X_nonurl_test = X_test[:, 21:]
    scaler_nonurl = StandardScaler()
    X_nonurl_train_s = scaler_nonurl.fit_transform(X_nonurl_train)
    X_nonurl_test_s = scaler_nonurl.transform(X_nonurl_test)

    model_nonurl = RandomForestClassifier(
        n_estimators=500, random_state=42, class_weight='balanced')
    model_nonurl.fit(X_nonurl_train_s, y_train)

    nonurl_clean_preds = model_nonurl.predict(X_nonurl_test_s)
    nonurl_clean_f1 = f1_score(y_test, nonurl_clean_preds, zero_division=0)
    print(f"Non-URL-only clean: F1={nonurl_clean_f1:.4f}")

    # URL-only model
    X_url_train = X_train[:, :21]
    X_url_test = X_test[:, :21]
    scaler_url = StandardScaler()
    X_url_train_s = scaler_url.fit_transform(X_url_train)
    X_url_test_s = scaler_url.transform(X_url_test)
    model_url = RandomForestClassifier(
        n_estimators=200, random_state=42, class_weight='balanced')
    model_url.fit(X_url_train_s, y_train)
    url_clean_preds = model_url.predict(X_url_test_s)
    url_clean_f1 = f1_score(y_test, url_clean_preds, zero_division=0)
    print(f"URL-only clean: F1={url_clean_f1:.4f}")

    phishing_mask = (y_test == 1)
    n_phishing = int(phishing_mask.sum())

    results = {
        "clean_baseline": {
            "full_system": {"acc": round(clean_acc, 4), "f1": round(clean_f1, 4)},
            "nonurl_only": {"f1": round(nonurl_clean_f1, 4)},
            "url_only": {"f1": round(url_clean_f1, 4)},
            "n_test": len(y_test),
            "n_phishing_test": n_phishing,
        }
    }

    attacks = {
        "decoy_navigation": _add_decoy_navigation,
        "url_camouflage": _add_url_camouflage,
        "form_splitting": _add_form_splitting,
        "combined": lambda t: _add_form_splitting(
            _add_url_camouflage(_add_decoy_navigation(t))),
    }

    for attack_name, attack_fn in attacks.items():
        print(f"\n  Attack: {attack_name}")

        # Re-extract features from perturbed traces
        adv_full, adv_nonurl, adv_url = [], [], []
        n_success = 0
        n_fail = 0

        for i, test_i in enumerate(idx_test):
            if y_test[i] == 1:  # Only attack phishing
                try:
                    perturbed = attack_fn(traces[test_i])
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
                    adv_full.append(scaler.transform(full_vec.reshape(1, -1))[0])
                    nonurl_vec = np.concatenate([n, r, inter, itg_eng, cross])
                    adv_nonurl.append(scaler_nonurl.transform(nonurl_vec.reshape(1, -1))[0])
                    adv_url.append(scaler_url.transform(u.reshape(1, -1))[0])
                    n_success += 1
                except Exception as e:
                    adv_full.append(X_test_s[i])
                    adv_nonurl.append(X_nonurl_test_s[i])
                    adv_url.append(X_url_test_s[i])
                    n_fail += 1
            else:
                adv_full.append(X_test_s[i])
                adv_nonurl.append(X_nonurl_test_s[i])
                adv_url.append(X_url_test_s[i])

        X_adv_full = np.array(adv_full)
        X_adv_nonurl = np.array(adv_nonurl)
        X_adv_url = np.array(adv_url)

        # Evaluate all three models
        full_preds = model.predict(X_adv_full)
        full_f1 = f1_score(y_test, full_preds, zero_division=0)
        full_acc = accuracy_score(y_test, full_preds)

        nonurl_preds = model_nonurl.predict(X_adv_nonurl)
        nonurl_f1 = f1_score(y_test, nonurl_preds, zero_division=0)

        url_preds = model_url.predict(X_adv_url)
        url_f1 = f1_score(y_test, url_preds, zero_division=0)

        # Evasion rate: fraction of phishing detected clean but missed under attack
        full_detected_clean = clean_preds[phishing_mask] == 1
        full_detected_adv = full_preds[phishing_mask] == 1
        full_evasion = float(np.sum(full_detected_clean & ~full_detected_adv)) / max(1, int(full_detected_clean.sum()))

        nonurl_detected_clean = nonurl_clean_preds[phishing_mask] == 1
        nonurl_detected_adv = nonurl_preds[phishing_mask] == 1
        nonurl_evasion = float(np.sum(nonurl_detected_clean & ~nonurl_detected_adv)) / max(1, int(nonurl_detected_clean.sum()))

        url_detected_clean = url_clean_preds[phishing_mask] == 1
        url_detected_adv = url_preds[phishing_mask] == 1
        url_evasion = float(np.sum(url_detected_clean & ~url_detected_adv)) / max(1, int(url_detected_clean.sum()))

        results[attack_name] = {
            "full_system": {
                "f1": round(full_f1, 4), "acc": round(full_acc, 4),
                "evasion_rate": round(full_evasion, 4),
            },
            "nonurl_only": {
                "f1": round(nonurl_f1, 4),
                "evasion_rate": round(nonurl_evasion, 4),
            },
            "url_only": {
                "f1": round(url_f1, 4),
                "evasion_rate": round(url_evasion, 4),
            },
            "n_perturbed": n_success,
            "n_failed": n_fail,
        }

        print(f"    Full system:  F1={full_f1:.4f}, Evasion={full_evasion*100:.1f}%")
        print(f"    Non-URL only: F1={nonurl_f1:.4f}, Evasion={nonurl_evasion*100:.1f}%")
        print(f"    URL-only:     F1={url_f1:.4f}, Evasion={url_evasion*100:.1f}%")
        print(f"    ({n_success} perturbed, {n_fail} fallback)")

    return results


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("Loading traces...")
    traces, y = load_all_traces()
    print(f"Loaded {len(traces)} traces ({int(sum(y))} phishing, "
          f"{int(sum(1-y))} benign)")

    print("Extracting all views...")
    X_url, X_net, X_redir, X_interact, X_itg, X_cross, itg_mask = \
        build_all_views(traces)
    X_itg_eng = engineer_itg_features(X_itg)
    print(f"Views: URL={X_url.shape[1]}, Net={X_net.shape[1]}, "
          f"Redir={X_redir.shape[1]}, Interact={X_interact.shape[1]}, "
          f"ITG+Eng={X_itg_eng.shape[1]}, Cross={X_cross.shape[1]}")

    all_results = {}

    # Experiment A: Behavioral Invariance
    all_results["behavioral_invariance"] = run_behavioral_invariance(
        y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross
    )

    # Experiment B: Full-System Decoy Navigation
    all_results["decoy_navigation"] = run_full_system_decoy_test(
        traces, y, X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross
    )

    # Save
    out_path = Path(__file__).parent / "results" / "invariance_decoy_results.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nAll results saved to {out_path}")

    # Summary table for paper
    print("\n" + "=" * 70)
    print("SUMMARY FOR PAPER")
    print("=" * 70)

    inv = all_results["behavioral_invariance"]
    if "invariance_ratios" in inv:
        print("\nInvariance Ratios (Clean-URL F1 / All F1):")
        for view, ratio in inv["invariance_ratios"].items():
            status = "STABLE" if ratio > 0.85 else "DEGRADED"
            print(f"  {view}: {ratio:.4f} [{status}]")

    decoy = all_results["decoy_navigation"]
    print("\nDecoy Navigation Robustness:")
    for attack in ["decoy_navigation", "url_camouflage", "form_splitting", "combined"]:
        if attack in decoy:
            d = decoy[attack]
            print(f"  {attack}:")
            print(f"    Full: F1={d['full_system']['f1']:.4f}, "
                  f"Evasion={d['full_system']['evasion_rate']*100:.1f}%")
            print(f"    NonURL: F1={d['nonurl_only']['f1']:.4f}, "
                  f"Evasion={d['nonurl_only']['evasion_rate']*100:.1f}%")

    return all_results


if __name__ == "__main__":
    main()
