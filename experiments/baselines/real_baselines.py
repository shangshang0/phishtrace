"""
Real Baseline Methods — Independent Feature Extraction from Raw Traces

Each baseline extracts its OWN features directly from the raw crawled trace
JSON files, independent of PhishTrace's ITG feature space. This ensures a
genuine "same dataset, different methodology" comparison.

Baselines:
  1. URLNet-RF       — URL lexical & structural features (URL-based detection)
  2. RedirectChain   — Redirect-chain analysis (cloaking/evasion detection)
  3. NetTraffic-RF   — Network request analysis (traffic-level detection)
  4. ContentHeur-LR  — Page content heuristics (content-based detection)
  5. StackEnsemble   — Calibrated ensemble of all independent features
"""

import json
import math
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# ─────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────

def _entropy(s: str) -> float:
    """Shannon entropy of a string."""
    if not s:
        return 0.0
    freq = Counter(s)
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.split(":")[0].lower()
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────
# Feature Extractors — each reads RAW trace JSON
# ─────────────────────────────────────────────────────────

def extract_url_features(trace_data: dict) -> np.ndarray:
    """URL lexical & structural features.

    Mimics URL-based detection approaches (e.g., PhishFarm, URLNet)
    that only inspect the URL string without page content.  21 features.
    """
    url = trace_data.get("url", "")
    final_url = trace_data.get("final_url", trace_data.get("trace", {}).get("final_url", url))

    parsed = urlparse(url)
    domain = (parsed.netloc or parsed.path).split(":")[0].lower()
    path = parsed.path or "/"
    query = parsed.query or ""

    url_len = len(url)
    domain_len = len(domain)
    path_len = len(path)
    num_dots = domain.count(".")
    num_dashes = domain.count("-")
    num_underscores = url.count("_")
    num_slashes = path.count("/")
    num_at = url.count("@")
    num_digits_domain = sum(c.isdigit() for c in domain)
    has_ip = bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
    has_https = int(parsed.scheme == "https")

    # TLD analysis
    suspicious_tlds = {
        ".xyz", ".top", ".club", ".buzz", ".tk", ".ml", ".ga",
        ".cf", ".gq", ".work", ".life", ".online", ".site",
        ".icu", ".fun", ".info", ".cc", ".pw",
    }
    tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
    has_suspicious_tld = int(tld in suspicious_tlds)

    # Subdomains
    num_subdomains = max(0, num_dots - 1)

    # Query params
    num_params = len(parse_qs(query))
    has_encoded = int("%" in url)

    # Entropy
    domain_entropy = _entropy(domain)
    path_entropy = _entropy(path)

    # Keyword signals
    kw_login = int(any(k in url.lower() for k in [
        "login", "signin", "verify", "secure", "update", "confirm",
        "account", "password", "credential", "banking",
    ]))
    brand_in_url = int(any(b in url.lower() for b in [
        "paypal", "apple", "microsoft", "google", "amazon",
        "facebook", "netflix", "chase", "wellsfargo",
    ]))

    # Domain / final-URL mismatch
    final_domain = _domain_of(final_url)
    domain_changed = int(domain != final_domain)

    return np.array([
        url_len, domain_len, path_len,
        num_dots, num_dashes, num_underscores, num_slashes,
        num_at, num_digits_domain, int(has_ip), has_https,
        has_suspicious_tld, num_subdomains, num_params, has_encoded,
        domain_entropy, path_entropy,
        kw_login, brand_in_url, domain_changed,
        len(path.split("/")) - 1,  # path depth
    ], dtype=float)

URL_FEATURE_NAMES = [
    "url_len", "domain_len", "path_len",
    "num_dots", "num_dashes", "num_underscores", "num_slashes",
    "num_at", "num_digits_domain", "has_ip", "has_https",
    "has_suspicious_tld", "num_subdomains", "num_params", "has_encoded",
    "domain_entropy", "path_entropy",
    "kw_login", "brand_in_url", "domain_changed", "path_depth",
]


def extract_redirect_features(trace_data: dict) -> np.ndarray:
    """Redirect-chain analysis features.

    Mimics CrawlPhish-style redirect/cloaking detection that analyses
    the redirect sequence without inspecting page content.  14 features.
    """
    trace = trace_data.get("trace", {})
    redirects = trace.get("redirects", [])
    url = trace_data.get("url", "")
    final_url = trace.get("final_url", url)
    net_reqs = trace.get("network_requests", [])

    num_redirects = len(redirects)
    initial_domain = _domain_of(url)
    final_domain = _domain_of(final_url)

    # Cross-domain redirect analysis
    domains_in_chain = [_domain_of(r) for r in redirects]
    unique_domains = len(set(d for d in domains_in_chain if d))
    cross_domain = sum(
        1 for i in range(1, len(domains_in_chain))
        if domains_in_chain[i] and domains_in_chain[i - 1]
        and domains_in_chain[i] != domains_in_chain[i - 1]
    )

    # HTTP→HTTPS transitions (common in cloaking)
    http_to_https = sum(
        1 for i in range(1, len(redirects))
        if redirects[i - 1].startswith("http://") and redirects[i].startswith("https://")
    )

    # Domain changed between start and end
    domain_changed = int(initial_domain != final_domain)

    # Redirect-related network requests (30x status codes)
    redirect_statuses = [r for r in net_reqs if (r.get("status") or 0) // 100 == 3]
    num_3xx = len(redirect_statuses)

    # Document vs. sub-resource requests
    doc_requests = [r for r in net_reqs if r.get("resource_type") == "document"]
    num_doc_requests = len(doc_requests)

    # External resource ratio (different domain from final_url)
    external_reqs = sum(
        1 for r in net_reqs
        if _domain_of(r.get("url", "")) and _domain_of(r.get("url", "")) != final_domain
    )
    external_ratio = external_reqs / max(1, len(net_reqs))

    # Chain length metric
    chain_length = max(num_redirects, num_3xx)

    # Free hosting indicator (many phishing sites use free hosting)
    free_hosting = {
        "weebly", "wix", "webflow", "000webhostapp", "sites.google",
        "firebase", "herokuapp", "netlify", "vercel", "github.io",
        "blogspot", "wordpress.com", "square.site",
    }
    uses_free_hosting = int(
        any(fh in final_domain for fh in free_hosting)
    )

    # URL shortener indicator
    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "rb.gy"}
    uses_shortener = int(initial_domain in shorteners)

    return np.array([
        num_redirects, unique_domains, cross_domain,
        http_to_https, domain_changed, num_3xx, num_doc_requests,
        external_ratio, chain_length,
        uses_free_hosting, uses_shortener,
        len(net_reqs),  # total network requests
        external_reqs,
        max(0, num_redirects - 1),  # extra hops beyond initial load
    ], dtype=float)

REDIRECT_FEATURE_NAMES = [
    "num_redirects", "unique_domains", "cross_domain_redirects",
    "http_to_https", "domain_changed", "num_3xx", "num_doc_requests",
    "external_ratio", "chain_length",
    "uses_free_hosting", "uses_shortener",
    "total_network_requests", "external_requests", "extra_hops",
]


def extract_network_features(trace_data: dict) -> np.ndarray:
    """Network traffic analysis features.

    Analyses HTTP request patterns, resource types, and domain
    diversity — the kind of signals traffic-level detectors use.
    18 features.
    """
    trace = trace_data.get("trace", {})
    net_reqs = trace.get("network_requests", [])
    final_url = trace.get("final_url", trace_data.get("url", ""))
    final_domain = _domain_of(final_url)

    n = len(net_reqs)

    # Resource-type counts
    res_types = Counter(r.get("resource_type", "other") for r in net_reqs)
    n_scripts = res_types.get("script", 0)
    n_stylesheets = res_types.get("stylesheet", 0)
    n_images = res_types.get("image", 0)
    n_xhr = res_types.get("xhr", 0) + res_types.get("fetch", 0)
    n_documents = res_types.get("document", 0)
    n_fonts = res_types.get("font", 0)

    # Domain diversity
    domains = [_domain_of(r.get("url", "")) for r in net_reqs]
    unique_req_domains = len(set(d for d in domains if d))

    # External resource ratio
    external = sum(1 for d in domains if d and d != final_domain)
    ext_ratio = external / max(1, n)

    # CDN usage (common in both legit and phishing)
    cdn_keywords = {"cdn", "cloudfront", "cloudflare", "akamai", "fastly", "jsdelivr"}
    cdn_requests = sum(1 for d in domains if any(c in d for c in cdn_keywords))
    cdn_ratio = cdn_requests / max(1, n)

    # Status code distribution
    statuses = [(r.get("status") or 0) for r in net_reqs]
    n_2xx = sum(1 for s in statuses if 200 <= s < 300)
    n_4xx = sum(1 for s in statuses if 400 <= s < 500)
    n_5xx = sum(1 for s in statuses if 500 <= s < 600)

    # Content-type diversity
    content_types = set(
        r.get("content_type", "").split(";")[0].strip()
        for r in net_reqs if r.get("content_type")
    )
    content_type_diversity = len(content_types)

    # Timing features (if timestamps available)
    timestamps = sorted(r.get("timestamp", 0) for r in net_reqs if r.get("timestamp"))
    if len(timestamps) >= 2:
        load_duration = timestamps[-1] - timestamps[0]
        avg_interval = load_duration / max(1, len(timestamps) - 1)
    else:
        load_duration = 0.0
        avg_interval = 0.0

    return np.array([
        n, n_scripts, n_stylesheets, n_images, n_xhr, n_documents, n_fonts,
        unique_req_domains, ext_ratio, cdn_ratio,
        n_2xx, n_4xx, n_5xx,
        content_type_diversity,
        load_duration, avg_interval,
        n_scripts / max(1, n),  # script ratio
        n_xhr / max(1, n),       # xhr ratio (data exfil indicator)
    ], dtype=float)

NETWORK_FEATURE_NAMES = [
    "total_requests", "n_scripts", "n_stylesheets", "n_images",
    "n_xhr", "n_documents", "n_fonts",
    "unique_domains", "ext_ratio", "cdn_ratio",
    "n_2xx", "n_4xx", "n_5xx",
    "content_type_diversity",
    "load_duration", "avg_interval",
    "script_ratio", "xhr_ratio",
]


def extract_content_features(trace_data: dict) -> np.ndarray:
    """Page content heuristic features.

    Examines page titles, interactive elements, form behavior, and
    event patterns — what content-level detectors analyse.  18 features.
    """
    trace = trace_data.get("trace", {})
    events = trace.get("events", [])
    title = (trace.get("page_title") or "").lower()
    url = trace_data.get("url", "").lower()
    final_url = (trace.get("final_url") or "").lower()

    # Title-based signals
    login_keywords = {"login", "signin", "sign in", "log in", "verify",
                      "verification", "secure", "update", "confirm", "password"}
    brand_keywords = {"paypal", "apple", "microsoft", "google", "amazon",
                      "facebook", "netflix", "chase", "wellsfargo", "bank",
                      "dropbox", "adobe", "office", "outlook", "icloud"}
    urgency_keywords = {"urgent", "immediate", "suspend", "locked", "expire",
                        "unauthorized", "alert", "warning", "limited"}

    title_has_login = int(any(k in title for k in login_keywords))
    title_has_brand = int(any(k in title for k in brand_keywords))
    title_has_urgency = int(any(k in title for k in urgency_keywords))
    title_length = len(title)
    title_entropy = _entropy(title)

    # Event analysis — filter noise events to match EITG preprocessing
    filtered_events = [e for e in events if e.get("event_type", "") not in ("screenshot", "page")]
    event_types = Counter(e.get("event_type", "") for e in filtered_events)
    n_links = event_types.get("link_detected", 0) + event_types.get("login_link_follow", 0)
    n_buttons = event_types.get("button_detected", 0) + event_types.get("button_click", 0)
    n_inputs = event_types.get("input", 0) + event_types.get("form_input", 0)
    n_submits = sum(v for k, v in event_types.items() if 'submit' in k)
    n_checks = event_types.get("check", 0) + event_types.get("select", 0)
    n_dual_submit = event_types.get("dual_submit_error", 0) + event_types.get("dual_submit_resubmit", 0)

    total_events = len(filtered_events)

    # Form analysis (from event data)
    has_password_field = int(any(
        "password" in (e.get("element_class", "") + e.get("element_id", "") +
                       str(e.get("form_data", ""))).lower()
        for e in events
    ))
    has_email_field = int(any(
        kw in (e.get("element_class", "") + e.get("element_id", "") +
               str(e.get("form_data", ""))).lower()
        for kw in ["email", "mail", "user"]
        for e in events
    ))

    # Temporal patterns
    timestamps = sorted(e.get("timestamp", 0) for e in events if e.get("timestamp"))
    if len(timestamps) >= 2:
        interaction_duration = timestamps[-1] - timestamps[0]
    else:
        interaction_duration = 0.0

    # Forms submitted
    forms_submitted = trace_data.get("forms_submitted", 0)
    dual_submissions = trace_data.get("dual_submissions", 0)

    return np.array([
        title_has_login, title_has_brand, title_has_urgency,
        title_length, title_entropy,
        n_links, n_buttons, n_inputs, n_submits, n_checks,
        total_events, has_password_field, has_email_field,
        interaction_duration, forms_submitted, dual_submissions,
        n_dual_submit,
        n_inputs / max(1, total_events),  # input ratio
    ], dtype=float)

CONTENT_FEATURE_NAMES = [
    "title_has_login", "title_has_brand", "title_has_urgency",
    "title_length", "title_entropy",
    "n_links", "n_buttons", "n_inputs", "n_submits", "n_checks",
    "total_events", "has_password_field", "has_email_field",
    "interaction_duration", "forms_submitted", "dual_submissions",
    "n_dual_submit", "input_ratio",
]


# ─────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────

def load_traces() -> Tuple[List[dict], np.ndarray]:
    """Load all raw trace JSON files and return (trace_dicts, labels)."""
    traces_dir = DATASET_DIR / "traces"
    all_data: List[dict] = []
    labels: List[int] = []

    for label_int, subdir in [(1, "phishing"), (0, "benign")]:
        d = traces_dir / subdir
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                if not data.get("success"):
                    continue
                all_data.append(data)
                labels.append(label_int)
            except Exception:
                continue

    return all_data, np.array(labels)


# ─────────────────────────────────────────────────────────
# Evaluation Harness
# ─────────────────────────────────────────────────────────

def evaluate_baseline(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    pipeline: Pipeline,
    n_folds: int = 10,
) -> Dict[str, float]:
    """Stratified k-fold CV for a single baseline."""
    minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    actual_folds = max(2, min(n_folds, minority))
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    metrics = {"accuracy": [], "precision": [], "recall": [], "f1": [], "auc": []}

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)

        auc = np.nan
        if len(np.unique(y_te)) > 1:
            try:
                y_prob = pipeline.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_te, y_prob)
            except (AttributeError, ValueError):
                pass

        metrics["accuracy"].append(accuracy_score(y_te, y_pred))
        metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
        metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        metrics["auc"].append(auc)

    return {k: round(float(np.nanmean(v)), 4) for k, v in metrics.items()}


# ─────────────────────────────────────────────────────────
# Main entry-point
# ─────────────────────────────────────────────────────────

def run_real_baselines(traces: List[dict] = None, labels: np.ndarray = None):
    """Run all real baselines on raw trace data.

    Returns dict {baseline_name: {metric: value}}.
    """
    if traces is None or labels is None:
        traces, labels = load_traces()

    print(f"\n{'='*70}")
    print("REAL BASELINES — Independent Feature Extraction from Raw Traces")
    print(f"{'='*70}")
    print(f"Dataset: {len(labels)} traces  ({int(np.sum(labels==1))} phishing, "
          f"{int(np.sum(labels==0))} benign)")

    results = {}

    # ── 1. URL Lexical Features ──
    print("\n[1/5] URLNet-RF: URL lexical & structural features ...")
    X_url = np.array([extract_url_features(t) for t in traces])
    pipe_url = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )),
    ])
    res = evaluate_baseline("URLNet-RF", X_url, labels, pipe_url)
    results["URLNet-RF"] = {**res, "features": len(URL_FEATURE_NAMES),
                            "paradigm": "URL lexical"}
    print(f"  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── 2. Redirect Chain Analysis ──
    print("[2/5] RedirectChain-GBM: Redirect-chain & cloaking features ...")
    X_redir = np.array([extract_redirect_features(t) for t in traces])
    pipe_redir = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42,
        )),
    ])
    res = evaluate_baseline("RedirectChain-GBM", X_redir, labels, pipe_redir)
    results["RedirectChain-GBM"] = {**res, "features": len(REDIRECT_FEATURE_NAMES),
                                     "paradigm": "Redirect chain"}
    print(f"  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── 3. Network Traffic Analysis ──
    print("[3/5] NetTraffic-RF: Network request analysis ...")
    X_net = np.array([extract_network_features(t) for t in traces])
    pipe_net = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=42,
            class_weight="balanced", n_jobs=-1,
        )),
    ])
    res = evaluate_baseline("NetTraffic-RF", X_net, labels, pipe_net)
    results["NetTraffic-RF"] = {**res, "features": len(NETWORK_FEATURE_NAMES),
                                 "paradigm": "Network traffic"}
    print(f"  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── 4. Content Heuristics ──
    print("[4/5] ContentHeur-LR: Page content heuristic features ...")
    X_content = np.array([extract_content_features(t) for t in traces])
    pipe_content = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42,
        )),
    ])
    res = evaluate_baseline("ContentHeur-LR", X_content, labels, pipe_content)
    results["ContentHeur-LR"] = {**res, "features": len(CONTENT_FEATURE_NAMES),
                                  "paradigm": "Page content"}
    print(f"  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── 5. Stacked Ensemble (all independent features) ──
    print("[5/5] StackEnsemble: Calibrated ensemble of all baselines ...")
    X_all = np.hstack([X_url, X_redir, X_net, X_content])
    total_feats = X_all.shape[1]
    estimators = [
        ("rf", RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced", n_jobs=-1)),
        ("gb", GradientBoostingClassifier(
            n_estimators=100, max_depth=5, random_state=42)),
        ("lr", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42)),
    ]
    pipe_stack = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=5, n_jobs=-1,
        )),
    ])
    res = evaluate_baseline("StackEnsemble", X_all, labels, pipe_stack)
    results["StackEnsemble"] = {**res, "features": total_feats,
                                 "paradigm": "Multi-signal ensemble"}
    print(f"  Acc={res['accuracy']:.4f}  F1={res['f1']:.4f}  AUC={res['auc']:.4f}")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"{'Method':<22} {'Paradigm':<20} {'|F|':>4} {'Acc':>7} {'Prec':>7} "
          f"{'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("-" * 90)
    for name, m in results.items():
        print(f"{name:<22} {m['paradigm']:<20} {m['features']:>4} "
              f"{m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")

    return results


if __name__ == "__main__":
    run_real_baselines()
