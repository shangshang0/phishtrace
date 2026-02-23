"""
Enhanced Interaction Trace Graph (EITG) Detector

Extends PhishTrace's ITG methodology to incorporate ALL signals available
in the interaction trace: URL structure, network traffic topology, redirect
chains, and page content events.

Key architectural improvements over base PhishTrace:
1. Full-trace coverage: works on ALL successful traces (not just graph-parseable ones)
2. Multi-signal ITG: URL + Network + Redirect + Event features as graph views
3. Unified multi-view ensemble: separate classifiers per signal view, soft voting
4. Cross-view interaction features: captures inter-view correlations

This is NOT feature stacking — each feature set represents a different
"view" of the same Interaction Trace Graph, and the ensemble architecture
reflects the multi-layered nature of web interactions.
"""

import json
import math
import os
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
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
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ═══════════════════════════════════════════════════════════════
# View 1: URL Structure Features (21D)
# Captures the URL-level signals that characterize phishing URLs
# ═══════════════════════════════════════════════════════════════

def _entropy(s: str) -> float:
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


def extract_url_view(trace_data: dict) -> np.ndarray:
    """URL lexical & structural features — 21D."""
    url = trace_data.get("url", "")
    final_url = trace_data.get("final_url",
                                trace_data.get("trace", {}).get("final_url", url))

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
    has_ip = bool(
        __import__("re").match(r"^\d{1,3}(\.\d{1,3}){3}$", domain)
    )
    has_https = int(parsed.scheme == "https")

    suspicious_tlds = {
        ".xyz", ".top", ".club", ".buzz", ".tk", ".ml", ".ga",
        ".cf", ".gq", ".work", ".life", ".online", ".site",
        ".icu", ".fun", ".info", ".cc", ".pw",
    }
    tld = "." + domain.rsplit(".", 1)[-1] if "." in domain else ""
    has_suspicious_tld = int(tld in suspicious_tlds)
    num_subdomains = max(0, num_dots - 1)
    num_params = len(parse_qs(query))
    has_encoded = int("%" in url)
    domain_entropy = _entropy(domain)
    path_entropy = _entropy(path)

    kw_login = int(any(k in url.lower() for k in [
        "login", "signin", "verify", "secure", "update", "confirm",
        "account", "password", "credential", "banking",
    ]))
    brand_in_url = int(any(b in url.lower() for b in [
        "paypal", "apple", "microsoft", "google", "amazon",
        "facebook", "netflix", "chase", "wellsfargo",
    ]))

    final_domain = _domain_of(final_url)
    domain_changed = int(domain != final_domain)

    return np.array([
        url_len, domain_len, path_len,
        num_dots, num_dashes, num_underscores, num_slashes,
        num_at, num_digits_domain, int(has_ip), has_https,
        has_suspicious_tld, num_subdomains, num_params, has_encoded,
        domain_entropy, path_entropy,
        kw_login, brand_in_url, domain_changed,
        len(path.split("/")) - 1,
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# View 2: Network Traffic Topology Features (18D)
# Captures the network request patterns characteristic of phishing
# ═══════════════════════════════════════════════════════════════

def extract_network_view(trace_data: dict) -> np.ndarray:
    """Network traffic analysis features — 18D."""
    trace = trace_data.get("trace", {})
    net_reqs = trace.get("network_requests", [])
    final_url = trace.get("final_url", trace_data.get("url", ""))
    final_domain = _domain_of(final_url)

    n = len(net_reqs)

    res_types = Counter(r.get("resource_type", "other") for r in net_reqs)
    n_scripts = res_types.get("script", 0)
    n_stylesheets = res_types.get("stylesheet", 0)
    n_images = res_types.get("image", 0)
    n_xhr = res_types.get("xhr", 0) + res_types.get("fetch", 0)
    n_documents = res_types.get("document", 0)
    n_fonts = res_types.get("font", 0)

    domains = [_domain_of(r.get("url", "")) for r in net_reqs]
    unique_req_domains = len(set(d for d in domains if d))
    external = sum(1 for d in domains if d and d != final_domain)
    ext_ratio = external / max(1, n)

    cdn_keywords = {"cdn", "cloudfront", "cloudflare", "akamai", "fastly", "jsdelivr"}
    cdn_requests = sum(1 for d in domains if any(c in d for c in cdn_keywords))
    cdn_ratio = cdn_requests / max(1, n)

    statuses = [(r.get("status") or 0) for r in net_reqs]
    n_2xx = sum(1 for s in statuses if 200 <= s < 300)
    n_4xx = sum(1 for s in statuses if 400 <= s < 500)
    n_5xx = sum(1 for s in statuses if 500 <= s < 600)

    content_types = set(
        r.get("content_type", "").split(";")[0].strip()
        for r in net_reqs if r.get("content_type")
    )
    content_type_diversity = len(content_types)

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
        n_scripts / max(1, n),
        n_xhr / max(1, n),
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# View 3: Redirect Chain Features (14D)
# Captures redirect/cloaking patterns
# ═══════════════════════════════════════════════════════════════

def extract_redirect_view(trace_data: dict) -> np.ndarray:
    """Redirect chain analysis features — 14D."""
    trace = trace_data.get("trace", {})
    redirects = trace.get("redirects", [])
    url = trace_data.get("url", "")
    final_url = trace.get("final_url", url)
    net_reqs = trace.get("network_requests", [])

    num_redirects = len(redirects)
    initial_domain = _domain_of(url)
    final_domain = _domain_of(final_url)

    domains_in_chain = [_domain_of(r) for r in redirects]
    unique_domains = len(set(d for d in domains_in_chain if d))
    cross_domain = sum(
        1 for i in range(1, len(domains_in_chain))
        if domains_in_chain[i] and domains_in_chain[i - 1]
        and domains_in_chain[i] != domains_in_chain[i - 1]
    )

    http_to_https = sum(
        1 for i in range(1, len(redirects))
        if redirects[i - 1].startswith("http://") and redirects[i].startswith("https://")
    )

    domain_changed = int(initial_domain != final_domain)
    redirect_statuses = [r for r in net_reqs if (r.get("status") or 0) // 100 == 3]
    num_3xx = len(redirect_statuses)
    doc_requests = [r for r in net_reqs if r.get("resource_type") == "document"]
    num_doc_requests = len(doc_requests)

    external_reqs = sum(
        1 for r in net_reqs
        if _domain_of(r.get("url", "")) and _domain_of(r.get("url", "")) != final_domain
    )
    external_ratio = external_reqs / max(1, len(net_reqs))
    chain_length = max(num_redirects, num_3xx)

    free_hosting = {
        "weebly", "wix", "webflow", "000webhostapp", "sites.google",
        "firebase", "herokuapp", "netlify", "vercel", "github.io",
        "blogspot", "wordpress.com", "square.site",
    }
    uses_free_hosting = int(any(fh in final_domain for fh in free_hosting))

    shorteners = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "rb.gy"}
    uses_shortener = int(initial_domain in shorteners)

    return np.array([
        num_redirects, unique_domains, cross_domain,
        http_to_https, domain_changed, num_3xx, num_doc_requests,
        external_ratio, chain_length,
        uses_free_hosting, uses_shortener,
        len(net_reqs), external_reqs,
        max(0, num_redirects - 1),
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# View 4: Interaction Event Features (18D)
# Captures page content and user interaction patterns
# ═══════════════════════════════════════════════════════════════

def _filter_interactions(events: list) -> list:
    """Remove screenshot/page events — they are visual bookmarks, not interactions."""
    return [e for e in events
            if e and e.get('event_type') != 'screenshot'
            and e.get('element_tag') != 'page']


def extract_interaction_view(trace_data: dict) -> np.ndarray:
    """Page content & interaction event features — 18D."""
    trace = trace_data.get("trace", {})
    events = _filter_interactions(trace.get("events", []))
    title = (trace.get("page_title") or "").lower()

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

    event_types = Counter(e.get("event_type", "") for e in events)
    # Count both old event names (link_detected, button_detected) and
    # new deep-interaction names (login_link_follow, button_click, etc.)
    n_links = (event_types.get("link_detected", 0)
               + event_types.get("login_link_follow", 0))
    n_buttons = (event_types.get("button_detected", 0)
                 + event_types.get("button_click", 0))
    n_inputs = (event_types.get("input", 0) + event_types.get("form_input", 0))
    n_submits = (event_types.get("submit", 0)
                 + event_types.get("js_form_submit", 0))
    n_checks = event_types.get("check", 0) + event_types.get("select", 0)
    n_dual_submit = (event_types.get("dual_submit_error", 0)
                     + event_types.get("dual_submit_resubmit", 0))
    n_page_transitions = event_types.get("page_transition", 0)
    total_events = len(events)

    has_password_field = int(any(
        "password" in (e.get("element_class", "") + e.get("element_id", "")
                       + str(e.get("form_data", ""))).lower()
        for e in events
    ))
    has_email_field = int(any(
        kw in (e.get("element_class", "") + e.get("element_id", "")
               + str(e.get("form_data", ""))).lower()
        for kw in ["email", "mail", "user"]
        for e in events
    ))

    timestamps = sorted(e.get("timestamp", 0) for e in events if e.get("timestamp"))
    interaction_duration = (timestamps[-1] - timestamps[0]) if len(timestamps) >= 2 else 0.0

    forms_submitted = trace_data.get("forms_submitted", 0)
    dual_submissions = trace_data.get("dual_submissions", 0)

    return np.array([
        title_has_login, title_has_brand, title_has_urgency,
        title_length, title_entropy,
        n_links, n_buttons, n_inputs, n_submits, n_checks,
        total_events, has_password_field, has_email_field,
        interaction_duration, forms_submitted, dual_submissions,
        n_dual_submit,
        n_inputs / max(1, total_events),
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# View 5: ITG Graph Features (30D)
# The original PhishTrace graph-based features
# ═══════════════════════════════════════════════════════════════

ITG_FEATURE_NAMES = [
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


def extract_itg_view(trace_data: dict) -> Optional[np.ndarray]:
    """ITG graph features — 30D.  Returns None if graph cannot be built."""
    try:
        from analyzer.graph_builder import InteractionGraphBuilder
        builder = InteractionGraphBuilder()
        # The graph builder expects trace dict with 'events', 'redirects', etc.
        inner = trace_data.get("trace", trace_data)
        if "url" not in inner and "url" in trace_data:
            inner = dict(inner)
            inner["url"] = trace_data["url"]
        graph = builder.build_graph_from_dict(inner)
        if graph.number_of_nodes() < 2:
            return None
        feats = builder.extract_features(graph)
        vec = []
        for name in ITG_FEATURE_NAMES:
            val = getattr(feats, name, 0)
            if isinstance(val, bool):
                val = 1 if val else 0
            vec.append(float(val))
        return np.array(vec, dtype=float)
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug("ITG extraction failed: %s", e)
        return None


# ═══════════════════════════════════════════════════════════════
# Cross-View Interaction Features (12D)
# Captures correlations between different views of the trace
# ═══════════════════════════════════════════════════════════════

def compute_cross_view_features(
    url_f: np.ndarray,
    net_f: np.ndarray,
    redir_f: np.ndarray,
    interaction_f: np.ndarray,
) -> np.ndarray:
    """Cross-view interaction features — 12D.

    These encode relationships that single-view detectors cannot capture:
    - URL complexity × network traffic volume
    - Redirect chain depth × content suspiciousness
    - Network external ratio × URL domain change
    """
    safe = lambda x: max(x, 1e-8)

    # URL × Network interactions
    url_entropy = url_f[15]  # domain_entropy
    net_requests = net_f[0]  # total_requests
    url_len = url_f[0]
    ext_ratio = net_f[8]     # ext_ratio

    cv1 = url_entropy * net_requests           # complex URL + heavy network
    cv2 = url_f[11] * ext_ratio                # suspicious_tld × external_ratio
    cv3 = url_f[19] * redir_f[4]               # domain_changed(url) × domain_changed(redirect)

    # Redirect × Interaction
    redir_count = redir_f[0]
    has_password = interaction_f[11]  # has_password_field
    has_email = interaction_f[12]  # has_email_field
    total_events = interaction_f[10]

    cv4 = redir_count * has_password            # redirects leading to password page
    cv5 = redir_f[9] * interaction_f[0]         # free_hosting × title_has_login
    cv6 = redir_f[2] * (has_password + has_email)  # cross_domain × credential harvesting

    # Network × Interaction
    n_xhr = net_f[4]
    n_scripts = net_f[1]
    interaction_dur = interaction_f[13]

    cv7 = n_xhr * has_password                  # XHR data exfil with password
    cv8 = n_scripts / safe(net_requests) * total_events  # script intensity × events
    cv9 = net_f[7] * interaction_f[5]           # unique_domains × n_links

    # Composite risk indicators
    cv10 = (has_password + has_email + url_f[17]) * redir_count  # credential signals × redirects
    cv11 = url_f[11] * redir_f[9]               # suspicious_tld × free_hosting
    cv12 = (n_xhr / safe(net_requests)) * (url_f[19])  # data exfil ratio × domain_changed

    return np.array([
        cv1, cv2, cv3, cv4, cv5, cv6,
        cv7, cv8, cv9, cv10, cv11, cv12,
    ], dtype=float)


# ═══════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════

def load_all_traces() -> Tuple[List[dict], np.ndarray]:
    """Load ALL successful trace JSON files."""
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


# ═══════════════════════════════════════════════════════════════
# Enhanced ITG Multi-View Ensemble
# ═══════════════════════════════════════════════════════════════

def build_all_views(traces: List[dict]) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """Extract all feature views from traces.

    Returns:
        X_url (N, 21), X_net (N, 18), X_redir (N, 14), X_interact (N, 18),
        X_itg (N, 30) [zeros where unavailable], X_cross (N, 12),
        itg_mask (N,) — boolean mask for available ITG features
    """
    X_url, X_net, X_redir, X_interact = [], [], [], []
    X_itg, X_cross = [], []
    itg_mask = []

    for t in traces:
        u = extract_url_view(t)
        n = extract_network_view(t)
        r = extract_redirect_view(t)
        i_ = extract_interaction_view(t)

        itg = extract_itg_view(t)
        if itg is not None:
            itg_mask.append(True)
        else:
            itg = np.zeros(30, dtype=float)
            itg_mask.append(False)

        cross = compute_cross_view_features(u, n, r, i_)

        X_url.append(u)
        X_net.append(n)
        X_redir.append(r)
        X_interact.append(i_)
        X_itg.append(itg)
        X_cross.append(cross)

    return (
        np.nan_to_num(np.array(X_url), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.array(X_net), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.array(X_redir), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.array(X_interact), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.array(X_itg), nan=0.0, posinf=0.0, neginf=0.0),
        np.nan_to_num(np.array(X_cross), nan=0.0, posinf=0.0, neginf=0.0),
        np.array(itg_mask),
    )


def engineer_itg_features(X_itg: np.ndarray) -> np.ndarray:
    """Add 12 engineered cross-features from ITG view (same as original PhishTrace)."""
    idx = {n: i for i, n in enumerate(ITG_FEATURE_NAMES)}
    derived = []

    derived.append(X_itg[:, idx['graph_density']] * X_itg[:, idx['form_to_node_ratio']])
    derived.append(X_itg[:, idx['clustering_coefficient']] * X_itg[:, idx['credential_fields_count']])
    safe_redirects = np.maximum(X_itg[:, idx['num_redirects']], 1e-8)
    derived.append(X_itg[:, idx['external_redirects']] / safe_redirects)
    derived.append(X_itg[:, idx['num_nodes']] * X_itg[:, idx['requests_sensitive_data']])

    derived.append(X_itg[:, idx['betweenness_centrality_max']] * X_itg[:, idx['total_interaction_time']])
    derived.append(X_itg[:, idx['pagerank_max']] * X_itg[:, idx['event_frequency']])
    derived.append(X_itg[:, idx['closeness_centrality_max']] * X_itg[:, idx['credential_fields_count']])

    safe_nodes = np.maximum(X_itg[:, idx['num_nodes']], 1e-8)
    derived.append(X_itg[:, idx['num_edges']] / safe_nodes)
    derived.append(X_itg[:, idx['diameter']] * X_itg[:, idx['max_redirect_depth']])
    safe_total = np.maximum(X_itg[:, idx['in_degree_max']] + X_itg[:, idx['out_degree_max']], 1e-8)
    derived.append(np.abs(X_itg[:, idx['in_degree_max']] - X_itg[:, idx['out_degree_max']]) / safe_total)

    derived.append(X_itg[:, idx['has_password_input']] + X_itg[:, idx['has_email_input']]
                   + X_itg[:, idx['financial_fields_count']])
    derived.append(X_itg[:, idx['avg_path_length']] * X_itg[:, idx['num_connected_components']])

    return np.hstack([X_itg, np.column_stack(derived)])


class EnhancedITGDetector:
    """
    Enhanced ITG Multi-View Ensemble Detector.

    Architecture:
    - View A: URL structure (21D) → GBM
    - View B: Network traffic (18D) → RF
    - View C: Redirect chain (14D) → GBM
    - View D: Interaction events (18D) → RF
    - View E: ITG graph + engineered (42D) → RF
    - View F: Cross-view interactions (12D) → GBM
    - Meta: Soft-voting ensemble → Logistic Regression
    """

    def __init__(self):
        self.view_names = [
            "URL Structure", "Network Traffic", "Redirect Chain",
            "Interaction Events", "ITG Graph", "Cross-View",
        ]

    def _make_view_pipelines(self) -> List[Tuple[str, Pipeline]]:
        """Create per-view classifiers."""
        return [
            ("url", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )),
            ])),
            ("net", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=None,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ])),
            ("redir", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=200, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )),
            ])),
            ("interact", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=None,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ])),
            ("itg", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=None,
                    min_samples_split=3, class_weight="balanced",
                    random_state=42, n_jobs=-1,
                )),
            ])),
            ("cross", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.1,
                    random_state=42,
                )),
            ])),
        ]

    def evaluate_multiview(
        self,
        X_views: List[np.ndarray],
        y: np.ndarray,
        n_folds: int = 10,
        view_weights: Optional[List[float]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Run multi-view ensemble with stratified k-fold CV.

        Returns: (mean_metrics, std_metrics, per_view_metrics)
        """
        minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
        actual_folds = max(2, min(n_folds, minority))
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        if view_weights is None:
            view_weights = [2.5, 1.5, 1.0, 1.0, 1.5, 1.0]

        ensemble_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}
        view_fold_metrics = {i: {k: [] for k in ["accuracy", "f1", "auc"]}
                            for i in range(len(X_views))}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_views[0], y)):
            y_train, y_test = y[train_idx], y[test_idx]

            # Train per-view classifiers and collect predictions
            view_probas = []
            pipelines = self._make_view_pipelines()

            for v_idx, (name, pipe) in enumerate(pipelines):
                X_v = X_views[v_idx]
                X_tr, X_te = X_v[train_idx], X_v[test_idx]

                pipe.fit(X_tr, y_train)
                y_pred_v = pipe.predict(X_te)

                try:
                    proba = pipe.predict_proba(X_te)[:, 1]
                except (AttributeError, IndexError):
                    proba = y_pred_v.astype(float)

                view_probas.append(proba)

                # Per-view metrics
                view_fold_metrics[v_idx]["accuracy"].append(accuracy_score(y_test, y_pred_v))
                view_fold_metrics[v_idx]["f1"].append(f1_score(y_test, y_pred_v, zero_division=0))
                if len(np.unique(y_test)) > 1:
                    view_fold_metrics[v_idx]["auc"].append(roc_auc_score(y_test, proba))
                else:
                    view_fold_metrics[v_idx]["auc"].append(np.nan)

            # Weighted soft voting
            weighted_proba = np.zeros(len(y_test))
            total_weight = sum(view_weights)
            for v_idx, proba in enumerate(view_probas):
                weighted_proba += view_weights[v_idx] * proba
            weighted_proba /= total_weight

            y_pred = (weighted_proba >= 0.5).astype(int)

            ensemble_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            ensemble_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            ensemble_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            ensemble_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            if len(np.unique(y_test)) > 1:
                ensemble_metrics["auc"].append(roc_auc_score(y_test, weighted_proba))
            else:
                ensemble_metrics["auc"].append(np.nan)

        mean_m = {k: round(float(np.nanmean(v)), 4) for k, v in ensemble_metrics.items()}
        std_m = {k: round(float(np.nanstd(v)), 4) for k, v in ensemble_metrics.items()}
        per_view = {}
        for v_idx, name in enumerate(self.view_names):
            per_view[name] = {k: round(float(np.nanmean(v)), 4)
                             for k, v in view_fold_metrics[v_idx].items()}

        return mean_m, std_m, per_view

    def evaluate_stacked_multiview(
        self,
        X_views: List[np.ndarray],
        y: np.ndarray,
        n_folds: int = 10,
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, float]]]:
        """Stacked Multi-View Ensemble: per-view classifiers feed a meta-learner.

        Level 0: Each view trains its own classifier → produces probability
        Level 1: Meta-learner (LR + RF) combines per-view probabilities +
                 raw features from the most discriminative views.

        This architecture captures non-linear view interactions that weighted
        soft voting cannot model.
        """
        from sklearn.base import clone

        minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
        actual_folds = max(2, min(n_folds, minority))
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        ensemble_metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}
        view_fold_metrics = {i: {k: [] for k in ["accuracy", "f1", "auc"]}
                            for i in range(len(X_views))}

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_views[0], y)):
            y_train, y_test = y[train_idx], y[test_idx]

            # ── Level 0: Per-view classifiers ──
            # Use inner CV (3-fold) on training set to generate out-of-fold predictions
            inner_folds = max(2, min(3, min(int(np.sum(y_train == 0)),
                                            int(np.sum(y_train == 1)))))
            inner_skf = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)

            # Collect OOF predictions for meta-learner training
            n_train = len(train_idx)
            oof_probas = np.zeros((n_train, len(X_views)))
            test_probas = np.zeros((len(test_idx), len(X_views)))

            pipelines = self._make_view_pipelines()

            for v_idx, (name, pipe) in enumerate(pipelines):
                X_v = X_views[v_idx]
                X_v_train = X_v[train_idx]
                X_v_test = X_v[test_idx]

                # Inner CV for OOF predictions
                for inner_train_idx, inner_val_idx in inner_skf.split(X_v_train, y_train):
                    X_inner_tr = X_v_train[inner_train_idx]
                    y_inner_tr = y_train[inner_train_idx]
                    X_inner_val = X_v_train[inner_val_idx]

                    inner_pipe = clone(pipe)
                    inner_pipe.fit(X_inner_tr, y_inner_tr)

                    try:
                        oof_probas[inner_val_idx, v_idx] = inner_pipe.predict_proba(X_inner_val)[:, 1]
                    except (AttributeError, IndexError):
                        oof_probas[inner_val_idx, v_idx] = inner_pipe.predict(X_inner_val).astype(float)

                # Retrain on full training set for test predictions
                full_pipe = clone(pipe)
                full_pipe.fit(X_v_train, y_train)
                y_pred_v = full_pipe.predict(X_v_test)

                try:
                    test_probas[:, v_idx] = full_pipe.predict_proba(X_v_test)[:, 1]
                except (AttributeError, IndexError):
                    test_probas[:, v_idx] = y_pred_v.astype(float)

                # Per-view metrics
                view_fold_metrics[v_idx]["accuracy"].append(accuracy_score(y_test, y_pred_v))
                view_fold_metrics[v_idx]["f1"].append(f1_score(y_test, y_pred_v, zero_division=0))
                if len(np.unique(y_test)) > 1:
                    view_fold_metrics[v_idx]["auc"].append(
                        roc_auc_score(y_test, test_probas[:, v_idx])
                    )
                else:
                    view_fold_metrics[v_idx]["auc"].append(np.nan)

            # ── Level 1: Meta-learner ──
            # Combine per-view probabilities + top raw features
            # Include raw URL features (most discriminative view) as passthrough
            X_url_train = X_views[0][train_idx]
            X_url_test = X_views[0][test_idx]

            # Meta features: view probabilities + URL raw features
            X_meta_train = np.hstack([oof_probas, X_url_train])
            X_meta_test = np.hstack([test_probas, X_url_test])

            # Ensemble meta-learner: NNet for non-linear combination
            meta_scaler = StandardScaler()
            X_meta_train_s = meta_scaler.fit_transform(X_meta_train)
            X_meta_test_s = meta_scaler.transform(X_meta_test)

            # Use stacking of LR + GBM as meta-learner for robustness
            meta_estimators = [
                ("lr", LogisticRegression(C=1.0, max_iter=1000, random_state=42)),
                ("gb", GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
                )),
            ]
            meta_model = VotingClassifier(
                estimators=meta_estimators, voting="soft"
            )
            meta_model.fit(X_meta_train_s, y_train)
            y_pred = meta_model.predict(X_meta_test_s)

            ensemble_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
            ensemble_metrics["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            ensemble_metrics["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            ensemble_metrics["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            if len(np.unique(y_test)) > 1:
                try:
                    y_proba = meta_model.predict_proba(X_meta_test_s)[:, 1]
                    ensemble_metrics["auc"].append(roc_auc_score(y_test, y_proba))
                except Exception:
                    ensemble_metrics["auc"].append(np.nan)
            else:
                ensemble_metrics["auc"].append(np.nan)

        mean_m = {k: round(float(np.nanmean(v)), 4) for k, v in ensemble_metrics.items()}
        std_m = {k: round(float(np.nanstd(v)), 4) for k, v in ensemble_metrics.items()}
        per_view = {}
        for v_idx, name in enumerate(self.view_names):
            per_view[name] = {k: round(float(np.nanmean(v)), 4)
                             for k, v in view_fold_metrics[v_idx].items()}

        return mean_m, std_m, per_view

    def evaluate_concatenated(
        self,
        X_all: np.ndarray,
        y: np.ndarray,
        n_folds: int = 10,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate with all features concatenated (stacking baseline)."""
        minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
        actual_folds = max(2, min(n_folds, minority))
        skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

        estimators = [
            ("rf", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", RandomForestClassifier(
                    n_estimators=300, max_depth=None,
                    class_weight="balanced", random_state=42, n_jobs=-1,
                )),
            ])),
            ("gb", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", GradientBoostingClassifier(
                    n_estimators=300, max_depth=5, learning_rate=0.1,
                    subsample=0.8, random_state=42,
                )),
            ])),
            ("mlp", Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(128, 64), max_iter=800,
                    activation="relu", random_state=42,
                    early_stopping=True, learning_rate="adaptive",
                )),
            ])),
        ]

        stacker = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=42),
            cv=3, stack_method="predict_proba", n_jobs=-1,
        )

        metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}

        for train_idx, test_idx in skf.split(X_all, y):
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            stacker.fit(X_tr, y_tr)
            y_pred = stacker.predict(X_te)

            metrics["accuracy"].append(accuracy_score(y_te, y_pred))
            metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
            if len(np.unique(y_te)) > 1:
                try:
                    y_proba = stacker.predict_proba(X_te)[:, 1]
                    metrics["auc"].append(roc_auc_score(y_te, y_proba))
                except Exception:
                    metrics["auc"].append(np.nan)
            else:
                metrics["auc"].append(np.nan)

        mean_m = {k: round(float(np.nanmean(v)), 4) for k, v in metrics.items()}
        std_m = {k: round(float(np.nanstd(v)), 4) for k, v in metrics.items()}
        return mean_m, std_m


# ═══════════════════════════════════════════════════════════════
# Main Experiment Runner
# ═══════════════════════════════════════════════════════════════

def run_enhanced_experiments():
    """Run the full enhanced ITG experiment suite."""
    print("=" * 80)
    print("Enhanced ITG (EITG) Multi-View Phishing Detection")
    print("=" * 80)

    # 1. Load ALL traces
    traces, y = load_all_traces()
    print(f"\nDataset: {len(y)} traces ({int(np.sum(y == 1))} phishing, {int(np.sum(y == 0))} benign)")

    # 2. Extract all views
    print("\nExtracting feature views...")
    X_url, X_net, X_redir, X_interact, X_itg, X_cross, itg_mask = build_all_views(traces)
    print(f"  URL Structure:     {X_url.shape[1]}D")
    print(f"  Network Traffic:   {X_net.shape[1]}D")
    print(f"  Redirect Chain:    {X_redir.shape[1]}D")
    print(f"  Interaction Events:{X_interact.shape[1]}D")
    print(f"  ITG Graph:         {X_itg.shape[1]}D (available: {int(itg_mask.sum())}/{len(itg_mask)})")
    print(f"  Cross-View:        {X_cross.shape[1]}D")

    # Engineer ITG features
    X_itg_eng = engineer_itg_features(X_itg)
    print(f"  ITG + Engineered:  {X_itg_eng.shape[1]}D")

    total_dim = X_url.shape[1] + X_net.shape[1] + X_redir.shape[1] + X_interact.shape[1] + X_itg_eng.shape[1] + X_cross.shape[1]
    print(f"\n  Total feature dimensionality: {total_dim}D")

    detector = EnhancedITGDetector()

    # ═══════════════════════════════════════════
    # A. Enhanced ITG Multi-View Ensemble (our method)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("A. Enhanced ITG Multi-View Ensemble (EITG-MVE)")
    print("=" * 70)

    X_views = [X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross]
    # Weights: URL gets highest (most discriminative), ITG+eng next, net, etc.
    view_weights = [2.5, 1.5, 1.0, 1.0, 1.5, 1.0]

    eitg_results, eitg_std, per_view = detector.evaluate_multiview(
        X_views, y, view_weights=view_weights
    )

    print(f"\n  EITG-MVE (Soft Voting) Results:")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"    {metric.upper():>10}: {eitg_results[metric]:.4f} (+/- {eitg_std[metric]:.4f})")

    print(f"\n  Per-View Performance:")
    for name, metrics in per_view.items():
        print(f"    {name:<22} Acc={metrics['accuracy']:.4f}  F1={metrics['f1']:.4f}  AUC={metrics['auc']:.4f}")

    # ═══════════════════════════════════════════
    # A2. Stacked Multi-View Ensemble (our best method)
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("A2. Stacked Multi-View Ensemble (EITG-SMVE)")
    print("   Level 0: Per-view specialized classifiers")
    print("   Level 1: Meta-learner (LR + GBM) on view probabilities + URL features")
    print("=" * 70)

    smve_results, smve_std, smve_per_view = detector.evaluate_stacked_multiview(
        X_views, y
    )

    print(f"\n  EITG-SMVE Results:")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"    {metric.upper():>10}: {smve_results[metric]:.4f} (+/- {smve_std[metric]:.4f})")

    # ═══════════════════════════════════════════
    # B. EITG Concatenated Stacking
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("B. EITG Concatenated Stacking")
    print("=" * 70)

    X_all = np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross])
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"  Full feature matrix: {X_all.shape[1]}D")

    stack_results, stack_std = detector.evaluate_concatenated(X_all, y)

    print(f"\n  EITG-Stacked Results:")
    for metric in ["accuracy", "precision", "recall", "f1", "auc"]:
        print(f"    {metric.upper():>10}: {stack_results[metric]:.4f} (+/- {stack_std[metric]:.4f})")

    # ═══════════════════════════════════════════
    # C. Ablation: Individual views
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("C. View Ablation Study")
    print("=" * 70)

    ablation_results = {}
    view_data = [
        ("URL-only", X_url, "GBM"),
        ("Network-only", X_net, "RF"),
        ("Redirect-only", X_redir, "GBM"),
        ("Interaction-only", X_interact, "RF"),
        ("ITG-only", X_itg_eng, "RF"),
        ("CrossView-only", X_cross, "GBM"),
    ]

    minority = min(int(np.sum(y == 0)), int(np.sum(y == 1)))
    actual_folds = max(2, min(10, minority))
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=42)

    for name, X_v, clf_type in view_data:
        metrics = {k: [] for k in ["accuracy", "precision", "recall", "f1", "auc"]}
        for train_idx, test_idx in skf.split(X_v, y):
            X_tr, X_te = X_v[train_idx], X_v[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            if clf_type == "RF":
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", RandomForestClassifier(
                        n_estimators=300, max_depth=None,
                        class_weight="balanced", random_state=42, n_jobs=-1,
                    )),
                ])
            else:
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf", GradientBoostingClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.1,
                        subsample=0.8, random_state=42,
                    )),
                ])

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_te)

            metrics["accuracy"].append(accuracy_score(y_te, y_pred))
            metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
            metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
            metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
            if len(np.unique(y_te)) > 1:
                try:
                    proba = pipe.predict_proba(X_te)[:, 1]
                    metrics["auc"].append(roc_auc_score(y_te, proba))
                except Exception:
                    metrics["auc"].append(np.nan)
            else:
                metrics["auc"].append(np.nan)

        ablation_results[name] = {k: round(float(np.nanmean(v)), 4) for k, v in metrics.items()}

    print(f"\n  {'View':<22} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "-" * 60)
    for name, m in ablation_results.items():
        print(f"  {name:<22} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['auc']:>7.4f}")

    # ═══════════════════════════════════════════
    # D. Comparison with independent baselines
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("D. Comparison with Independent Baselines")
    print("=" * 70)

    from baselines.real_baselines import run_real_baselines
    baseline_results = run_real_baselines(traces, y)

    # ═══════════════════════════════════════════
    # E. Summary Table
    # ═══════════════════════════════════════════
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON TABLE")
    print("=" * 80)

    all_results = {}
    all_results["EITG-SMVE (Ours)"] = smve_results
    all_results["EITG-MVE (Ours)"] = eitg_results
    all_results["EITG-Stacked (Ours)"] = stack_results
    for name, m in ablation_results.items():
        all_results[f"EITG-{name}"] = m
    for name, m in baseline_results.items():
        # Only keep the core metrics
        all_results[name] = {k: m[k] for k in ["accuracy", "precision", "recall", "f1", "auc"]}

    print(f"\n  {'Method':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7}")
    print("  " + "-" * 72)

    for method, metrics in sorted(all_results.items(), key=lambda x: -x[1].get("f1", 0)):
        marker = " ***" if "Ours" in method else ""
        print(f"  {method:<30} {metrics['accuracy']:>7.4f} {metrics['precision']:>7.4f} "
              f"{metrics['recall']:>7.4f} {metrics['f1']:>7.4f} {metrics['auc']:>7.4f}{marker}")

    # ═══════════════════════════════════════════
    # F. Improvement analysis
    # ═══════════════════════════════════════════
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)

    non_ours = {k: v for k, v in all_results.items() if "Ours" not in k and "EITG-" not in k}
    if non_ours:
        best_name = max(non_ours.items(), key=lambda x: x[1].get("f1", 0))[0]
        best_f1 = non_ours[best_name]["f1"]
        our_f1 = smve_results["f1"]

        print(f"\n  Best baseline:  {best_name} (F1={best_f1:.4f})")
        print(f"  Our method:     EITG-SMVE (F1={our_f1:.4f})")
        improvement = ((our_f1 - best_f1) / max(best_f1, 1e-8)) * 100
        print(f"  Improvement:    {improvement:+.2f}%")

        if smve_results["f1"] < best_f1:
            print("\n  [!] WARNING: Our method underperforms the best baseline!")
            print("  → Consider adjusting view weights or ensemble architecture")

    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    from datetime import datetime
    results_data = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "dataset_size": int(len(y)),
        "phishing_count": int(np.sum(y == 1)),
        "benign_count": int(np.sum(y == 0)),
        "itg_available": int(itg_mask.sum()),
        "eitg_smve": smve_results,
        "eitg_smve_std": smve_std,
        "eitg_mve": eitg_results,
        "eitg_mve_std": eitg_std,
        "eitg_stacked": stack_results,
        "eitg_stacked_std": stack_std,
        "per_view": per_view,
        "ablation": ablation_results,
        "baselines": {k: {kk: vv for kk, vv in v.items()
                         if kk in ["accuracy", "precision", "recall", "f1", "auc"]}
                     for k, v in baseline_results.items()},
    }

    with open(output_dir / "eitg_results.json", "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\nResults saved to {output_dir / 'eitg_results.json'}")

    return eitg_results, all_results


if __name__ == "__main__":
    run_enhanced_experiments()
