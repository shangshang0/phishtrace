#!/usr/bin/env python3
"""
PhishTrace — URLScan Feed Live Crawl Pipeline
===============================================
Reuses the proven crawl→classify pipeline from serp_crawl_all.py,
but sources URLs from URLScan.io's latest-scans feed instead of SERP.

Output structure:
  dataset/discovery_live/
    urlscan_feed/
      phishing/{hash}/trace.json + shot.png + screenshots/
      benign/{hash}/trace.json + shot.png + screenshots/
    typosquat/
      phishing/{hash}/...
      benign/{hash}/...

Usage:
  py -3 scripts/urlscan_live_crawl.py --hours 12
  py -3 scripts/urlscan_live_crawl.py --hours 1 --cycle-interval 120
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import random
import re
import shutil
import socket
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import tldextract
import urllib3
urllib3.disable_warnings()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from experiments.enhanced_itg_detector import (
    load_all_traces, extract_url_view, extract_network_view,
    extract_redirect_view, extract_interaction_view, extract_itg_view,
    compute_cross_view_features, engineer_itg_features, build_all_views,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
LIVE_DIR = PROJECT_ROOT / "dataset" / "discovery_live"
CHECKPOINT_FILE = LIVE_DIR / "checkpoint.json"
PROXY_URL = "http://127.0.0.1:10809"
PROXY_DICT = {"http": PROXY_URL, "https": PROXY_URL}
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/122.0.0.0 Safari/537.36"
    )
}

logger = logging.getLogger("urlscan_live")

# ── Filtering constants ──
BRANDS_KW = {
    'paypal', 'apple', 'microsoft', 'google', 'amazon', 'netflix',
    'chase', 'facebook', 'instagram', 'coinbase', 'binance', 'dhl',
    'usps', 'wellsfargo', 'dropbox', 'linkedin', 'docusign', 'fedex',
    'outlook', 'office365', 'gmail', 'icloud', 'whatsapp', 'telegram',
    'steam', 'roblox', 'stripe', 'venmo', 'adobe', 'twitter',
}
SUSP_TLDS = {
    'xyz', 'top', 'club', 'online', 'site', 'info', 'click', 'buzz',
    'icu', 'cam', 'rest', 'monster', 'cfd', 'sbs', 'gq', 'cf', 'ga',
    'ml', 'tk', 'pw', 'cc', 'app', 'dev', 'io',
}
PHISH_KW = {
    'login', 'signin', 'verify', 'secure', 'account', 'update', 'unlock',
    'billing', 'payment', 'wallet', 'password', 'reset', 'suspended',
    'confirm', 'auth', 'bank', 'webmail', 'recover',
}
# Known legitimate domains to SKIP
WHITELIST_DOMAINS = {
    'google.com', 'microsoft.com', 'apple.com', 'amazon.com', 'facebook.com',
    'netflix.com', 'paypal.com', 'github.com', 'linkedin.com', 'twitter.com',
    'instagram.com', 'youtube.com', 'wikipedia.org', 'reddit.com',
    'stackoverflow.com', 'cloudflare.com', 'akamai.com', 'fastly.net',
    'login.microsoftonline.com', 'accounts.google.com', 'appleid.apple.com',
    'outlook.live.com', 'live.com', 'office.com', 'office365.com',
    'googleapis.com', 'gstatic.com', 'microsoftonline.com',
}
LEGIT_SUFFIXES = {'.gov', '.edu', '.mil', '.gov.uk', '.ac.uk'}

# Typosquatting targets
TYPO_BRANDS = {
    "paypal": ["paypal.com"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "outlook.com"],
    "google": ["google.com", "gmail.com"],
    "amazon": ["amazon.com"],
    "netflix": ["netflix.com"],
    "chase": ["chase.com"],
    "facebook": ["facebook.com"],
}
TYPO_TLDS = ['com', 'net', 'org', 'xyz', 'top', 'app', 'site', 'online', 'info', 'club']
TYPO_PREFIXES = ['login-', 'secure-', 'verify-', 'account-', 'my-', 'web-', 'signin-']
TYPO_SUFFIXES = ['-login', '-secure', '-verify', '-account', '-support', '-help']


# ═══════════════════════════════════════════════════════════════════════════════
# ML Model (same as serp_crawl_all)
# ═══════════════════════════════════════════════════════════════════════════════

def train_detector():
    """Train the detector on existing dataset (114D features, RF 500 trees)."""
    logger.info("Training ML detector on existing dataset...")
    traces, y = load_all_traces()
    logger.info(f"  Training data: {len(traces)} traces ({sum(y==1)} phish, {sum(y==0)} benign)")
    X_url, X_net, X_redir, X_interact, X_itg, X_cross, itg_mask = build_all_views(traces)
    X_itg_eng = engineer_itg_features(X_itg)
    X_full = np.hstack([X_url, X_net, X_redir, X_interact, X_itg_eng, X_cross])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_full)
    model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
    model.fit(X_scaled, y)
    logger.info(f"  Model trained: {X_full.shape[1]}D features")
    return model, scaler


def extract_features_from_trace(trace_data):
    """Extract 114D feature vector from a single trace."""
    u = extract_url_view(trace_data)
    n = extract_network_view(trace_data)
    r = extract_redirect_view(trace_data)
    inter = extract_interaction_view(trace_data)
    itg = extract_itg_view(trace_data)
    if itg is None:
        itg = np.zeros(30, dtype=float)
    itg_eng = engineer_itg_features(itg.reshape(1, -1))[0]
    cross = compute_cross_view_features(u, n, r, inter)
    return np.concatenate([u, n, r, inter, itg_eng, cross])


def classify_trace(model, scaler, trace_data):
    """Classify a trace as phishing or benign."""
    features = extract_features_from_trace(trace_data)
    X = scaler.transform(features.reshape(1, -1))
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    return {
        "prediction": int(pred),
        "label": "phishing" if pred == 1 else "benign",
        "confidence": round(float(max(prob)), 4),
        "prob_phishing": round(float(prob[1]), 4),
        "prob_benign": round(float(prob[0]), 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# URL Sources
# ═══════════════════════════════════════════════════════════════════════════════

def is_interesting(domain: str) -> bool:
    """Check if a domain from URLScan feed is worth crawling."""
    dl = domain.lower()
    ext = tldextract.extract(dl)
    sld = ext.domain or ''
    subdomain = ext.subdomain or ''
    suffix = ext.suffix or ''
    rd = (getattr(ext, 'top_domain_under_public_suffix', None)
          or ext.registered_domain or '').lower()
    full = f'{subdomain}.{sld}'.lower()

    # Skip known legitimate
    if rd in WHITELIST_DOMAINS or dl in WHITELIST_DOMAINS:
        return False
    for ls in LEGIT_SUFFIXES:
        if ('.' + suffix).endswith(ls):
            return False
    # Skip enterprise infrastructure patterns
    if 'sslproxy.gateway' in dl or '.internal.' in dl:
        return False

    has_brand = any(b in full for b in BRANDS_KW)
    has_susp_tld = suffix in SUSP_TLDS
    has_phish_kw = any(k in full for k in PHISH_KW)
    long_sub = len(subdomain) > 30
    many_hyphens = sld.count('-') >= 3

    return has_brand or (has_susp_tld and has_phish_kw) or long_sub or many_hyphens


def fetch_urlscan_feed(seen_domains: Set[str],
                       time_range: str = "now-2h",
                       max_pages: int = 3) -> List[Dict]:
    """
    Fetch latest scans from URLScan.io feed (NOT phishing feed).
    Uses `date:>now-Xh` to get recently scanned sites.
    Returns list of {url, domain, source} for interesting domains.
    """
    candidates = []
    search_after = None

    for page in range(max_pages):
        params = {"q": f"date:>{time_range}", "size": 100}
        if search_after:
            params["search_after"] = search_after

        try:
            r = requests.get(
                "https://urlscan.io/api/v1/search/",
                params=params, headers=UA_HEADERS,
                proxies=PROXY_DICT, timeout=25, verify=False,
            )
            if r.status_code != 200:
                logger.warning(f"URLScan feed HTTP {r.status_code}")
                break

            data = r.json()
            results = data.get("results", [])
            if not results:
                break

            for hit in results:
                pg = hit.get("page", {})
                domain = pg.get("domain", "")
                url = pg.get("url", "")
                if not domain or not url or domain in seen_domains:
                    continue
                seen_domains.add(domain)

                if is_interesting(domain):
                    candidates.append({
                        "url": url,
                        "domain": domain,
                        "source": "urlscan_feed",
                    })

            # Pagination: get search_after for next page
            if data.get("has_more") and results:
                last = results[-1]
                sort_vals = last.get("sort")
                if sort_vals:
                    search_after = ",".join(str(s) for s in sort_vals)
                else:
                    break
            else:
                break

            logger.info(f"  URLScan feed page {page+1}: {len(results)} scans, "
                       f"{len(candidates)} interesting so far")
            time.sleep(3)  # Be nice to API

        except Exception as e:
            logger.error(f"URLScan feed error: {e}")
            break

    return candidates


def discover_typosquat_domains(seen_domains: Set[str]) -> List[Dict]:
    """Generate and DNS-resolve typosquatting domains."""
    candidates_to_check = []
    for brand, base_domains in TYPO_BRANDS.items():
        for tld in TYPO_TLDS:
            for prefix in TYPO_PREFIXES:
                d = f"{prefix}{brand}.{tld}"
                if d not in seen_domains:
                    candidates_to_check.append(d)
            for suffix in TYPO_SUFFIXES:
                d = f"{brand}{suffix}.{tld}"
                if d not in seen_domains:
                    candidates_to_check.append(d)

    # Concurrent DNS resolution
    alive = []
    def check_dns(domain):
        try:
            socket.setdefaulttimeout(3)
            socket.getaddrinfo(domain, 80)
            return domain
        except Exception:
            return None

    logger.info(f"  Typosquat: checking {len(candidates_to_check)} domains via DNS...")
    with ThreadPoolExecutor(max_workers=50) as pool:
        results = list(pool.map(check_dns, candidates_to_check))

    for domain in results:
        if domain:
            seen_domains.add(domain)
            alive.append({
                "url": f"https://{domain}",
                "domain": domain,
                "source": "typosquat",
            })

    logger.info(f"  Typosquat: {len(alive)} alive domains")
    return alive


# ═══════════════════════════════════════════════════════════════════════════════
# Crawling (reused from serp_crawl_all with output structure changes)
# ═══════════════════════════════════════════════════════════════════════════════

ERROR_PAGE_RE = re.compile(
    r'\b(403|404|500|502|503)\b|forbidden|not\s*found|access\s*denied|'
    r'error\s*page|server\s*error|bad\s*gateway|service\s*unavailable|'
    r'page\s*not\s*found|site\s*not\s*found|domain\s*for\s*sale|'
    r'parked\s*domain|coming\s*soon|under\s*construction',
    re.IGNORECASE,
)


async def preflight_check(url: str, proxy: str = None, timeout: int = 10) -> bool:
    """Quick HTTP check before committing to full crawl."""
    import aiohttp
    for attempt_proxy in [proxy, None]:
        try:
            conn = aiohttp.TCPConnector(ssl=False)
            async with aiohttp.ClientSession(
                connector=conn, headers=UA_HEADERS,
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as session:
                kw = {}
                if attempt_proxy:
                    kw["proxy"] = attempt_proxy
                async with session.get(url, allow_redirects=True, **kw) as resp:
                    if resp.status < 500:
                        return True
        except Exception:
            if attempt_proxy is None:
                return False
            continue
    return False


async def deep_crawl_url(url: str, output_dir: Path,
                         timeout_ms: int = 25000,
                         proxy: str = None) -> Optional[dict]:
    """Deep-crawl a URL with PhishingCrawler, save trace + screenshots."""
    from src.crawler.phishing_crawler import PhishingCrawler

    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    ss_dir = output_dir / "screenshots"

    reachable = await preflight_check(url, proxy=proxy, timeout=10)
    if not reachable:
        return None

    for attempt, use_proxy in enumerate([proxy, None]):
        crawler = PhishingCrawler(
            headless=True,
            timeout=timeout_ms,
            capture_screenshots=True,
            screenshot_dir=str(ss_dir),
            proxy=use_proxy,
        )
        try:
            trace = await asyncio.wait_for(
                crawler.crawl(url, max_depth=2),
                timeout=timeout_ms / 1000 + 20,
            )
            if not trace or not trace.events:
                if attempt == 0 and use_proxy:
                    continue
                return None

            events_ser = [
                asdict(e) if hasattr(e, '__dataclass_fields__') else
                (e.__dict__ if hasattr(e, '__dict__') else e)
                for e in trace.events
            ]
            net_reqs_ser = [
                asdict(nr) if hasattr(nr, '__dataclass_fields__') else
                (nr.__dict__ if hasattr(nr, '__dict__') else nr)
                for nr in (trace.network_requests or [])
            ]

            wrapped = {
                "url": url, "hash": url_hash,
                "label": "unknown", "success": True,
                "domain": urlparse(url).netloc,
                "crawled_at": datetime.now().isoformat(),
                "final_url": getattr(trace, "final_url", url),
                "page_title": getattr(trace, "page_title", ""),
                "forms_submitted": trace.forms_submitted,
                "dual_submissions": trace.dual_submissions_detected,
                "elements_interacted": trace.elements_interacted,
                "trace": {
                    "events": events_ser,
                    "network_requests": net_reqs_ser,
                    "redirects": getattr(trace, "redirects", []),
                    "final_url": getattr(trace, "final_url", url),
                    "page_title": getattr(trace, "page_title", ""),
                    "cookies": getattr(trace, "cookies", []),
                    "console_logs": (trace.console_logs or [])[:50],
                },
                "screenshots": [
                    e.screenshot_path for e in trace.events
                    if getattr(e, 'screenshot_path', None)
                ],
            }

            # Write trace to output_dir (will be re-saved later to verdict dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "trace.json").write_text(
                json.dumps(wrapped, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )

            # Copy first screenshot as shot.png
            try:
                screenshots = sorted(ss_dir.glob("step_*.png")) if ss_dir.exists() else []
                if screenshots:
                    shutil.copy2(screenshots[0], output_dir / "shot.png")
            except Exception:
                pass

            del trace, events_ser, net_reqs_ser
            return wrapped

        except asyncio.TimeoutError:
            if attempt == 0 and use_proxy:
                continue
            return None
        except Exception as e:
            logger.debug(f"  Crawl error [{url[:40]}]: {type(e).__name__}: {str(e)[:80]}")
            if attempt == 0 and use_proxy:
                continue
            return None

    return None


def post_classify(result: dict) -> dict:
    """Post-classification checks: error pages, content-only FPs, legit redirects."""
    classification = result.get("classification", {})
    if classification.get("prediction") != 1:
        return result

    title = result.get("page_title", "")
    n_events = result.get("n_events", 0)
    forms = result.get("forms_submitted", 0)
    n_net = result.get("n_network_requests", 0)

    # Error page
    if title and ERROR_PAGE_RE.search(title):
        classification["label"] = "error_page"
        classification["prediction"] = -2
        result["is_error_page"] = True
    elif n_events <= 2 and forms == 0 and n_net <= 3:
        classification["label"] = "error_page"
        classification["prediction"] = -2
        result["is_error_page"] = True
    # Content-only page (no forms → likely FP)
    elif forms == 0 and n_events <= 5:
        trace = result.get("trace", {})
        events = trace.get("events", []) if isinstance(trace, dict) else []
        event_types = {e.get("event_type", "") for e in events}
        if "form_submit" not in event_types and "submit" not in event_types:
            classification["label"] = "content_page_fp"
            classification["prediction"] = 0
            result["is_content_fp"] = True
    # Final URL redirected to official brand
    final_url = result.get("final_url", "")
    if final_url:
        ext = tldextract.extract(final_url)
        rd = (getattr(ext, 'top_domain_under_public_suffix', None)
              or ext.registered_domain or '').lower()
        if rd in WHITELIST_DOMAINS:
            classification["label"] = "false_positive"
            classification["prediction"] = 0
            result["validated_fp"] = True

    result["classification"] = classification
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

async def process_and_store(url_info: dict, model, scaler,
                            timeout_ms: int) -> Optional[dict]:
    """
    Crawl a single URL, classify it, and store to the correct directory:
      discovery_live/{source}/{verdict}/{hash}/trace.json + shot.png
    """
    url = url_info["url"]
    source = url_info["source"]
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

    # Temp dir for crawling
    tmp_dir = LIVE_DIR / "_tmp" / url_hash

    trace_data = await deep_crawl_url(url, tmp_dir,
                                       timeout_ms=timeout_ms,
                                       proxy=PROXY_URL)

    if trace_data is None or not trace_data.get("success"):
        # Clean up tmp
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # Add source info
    trace_data["source"] = source

    # Classify
    try:
        classification = classify_trace(model, scaler, trace_data)
    except Exception as e:
        logger.debug(f"  Classification error: {e}")
        classification = {"prediction": -1, "label": "classification_error", "confidence": 0.0}

    trace = trace_data.get("trace", {})
    result = {
        "url": url,
        "url_hash": url_hash,
        "source": source,
        "domain": url_info.get("domain", urlparse(url).netloc),
        "crawl_success": True,
        "classification": classification,
        "final_url": trace_data.get("final_url", url),
        "page_title": trace.get("page_title", ""),
        "n_events": len(trace.get("events", [])),
        "n_network_requests": len(trace.get("network_requests", [])),
        "n_redirects": len(trace.get("redirects", [])),
        "forms_submitted": trace_data.get("forms_submitted", 0),
        "dual_submissions": trace_data.get("dual_submissions", 0),
        "elements_interacted": trace_data.get("elements_interacted", 0),
        "timestamp": datetime.now().isoformat(),
        "trace": trace,
    }

    # Post-classification checks
    result = post_classify(result)

    # Determine verdict directory
    pred = result["classification"].get("prediction", -1)
    if pred == 1:
        verdict = "phishing"
    elif pred == 0:
        verdict = "benign"
    else:
        verdict = "other"  # error pages, classification errors

    # Move to final directory: discovery_live/{source}/{verdict}/{hash}/
    final_dir = LIVE_DIR / source / verdict / url_hash
    final_dir.mkdir(parents=True, exist_ok=True)

    # Save trace.json with classification info appended
    trace_data["classification"] = result["classification"]
    trace_data["verdict"] = verdict
    (final_dir / "trace.json").write_text(
        json.dumps(trace_data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )

    # Move screenshots
    tmp_ss = tmp_dir / "screenshots"
    final_ss = final_dir / "screenshots"
    if tmp_ss.exists():
        if final_ss.exists():
            shutil.rmtree(final_ss, ignore_errors=True)
        shutil.move(str(tmp_ss), str(final_ss))

    # Move shot.png
    tmp_shot = tmp_dir / "shot.png"
    if tmp_shot.exists():
        shutil.move(str(tmp_shot), str(final_dir / "shot.png"))

    # Clean up tmp
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def load_checkpoint() -> Tuple[Set[str], Set[str], dict]:
    """Load checkpoint: crawled hashes, seen domains, stats."""
    if not CHECKPOINT_FILE.exists():
        return set(), set(), {"phishing": 0, "benign": 0, "other": 0, "failed": 0}
    try:
        data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
        return (
            set(data.get("crawled_hashes", [])),
            set(data.get("seen_domains", [])),
            data.get("stats", {"phishing": 0, "benign": 0, "other": 0, "failed": 0}),
        )
    except Exception:
        return set(), set(), {"phishing": 0, "benign": 0, "other": 0, "failed": 0}


def save_checkpoint(crawled: Set[str], seen: Set[str], stats: dict):
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "crawled_hashes": list(crawled),
        "seen_domains": list(seen),
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }
    CHECKPOINT_FILE.write_text(
        json.dumps(data, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )


async def main_async(args):
    logger.info("=" * 60)
    logger.info("PhishTrace — URLScan Feed Live Crawl Pipeline")
    logger.info(f"  Duration: {args.hours}h | Cycle interval: {args.cycle_interval}s")
    logger.info(f"  Output: {LIVE_DIR}")
    logger.info("=" * 60)

    # Load checkpoint
    crawled_hashes, seen_domains, stats = load_checkpoint()
    logger.info(f"  Checkpoint: {len(crawled_hashes)} already crawled, "
               f"{len(seen_domains)} seen domains")

    # Train model once
    logger.info("\nTraining ML detector...")
    model, scaler = train_detector()

    deadline = time.time() + args.hours * 3600
    cycle = 0

    while time.time() < deadline:
        cycle += 1
        remaining = (deadline - time.time()) / 3600
        logger.info(f"\n{'='*40} CYCLE {cycle} ({remaining:.1f}h remaining) {'='*40}")

        # ── Discover new URLs ──
        candidates = []

        # Source 1: URLScan feed
        logger.info("\n[URLScan Feed] Fetching latest scans...")
        feed_urls = fetch_urlscan_feed(
            seen_domains, time_range="now-2h", max_pages=3,
        )
        candidates.extend(feed_urls)
        logger.info(f"  → {len(feed_urls)} interesting URLs from URLScan feed")

        # Source 2: Typosquatting (every 5th cycle to avoid repetition)
        if cycle % 5 == 1:
            logger.info("\n[Typosquat] Checking typosquatting domains...")
            typo_urls = discover_typosquat_domains(seen_domains)
            candidates.extend(typo_urls)
            logger.info(f"  → {len(typo_urls)} alive typosquatting domains")

        # Dedup against already crawled
        new_candidates = []
        for c in candidates:
            h = hashlib.md5(c["url"].encode()).hexdigest()[:12]
            if h not in crawled_hashes:
                new_candidates.append(c)

        if args.max_per_cycle and len(new_candidates) > args.max_per_cycle:
            new_candidates = new_candidates[:args.max_per_cycle]

        logger.info(f"\n[CRAWL] {len(new_candidates)} new URLs to process")

        if not new_candidates:
            logger.info("  No new candidates. Sleeping...")
            time.sleep(args.cycle_interval)
            continue

        # ── Crawl and classify each URL ──
        for i, url_info in enumerate(new_candidates):
            if time.time() >= deadline:
                logger.info("  ⏰ Time's up!")
                break

            url = url_info["url"]
            url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

            try:
                result = await process_and_store(url_info, model, scaler, args.timeout)
            except Exception as e:
                logger.debug(f"  Error [{url[:40]}]: {e}")
                result = None

            crawled_hashes.add(url_hash)

            if result is None:
                stats["failed"] = stats.get("failed", 0) + 1
                if (i + 1) % 5 == 0:
                    logger.info(f"  [{i+1}/{len(new_candidates)}] FAIL")
            else:
                verdict = result["classification"].get("label", "?")
                pred = result["classification"].get("prediction", -1)
                conf = result["classification"].get("confidence", 0)

                if pred == 1:
                    stats["phishing"] = stats.get("phishing", 0) + 1
                    logger.info(
                        f"  ★ PHISHING [{i+1}/{len(new_candidates)}] "
                        f"conf={conf:.3f} forms={result.get('forms_submitted',0)} "
                        f"src={url_info['source']} "
                        f"[total phish: {stats['phishing']}]"
                    )
                    logger.info(f"    → {url[:100]}")
                elif pred == 0:
                    stats["benign"] = stats.get("benign", 0) + 1
                    if (i + 1) % 5 == 0:
                        logger.info(f"  [{i+1}/{len(new_candidates)}] {verdict}")
                else:
                    stats["other"] = stats.get("other", 0) + 1

            # Checkpoint every 10
            if (i + 1) % 10 == 0:
                save_checkpoint(crawled_hashes, seen_domains, stats)
                gc.collect()

            # Rate limiting: 3-8s between crawls
            await asyncio.sleep(random.uniform(3, 8))

        # End-of-cycle checkpoint
        save_checkpoint(crawled_hashes, seen_domains, stats)

        logger.info(f"\n[Cycle {cycle} done] "
                   f"phishing={stats.get('phishing',0)} "
                   f"benign={stats.get('benign',0)} "
                   f"other={stats.get('other',0)} "
                   f"failed={stats.get('failed',0)} "
                   f"total_crawled={len(crawled_hashes)}")

        # Sleep until next cycle
        if time.time() < deadline:
            logger.info(f"  Sleeping {args.cycle_interval}s until next cycle...")
            time.sleep(args.cycle_interval)

    # ── Final summary ──
    logger.info(f"\n{'='*60}")
    logger.info("FINAL SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"  Cycles completed:  {cycle}")
    logger.info(f"  Total crawled:     {len(crawled_hashes)}")
    logger.info(f"  Phishing found:    {stats.get('phishing', 0)}")
    logger.info(f"  Benign:            {stats.get('benign', 0)}")
    logger.info(f"  Other:             {stats.get('other', 0)}")
    logger.info(f"  Failed:            {stats.get('failed', 0)}")

    # List what we found
    for source_dir in LIVE_DIR.iterdir():
        if source_dir.is_dir() and source_dir.name not in ('_tmp',):
            phish_dir = source_dir / "phishing"
            benign_dir = source_dir / "benign"
            n_phish = len(list(phish_dir.iterdir())) if phish_dir.exists() else 0
            n_benign = len(list(benign_dir.iterdir())) if benign_dir.exists() else 0
            if n_phish or n_benign:
                logger.info(f"  [{source_dir.name}] {n_phish} phishing / {n_benign} benign")


def main():
    parser = argparse.ArgumentParser(
        description="PhishTrace — URLScan Feed Live Crawl Pipeline"
    )
    parser.add_argument("--hours", type=float, default=12,
                       help="How many hours to run (default: 12)")
    parser.add_argument("--cycle-interval", type=int, default=300,
                       help="Seconds between discovery cycles (default: 300)")
    parser.add_argument("--timeout", type=int, default=25000,
                       help="Crawl timeout in ms (default: 25000)")
    parser.add_argument("--max-per-cycle", type=int, default=30,
                       help="Max URLs per cycle (default: 30)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LIVE_DIR / "live_crawl.log"
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    import warnings
    warnings.filterwarnings("ignore")

    logger.info(f"Log file: {log_file}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
