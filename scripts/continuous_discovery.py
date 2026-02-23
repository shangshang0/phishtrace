#!/usr/bin/env python3
"""
PhishTrace — Continuous Discovery & Classification Daemon
============================================================
Runs continuously for up to 12 hours, periodically discovering
new suspicious URLs from multiple sources, crawling them, and
classifying using our multi-view ensemble detector.

Results stored in:
  dataset/discovery_live/
    urlscan/phishing/{hash}/  — trace.json, shot.png, screenshots/
    urlscan/benign/{hash}/
    typosquat/phishing/{hash}/
    typosquat/benign/{hash}/
    summary.json              — running summary of all results

Usage:
    py -3 scripts/continuous_discovery.py          # run 12h (background)
    py -3 scripts/continuous_discovery.py --hours 1  # run 1 hour (testing)
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import shutil
import socket
import sys
import time
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import tldextract

# ── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

LIVE_DIR = PROJECT_ROOT / "dataset" / "discovery_live"
LOG_FILE = LIVE_DIR / "daemon.log"
SUMMARY_FILE = LIVE_DIR / "summary.json"
PROXY = "http://127.0.0.1:10809"
PROXIES = {"https": PROXY, "http": PROXY}

# ── Logging ─────────────────────────────────────────────────────────────────
LIVE_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
    ],
)
logger = logging.getLogger("daemon")

# ── Brand Intelligence (from ct_domain_discovery.py) ────────────────────────
BRANDS = {
    'paypal':        {'official': {'paypal.com','paypalinc.com','braintreepayments.com','hyperwallet.com','paypal.me'},
                      'keywords': ['paypal','paypa1','paipal']},
    'apple':         {'official': {'apple.com','icloud.com','cdn-apple.com','apple.news'},
                      'keywords': ['apple','app1e','icloud']},
    'microsoft':     {'official': {'microsoft.com','live.com','outlook.com','office.com','hotmail.com',
                                    'azure.com','msft.net','onedrive.com','sharepoint.com','github.com'},
                      'keywords': ['microsoft','office365','outlook']},
    'google':        {'official': {'google.com','googleapis.com','gstatic.com','youtube.com','gmail.com'},
                      'keywords': ['google','gmail']},
    'amazon':        {'official': {'amazon.com','amazonaws.com','amazontrust.com','amzn.com','amazon.co.uk'},
                      'keywords': ['amazon','amaz0n']},
    'netflix':       {'official': {'netflix.com','nflxext.com','nflximg.net'},
                      'keywords': ['netflix','netfIix']},
    'chase':         {'official': {'chase.com','jpmorganchase.com'},
                      'keywords': ['chase']},
    'facebook':      {'official': {'facebook.com','fb.com','fbcdn.net','meta.com','instagram.com'},
                      'keywords': ['facebook','faceb00k']},
    'instagram':     {'official': {'instagram.com','cdninstagram.com'},
                      'keywords': ['instagram','lnstagram']},
    'coinbase':      {'official': {'coinbase.com'},
                      'keywords': ['coinbase']},
    'binance':       {'official': {'binance.com','binance.org'},
                      'keywords': ['binance']},
    'dhl':           {'official': {'dhl.com','dhl.de'},
                      'keywords': ['dhl']},
    'usps':          {'official': {'usps.com'},
                      'keywords': ['usps']},
    'wellsfargo':    {'official': {'wellsfargo.com','wf.com'},
                      'keywords': ['wellsfargo','wells-fargo']},
    'bankofamerica': {'official': {'bankofamerica.com','bofa.com'},
                      'keywords': ['bankofamerica','bofa']},
}

PHISH_INDICATORS = [
    'login','signin','sign-in','logon','verify','verification','validate',
    'confirm','secure','security','account','update','recover','unlock',
    'support','billing','payment','wallet','password','reset','alert',
    'suspended','limited','restricted','reward','giveaway',
]

SUSPICIOUS_TLDS = {
    'xyz','top','club','online','site','info','click','link','buzz','icu',
    'cam','rest','monster','cfd','sbs','quest','gq','cf','ga','ml','tk',
    'pw','cc','ws','ru','cn','su',
}


# ═══════════════════════════════════════════════════════════════════════════════
# Domain Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_domain(domain: str) -> dict:
    """Score a domain for phishing suspiciousness."""
    if not domain:
        return {'score': 0, 'reasons': [], 'brand': None}
    dl = domain.lower().replace('*.', '').strip()
    if not dl:
        return {'score': 0, 'reasons': [], 'brand': None}

    ext = tldextract.extract(dl)
    registered = getattr(ext, 'top_domain_under_public_suffix', None) or ext.registered_domain or ''
    subdomain = ext.subdomain or ''
    suffix = ext.suffix or ''
    sld = ext.domain or ''

    score = 0
    reasons = []
    matched_brand = None

    for brand_name, brand_info in BRANDS.items():
        for kw in brand_info['keywords']:
            if kw in sld or kw in subdomain:
                if any(registered.endswith(od) or registered == od
                       for od in brand_info['official']):
                    return {'score': -1, 'reasons': ['official_domain'], 'brand': brand_name}
                matched_brand = brand_name
                score += 30
                reasons.append(f'brand_{brand_name}')
                break
        if matched_brand:
            break

    phish_hits = [p for p in PHISH_INDICATORS if p in dl]
    if phish_hits:
        score += 15 * min(len(phish_hits), 3)
        reasons.append(f'phish_kw:{",".join(phish_hits[:3])}')

    if suffix in SUSPICIOUS_TLDS:
        score += 20
        reasons.append(f'susp_tld:{suffix}')

    if len(sld) > 20:
        score += 10
        reasons.append('long_sld')

    hyphen_count = sld.count('-')
    if hyphen_count >= 2:
        score += 10 * min(hyphen_count, 3)
        reasons.append(f'hyphens:{hyphen_count}')

    subdomain_parts = [p for p in subdomain.split('.') if p]
    if len(subdomain_parts) >= 2:
        score += 5 * len(subdomain_parts)
        reasons.append(f'deep_sub:{len(subdomain_parts)}')

    return {'score': score, 'reasons': reasons, 'brand': matched_brand}


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: URLScan.io — latest public scans
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_urlscan_latest(seen_domains: Set[str], batch_brands: list = None) -> List[dict]:
    """Fetch latest URLScan.io results for brand-impersonating domains.
    Rotates through brands across calls to distribute rate limit."""
    if batch_brands is None:
        batch_brands = list(BRANDS.keys())[:4]  # 4 brands per cycle

    results = []
    for brand_name in batch_brands:
        brand_info = BRANDS[brand_name]
        keyword = brand_info['keywords'][0]
        official = brand_info['official']

        excl = ' '.join(f'AND NOT domain:{od}' for od in list(official)[:3])
        q = f'domain:{keyword} {excl}'

        try:
            r = requests.get(
                'https://urlscan.io/api/v1/search/',
                params={'q': q, 'size': 100},
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                proxies=PROXIES, timeout=25,
            )
            if r.status_code == 200:
                hits = r.json().get('results', [])
                for hit in hits:
                    page = hit.get('page', {})
                    url = page.get('url', '')
                    domain = page.get('domain', '')
                    if not url or not domain or domain in seen_domains:
                        continue

                    if any(domain.endswith(od) or domain == od for od in official):
                        continue

                    analysis = score_domain(domain)
                    ds = analysis['score']
                    verdicts = hit.get('verdicts', {}).get('overall', {})
                    if verdicts.get('malicious'):
                        ds += 40
                    ds += 10  # bonus for being in URLScan

                    if ds > 0:
                        results.append({
                            'url': url,
                            'domain': domain,
                            'source': 'urlscan',
                            'brand': brand_name,
                            'score': ds,
                            'reasons': analysis.get('reasons', []),
                            'malicious_flag': verdicts.get('malicious', False),
                        })
                logger.info(f"  URLScan {keyword}: {len(hits)} scans")
            elif r.status_code in (403, 429):
                logger.warning(f"  URLScan {keyword}: rate limited ({r.status_code})")
            else:
                logger.warning(f"  URLScan {keyword}: HTTP {r.status_code}")
        except Exception as e:
            logger.warning(f"  URLScan {keyword}: {type(e).__name__}")

        time.sleep(8)  # Conservative rate limit

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: Typosquatting DNS check
# ═══════════════════════════════════════════════════════════════════════════════

def check_typosquat_batch(seen_domains: Set[str], batch_brands: list = None) -> List[dict]:
    """Check a batch of combosquatting domains via concurrent DNS."""
    if batch_brands is None:
        batch_brands = list(BRANDS.keys())[:3]

    combos = []
    top_indicators = ['login', 'signin', 'verify', 'secure', 'account', 'update', 'wallet']
    top_tlds = ['xyz', 'top', 'club', 'online', 'site', 'info', 'app', 'icu']

    for brand_name in batch_brands:
        if brand_name not in BRANDS:
            continue
        kw = BRANDS[brand_name]['keywords'][0]
        for ind in top_indicators:
            for tld in top_tlds:
                d1 = f'{kw}-{ind}.{tld}'
                d2 = f'{ind}-{kw}.{tld}'
                d3 = f'{kw}{ind}.{tld}'
                for d in (d1, d2, d3):
                    if d not in seen_domains:
                        combos.append((d, brand_name))

    if not combos:
        return []

    def check_domain(domain_brand):
        domain, brand_name = domain_brand
        try:
            socket.setdefaulttimeout(2)
            ip = socket.gethostbyname(domain)
            return (domain, brand_name, ip)
        except (socket.gaierror, socket.timeout, OSError):
            return None

    resolved = []
    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(check_domain, c): c for c in combos}
        for future in as_completed(futures):
            result = future.result()
            if result:
                resolved.append(result)

    results = []
    for domain, brand_name, ip in resolved:
        if domain in seen_domains:
            continue
        analysis = score_domain(domain)
        results.append({
            'url': f'https://{domain}',
            'domain': domain,
            'source': 'typosquat',
            'brand': brand_name,
            'score': analysis['score'],
            'reasons': analysis['reasons'],
            'ip': ip,
        })
        logger.info(f"  Typosquat RESOLVED: {domain} → {ip} (score={analysis['score']})")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Model Training (once at startup)
# ═══════════════════════════════════════════════════════════════════════════════

def train_model():
    """Train the multi-view ensemble on existing intersection dataset.
    Uses a simple concatenated RF for speed (not full SMVE which needs CV)."""
    from experiments.enhanced_itg_detector import (
        load_all_traces, build_all_views,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    logger.info("  Loading training data...")
    traces, labels = load_all_traces()
    logger.info(f"  {len(traces)} traces loaded ({sum(labels)} phishing, {len(labels)-sum(labels)} benign)")

    logger.info("  Extracting features...")
    views = build_all_views(traces)
    # views = (X_url, X_net, X_redir, X_interact, X_itg, X_cross, itg_mask)
    X = np.hstack(views[:6])  # concatenate all 6 views → 113D
    y = labels

    logger.info(f"  Feature matrix: {X.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(
        n_estimators=500, max_depth=None,
        min_samples_split=5, random_state=42, n_jobs=-1,
    )
    model.fit(X_scaled, y)
    logger.info(f"  Model trained: {X.shape[1]}D, {len(y)} samples")
    return model, scaler


def classify_trace(trace_dict: dict, model, scaler) -> Tuple[int, float]:
    """Classify a single trace using the trained model.
    Returns (prediction: 0=benign/1=phishing, confidence)."""
    from experiments.enhanced_itg_detector import build_all_views

    try:
        views = build_all_views([trace_dict])
        X = np.hstack(views[:6])  # (1, 113)
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        proba = model.predict_proba(X_scaled)[0]
        conf = float(max(proba))
        return int(pred), conf
    except Exception as e:
        logger.warning(f"  Classify error: {e}")
        return -1, 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Crawler
# ═══════════════════════════════════════════════════════════════════════════════

async def crawl_single(url: str, output_dir: Path, timeout_ms: int = 30000) -> Optional[dict]:
    """Crawl a single URL with PhishTrace's Playwright crawler.
    Returns serialized trace dict or None on failure."""
    from src.crawler.phishing_crawler import PhishingCrawler

    ss_dir = output_dir / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)

    crawler = PhishingCrawler(
        headless=True, timeout=timeout_ms,
        capture_screenshots=True,
        screenshot_dir=str(ss_dir),
        proxy=PROXY,
    )

    try:
        trace = await asyncio.wait_for(
            crawler.crawl(url, max_depth=2),
            timeout=timeout_ms / 1000 + 30,
        )
        if not trace or not trace.events:
            return None

        # Serialize dataclasses
        events_ser = [asdict(e) if hasattr(e, '__dataclass_fields__') else
                      (e.__dict__ if hasattr(e, '__dict__') else e)
                      for e in trace.events]
        net_reqs_ser = [asdict(nr) if hasattr(nr, '__dataclass_fields__') else
                        (nr.__dict__ if hasattr(nr, '__dict__') else nr)
                        for nr in (trace.network_requests or [])]

        wrapped = {
            "url": url,
            "hash": hashlib.md5(url.encode()).hexdigest()[:12],
            "success": True,
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
        }

        # Save first screenshot as landing page thumbnail
        screenshots = sorted(ss_dir.glob("step_*.png"))
        if screenshots:
            shutil.copy2(screenshots[0], output_dir / "shot.png")

        del trace
        return wrapped

    except asyncio.TimeoutError:
        logger.debug(f"  Crawl timeout: {url[:60]}")
        return None
    except Exception as e:
        logger.debug(f"  Crawl error: {type(e).__name__}: {str(e)[:60]}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Process a batch of discovered URLs
# ═══════════════════════════════════════════════════════════════════════════════

async def process_batch(candidates: List[dict], model, scaler,
                         seen_domains: Set[str], stats: dict,
                         timeout_ms: int = 30000):
    """Crawl and classify a batch of discovered URLs."""

    # Sort by score descending — process most suspicious first
    candidates.sort(key=lambda x: -x.get('score', 0))

    for entry in candidates:
        url = entry['url']
        domain = entry.get('domain', urlparse(url).netloc)
        source = entry.get('source', 'unknown')
        brand = entry.get('brand', 'unknown')
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

        if domain in seen_domains:
            continue
        seen_domains.add(domain)

        # Liveness check (HEAD request)
        try:
            r = requests.head(url, timeout=8, proxies=PROXIES,
                            allow_redirects=True,
                            headers={'User-Agent': 'Mozilla/5.0'})
            if r.status_code >= 500:
                stats['dead'] += 1
                continue
        except requests.exceptions.SSLError:
            pass  # SSL error = server exists, interesting for phishing
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            stats['dead'] += 1
            continue
        except Exception:
            stats['dead'] += 1
            continue

        # Determine output directory
        source_dir = LIVE_DIR / source
        tmp_dir = source_dir / "pending" / url_hash

        logger.info(f"  [{stats['processed']+1}] Crawling {url[:70]} (score={entry.get('score',0)}, brand={brand})")

        # Crawl
        trace_dict = await crawl_single(url, tmp_dir, timeout_ms)
        stats['processed'] += 1

        if trace_dict is None:
            stats['crawl_fail'] += 1
            # Clean up
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir, ignore_errors=True)
            continue

        stats['crawl_ok'] += 1

        # Classify
        pred, conf = classify_trace(trace_dict, model, scaler)

        if pred == 1:
            label = "phishing"
            stats['phishing'] += 1
            logger.info(f"    ★ PHISHING (conf={conf:.3f}) forms={trace_dict.get('forms_submitted',0)} "
                       f"dual={trace_dict.get('dual_submissions',0)} brand={brand}")
        elif pred == 0:
            label = "benign"
            stats['benign'] += 1
            logger.info(f"    ✓ benign (conf={conf:.3f})")
        else:
            label = "error"
            stats['classify_fail'] += 1
            continue

        # Move to final directory: source/label/hash/
        final_dir = source_dir / label / url_hash
        if tmp_dir.exists():
            if final_dir.exists():
                shutil.rmtree(final_dir, ignore_errors=True)
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_dir), str(final_dir))

        # Enrich and save trace JSON
        trace_dict['label'] = label
        trace_dict['prediction'] = pred
        trace_dict['confidence'] = conf
        trace_dict['source'] = source
        trace_dict['brand'] = brand
        trace_dict['domain_score'] = entry.get('score', 0)
        trace_dict['discovery_reasons'] = entry.get('reasons', [])

        trace_file = final_dir / "trace.json"
        trace_file.parent.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            json.dumps(trace_dict, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        # Clean up pending dir if it still exists
        pending_dir = source_dir / "pending"
        if pending_dir.exists() and not list(pending_dir.iterdir()):
            pending_dir.rmdir()

        # Slow down between crawls to avoid rate limiting
        await asyncio.sleep(3)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary(stats: dict, start_time: datetime, seen_domains: Set[str]):
    """Save running summary to JSON."""
    elapsed = (datetime.now() - start_time).total_seconds()

    # Count files in each category
    source_counts = {}
    for source_dir in LIVE_DIR.iterdir():
        if not source_dir.is_dir() or source_dir.name in ('pending',):
            continue
        for label_dir in source_dir.iterdir():
            if label_dir.is_dir() and label_dir.name in ('phishing', 'benign'):
                key = f"{source_dir.name}/{label_dir.name}"
                source_counts[key] = len(list(label_dir.iterdir()))

    summary = {
        'start_time': start_time.isoformat(),
        'last_update': datetime.now().isoformat(),
        'elapsed_hours': round(elapsed / 3600, 2),
        'stats': stats,
        'domains_seen': len(seen_domains),
        'source_counts': source_counts,
    }

    SUMMARY_FILE.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Loop
# ═══════════════════════════════════════════════════════════════════════════════

async def main_loop(args):
    start_time = datetime.now()
    end_time = start_time + timedelta(hours=args.hours)
    logger.info("=" * 60)
    logger.info(f"PhishTrace Continuous Discovery Daemon")
    logger.info(f"  Start: {start_time.isoformat()}")
    logger.info(f"  End:   {end_time.isoformat()} ({args.hours}h)")
    logger.info("=" * 60)

    # Train model once
    logger.info("\n[INIT] Training ML detector...")
    model, scaler = train_model()

    # Track seen domains to avoid duplicates
    seen_domains: Set[str] = set()

    # Also load existing URLs from prior runs
    for label in ('phishing', 'benign'):
        for source_name in ('urlscan', 'typosquat'):
            d = LIVE_DIR / source_name / label
            if d.exists():
                for item in d.iterdir():
                    if item.is_dir():
                        trace_file = item / "trace.json"
                        if trace_file.exists():
                            try:
                                data = json.loads(trace_file.read_text(encoding='utf-8'))
                                seen_domains.add(data.get('domain', ''))
                            except Exception:
                                pass
    logger.info(f"  Already seen: {len(seen_domains)} domains from prior runs")

    stats = {
        'processed': 0, 'crawl_ok': 0, 'crawl_fail': 0,
        'phishing': 0, 'benign': 0, 'classify_fail': 0,
        'dead': 0, 'cycles': 0,
    }

    # Brand rotation — cycle through brands across loops
    all_brands = list(BRANDS.keys())
    brand_idx = 0

    logger.info("\n[RUNNING] Starting continuous discovery loop...")

    while datetime.now() < end_time:
        cycle_start = datetime.now()
        stats['cycles'] += 1
        logger.info(f"\n{'─'*50}")
        logger.info(f"  Cycle {stats['cycles']} | "
                    f"Elapsed: {(datetime.now() - start_time).total_seconds()/3600:.1f}h | "
                    f"Processed: {stats['processed']} | "
                    f"Phishing: {stats['phishing']} | Benign: {stats['benign']}")
        logger.info(f"{'─'*50}")

        all_candidates = []

        # ── URLScan.io (4 brands per cycle, rotate) ──
        batch_end = brand_idx + 4
        if batch_end > len(all_brands):
            batch = all_brands[brand_idx:] + all_brands[:batch_end - len(all_brands)]
        else:
            batch = all_brands[brand_idx:batch_end]
        brand_idx = (brand_idx + 4) % len(all_brands)

        logger.info(f"  URLScan brands: {batch}")
        try:
            urlscan_results = fetch_urlscan_latest(seen_domains, batch)
            all_candidates.extend(urlscan_results)
            logger.info(f"  URLScan: {len(urlscan_results)} new candidates")
        except Exception as e:
            logger.warning(f"  URLScan error: {e}")

        # ── Typosquatting DNS (3 brands per cycle, rotate) ──
        typo_batch = batch[:3]
        logger.info(f"  Typosquat brands: {typo_batch}")
        try:
            typo_results = check_typosquat_batch(seen_domains, typo_batch)
            all_candidates.extend(typo_results)
            logger.info(f"  Typosquat: {len(typo_results)} new candidates")
        except Exception as e:
            logger.warning(f"  Typosquat error: {e}")

        # ── Process batch ──
        if all_candidates:
            logger.info(f"  Processing {len(all_candidates)} candidates...")
            await process_batch(all_candidates, model, scaler, seen_domains, stats)
        else:
            logger.info(f"  No new candidates this cycle")

        # Save summary
        save_summary(stats, start_time, seen_domains)

        # Wait between cycles (respect rate limits)
        cycle_elapsed = (datetime.now() - cycle_start).total_seconds()
        wait_time = max(0, args.cycle_interval - cycle_elapsed)
        if wait_time > 0 and datetime.now() < end_time:
            logger.info(f"  Sleeping {wait_time:.0f}s until next cycle...")
            await asyncio.sleep(wait_time)

    # Final summary
    elapsed_total = (datetime.now() - start_time).total_seconds()
    logger.info("\n" + "=" * 60)
    logger.info("DAEMON COMPLETED")
    logger.info(f"  Total time:     {elapsed_total/3600:.1f}h")
    logger.info(f"  Cycles:         {stats['cycles']}")
    logger.info(f"  Processed:      {stats['processed']}")
    logger.info(f"  Crawl success:  {stats['crawl_ok']}")
    logger.info(f"  Crawl fail:     {stats['crawl_fail']}")
    logger.info(f"  Dead (offline): {stats['dead']}")
    logger.info(f"  Phishing:       {stats['phishing']}")
    logger.info(f"  Benign:         {stats['benign']}")
    logger.info(f"  Classify fail:  {stats['classify_fail']}")
    logger.info(f"  Domains seen:   {len(seen_domains)}")
    if stats['crawl_ok'] > 0:
        logger.info(f"  Detection rate: {stats['phishing']/stats['crawl_ok']:.1%}")
    logger.info("=" * 60)

    save_summary(stats, start_time, seen_domains)


def main():
    parser = argparse.ArgumentParser(description="PhishTrace Continuous Discovery Daemon")
    parser.add_argument("--hours", type=float, default=12,
                       help="How many hours to run (default: 12)")
    parser.add_argument("--cycle-interval", type=int, default=120,
                       help="Seconds between discovery cycles (default: 120)")
    parser.add_argument("--timeout", type=int, default=30000,
                       help="Crawl timeout in ms (default: 30000)")
    args = parser.parse_args()

    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_loop(args))


if __name__ == "__main__":
    main()
