#!/usr/bin/env python3
"""
PhishTrace — CT Log & Suspicious Domain Discovery Pipeline
=============================================================
Discovers potential phishing URLs from:
1. crt.sh Certificate Transparency logs (brand-impersonating certs)
2. Domain pattern analysis (typosquatting, combosquatting, homoglyphs)
3. Recently created suspicious domains

NOT a phishing feed — purely infrastructure-based discovery.

Usage:
    py -3 scripts/ct_domain_discovery.py               # discover + crawl
    py -3 scripts/ct_domain_discovery.py --discover-only  # just discover URLs
    py -3 scripts/ct_domain_discovery.py --limit 100    # limit crawl count
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import requests
import tldextract

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SERP_DIR = PROJECT_ROOT / "dataset" / "SERP"
DISCOVERY_DIR = PROJECT_ROOT / "dataset" / "ct_discovery"
RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
CT_URLS_FILE = DISCOVERY_DIR / "ct_urls.txt"
CT_PROGRESS_FILE = DISCOVERY_DIR / "ct_progress.json"
PROXY = "http://127.0.0.1:10809"
PROXIES = {"https": PROXY, "http": PROXY}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DISCOVERY_DIR / "ct_discovery.log", mode="a", encoding="utf-8")
        if DISCOVERY_DIR.exists() else logging.StreamHandler(),
    ]
)
logger = logging.getLogger("ct_discovery")

# ═══════════════════════════════════════════════════════════════════════════════
# Brand Intelligence
# ═══════════════════════════════════════════════════════════════════════════════

BRANDS = {
    'paypal': {
        'official': ['paypal.com', 'paypalinc.com', 'paypalcorp.com', 'paypal.cn',
                     'paypal.me', 'paypalcredit.com', 'braintreepayments.com',
                     'braintree-api.com', 'braintree.tools', 'hyperwallet.com',
                     'honey-images.com', 'loanbuilder.com'],
        'keywords': ['paypal', 'paypa1', 'paipal', 'paypaì', 'paypol'],
    },
    'apple': {
        'official': ['apple.com', 'icloud.com', 'apple.co', 'cdn-apple.com',
                     'apple.news', 'apple-dns.net', 'apple-cloudkit.com',
                     'apple-mapkit.com', 'appstoreconnect.apple.com'],
        'keywords': ['apple', 'app1e', 'appie', 'icloud', 'icl0ud'],
    },
    'microsoft': {
        'official': ['microsoft.com', 'microsoftonline.com', 'msft.net', 'azure.com',
                     'office.com', 'live.com', 'outlook.com', 'hotmail.com',
                     'msedge.net', 'windowsupdate.com', 'msftconnecttest.com',
                     'windows.com', 'microsoft.net', 'office365.com', 'onedrive.com',
                     'sharepoint.com', 'skype.com', 'xbox.com', 'bing.com',
                     'visualstudio.com', 'github.com'],
        'keywords': ['microsoft', 'micr0soft', 'microsft', 'microsoftt', 'office365',
                     'outlook', '0utlook', 'onedrive'],
    },
    'google': {
        'official': ['google.com', 'googleapis.com', 'gstatic.com', 'google.co',
                     'googlesyndication.com', 'googleadservices.com', 'google.cloud',
                     'google.org', 'youtube.com', 'goo.gl', 'google.cn', 'g.co',
                     'withgoogle.com', 'googleusercontent.com', 'googlevideo.com'],
        'keywords': ['google', 'go0gle', 'g00gle', 'gogle'],
    },
    'amazon': {
        'official': ['amazon.com', 'amazon.co.uk', 'amazon.de', 'amazon.fr',
                     'amazon.co.jp', 'amazonaws.com', 'amazontrust.com',
                     'amazon.in', 'amazon.ca', 'amazon.cn', 'amzn.com',
                     'prime.amazon', 'amazon.com.au'],
        'keywords': ['amazon', 'amaz0n', 'amazone', 'arnazon'],
    },
    'netflix': {
        'official': ['netflix.com', 'netflix.net', 'nflxext.com', 'nflximg.net',
                     'nflxso.net', 'nflxvideo.net', 'netflix.ca'],
        'keywords': ['netflix', 'netfIix', 'netfl1x', 'neftlix'],
    },
    'chase': {
        'official': ['chase.com', 'jpmorganchase.com', 'jpmorgan.com', 'jpmchase.net',
                     'jpmchase.com'],
        'keywords': ['chase', 'jpmorgan'],
    },
    'wellsfargo': {
        'official': ['wellsfargo.com', 'wf.com', 'wellsfargoadvisors.com'],
        'keywords': ['wellsfargo', 'wells-fargo', 'wf-login'],
    },
    'bankofamerica': {
        'official': ['bankofamerica.com', 'bofa.com', 'bofaml.com', 'ml.com',
                     'merrilledge.com', 'merrilllynch.com'],
        'keywords': ['bankofamerica', 'bofa', 'bank0famerica'],
    },
    'coinbase': {
        'official': ['coinbase.com', 'coinbase-staking.com'],
        'keywords': ['coinbase', 'c0inbase', 'coinbse'],
    },
    'binance': {
        'official': ['binance.com', 'binance.org', 'binance.us', 'bnbchain.org'],
        'keywords': ['binance', 'b1nance', 'binanсe'],
    },
    'facebook': {
        'official': ['facebook.com', 'fb.com', 'fbcdn.net', 'facebook.net',
                     'meta.com', 'instagram.com', 'whatsapp.com'],
        'keywords': ['facebook', 'faceb00k', 'facebok', 'fb-login'],
    },
    'instagram': {
        'official': ['instagram.com', 'cdninstagram.com'],
        'keywords': ['instagram', 'instagr4m', 'lnstagram', 'instagramm'],
    },
    'linkedin': {
        'official': ['linkedin.com', 'licdn.com'],
        'keywords': ['linkedin', 'linked1n', 'linkediη'],
    },
    'dropbox': {
        'official': ['dropbox.com', 'dropboxapi.com'],
        'keywords': ['dropbox', 'dr0pbox', 'dropb0x'],
    },
    'docusign': {
        'official': ['docusign.com', 'docusign.net'],
        'keywords': ['docusign', 'docu-sign', 'd0cusign'],
    },
    'dhl': {
        'official': ['dhl.com', 'dhl.de'],
        'keywords': ['dhl', 'dh1'],
    },
    'usps': {
        'official': ['usps.com'],
        'keywords': ['usps', 'us-postal', 'uspostal'],
    },
    'fedex': {
        'official': ['fedex.com'],
        'keywords': ['fedex', 'f3dex', 'fed-ex'],
    },
}

# Phishing indicator words in domain names
PHISH_INDICATORS = [
    'login', 'signin', 'sign-in', 'logon', 'signon', 'sign-on',
    'verify', 'verification', 'validate', 'confirm', 'confirmation',
    'secure', 'security', 'safety', 'protect',
    'account', 'acct', 'myaccount',
    'update', 'upgrade', 'renew', 'renewal',
    'recover', 'recovery', 'restore', 'unlock',
    'support', 'helpdesk', 'help-desk', 'service',
    'billing', 'invoice', 'payment', 'pay',
    'wallet', 'password', 'reset', 'credential',
    'alert', 'warning', 'suspended', 'limited', 'restricted',
    'claim', 'reward', 'prize', 'giveaway', 'free',
    'web', 'online', 'portal', 'app', 'mobile',
]

# Suspicious TLDs commonly used in phishing
SUSPICIOUS_TLDS = {
    'xyz', 'top', 'club', 'online', 'site', 'info', 'click', 'link',
    'buzz', 'space', 'fun', 'icu', 'cam', 'rest', 'monster', 'surf',
    'wang', 'work', 'gq', 'cf', 'ga', 'ml', 'tk', 'pw', 'cc',
    'us', 'ws', 'to', 'ly', 'life', 'live', 'today', 'world',
    'digital', 'tech', 'store', 'shop', 'app', 'dev', 'cfd',
}


def score_domain(domain: str) -> dict:
    """Score a domain for phishing likelihood. Returns score and analysis."""
    if not domain:
        return {'score': 0, 'reasons': ['empty'], 'brand': None}
    dl = domain.lower().replace('*.' , '').strip()
    if not dl:
        return {'score': 0, 'reasons': ['empty'], 'brand': None}
    ext = tldextract.extract(dl)
    registered = getattr(ext, 'top_domain_under_public_suffix', None) or ext.registered_domain or ''
    subdomain = ext.subdomain or ''
    suffix = ext.suffix or ''
    sld = ext.domain or ''  # second-level domain

    score = 0
    reasons = []

    # Check brand presence
    matched_brand = None
    for brand_name, brand_info in BRANDS.items():
        for kw in brand_info['keywords']:
            if kw in sld or kw in subdomain:
                # Check if it's an official domain
                if any(registered.endswith(od) or registered == od
                       for od in brand_info['official']):
                    return {'score': -1, 'reasons': ['official_domain'], 'brand': brand_name}
                matched_brand = brand_name
                score += 30
                reasons.append(f'brand_{brand_name}_in_domain')
                break
        if matched_brand:
            break

    if not matched_brand:
        return {'score': 0, 'reasons': ['no_brand'], 'brand': None}

    # Phishing indicator words
    phish_hits = [p for p in PHISH_INDICATORS if p in dl]
    if phish_hits:
        score += 15 * min(len(phish_hits), 3)
        reasons.append(f'phish_keywords:{",".join(phish_hits[:3])}')

    # Suspicious TLD
    if suffix in SUSPICIOUS_TLDS:
        score += 20
        reasons.append(f'suspicious_tld:{suffix}')

    # Domain length (long domains are more suspicious)
    if len(sld) > 20:
        score += 10
        reasons.append(f'long_sld:{len(sld)}')

    # Hyphens in domain (common in phishing)
    hyphen_count = sld.count('-')
    if hyphen_count >= 2:
        score += 10 * min(hyphen_count, 3)
        reasons.append(f'hyphens:{hyphen_count}')

    # Multiple subdomains (suspicious depth)
    subdomain_parts = [p for p in subdomain.split('.') if p]
    if len(subdomain_parts) >= 2:
        score += 5 * len(subdomain_parts)
        reasons.append(f'deep_subdomains:{len(subdomain_parts)}')

    # Numeric characters mixed with brand
    if any(c.isdigit() for c in sld) and matched_brand:
        score += 10
        reasons.append('digits_in_sld')

    # Free hosting / dynamic DNS patterns
    FREE_HOSTING = ['blogspot', 'wordpress', 'wixsite', 'weebly', 'netlify',
                    'herokuapp', 'github.io', 'firebase', 'web.app',
                    'pages.dev', 'vercel', 'duckdns', 'no-ip', 'ddns']
    if any(fh in dl for fh in FREE_HOSTING):
        score += 15
        reasons.append('free_hosting')

    return {
        'score': score,
        'reasons': reasons,
        'brand': matched_brand,
        'registered_domain': registered,
        'tld': suffix,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Source 1: crt.sh Certificate Transparency
# ═══════════════════════════════════════════════════════════════════════════════

def discover_from_crtsh(brands: List[str] = None, min_score: int = 40) -> List[dict]:
    """Query crt.sh for brand-impersonating certificates."""
    if brands is None:
        brands = list(BRANDS.keys())

    all_suspicious = []
    seen_domains = set()

    for brand_name in brands:
        brand_info = BRANDS[brand_name]
        for keyword in brand_info['keywords'][:2]:  # limit to avoid rate limit
            try:
                logger.info(f"  crt.sh query: %{keyword}%")
                r = requests.get(
                    f'https://crt.sh/?q=%25{keyword}%25&output=json&exclude=expired',
                    proxies=PROXIES, timeout=60
                )
                if r.status_code != 200:
                    logger.warning(f"  crt.sh {keyword}: HTTP {r.status_code}")
                    time.sleep(5)
                    continue

                data = r.json()
                logger.info(f"  crt.sh {keyword}: {len(data)} certificates")

                for entry in data:
                    cn = (entry.get('common_name') or '').lower().strip()
                    if not cn or cn in seen_domains:
                        continue
                    seen_domains.add(cn)

                    analysis = score_domain(cn)
                    if analysis['score'] >= min_score:
                        all_suspicious.append({
                            'domain': cn.replace('*.', ''),
                            'source': 'crt.sh',
                            'brand': analysis['brand'],
                            'score': analysis['score'],
                            'reasons': analysis['reasons'],
                            'issued': entry.get('not_before', ''),
                            'issuer': entry.get('issuer_name', ''),
                        })

                # Also check SAN (subject alternative names)
                for entry in data:
                    name_value = entry.get('name_value') or ''
                    for line in name_value.split('\n'):
                        cn = line.strip().lower()
                        if not cn or cn in seen_domains:
                            continue
                        seen_domains.add(cn)
                        analysis = score_domain(cn)
                        if analysis['score'] >= min_score:
                            all_suspicious.append({
                                'domain': cn.replace('*.', ''),
                                'source': 'crt.sh_san',
                                'brand': analysis['brand'],
                                'score': analysis['score'],
                                'reasons': analysis['reasons'],
                                'issued': entry.get('not_before', ''),
                            })

            except Exception as e:
                logger.warning(f"  crt.sh {keyword}: {type(e).__name__}: {str(e)[:80]}")

            time.sleep(5)  # Rate limit

    all_suspicious.sort(key=lambda x: -x['score'])
    return all_suspicious


# ═══════════════════════════════════════════════════════════════════════════════
# Source 2: Domain Pattern Generation (Typosquatting)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_typosquat_candidates() -> List[dict]:
    """Generate and check typosquatting / combosquatting domain candidates.
    Uses concurrent DNS resolution for speed."""
    import socket
    from concurrent.futures import ThreadPoolExecutor, as_completed

    candidates = []
    all_combos = []  # (domain, brand_name)

    # Top brands only for speed
    top_brands = ['paypal', 'apple', 'microsoft', 'google', 'amazon',
                  'netflix', 'chase', 'facebook']
    top_indicators = ['login', 'signin', 'verify', 'secure', 'account',
                      'update', 'support', 'wallet']
    top_tlds = ['xyz', 'top', 'club', 'online', 'site', 'info', 'app',
                'icu', 'cc', 'buzz']

    for brand_name in top_brands:
        if brand_name not in BRANDS:
            continue
        base_keyword = BRANDS[brand_name]['keywords'][0]

        for indicator in top_indicators:
            for tld in top_tlds:
                all_combos.append((f'{base_keyword}-{indicator}.{tld}', brand_name))
                all_combos.append((f'{indicator}-{base_keyword}.{tld}', brand_name))
                all_combos.append((f'{base_keyword}{indicator}.{tld}', brand_name))

    logger.info(f"  Checking {len(all_combos)} typosquat candidates (concurrent)...")

    def check_domain(domain_brand):
        domain, brand_name = domain_brand
        try:
            socket.setdefaulttimeout(2)
            ip = socket.gethostbyname(domain)
            return (domain, brand_name, ip)
        except (socket.gaierror, socket.timeout, OSError):
            return None

    # Use 20 threads for concurrent DNS resolution
    resolved = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(check_domain, combo): combo for combo in all_combos}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            result = future.result()
            if result:
                resolved.append(result)
            if done_count % 200 == 0:
                logger.info(f"  DNS checked: {done_count}/{len(all_combos)}, "
                           f"resolved: {len(resolved)}")

    logger.info(f"  DNS check complete: {len(resolved)} resolved out of {len(all_combos)}")

    seen = set()
    for domain, brand_name, ip in resolved:
        if domain in seen:
            continue
        seen.add(domain)
        analysis = score_domain(domain)
        candidates.append({
            'domain': domain,
            'source': 'typosquat_check',
            'brand': brand_name,
            'score': analysis['score'],
            'reasons': analysis['reasons'],
            'ip': ip,
            'checked_at': datetime.now().isoformat(),
        })
        logger.info(f"  RESOLVED: {domain} → {ip} (score={analysis['score']})")

    candidates.sort(key=lambda x: -x['score'])
    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Source 3: URLhaus (abuse reports, NOT a phishing feed)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_from_urlhaus() -> List[dict]:
    """Get recently reported malicious URLs from URLhaus (abuse.ch).
    NOTE: URLhaus primarily contains malware_download entries.
    Very few phishing entries — kept for completeness but low priority."""
    urls = []
    try:
        r = requests.get(
            'https://urlhaus.abuse.ch/downloads/json_recent/',
            proxies=PROXIES, timeout=60
        )
        if r.status_code == 200:
            data = r.json()
            for key, value in data.items():
                # Values can be lists or dicts depending on version
                entries = value if isinstance(value, list) else [value] if isinstance(value, dict) else []
                for entry in entries:
                    if not isinstance(entry, dict):
                        continue
                    url = entry.get('url', '')
                    tags = entry.get('tags', []) or []
                    threat = str(entry.get('threat', '') or '').lower()
                    tags_str = ' '.join(str(t).lower() for t in tags)
                    if 'phish' in threat or 'phish' in tags_str:
                        parsed = urlparse(url)
                        analysis = score_domain(parsed.netloc)
                        urls.append({
                            'url': url,
                            'domain': parsed.netloc,
                            'source': 'urlhaus',
                            'brand': analysis.get('brand'),
                            'score': analysis.get('score', 0) + 20,
                            'tags': tags,
                            'threat': threat,
                        })
            logger.info(f"  URLhaus: {len(urls)} phishing-tagged URLs (of {len(data)} total)")
        else:
            logger.warning(f"  URLhaus: HTTP {r.status_code}")
    except Exception as e:
        logger.warning(f"  URLhaus: {type(e).__name__}: {str(e)[:80]}")

    return urls


# ═══════════════════════════════════════════════════════════════════════════════
# Source 4: URLScan.io (community-submitted URL scans)
# ═══════════════════════════════════════════════════════════════════════════════

def discover_from_urlscan(brands: List[str] = None) -> List[dict]:
    """Search URLScan.io for recently scanned suspicious pages.
    URLScan.io is a community sandbox — NOT a phishing feed.
    We search for brand-related pages on non-official domains.
    PRIMARY DISCOVERY SOURCE - most reliable and productive."""
    if brands is None:
        brands = list(BRANDS.keys())

    results = []
    seen = set()
    rate_limit_hits = 0

    for brand_name in brands[:12]:  # Top 12 brands
        brand_info = BRANDS[brand_name]
        keyword = brand_info['keywords'][0]
        official = set(brand_info['official'])

        # Build exclusion clause
        official_main = f'{keyword}.com'
        excl_parts = [f'AND NOT domain:{od}' for od in list(official)[:5]]
        excl = ' '.join(excl_parts)
        # Use plain domain: prefix (NOT page.domain:* wildcards — those get 403)
        q = f'domain:{keyword} {excl}'

        try:
            r = requests.get(
                'https://urlscan.io/api/v1/search/',
                params={'q': q, 'size': 100},
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'},
                proxies=PROXIES, timeout=25,
            )
            if r.status_code == 429:
                rate_limit_hits += 1
                wait = 30 if rate_limit_hits < 3 else 60
                logger.warning(f"  URLScan {keyword}: rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code == 403:
                rate_limit_hits += 1
                wait = 15 if rate_limit_hits < 3 else 45
                logger.warning(f"  URLScan {keyword}: HTTP 403, waiting {wait}s...")
                time.sleep(wait)
                continue
            if r.status_code != 200:
                logger.warning(f"  URLScan {keyword}: HTTP {r.status_code}")
                time.sleep(5)
                continue

            data = r.json()
            hits = data.get('results', [])
            brand_suspicious = 0

            for hit in hits:
                page = hit.get('page', {})
                url = page.get('url', '')
                domain = page.get('domain', '')
                verdicts = hit.get('verdicts', {})
                overall = verdicts.get('overall', {})
                malicious_flag = overall.get('malicious', False)
                scan_score = overall.get('score', 0)

                if not url or not domain or domain in seen:
                    continue
                seen.add(domain)

                # Skip official domains
                if any(domain.endswith(od) or domain == od for od in official):
                    continue

                analysis = score_domain(domain)
                domain_score = analysis['score']
                # Bonus points
                if malicious_flag:
                    domain_score += 40  # flagged malicious by URLScan
                if scan_score >= 50:
                    domain_score += 20
                domain_score += 10  # base bonus for appearing in URLScan

                if domain_score > 0:
                    brand_suspicious += 1
                    results.append({
                        'url': url,
                        'domain': domain,
                        'source': 'urlscan',
                        'brand': brand_name,
                        'score': domain_score,
                        'reasons': analysis.get('reasons', []) + (['urlscan_malicious'] if malicious_flag else []),
                        'scan_time': hit.get('task', {}).get('time', ''),
                        'malicious_flag': malicious_flag,
                    })

            logger.info(f"  URLScan {keyword}: {len(hits)} scans, {brand_suspicious} suspicious")

        except Exception as e:
            logger.warning(f"  URLScan {keyword}: {type(e).__name__}: {str(e)[:80]}")

        time.sleep(5)  # Rate limit — 5s between brands

    results.sort(key=lambda x: -x['score'])
    logger.info(f"  URLScan total: {len(results)} suspicious domains (rate limits hit: {rate_limit_hits})")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Source 5: DDG Meta-Search for Suspicious Domains
# ═══════════════════════════════════════════════════════════════════════════════

def discover_from_ddg_suspicious(brands: List[str] = None) -> List[dict]:
    """Use DDG meta-search to find recently indexed suspicious domains.
    Searches for patterns like 'paypal login verify site:xyz' etc."""
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        logger.warning("  DDG: duckduckgo_search not installed. pip install duckduckgo_search")
        return []

    if brands is None:
        brands = list(BRANDS.keys())

    results = []
    seen = set()

    # Generate targeted queries
    queries = []
    suspicious_tld_list = ['xyz', 'top', 'club', 'online', 'site', 'info',
                            'click', 'link', 'buzz', 'icu', 'cfd']
    for brand_name in brands[:10]:
        kw = BRANDS[brand_name]['keywords'][0]
        # Search for brand on suspicious TLDs
        for tld in suspicious_tld_list[:6]:
            queries.append((f'{kw} site:{tld}', brand_name))
        # Search for brand with phishing indicators
        for indicator in ['login', 'verify', 'secure', 'account', 'update']:
            queries.append((f'{kw} {indicator} -site:{kw}.com', brand_name))

    random.shuffle(queries)

    ddgs = DDGS(proxy=PROXY)
    attempted = 0
    for query, brand_name in queries[:60]:  # Limit total queries
        try:
            brand_info = BRANDS[brand_name]
            official = set(brand_info['official'])

            hits = list(ddgs.text(query, max_results=10))
            attempted += 1

            for hit in hits:
                url = hit.get('href', '')
                if not url:
                    continue
                parsed = urlparse(url)
                domain = parsed.netloc
                if not domain or domain in seen:
                    continue
                seen.add(domain)

                # Skip official
                if any(domain.endswith(od) for od in official):
                    continue

                analysis = score_domain(domain)
                if analysis['score'] >= 30:
                    results.append({
                        'url': url,
                        'domain': domain,
                        'source': 'ddg_suspicious',
                        'brand': brand_name,
                        'score': analysis['score'],
                        'reasons': analysis.get('reasons', []),
                        'title': hit.get('title', ''),
                        'query': query,
                    })

            if attempted % 10 == 0:
                logger.info(f"  DDG: {attempted} queries, {len(results)} suspicious domains so far")

        except Exception as e:
            if 'ratelimit' in str(e).lower():
                logger.warning(f"  DDG rate limited, pausing 30s...")
                time.sleep(30)
            else:
                logger.debug(f"  DDG query error: {str(e)[:60]}")

        time.sleep(2)  # Polite delay

    logger.info(f"  DDG: completed {attempted} queries, {len(results)} suspicious domains")
    results.sort(key=lambda x: -x['score'])
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Combine & Deduplicate
# ═══════════════════════════════════════════════════════════════════════════════

def combine_discoveries(crtsh_results, typosquat_results, urlhaus_results,
                        urlscan_results=None, ddg_results=None,
                        existing_urls: Set[str] = None) -> List[dict]:
    """Combine all sources, deduplicate, and rank."""
    if existing_urls is None:
        existing_urls = set()

    all_results = []
    seen_domains = set()

    def add_results(result_list, has_url=False):
        for r in result_list:
            domain = r.get('domain', '')
            if not domain or domain in seen_domains:
                continue
            seen_domains.add(domain)
            url = r.get('url', f'https://{domain}') if has_url else f'https://{domain}'
            if url in existing_urls:
                continue
            all_results.append({
                'url': url,
                'domain': domain,
                'source': r['source'],
                'brand': r.get('brand'),
                'score': r['score'],
                'reasons': r.get('reasons', []),
                'malicious_flag': r.get('malicious_flag', False),
            })

    # URLScan first (best source), then others
    add_results(urlscan_results or [], has_url=True)
    add_results(crtsh_results, has_url=False)
    add_results(typosquat_results, has_url=False)
    add_results(urlhaus_results, has_url=True)
    add_results(ddg_results or [], has_url=True)

    all_results.sort(key=lambda x: -x['score'])
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# Liveness Check
# ═══════════════════════════════════════════════════════════════════════════════

def check_liveness(candidates: List[dict], timeout: int = 10) -> List[dict]:
    """Check which discovered URLs are still reachable (alive).
    Filters out dead domains before expensive crawling."""
    alive = []
    dead = 0

    for i, entry in enumerate(candidates):
        url = entry['url']
        try:
            r = requests.head(url, timeout=timeout, proxies=PROXIES,
                            allow_redirects=True,
                            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})
            if r.status_code < 500:
                entry['live_status'] = r.status_code
                entry['final_url_head'] = r.url
                alive.append(entry)
            else:
                dead += 1
        except requests.exceptions.SSLError:
            # SSL errors mean server exists but cert is bad — still interesting for phishing
            entry['live_status'] = 'ssl_error'
            entry['final_url_head'] = url
            alive.append(entry)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            dead += 1
        except Exception:
            dead += 1

        if (i + 1) % 20 == 0:
            logger.info(f"  Liveness: checked {i+1}/{len(candidates)}, "
                       f"alive={len(alive)}, dead={dead}")

    logger.info(f"  Liveness: {len(alive)} alive, {dead} dead out of {len(candidates)}")
    return alive


# ═══════════════════════════════════════════════════════════════════════════════
# Crawl Integration
# ═══════════════════════════════════════════════════════════════════════════════

async def crawl_and_classify(urls: List[dict], model, scaler,
                              timeout_ms: int = 30000) -> List[dict]:
    """Crawl discovered URLs and classify using PhishTrace."""
    from dataclasses import asdict
    from src.crawler.phishing_crawler import PhishingCrawler

    results = []
    for i, entry in enumerate(urls):
        url = entry['url']
        url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
        output_dir = DISCOVERY_DIR / url_hash

        logger.info(f"  [{i+1}/{len(urls)}] Crawling {url[:80]}...")

        trace_file = output_dir / "trace.json"
        if trace_file.exists():
            try:
                data = json.loads(trace_file.read_text(encoding="utf-8"))
                if data.get("success"):
                    logger.info(f"    Cache hit")
                    results.append({**entry, 'url_hash': url_hash, 'trace': data, 'cached': True})
                    continue
            except Exception:
                pass

        # Crawl
        crawler = PhishingCrawler(
            headless=True, timeout=timeout_ms,
            capture_screenshots=True,
            screenshot_dir=str(output_dir / "screenshots"),
            proxy=PROXY,
        )
        try:
            trace = await asyncio.wait_for(
                crawler.crawl(url, max_depth=2),
                timeout=timeout_ms / 1000 + 20,
            )
            if not trace or not trace.events:
                results.append({**entry, 'url_hash': url_hash, 'crawl_success': False})
                continue

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
                "source": entry['source'],
                "brand": entry.get('brand', ''),
                "domain_score": entry['score'],
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
            }

            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "trace.json").write_text(
                json.dumps(wrapped, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8")

            # Save screenshot (use FIRST = landing page)
            ss_dir = output_dir / "screenshots"
            if ss_dir.exists():
                screenshots = sorted(ss_dir.glob("step_*.png"))
                if screenshots:
                    shutil.copy2(screenshots[0], output_dir / "shot.png")

            # Classify
            try:
                from experiments.enhanced_itg_detector import build_all_views
                from sklearn.ensemble import RandomForestClassifier

                views = build_all_views(wrapped)
                if views is not None:
                    features = views.reshape(1, -1)
                    features_scaled = scaler.transform(features)
                    pred = model.predict(features_scaled)[0]
                    proba = model.predict_proba(features_scaled)[0]
                    conf = max(proba)
                    label = 'phishing' if pred == 1 else 'benign'
                else:
                    pred, conf, label = -1, 0.0, 'feature_error'
            except Exception as e:
                pred, conf, label = -1, 0.0, f'classify_error:{str(e)[:50]}'

            result = {
                **entry,
                'url_hash': url_hash,
                'crawl_success': True,
                'prediction': int(pred),
                'confidence': float(conf),
                'label': label,
                'final_url': getattr(trace, 'final_url', url),
                'page_title': getattr(trace, 'page_title', ''),
                'forms_submitted': trace.forms_submitted,
                'dual_submissions': trace.dual_submissions_detected,
                'events_count': len(trace.events),
            }

            if pred == 1:
                logger.info(f"    ★ PHISHING conf={conf:.3f} forms={trace.forms_submitted} "
                           f"domain_score={entry['score']}")
            else:
                logger.info(f"    {label} conf={conf:.3f}")

            results.append(result)
            del trace

        except asyncio.TimeoutError:
            results.append({**entry, 'url_hash': url_hash, 'crawl_success': False, 'error': 'timeout'})
        except Exception as e:
            results.append({**entry, 'url_hash': url_hash, 'crawl_success': False,
                           'error': f'{type(e).__name__}: {str(e)[:100]}'})

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def train_detector():
    """Train ML detector on existing dataset."""
    from experiments.enhanced_itg_detector import (
        load_all_traces, build_all_views,
    )
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    traces_dir = PROJECT_ROOT / "dataset" / "traces"
    traces, labels = load_all_traces(str(traces_dir))
    logger.info(f"  Training data: {len(traces)} traces")

    X_list, y_list = [], []
    for trace, label in zip(traces, labels):
        views = build_all_views(trace)
        if views is not None:
            X_list.append(views)
            y_list.append(label)

    X = np.array(X_list)
    y = np.array(y_list)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=500, max_depth=None,
                                    min_samples_split=5, random_state=42, n_jobs=-1)
    model.fit(X_scaled, y)
    logger.info(f"  Model trained: {X.shape[1]}D features, {len(y)} samples")
    return model, scaler


async def main_async(args):
    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)

    # Re-setup logging with file handler
    fh = logging.FileHandler(DISCOVERY_DIR / "ct_discovery.log", mode="a", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s"))
    logger.addHandler(fh)

    logger.info("=" * 60)
    logger.info("PhishTrace — CT Log & Domain Discovery Pipeline")
    logger.info("=" * 60)

    # Load existing URLs to avoid duplicates
    existing_urls = set()
    for url_file in [SERP_DIR / "urls.txt", CT_URLS_FILE]:
        if url_file.exists():
            for line in url_file.read_text(encoding='utf-8').splitlines():
                line = line.strip()
                if line and line.startswith('http'):
                    existing_urls.add(line)
    logger.info(f"  Existing URLs: {len(existing_urls)}")

    # ── Phase 1: Discovery ──────────────────────────────────────────────────
    logger.info("\n[PHASE 1] Discovering suspicious domains...")

    # Source 1 (PRIMARY): URLScan.io — most productive source
    logger.info("\n  --- URLScan.io Community Scans (PRIMARY) ---")
    urlscan_results = discover_from_urlscan()
    logger.info(f"  URLScan: {len(urlscan_results)} suspicious domains")

    # Source 2: Typosquatting checks
    logger.info("\n  --- Typosquatting DNS Resolution ---")
    if not args.skip_typosquat:
        typosquat_results = generate_typosquat_candidates()
        logger.info(f"  Typosquat: {len(typosquat_results)} registered suspicious domains")
    else:
        typosquat_results = []

    # Source 3 (OPTIONAL): crt.sh — unreliable, skip by default with --skip-crtsh
    if not args.skip_crtsh:
        logger.info("\n  --- crt.sh Certificate Transparency (may be slow/unreliable) ---")
        crtsh_results = discover_from_crtsh(min_score=args.min_score)
        logger.info(f"  crt.sh: {len(crtsh_results)} suspicious domains")
    else:
        crtsh_results = []
        logger.info("\n  --- crt.sh: SKIPPED (use --no-skip-crtsh to enable) ---")

    # Source 4: URLhaus (rarely has phishing, mostly malware)
    logger.info("\n  --- URLhaus Abuse Reports ---")
    urlhaus_results = discover_from_urlhaus()

    # Source 5: DDG Suspicious Domain Search
    logger.info("\n  --- DDG Meta-Search for Suspicious Domains ---")
    if not args.skip_ddg:
        ddg_results = discover_from_ddg_suspicious()
        logger.info(f"  DDG: {len(ddg_results)} suspicious domains")
    else:
        ddg_results = []

    # Combine
    combined = combine_discoveries(
        crtsh_results, typosquat_results, urlhaus_results,
        urlscan_results, ddg_results, existing_urls
    )
    logger.info(f"\n  Combined unique candidates: {len(combined)}")

    # Save discovered URLs
    CT_URLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CT_URLS_FILE, 'w', encoding='utf-8') as f:
        for entry in combined:
            f.write(entry['url'] + '\n')

    # Save full discovery data
    discovery_file = DISCOVERY_DIR / "discovery_results.json"
    with open(discovery_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'sources': {
                'crt.sh': len(crtsh_results),
                'typosquat': len(typosquat_results),
                'urlhaus': len(urlhaus_results),
                'urlscan': len(urlscan_results),
                'ddg_suspicious': len(ddg_results),
            },
            'total_candidates': len(combined),
            'candidates': combined[:500],  # Save top 500
        }, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"\n  Top 20 suspicious domains:")
    for i, c in enumerate(combined[:20], 1):
        reasons = ','.join(c.get('reasons', [])[:2])
        logger.info(f"    {i:2d}. [{c['score']:3d}] {c['domain'][:60]}  ({reasons})")

    if args.discover_only:
        logger.info("\n  Discovery-only mode. Skipping crawl.")
        return

    # ── Phase 1.5: Liveness Check ──────────────────────────────────────────
    logger.info("\n[PHASE 1.5] Checking liveness of discovered URLs...")
    to_check = combined[:args.limit * 2] if args.limit else combined  # check 2x limit
    alive = check_liveness(to_check)

    # ── Phase 2: Crawl & Classify ───────────────────────────────────────────
    logger.info("\n[PHASE 2] Crawling & classifying...")

    # Train model
    logger.info("  Training ML detector...")
    model, scaler = train_detector()

    # Limit crawl
    to_crawl = alive[:args.limit] if args.limit else alive
    logger.info(f"  Crawling {len(to_crawl)} live URLs...")

    results = await crawl_and_classify(to_crawl, model, scaler, timeout_ms=args.timeout)

    # ── Phase 3: Results ────────────────────────────────────────────────────
    successful = [r for r in results if r.get('crawl_success')]
    phishing = [r for r in successful if r.get('prediction') == 1]
    benign = [r for r in successful if r.get('prediction') == 0]
    failed = [r for r in results if not r.get('crawl_success')]

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Total processed:     {len(results)}")
    logger.info(f"  Crawl success:       {len(successful)}")
    logger.info(f"  Crawl failure:       {len(failed)}")
    logger.info(f"  Classified phishing: {len(phishing)}")
    logger.info(f"  Classified benign:   {len(benign)}")
    if successful:
        logger.info(f"  Detection rate:      {len(phishing)/len(successful):.1%}")

    if phishing:
        logger.info(f"\n  Phishing sites ({len(phishing)}):")
        for r in phishing:
            logger.info(f"    [{r['url_hash']}] conf={r['confidence']:.3f} "
                       f"score={r['score']} brand={r.get('brand','')} "
                       f"forms={r.get('forms_submitted',0)}")
            logger.info(f"      {r['url'][:90]}")

    # Save results
    results_file = RESULTS_DIR / "ct_discovery_results.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'experiment': 'CT Log & Domain Discovery Pipeline',
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_processed': len(results),
                'crawl_success': len(successful),
                'crawl_failure': len(failed),
                'classified_phishing': len(phishing),
                'classified_benign': len(benign),
                'detection_rate': round(len(phishing)/max(1,len(successful)), 4),
            },
            'phishing': [{k:v for k,v in r.items() if k != 'trace'} for r in phishing],
            'all_results': [{k:v for k,v in r.items() if k != 'trace'} for r in results],
        }, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\n  Results saved to {results_file}")


def main():
    parser = argparse.ArgumentParser(description="PhishTrace CT Log Discovery")
    parser.add_argument("--discover-only", action="store_true",
                       help="Only discover URLs, don't crawl")
    parser.add_argument("--skip-typosquat", action="store_true",
                       help="Skip typosquatting DNS checks")
    parser.add_argument("--skip-crtsh", action="store_true", default=True,
                       help="Skip crt.sh (default: skip)")
    parser.add_argument("--no-skip-crtsh", dest="skip_crtsh",
                       action="store_false",
                       help="Enable crt.sh queries")
    parser.add_argument("--skip-ddg", action="store_true",
                       help="Skip DDG meta-search")
    parser.add_argument("--limit", type=int, default=200,
                       help="Max URLs to crawl")
    parser.add_argument("--timeout", type=int, default=30000,
                       help="Crawl timeout ms")
    parser.add_argument("--min-score", type=int, default=40,
                       help="Minimum domain suspiciousness score")
    args = parser.parse_args()

    DISCOVERY_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
