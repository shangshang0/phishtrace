"""
PhishTrace Wild Domain Discovery Scanner
=========================================
Discovers new potentially-phishing domains from multiple open-source
intelligence feeds **and** scores them using heuristic phishing-domain
analysis (brand impersonation, TLD risk, structural anomalies, etc.).

Sources
-------
1. crt.sh  - Certificate Transparency log search (multi-brand queries)
2. URLScan.io - community-submitted recent scans (multi-query)
3. WhoisDS  - Newly Observed Domain (NOD) daily feed
4. OpenPhish - community phishing feed
5. PhishTank - verified phishing URLs
6. URLhaus  - abuse.ch fresh malicious-URL feed (CSV, web-only filtering)

Scoring
-------
``PhishingDomainScorer`` evaluates every discovered domain on five axes:
  * Brand impersonation (Levenshtein typosquatting)
  * Suspicious-keyword density
  * High-risk TLD
  * Structural anomalies (subdomains, hyphens, entropy)
  * Abused-hosting platform detection

Domains that appear in a *known* phishing feed (OpenPhish, PhishTank)
receive a score boost so they always pass the threshold.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import math
import re
import time
import zipfile
from collections import Counter
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urlparse

import requests

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DomainScore:
    """Score indicating how likely a domain is phishing."""
    domain: str
    score: float              # 0.0 - 1.0
    reasons: List[str] = field(default_factory=list)
    source: str = ""
    discovered_at: str = ""


@dataclass
class DiscoveredDomain:
    """A single domain/URL surfaced by one of the discovery sources."""

    domain: str
    url: str
    source: str
    phishing_score: float = 0.0
    score_reasons: List[str] = field(default_factory=list)
    discovered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    metadata: Dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# Domain Intelligence: Phishing Heuristics for New Domains
# ============================================================================

class PhishingDomainScorer:
    """
    Heuristic-based scorer for detecting potentially phishing domains.
    Uses brand impersonation patterns, suspicious TLD analysis,
    and structural anomaly detection.

    Scoring axes (max contribution):
      1. Brand impersonation  - 0.40
      2. Suspicious keywords  - 0.20
      3. High-risk TLD        - 0.15
      4. Structural anomalies - 0.15
      5. Abused hosting       - 0.10
                        Total = 1.00
    """

    # Top impersonated brands (PhishIntention 277-brand list + APWG data)
    TARGET_BRANDS: Set[str] = {
        'paypal', 'apple', 'microsoft', 'google', 'amazon', 'netflix',
        'facebook', 'instagram', 'whatsapp', 'chase', 'wellsfargo',
        'bankofamerica', 'citibank', 'usbank', 'capitalone', 'amex',
        'americanexpress', 'discover', 'coinbase', 'binance', 'metamask',
        'blockchain', 'ledger', 'opensea', 'dropbox', 'onedrive',
        'icloud', 'outlook', 'office365', 'office', 'linkedin',
        'twitter', 'tiktok', 'snapchat', 'telegram', 'signal',
        'zoom', 'webex', 'teams', 'slack', 'discord',
        'ebay', 'walmart', 'target', 'bestbuy', 'costco',
        'usps', 'fedex', 'ups', 'dhl', 'royalmail',
        'irs', 'hmrc', 'gov', 'tax', 'medicare',
        'hsbc', 'barclays', 'natwest', 'santander', 'ing',
        'docusign', 'adobe', 'salesforce', 'zendesk', 'shopify',
        'stripe', 'square', 'venmo', 'cashapp', 'zelle',
        'steam', 'epic', 'roblox', 'playstation', 'nintendo',
    }

    PHISH_KEYWORDS: Set[str] = {
        'login', 'signin', 'sign-in', 'logon', 'log-on', 'verify',
        'verification', 'secure', 'security', 'update', 'confirm',
        'account', 'suspend', 'locked', 'unlock', 'restore',
        'recover', 'reset', 'password', 'credential', 'authenticate',
        'validate', 'billing', 'payment', 'invoice', 'refund',
        'alert', 'urgent', 'immediate', 'expire', 'expired',
        'limited', 'unusual', 'activity', 'unauthorized', 'compromised',
        'wallet', 'connect', 'swap', 'bridge', 'claim', 'reward',
        'airdrop', 'mint', 'stake', 'defi',
    }

    HIGH_RISK_TLDS: Set[str] = {
        '.tk', '.ml', '.ga', '.cf', '.gq',
        '.xyz', '.top', '.club', '.online', '.site',
        '.icu', '.buzz', '.cyou', '.rest', '.shop',
        '.work', '.life', '.fun', '.uno', '.cam',
        '.wang', '.vip', '.cc', '.live', '.best',
    }

    ABUSED_HOSTS: Set[str] = {
        'pages.dev', 'workers.dev', 'netlify.app', 'vercel.app',
        'herokuapp.com', 'firebaseapp.com', 'web.app',
        'github.io', 'gitlab.io', 'blogspot.com',
        'weebly.com', 'wixsite.com', 'godaddysites.com',
        'square.site', 'myshopify.com', 'carrd.co',
        '000webhostapp.com', 'infinityfreeapp.com',
        'repl.co', 'glitch.me', 'ngrok.io', 'trycloudflare.com',
    }

    def score_domain(self, domain: str) -> DomainScore:
        """Score a domain for phishing likelihood (0.0 - 1.0)."""
        domain_lower = domain.lower().strip()
        score = 0.0
        reasons: List[str] = []

        brand_score = self._check_brand_impersonation(domain_lower)
        if brand_score > 0:
            score += brand_score
            reasons.append(f"brand_impersonation:{brand_score:.2f}")

        keyword_score = self._check_keywords(domain_lower)
        if keyword_score > 0:
            score += keyword_score
            reasons.append(f"suspicious_keywords:{keyword_score:.2f}")

        tld_score = self._check_tld(domain_lower)
        if tld_score > 0:
            score += tld_score
            reasons.append(f"high_risk_tld:{tld_score:.2f}")

        struct_score = self._check_structure(domain_lower)
        if struct_score > 0:
            score += struct_score
            reasons.append(f"structural_anomaly:{struct_score:.2f}")

        host_score = self._check_abused_hosting(domain_lower)
        if host_score > 0:
            score += host_score
            reasons.append(f"abused_hosting:{host_score:.2f}")

        return DomainScore(
            domain=domain,
            score=min(score, 1.0),
            reasons=reasons,
            discovered_at=datetime.now(timezone.utc).isoformat(),
        )

    def _check_brand_impersonation(self, domain: str) -> float:
        parts = domain.split('.')
        domain_name = parts[0] if parts else domain
        score = 0.0
        for brand in self.TARGET_BRANDS:
            if brand in domain_name and brand != domain_name:
                score = max(score, 0.35)
            elif self._levenshtein_distance(brand, domain_name) <= 2 and len(brand) > 4:
                score = max(score, 0.4)
            elif brand in domain and len(parts) > 2:
                score = max(score, 0.3)
        return score

    def _check_keywords(self, domain: str) -> float:
        count = sum(1 for kw in self.PHISH_KEYWORDS if kw in domain)
        if count >= 3:
            return 0.2
        elif count >= 2:
            return 0.15
        elif count >= 1:
            return 0.08
        return 0.0

    def _check_tld(self, domain: str) -> float:
        for tld in self.HIGH_RISK_TLDS:
            if domain.endswith(tld):
                return 0.15
        return 0.0

    def _check_structure(self, domain: str) -> float:
        score = 0.0
        parts = domain.split('.')
        if len(parts) > 4:
            score += 0.08
        if len(domain) > 50:
            score += 0.05
        if domain.count('-') >= 3:
            score += 0.07
        if re.search(r'\d{1,3}[-\.]\d{1,3}[-\.]\d{1,3}', domain):
            score += 0.05
        domain_name = parts[0] if parts else domain
        if len(domain_name) > 8:
            entropy = self._shannon_entropy(domain_name)
            if entropy > 3.5:
                score += 0.05
        return min(score, 0.15)

    def _check_abused_hosting(self, domain: str) -> float:
        for host in self.ABUSED_HOSTS:
            if domain.endswith(host):
                return 0.1
        return 0.0

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        if len(s1) < len(s2):
            return PhishingDomainScorer._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        prev = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            curr = [i + 1]
            for j, c2 in enumerate(s2):
                cost = 0 if c1 == c2 else 1
                curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
            prev = curr
        return prev[len(s2)]

    @staticmethod
    def _shannon_entropy(s: str) -> float:
        if not s:
            return 0.0
        freq = Counter(s)
        length = len(s)
        return -sum((c / length) * math.log2(c / length) for c in freq.values())


# ============================================================================
# URL-quality filters (skip malware downloads, infrastructure, etc.)
# ============================================================================

_MALWARE_EXT_RE = re.compile(
    r'\.(?:exe|msi|dll|sh|bin|elf|arm[v\d]*|bat|cmd|ps1|vbs|jar|apk|deb|rpm|iso|img)$',
    re.IGNORECASE,
)

_MALWARE_PATH_RE = re.compile(
    r'^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?/[a-zA-Z]/?$'
)


def _is_web_page_url(url: str) -> bool:
    """Return True if the URL likely points to a web page (not a binary)."""
    try:
        parsed = urlparse(url)
        path = parsed.path.lower()
        if _MALWARE_EXT_RE.search(path):
            return False
        if _MALWARE_PATH_RE.match(url):
            return False
        return True
    except Exception:
        return True


_INFRA_PATTERNS = re.compile(
    r'(?:'
    r'awsdns-|akamai|cloudflare-dns|fastly-analytics|edgecast|limelight|'
    r'gtld-servers|root-servers|registrar-servers|'
    r'in-addr\.arpa|ip6\.arpa|'
    r'ntp\d*\.|dns\d*\.|ns\d*\.|mx\d*\.|smtp\d*\.'
    r')',
    re.IGNORECASE,
)

_INFRA_SUFFIXES = {
    'amazonaws.com', 'cloudfront.net', 'akamaiedge.net', 'akamai.net',
    'akadns.net', 'edgekey.net', 'edgesuite.net', 'fastly.net',
    'azurefd.net', 'trafficmanager.net', 'vo.msecnd.net', 'azureedge.net',
}


def _is_browsable_domain(domain: str) -> bool:
    """Return True if the domain is likely a browsable website."""
    d = domain.lower().strip()
    if _INFRA_PATTERNS.search(d):
        return False
    for suffix in _INFRA_SUFFIXES:
        if d.endswith(suffix):
            return False
    return True


# ============================================================================
# WildDomainDiscovery - multi-source OSINT with scoring
# ============================================================================

class WildDomainDiscovery:
    """Aggregate domain discovery from multiple OSINT sources.

    Parameters
    ----------
    use_proxy : bool
        Route HTTP requests through a proxy.
    proxy_url : str | None
        Proxy address (e.g. ``http://127.0.0.1:10809``).
    keyword_filter : bool
        If True, keep only domains matching suspicious-keyword heuristic.
    score_threshold : float
        Minimum phishing score (0.0-1.0) for non-feed sources.
        Domains from known-phishing feeds always pass.
    """

    def __init__(
        self,
        use_proxy: bool = False,
        proxy_url: Optional[str] = None,
        keyword_filter: bool = True,
        score_threshold: float = 0.20,
    ) -> None:
        self.keyword_filter = keyword_filter
        self.score_threshold = score_threshold
        self.scorer = PhishingDomainScorer()
        self._session = requests.Session()
        self._session.headers["User-Agent"] = (
            "PhishTrace-Discovery/2.0 (research; +https://github.com/anthropics/phish)"
        )
        if use_proxy and proxy_url:
            self._session.proxies = {
                "http": proxy_url,
                "https": proxy_url,
            }
        self._seen: Set[str] = set()

    # ------------------------------------------------------------------
    # Individual source methods
    # ------------------------------------------------------------------

    def discover_ct_logs(self, limit: int = 500) -> List[DiscoveredDomain]:
        """Query crt.sh Certificate Transparency - brand-specific queries."""
        results: List[DiscoveredDomain] = []
        brand_queries = [
            '%paypal%', '%apple%login%', '%microsoft%verify%',
            '%google%secure%', '%amazon%update%', '%netflix%account%',
            '%chase%secure%', '%wellsfargo%verify%', '%facebook%login%',
            '%instagram%verify%', '%coinbase%wallet%', '%metamask%connect%',
            '%binance%secure%', '%docusign%sign%', '%dropbox%share%',
        ]
        for query in brand_queries[:10]:
            try:
                resp = self._session.get(
                    "https://crt.sh/",
                    params={"q": query, "output": "json"},
                    timeout=30,
                )
                if resp.status_code != 200:
                    continue
                entries = resp.json()[:100]
                for entry in entries:
                    cn = entry.get("common_name", "").strip().lstrip("*.")
                    name_value = entry.get("name_value", "")
                    domains: Set[str] = set()
                    if cn and '*' not in cn and '.' in cn:
                        domains.add(cn)
                    for name in name_value.split('\n'):
                        name = name.strip()
                        if name and '*' not in name and '.' in name:
                            domains.add(name)
                    for name in domains:
                        if name in self._seen:
                            continue
                        self._seen.add(name)
                        ds = self.scorer.score_domain(name)
                        if ds.score < self.score_threshold:
                            continue
                        results.append(
                            DiscoveredDomain(
                                domain=name,
                                url=f"https://{name}",
                                source="crt.sh",
                                phishing_score=ds.score,
                                score_reasons=ds.reasons,
                                metadata={"issuer": entry.get("issuer_name", "")},
                            )
                        )
                time.sleep(3)
            except Exception as exc:
                logger.debug("crt.sh query %s failed: %s", query, exc)
                continue
        logger.info("crt.sh: discovered %d domains (from %d brand queries)",
                     len(results), len(brand_queries[:10]))
        return results[:limit]

    def discover_urlscan(self, limit: int = 500) -> List[DiscoveredDomain]:
        """Fetch recent phishing-tagged scans from URLScan.io - multi-query."""
        results: List[DiscoveredDomain] = []
        queries = [
            "date:>now-7d AND task.tags:phishing",
            "date:>now-7d AND page.domain:*.pages.dev AND task.tags:phishing",
            "date:>now-7d AND page.domain:*.workers.dev AND task.tags:phishing",
            "date:>now-7d AND page.domain:*.netlify.app AND task.tags:phishing",
            "date:>now-7d AND category:phishing",
        ]
        per_query = max(limit // len(queries), 50)
        for query in queries:
            try:
                resp = self._session.get(
                    "https://urlscan.io/api/v1/search/",
                    params={"q": query, "size": per_query},
                    timeout=30,
                )
                if resp.status_code != 200:
                    continue
                data = resp.json()
                for result in data.get("results", []):
                    page = result.get("page", {})
                    domain = page.get("domain", "")
                    page_url = page.get("url", "")
                    if not domain or domain in self._seen:
                        continue
                    self._seen.add(domain)
                    ds = self.scorer.score_domain(domain)
                    ds.score = min(ds.score + 0.3, 1.0)
                    ds.reasons.append("urlscan_flagged")
                    if ds.score < self.score_threshold:
                        continue
                    results.append(
                        DiscoveredDomain(
                            domain=domain,
                            url=page_url or f"https://{domain}",
                            source="urlscan.io",
                            phishing_score=ds.score,
                            score_reasons=ds.reasons,
                            metadata={
                                "scan_id": result.get("_id", ""),
                                "country": page.get("country", ""),
                            },
                        )
                    )
                time.sleep(2)
            except Exception as exc:
                logger.debug("URLScan query failed: %s", exc)
                continue
        logger.info("urlscan.io: discovered %d domains", len(results))
        return results[:limit]

    def discover_nod_whoisds(self) -> List[DiscoveredDomain]:
        """Download today's Newly Observed Domains feed from WhoisDS."""
        results: List[DiscoveredDomain] = []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        url = f"https://whoisds.com/whois-database/newly-registered-domains/{today}.zip/nrd"
        try:
            resp = self._session.get(url, timeout=60)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                for name in zf.namelist():
                    with zf.open(name) as f:
                        text = f.read().decode("utf-8", errors="ignore")
                    for line in text.splitlines():
                        domain = line.strip().strip(".")
                        if not domain or '.' not in domain:
                            continue
                        if domain in self._seen:
                            continue
                        self._seen.add(domain)
                        ds = self.scorer.score_domain(domain)
                        if ds.score < self.score_threshold:
                            continue
                        results.append(
                            DiscoveredDomain(
                                domain=domain,
                                url=f"https://{domain}",
                                source="whoisds-nod",
                                phishing_score=ds.score,
                                score_reasons=ds.reasons,
                            )
                        )
        except Exception as exc:
            logger.warning("WhoisDS NOD feed failed: %s", exc)
        logger.info("whoisds-nod: discovered %d domains", len(results))
        return results

    def discover_openphish(self, limit: int = 500) -> List[DiscoveredDomain]:
        """Fetch active phishing URLs from OpenPhish community feed."""
        results: List[DiscoveredDomain] = []
        url = "https://openphish.com/feed.txt"
        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
            for line in resp.text.strip().splitlines()[:limit]:
                line = line.strip()
                if not line or not line.startswith("http"):
                    continue
                if not _is_web_page_url(line):
                    continue
                parsed = urlparse(line)
                domain = parsed.hostname or ""
                if not domain or domain in self._seen:
                    continue
                self._seen.add(domain)
                ds = self.scorer.score_domain(domain)
                ds.score = max(ds.score, 0.5)  # Known phishing
                ds.reasons.append("openphish_confirmed")
                results.append(
                    DiscoveredDomain(
                        domain=domain,
                        url=line,
                        source="openphish",
                        phishing_score=ds.score,
                        score_reasons=ds.reasons,
                    )
                )
        except Exception as exc:
            logger.warning("OpenPhish feed failed: %s", exc)
        logger.info("openphish: discovered %d domains", len(results))
        return results

    def discover_phishtank(self, limit: int = 500) -> List[DiscoveredDomain]:
        """Fetch verified phishing URLs from PhishTank."""
        results: List[DiscoveredDomain] = []
        url = "http://data.phishtank.com/data/online-valid.json"
        try:
            resp = self._session.get(
                url, timeout=20,
                headers={"User-Agent": "phishtrace/research (https://github.com/shangshang0/phish)"},
            )
            resp.raise_for_status()
            entries = resp.json()[:limit]
            for entry in entries:
                raw_url = entry.get("url", "")
                if not _is_web_page_url(raw_url):
                    continue
                parsed = urlparse(raw_url)
                domain = parsed.hostname or ""
                if not domain or domain in self._seen:
                    continue
                self._seen.add(domain)
                ds = self.scorer.score_domain(domain)
                ds.score = max(ds.score, 0.5)  # Verified phishing
                ds.reasons.append("phishtank_verified")
                results.append(
                    DiscoveredDomain(
                        domain=domain,
                        url=raw_url,
                        source="phishtank",
                        phishing_score=ds.score,
                        score_reasons=ds.reasons,
                        metadata={"phish_id": entry.get("phish_id", "")},
                    )
                )
        except Exception as exc:
            logger.warning("PhishTank feed failed: %s", exc)
        logger.info("phishtank: discovered %d domains", len(results))
        return results

    def discover_urlhaus(self, limit: int = 500) -> List[DiscoveredDomain]:
        """Fetch currently-online malicious URLs from abuse.ch URLhaus.

        Uses CSV download with fallback to plain-text endpoint.
        Filters out binary downloads to keep only web-page URLs.
        """
        results: List[DiscoveredDomain] = []
        url_csv = "https://urlhaus.abuse.ch/downloads/csv_online/"
        url_txt = "https://urlhaus.abuse.ch/downloads/text_online/"

        resp = None
        use_csv = False
        try:
            resp = self._session.get(url_csv, timeout=60)
            if resp.status_code == 200:
                use_csv = True
            else:
                resp = self._session.get(url_txt, timeout=30)
                resp.raise_for_status()
        except Exception as exc:
            logger.warning("URLhaus feed failed: %s", exc)
            return results

        count = 0
        if use_csv and resp:
            for line in resp.text.strip().split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                if count >= limit:
                    break
                parts = line.split('","')
                if len(parts) >= 3:
                    raw_url = parts[2].strip('"')
                else:
                    continue
                if not raw_url.startswith('http'):
                    continue
                if not _is_web_page_url(raw_url):
                    continue
                parsed = urlparse(raw_url)
                domain = parsed.hostname or ""
                if not domain or domain in self._seen:
                    continue
                self._seen.add(domain)
                ds = self.scorer.score_domain(domain)
                ds.score = max(ds.score, 0.4)
                ds.reasons.append("urlhaus_flagged")
                results.append(
                    DiscoveredDomain(
                        domain=domain,
                        url=raw_url,
                        source="urlhaus",
                        phishing_score=ds.score,
                        score_reasons=ds.reasons,
                    )
                )
                count += 1
        elif resp:
            for line in resp.text.strip().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if count >= limit:
                    break
                if not _is_web_page_url(line):
                    continue
                parsed = urlparse(line)
                domain = parsed.hostname or ""
                if not domain or domain in self._seen:
                    continue
                self._seen.add(domain)
                ds = self.scorer.score_domain(domain)
                ds.score = max(ds.score, 0.4)
                ds.reasons.append("urlhaus_flagged")
                results.append(
                    DiscoveredDomain(
                        domain=domain,
                        url=line,
                        source="urlhaus",
                        phishing_score=ds.score,
                        score_reasons=ds.reasons,
                    )
                )
                count += 1

        logger.info("urlhaus: discovered %d domains (filtered web-only)", len(results))
        return results

    # -- aggregate ---------------------------------------------------------

    def discover_all(self) -> List[DiscoveredDomain]:
        """Run every configured source and return de-duplicated, scored results."""
        all_results: List[DiscoveredDomain] = []
        for method in [
            self.discover_openphish,
            self.discover_phishtank,
            self.discover_ct_logs,
            self.discover_urlscan,
            self.discover_nod_whoisds,
            self.discover_urlhaus,
        ]:
            try:
                all_results.extend(method())
            except Exception as exc:
                logger.error("Source %s failed: %s", method.__name__, exc)

        # De-duplicate by domain (keep highest score)
        best: Dict[str, DiscoveredDomain] = {}
        for d in all_results:
            key = d.domain
            if key not in best or d.phishing_score > best[key].phishing_score:
                best[key] = d
        unique = sorted(best.values(), key=lambda x: x.phishing_score, reverse=True)

        high = sum(1 for d in unique if d.phishing_score >= 0.7)
        med = sum(1 for d in unique if 0.4 <= d.phishing_score < 0.7)
        low = sum(1 for d in unique if d.phishing_score < 0.4)
        logger.info(
            "Total discovered: %d unique domains (high=%d, med=%d, low=%d)",
            len(unique), high, med, low,
        )
        return unique

    # -- persistence -------------------------------------------------------

    def save(
        self,
        domains: List[DiscoveredDomain],
        output_path: Path,
        append: bool = True,
    ) -> None:
        """Write discovered domains to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with open(output_path, mode) as fh:
            for d in domains:
                fh.write(json.dumps(d.as_dict()) + "\n")
        logger.info("Saved %d domains to %s", len(domains), output_path)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def run_discovery(
    output_dir: Path,
    loop: bool = False,
    interval: int = 3600,
    proxy: Optional[str] = None,
) -> None:
    """Entry-point for CLI or orchestrator invocation."""
    scanner = WildDomainDiscovery(
        use_proxy=bool(proxy),
        proxy_url=proxy,
        keyword_filter=True,
        score_threshold=0.20,
    )
    output_file = output_dir / "discovered.jsonl"

    while True:
        logger.info("Starting discovery cycle ...")
        domains = scanner.discover_all()
        if domains:
            scanner.save(domains, output_file)
        logger.info("Cycle complete - sleeping %ds", interval)
        if not loop:
            break
        time.sleep(interval)


def main() -> None:  # pragma: no cover
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="PhishTrace Wild Domain Discovery"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/discovery"),
        help="Directory to write discovered.jsonl",
    )
    parser.add_argument("--loop", action="store_true", help="Run continuously")
    parser.add_argument(
        "--interval",
        type=int,
        default=3600,
        help="Seconds between discovery cycles (default: 3600)",
    )
    parser.add_argument("--proxy", type=str, default=None, help="HTTP proxy URL")

    args = parser.parse_args()
    run_discovery(
        output_dir=args.output_dir,
        loop=args.loop,
        interval=args.interval,
        proxy=args.proxy,
    )


if __name__ == "__main__":
    main()
