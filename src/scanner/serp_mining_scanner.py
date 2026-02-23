"""
PhishTrace Search Engine Mining Scanner
=========================================
Proactive phishing website discovery via search engine result page (SERP)
mining. Inspired by LOKI (NDSS'26): "Proactively Discovering Online Scam
Websites by Mining Toxic Search Queries" (Paudel & Stringhini).

Core idea: Generate candidate search queries that are likely to surface
phishing/scam websites in SERP results. Score each query for "toxicity"
(fraction of malicious results), then systematically collect and verify
the discovered websites.

Pipeline:
  1. Seed Query Generation — from known phishing URLs, brand lists,
     and credential-harvesting keyword templates
  2. Query Expansion — auto-suggest, related queries, combinatorial
  3. SERP Collection — submit queries to multiple search engines
  4. Result Scoring — heuristic + ML scoring of SERP results
  5. Candidate Verification — feed high-score URLs to PhishTrace crawler
  6. Continuous Discovery — scheduled re-scanning for fresh phishing sites

Search Engines Supported:
  - Google Custom Search API (100 free queries/day)
  - Bing Web Search API (1000 free queries/month)
  - DuckDuckGo (scraping, no API key needed)
  - Google Suggest API (auto-complete, free)

This module integrates with:
  - src/scanner/wild_scanner.py (PhishingDomainScorer)
  - src/crawler/deep_recursive_crawler.py (trace collection)
  - src/validator/trace_validator.py (quality filtering)
"""

import asyncio
import json
import time
import re
import logging
import hashlib
import os
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Tuple, Iterator
from urllib.parse import urlparse, quote_plus, urljoin
from dataclasses import dataclass, field, asdict
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product

import requests

logger = logging.getLogger(__name__)

_proxy_url = os.environ.get("PHISHTRACE_PROXY", "http://127.0.0.1:10809")
PROXY = {"http": _proxy_url, "https": _proxy_url}

# ============================================================================
# Toxic Query Generator
# ============================================================================

@dataclass
class ToxicQuery:
    """A search query scored for phishing toxicity."""
    query: str
    toxicity_score: float  # 0.0 - 1.0 (higher = more likely to return phishing)
    category: str  # e.g. 'credential_harvest', 'brand_impersonation'
    source: str  # how this query was generated
    expansion_score: float = 0.0  # how many unique phishing sites discovered
    results_collected: int = 0
    phishing_found: int = 0


class ToxicQueryGenerator:
    """
    Generate candidate search queries likely to surface phishing websites.

    Strategies (inspired by LOKI Section IV):
      1. Brand + Action templates (e.g., "paypal login verify account")
      2. Credential harvesting keywords (e.g., "enter password secure")
      3. Urgency + Brand combinations (e.g., "urgent apple id suspended")
      4. Typosquatting brand variations (e.g., "paypa1 login")
      5. URL-pattern queries (e.g., "webmail login.php")
      6. Multi-language phishing queries (e.g., "连接钱包 验证")
    """

    # --- Top targeted brands (from APWG, PhishIntention, our data) ---
    BRANDS = [
        'paypal', 'apple', 'microsoft', 'google', 'amazon', 'netflix',
        'facebook', 'instagram', 'chase', 'wellsfargo', 'bank of america',
        'coinbase', 'binance', 'metamask', 'outlook', 'office 365',
        'linkedin', 'twitter', 'dropbox', 'docusign', 'adobe',
        'usps', 'fedex', 'dhl', 'whatsapp', 'telegram',
        'steam', 'roblox', 'epic games', 'stripe', 'venmo', 'cashapp',
        'walmart', 'ebay', 'target', 'costco', 'bestbuy',
        'hsbc', 'barclays', 'citibank', 'capital one', 'american express',
        'blockchain', 'opensea', 'ledger', 'crypto.com', 'trust wallet',
    ]

    # --- Credential harvesting action verbs ---
    ACTIONS = [
        'login', 'sign in', 'log in', 'verify', 'confirm',
        'update', 'secure', 'unlock', 'restore', 'recover',
        'reset password', 'authenticate', 'validate',
        'connect wallet', 'claim reward', 'airdrop',
    ]

    # --- Urgency / social engineering phrases ---
    URGENCY = [
        'urgent', 'immediately', 'suspended', 'locked',
        'unusual activity', 'unauthorized', 'expire',
        'limited time', 'action required', 'verify identity',
        'security alert', 'compromised', 'confirm account',
    ]

    # --- URL/technical patterns common in phishing ---
    TECHNICAL_PATTERNS = [
        'login.php', 'signin.html', 'verify.php', 'webmail',
        'secure-login', 'account-update', 'billing-update',
        'password-reset', 'confirm-identity', 'wallet-connect',
        'customer-support', 'help-center',
    ]

    # --- Scam categories from LOKI paper ---
    SCAM_CATEGORIES = {
        'credential_harvest': {
            'templates': ['{brand} login', '{brand} sign in page', '{brand} verify account'],
            'weight': 3.0,
        },
        'brand_impersonation': {
            'templates': ['{brand} official site', '{brand} customer service', '{brand} help center'],
            'weight': 2.0,
        },
        'financial_fraud': {
            'templates': ['{brand} payment update', '{brand} billing verify', '{brand} refund process'],
            'weight': 2.5,
        },
        'crypto_scam': {
            'templates': ['connect wallet {brand}', '{brand} airdrop claim', '{brand} free tokens', 'defi swap {brand}'],
            'weight': 2.0,
        },
        'urgency_social_eng': {
            'templates': ['{brand} {urgency}', '{brand} account {urgency}'],
            'weight': 2.5,
        },
        'technical_phishing': {
            'templates': ['{brand} {tech_pattern}', 'webmail {brand} login'],
            'weight': 1.5,
        },
    }

    def generate_queries(
        self,
        max_queries: int = 2000,
        brands: List[str] = None,
    ) -> List[ToxicQuery]:
        """
        Generate toxic search queries using template-based expansion.

        Returns sorted by estimated toxicity score.
        """
        brands = brands or self.BRANDS
        queries: Dict[str, ToxicQuery] = {}

        # Strategy 1: Brand + Action combinations
        for brand, action in product(brands, self.ACTIONS):
            q = f"{brand} {action}"
            queries[q.lower()] = ToxicQuery(
                query=q, toxicity_score=0.0,
                category='credential_harvest',
                source='brand_action_template',
            )

        # Strategy 2: Brand + Urgency combinations
        for brand, urgency in product(brands[:20], self.URGENCY):
            q = f"{brand} {urgency}"
            queries[q.lower()] = ToxicQuery(
                query=q, toxicity_score=0.0,
                category='urgency_social_eng',
                source='brand_urgency_template',
            )

        # Strategy 3: Category-specific templates
        for cat_name, cat_config in self.SCAM_CATEGORIES.items():
            for template in cat_config['templates']:
                for brand in brands:
                    q = template.replace('{brand}', brand)
                    if '{urgency}' in q:
                        for urgency in random.sample(self.URGENCY, min(3, len(self.URGENCY))):
                            q2 = q.replace('{urgency}', urgency)
                            queries[q2.lower()] = ToxicQuery(
                                query=q2, toxicity_score=0.0,
                                category=cat_name, source='category_template',
                            )
                    elif '{tech_pattern}' in q:
                        for tech in random.sample(self.TECHNICAL_PATTERNS, min(3, len(self.TECHNICAL_PATTERNS))):
                            q2 = q.replace('{tech_pattern}', tech)
                            queries[q2.lower()] = ToxicQuery(
                                query=q2, toxicity_score=0.0,
                                category=cat_name, source='category_template',
                            )
                    else:
                        queries[q.lower()] = ToxicQuery(
                            query=q, toxicity_score=0.0,
                            category=cat_name, source='category_template',
                        )

        # Strategy 4: Technical URL-pattern queries
        for tech in self.TECHNICAL_PATTERNS:
            queries[tech.lower()] = ToxicQuery(
                query=tech, toxicity_score=0.0,
                category='technical_phishing',
                source='tech_pattern',
            )

        # Score all queries
        scored = list(queries.values())
        for tq in scored:
            tq.toxicity_score = self._estimate_toxicity(tq)

        # Sort by estimated toxicity (highest first)
        scored.sort(key=lambda x: x.toxicity_score, reverse=True)

        return scored[:max_queries]

    def _estimate_toxicity(self, tq: ToxicQuery) -> float:
        """
        Estimate the toxicity of a query before submitting it to search engines.
        Uses heuristic scoring based on query composition.

        This is a pre-filter; actual toxicity is measured after SERP collection.
        """
        score = 0.0
        q = tq.query.lower()

        # Category weight
        cat_weight = self.SCAM_CATEGORIES.get(tq.category, {}).get('weight', 1.0)
        score += cat_weight * 0.1

        # Brand presence
        brand_count = sum(1 for b in self.BRANDS if b.lower() in q)
        score += min(brand_count * 0.15, 0.3)

        # Action verb presence
        action_count = sum(1 for a in self.ACTIONS if a.lower() in q)
        score += min(action_count * 0.1, 0.2)

        # Urgency keyword presence
        urgency_count = sum(1 for u in self.URGENCY if u.lower() in q)
        score += min(urgency_count * 0.1, 0.2)

        # Technical pattern presence
        tech_count = sum(1 for t in self.TECHNICAL_PATTERNS if t.lower() in q)
        score += min(tech_count * 0.05, 0.1)

        # Multi-word queries are usually more targeted
        word_count = len(q.split())
        if word_count >= 3:
            score += 0.05
        elif word_count >= 4:
            score += 0.1

        return min(score, 1.0)


# ============================================================================
# SERP Collector (Search Engine Results Page)
# ============================================================================

@dataclass
class SERPResult:
    """A single search engine result."""
    url: str
    title: str
    snippet: str
    position: int  # rank in results
    search_engine: str
    query: str
    domain: str = ""
    phishing_score: float = 0.0
    collected_at: str = ""

    def __post_init__(self):
        if not self.domain:
            try:
                self.domain = urlparse(self.url).netloc
            except Exception:
                self.domain = ""
        if not self.collected_at:
            self.collected_at = datetime.utcnow().isoformat()


class SERPCollector:
    """
    Collect search engine results for toxic queries.

    Supports multiple search engines for broader coverage
    (as LOKI uses Google, Bing, Baidu, Naver).
    """

    def __init__(self, use_proxy: bool = True):
        self.session = requests.Session()
        if use_proxy:
            self.session.proxies = PROXY
        self.session.headers['User-Agent'] = (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/120.0.0.0 Safari/537.36'
        )
        self.session.verify = False
        self.stats = Counter()

        # API keys (from environment)
        self.google_api_key = os.environ.get('GOOGLE_API_KEY', '')
        self.google_cx = os.environ.get('GOOGLE_CX', '')
        self.bing_api_key = os.environ.get('BING_API_KEY', '')

    def search_duckduckgo(
        self, query: str, max_results: int = 30
    ) -> List[SERPResult]:
        """
        Search DuckDuckGo HTML (no API key required).
        Uses the HTML version to extract organic results.
        """
        results = []
        try:
            resp = self.session.get(
                'https://html.duckduckgo.com/html/',
                params={'q': query},
                timeout=30,
            )
            if resp.status_code == 200:
                # Parse results from HTML
                from html.parser import HTMLParser
                results = self._parse_ddg_html(resp.text, query, max_results)
                self.stats['duckduckgo'] += len(results)
        except Exception as e:
            logger.debug(f"DuckDuckGo search error: {e}")
        return results

    def _parse_ddg_html(
        self, html: str, query: str, max_results: int
    ) -> List[SERPResult]:
        """Parse DuckDuckGo HTML results."""
        results = []
        # Extract result links - DDG uses class="result__a"
        link_pattern = re.compile(
            r'<a\s+[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            re.DOTALL
        )
        snippet_pattern = re.compile(
            r'<a\s+class="result__snippet"[^>]*>(.*?)</a>',
            re.DOTALL
        )

        links = link_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i, (url, title) in enumerate(links[:max_results]):
            # DDG wraps URLs through their redirect
            if 'uddg=' in url:
                import urllib.parse
                parsed = urllib.parse.parse_qs(
                    urllib.parse.urlparse(url).query
                )
                url = parsed.get('uddg', [url])[0]

            # Clean HTML tags from title
            title = re.sub(r'<[^>]+>', '', title).strip()
            snippet = ''
            if i < len(snippets):
                snippet = re.sub(r'<[^>]+>', '', snippets[i]).strip()

            if url.startswith('http'):
                results.append(SERPResult(
                    url=url, title=title, snippet=snippet,
                    position=i + 1, search_engine='duckduckgo',
                    query=query,
                ))

        return results

    def search_bing_api(
        self, query: str, max_results: int = 50
    ) -> List[SERPResult]:
        """
        Search via Bing Web Search API (requires API key).
        Free tier: 1000 queries/month, 3 req/sec.
        """
        if not self.bing_api_key:
            return []

        results = []
        try:
            resp = self.session.get(
                'https://api.bing.microsoft.com/v7.0/search',
                params={'q': query, 'count': min(max_results, 50)},
                headers={'Ocp-Apim-Subscription-Key': self.bing_api_key},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                for i, result in enumerate(
                    data.get('webPages', {}).get('value', [])
                ):
                    results.append(SERPResult(
                        url=result.get('url', ''),
                        title=result.get('name', ''),
                        snippet=result.get('snippet', ''),
                        position=i + 1,
                        search_engine='bing',
                        query=query,
                    ))
                self.stats['bing'] += len(results)
        except Exception as e:
            logger.debug(f"Bing API error: {e}")
        return results

    def search_google_api(
        self, query: str, max_results: int = 10
    ) -> List[SERPResult]:
        """
        Search via Google Custom Search API (requires API key + CX).
        Free tier: 100 queries/day.
        """
        if not self.google_api_key or not self.google_cx:
            return []

        results = []
        try:
            resp = self.session.get(
                'https://www.googleapis.com/customsearch/v1',
                params={
                    'key': self.google_api_key,
                    'cx': self.google_cx,
                    'q': query,
                    'num': min(max_results, 10),
                },
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                for i, item in enumerate(data.get('items', [])):
                    results.append(SERPResult(
                        url=item.get('link', ''),
                        title=item.get('title', ''),
                        snippet=item.get('snippet', ''),
                        position=i + 1,
                        search_engine='google',
                        query=query,
                    ))
                self.stats['google'] += len(results)
        except Exception as e:
            logger.debug(f"Google API error: {e}")
        return results

    def search_google_suggest(self, query: str) -> List[str]:
        """
        Get Google query suggestions (auto-complete).
        Useful for query expansion — finds related queries
        that real users search for.
        """
        try:
            resp = self.session.get(
                'https://suggestqueries.google.com/complete/search',
                params={'client': 'firefox', 'q': query},
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if isinstance(data, list) and len(data) > 1:
                    return data[1][:10]
        except Exception as e:
            logger.debug(f"Google suggest error: {e}")
        return []

    def search_all(
        self, query: str, max_per_engine: int = 30
    ) -> List[SERPResult]:
        """
        Search across all available engines for a query.
        Returns deduplicated results sorted by position.
        """
        all_results = []

        # DuckDuckGo (always available, no API key)
        all_results.extend(self.search_duckduckgo(query, max_per_engine))

        # Bing (if API key available)
        if self.bing_api_key:
            time.sleep(0.5)
            all_results.extend(self.search_bing_api(query, max_per_engine))

        # Google (if API key available)
        if self.google_api_key:
            time.sleep(0.5)
            all_results.extend(self.search_google_api(query, min(max_per_engine, 10)))

        # Deduplicate by domain
        seen_urls = set()
        unique = []
        for r in all_results:
            url_key = r.url.lower().rstrip('/')
            if url_key not in seen_urls:
                seen_urls.add(url_key)
                unique.append(r)

        return unique


# ============================================================================
# SERP Result Scorer (Phishing Likelihood)
# ============================================================================

class SERPPhishingScorer:
    """
    Score SERP results for phishing likelihood.

    Uses signals from the SERP metadata itself (without crawling):
      - Domain reputation heuristics
      - Title/snippet keyword analysis
      - URL structure analysis
      - Position in search results
      - Cross-engine consistency

    This is a lightweight pre-filter before expensive crawling.
    """

    # Import the domain scorer from wild_scanner
    def __init__(self):
        try:
            from .wild_scanner import PhishingDomainScorer
            self.domain_scorer = PhishingDomainScorer()
        except ImportError:
            try:
                from src.scanner.wild_scanner import PhishingDomainScorer
                self.domain_scorer = PhishingDomainScorer()
            except ImportError:
                self.domain_scorer = None

    # Suspicious snippet/title keywords
    PHISH_SNIPPET_KEYWORDS = {
        'verify your', 'confirm your', 'update your', 'secure your',
        'unusual activity', 'suspended', 'locked', 'unauthorized',
        'click here to', 'enter your password', 'login required',
        'account verification', 'identity verification', 'security alert',
        'reset your password', 'claim your', 'free reward',
        'limited time offer', 'act now', 'immediate action',
        'connect your wallet', 'claim airdrop', 'mint now',
    }

    def score_result(self, result: SERPResult) -> float:
        """Score a single SERP result for phishing likelihood (0.0-1.0)."""
        score = 0.0

        # 1. Domain-level scoring
        if self.domain_scorer:
            ds = self.domain_scorer.score_domain(result.domain)
            score += ds.score * 0.4  # Weight domain analysis

        # 2. Snippet keyword analysis
        combined = (result.title + ' ' + result.snippet).lower()
        keyword_hits = sum(
            1 for kw in self.PHISH_SNIPPET_KEYWORDS
            if kw in combined
        )
        score += min(keyword_hits * 0.08, 0.25)

        # 3. URL structure analysis
        url_lower = result.url.lower()
        # Check for IP addresses in URL
        if re.search(r'https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', url_lower):
            score += 0.15

        # Check for suspicious paths
        suspicious_paths = [
            '/login', '/signin', '/verify', '/secure', '/update',
            '/confirm', '/account', '/wallet', '/connect',
        ]
        path_hits = sum(1 for p in suspicious_paths if p in url_lower)
        score += min(path_hits * 0.05, 0.15)

        # 4. Low search position (phishing often appears lower)
        if result.position > 10:
            score += 0.05
        elif result.position > 5:
            score += 0.03

        # 5. Title mismatch with domain (e.g., title says "PayPal" but domain isn't paypal.com)
        title_lower = result.title.lower()
        try:
            from .wild_scanner import PhishingDomainScorer as _PDS
        except ImportError:
            from src.scanner.wild_scanner import PhishingDomainScorer as _PDS
        for brand in _PDS.TARGET_BRANDS:
            if brand in title_lower and brand not in result.domain.lower():
                score += 0.2
                break

        return min(score, 1.0)

    def score_results(
        self, results: List[SERPResult], threshold: float = 0.25
    ) -> List[SERPResult]:
        """Score all results and filter by threshold."""
        scored = []
        for r in results:
            r.phishing_score = self.score_result(r)
            if r.phishing_score >= threshold:
                scored.append(r)
        scored.sort(key=lambda x: x.phishing_score, reverse=True)
        return scored


# ============================================================================
# Search Engine Mining Pipeline
# ============================================================================

class SearchEngineMiningPipeline:
    """
    End-to-end pipeline for proactive phishing discovery via SERP mining.

    Stages:
      1. Generate toxic queries
      2. Submit to search engines
      3. Score SERP results for phishing
      4. Output candidate URLs for crawling + verification
      5. Compute query toxicity scores (actual, based on results)

    This implements a simplified version of LOKI's approach, adapted
    for PhishTrace's trace-based detection paradigm.
    """

    def __init__(
        self,
        use_proxy: bool = True,
        max_queries: int = 200,
        results_per_query: int = 30,
        phishing_score_threshold: float = 0.25,
        delay_between_queries: float = 3.0,
    ):
        self.query_gen = ToxicQueryGenerator()
        self.serp_collector = SERPCollector(use_proxy=use_proxy)
        self.scorer = SERPPhishingScorer()
        self.max_queries = max_queries
        self.results_per_query = results_per_query
        self.threshold = phishing_score_threshold
        self.delay = delay_between_queries

        self.all_results: List[SERPResult] = []
        self.scored_results: List[SERPResult] = []
        self.query_toxicity: Dict[str, float] = {}
        self.seen_domains: Set[str] = set()
        self.stats = Counter()

    def run(
        self,
        brands: List[str] = None,
        categories: List[str] = None,
        output_dir: Path = None,
    ) -> Dict:
        """
        Execute the full mining pipeline.

        Returns a report dict with discovered candidates.
        """
        output_dir = output_dir or Path(__file__).parent.parent.parent / 'dataset'
        start_time = time.time()

        # Stage 1: Generate queries
        logger.info("=" * 60)
        logger.info("Stage 1: Generating toxic search queries")
        logger.info("=" * 60)
        queries = self.query_gen.generate_queries(
            max_queries=self.max_queries,
            brands=brands,
        )

        # Filter by category if specified
        if categories:
            queries = [q for q in queries if q.category in categories]

        logger.info(f"Generated {len(queries)} candidate queries")
        cat_dist = Counter(q.category for q in queries)
        for cat, count in cat_dist.most_common():
            logger.info(f"  {cat}: {count}")

        # Stage 2: Collect SERP results
        logger.info("\n" + "=" * 60)
        logger.info("Stage 2: Collecting SERP results")
        logger.info("=" * 60)

        for i, tq in enumerate(queries):
            if i >= self.max_queries:
                break

            logger.info(
                f"[{i+1}/{min(len(queries), self.max_queries)}] "
                f"Query: '{tq.query}' (est_tox={tq.toxicity_score:.2f})"
            )

            try:
                results = self.serp_collector.search_all(
                    tq.query, max_per_engine=self.results_per_query
                )
                tq.results_collected = len(results)

                # Score results
                scored = self.scorer.score_results(results, self.threshold)
                tq.phishing_found = len(scored)

                # Compute actual query toxicity
                if results:
                    tq.toxicity_score = len(scored) / len(results)
                    self.query_toxicity[tq.query] = tq.toxicity_score

                # Collect unique results
                for r in scored:
                    if r.domain not in self.seen_domains:
                        self.seen_domains.add(r.domain)
                        self.scored_results.append(r)
                        self.stats['unique_candidates'] += 1

                self.all_results.extend(results)
                self.stats['total_results'] += len(results)
                self.stats['queries_completed'] += 1

                logger.info(
                    f"  Got {len(results)} results, "
                    f"{len(scored)} suspicious (toxicity={tq.toxicity_score:.2f})"
                )

            except Exception as e:
                logger.error(f"  Error: {e}")
                self.stats['query_errors'] += 1

            # Rate limiting
            time.sleep(self.delay)

        # Stage 3: Aggregate and rank
        logger.info("\n" + "=" * 60)
        logger.info("Stage 3: Aggregating and ranking candidates")
        logger.info("=" * 60)

        self.scored_results.sort(
            key=lambda x: x.phishing_score, reverse=True
        )

        # Stage 4: Query expansion (use high-toxicity queries for auto-suggest)
        high_tox_queries = [
            q for q in queries if q.toxicity_score > 0.3
        ][:20]

        if high_tox_queries:
            logger.info(f"\nExpanding {len(high_tox_queries)} high-toxicity queries via auto-suggest")
            expanded_queries = []
            for tq in high_tox_queries:
                suggestions = self.serp_collector.search_google_suggest(tq.query)
                for sugg in suggestions:
                    if sugg.lower() not in self.query_toxicity:
                        expanded_queries.append(ToxicQuery(
                            query=sugg,
                            toxicity_score=tq.toxicity_score * 0.8,  # inherit parent score
                            category=tq.category,
                            source='auto_suggest_expansion',
                        ))
                time.sleep(1)

            logger.info(f"  Generated {len(expanded_queries)} expanded queries")

            # Search expanded queries (limited batch)
            for i, eq in enumerate(expanded_queries[:50]):
                try:
                    results = self.serp_collector.search_duckduckgo(
                        eq.query, max_results=20
                    )
                    scored = self.scorer.score_results(results, self.threshold)
                    for r in scored:
                        if r.domain not in self.seen_domains:
                            self.seen_domains.add(r.domain)
                            self.scored_results.append(r)
                            self.stats['expanded_candidates'] += 1
                    time.sleep(self.delay)
                except Exception as e:
                    logger.debug(f"Expansion search error: {e}")

        # Stage 5: Generate report
        elapsed = time.time() - start_time
        report = self._generate_report(queries, elapsed)

        # Save results
        self._save_results(output_dir, report)

        return report

    def _generate_report(self, queries: List[ToxicQuery], elapsed: float) -> Dict:
        """Generate a comprehensive mining report."""
        # Top toxic queries
        top_queries = sorted(
            [q for q in queries if q.results_collected > 0],
            key=lambda x: x.toxicity_score,
            reverse=True,
        )[:20]

        # Score distribution
        score_bins = Counter()
        for r in self.scored_results:
            if r.phishing_score >= 0.7:
                score_bins['high (>=0.7)'] += 1
            elif r.phishing_score >= 0.4:
                score_bins['medium (0.4-0.7)'] += 1
            else:
                score_bins['low (<0.4)'] += 1

        # Category distribution
        cat_dist = Counter(r.query.split()[0] for r in self.scored_results)

        # Search engine distribution
        engine_dist = Counter(r.search_engine for r in self.scored_results)

        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'elapsed_seconds': elapsed,
            'stats': dict(self.stats),
            'summary': {
                'total_queries': self.stats.get('queries_completed', 0),
                'total_results': self.stats.get('total_results', 0),
                'unique_candidates': len(self.scored_results),
                'unique_domains': len(self.seen_domains),
            },
            'top_toxic_queries': [
                {
                    'query': q.query,
                    'toxicity': q.toxicity_score,
                    'category': q.category,
                    'results': q.results_collected,
                    'phishing_found': q.phishing_found,
                }
                for q in top_queries
            ],
            'score_distribution': dict(score_bins),
            'engine_distribution': dict(engine_dist),
            'candidates': [
                {
                    'url': r.url,
                    'domain': r.domain,
                    'title': r.title,
                    'phishing_score': r.phishing_score,
                    'search_engine': r.search_engine,
                    'query': r.query,
                    'position': r.position,
                }
                for r in self.scored_results[:1000]  # Cap at 1000
            ],
        }

        return report

    def _save_results(self, output_dir: Path, report: Dict):
        """Save mining results to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save full report
        report_path = output_dir / 'serp_mining_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Report saved to: {report_path}")

        # Save candidate URLs as crawl targets (compatible with wild_scanner)
        targets_path = output_dir / 'serp_crawl_targets.json'
        targets = [
            {
                'url': r.url,
                'label': 'phishing',
                'source': f'serp_mining_{r.search_engine}',
                'phishing_score': r.phishing_score,
                'reasons': [f'query:{r.query}', f'position:{r.position}'],
            }
            for r in self.scored_results
        ]
        with open(targets_path, 'w', encoding='utf-8') as f:
            json.dump(targets, f, indent=2, default=str)
        logger.info(f"Crawl targets ({len(targets)}) saved to: {targets_path}")

    def get_crawl_targets(self, max_targets: int = 500) -> List[Dict]:
        """Get crawl-ready targets from mining results."""
        return [
            {
                'url': r.url,
                'label': 'phishing',
                'source': f'serp_mining_{r.search_engine}',
                'phishing_score': r.phishing_score,
                'reasons': [f'query:{r.query}'],
            }
            for r in self.scored_results[:max_targets]
        ]


# ============================================================================
# Continuous Discovery Scheduler
# ============================================================================

class ContinuousDiscoveryScheduler:
    """
    Runs search engine mining on a schedule for continuous phishing discovery.

    Maintains state between runs:
      - Previously discovered domains (avoid re-crawling)
      - Query performance history (prioritize high-toxicity queries)
      - Discovery rate tracking
    """

    def __init__(
        self,
        state_dir: Path = None,
        scan_interval_hours: int = 6,
        max_queries_per_run: int = 100,
    ):
        self.state_dir = state_dir or Path(__file__).parent.parent.parent / 'dataset' / 'discovery_state'
        self.scan_interval = timedelta(hours=scan_interval_hours)
        self.max_queries = max_queries_per_run
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self.discovered_domains: Set[str] = set()
        self.query_history: Dict[str, Dict] = {}
        self._load_state()

    def _load_state(self):
        """Load persistent state from disk."""
        domains_file = self.state_dir / 'discovered_domains.json'
        if domains_file.exists():
            with open(domains_file, 'r') as f:
                self.discovered_domains = set(json.load(f))
            logger.info(f"Loaded {len(self.discovered_domains)} previously discovered domains")

        history_file = self.state_dir / 'query_history.json'
        if history_file.exists():
            with open(history_file, 'r') as f:
                self.query_history = json.load(f)

    def _save_state(self):
        """Persist state to disk."""
        domains_file = self.state_dir / 'discovered_domains.json'
        with open(domains_file, 'w') as f:
            json.dump(list(self.discovered_domains), f)

        history_file = self.state_dir / 'query_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.query_history, f, indent=2, default=str)

    def run_discovery_cycle(self) -> Dict:
        """
        Run a single discovery cycle.
        Returns report with newly discovered candidates.
        """
        pipeline = SearchEngineMiningPipeline(
            max_queries=self.max_queries,
            delay_between_queries=3.0,
        )

        # Run pipeline
        report = pipeline.run(
            output_dir=self.state_dir.parent,
        )

        # Update state with newly discovered domains
        new_domains = 0
        for candidate in report.get('candidates', []):
            domain = candidate.get('domain', '')
            if domain and domain not in self.discovered_domains:
                self.discovered_domains.add(domain)
                new_domains += 1

        # Update query history
        for tq_data in report.get('top_toxic_queries', []):
            query = tq_data['query']
            if query not in self.query_history:
                self.query_history[query] = {
                    'first_seen': datetime.utcnow().isoformat(),
                    'runs': 0,
                    'total_phishing': 0,
                    'avg_toxicity': 0,
                }
            h = self.query_history[query]
            h['runs'] += 1
            h['total_phishing'] += tq_data.get('phishing_found', 0)
            h['avg_toxicity'] = (
                (h['avg_toxicity'] * (h['runs'] - 1) + tq_data['toxicity'])
                / h['runs']
            )
            h['last_run'] = datetime.utcnow().isoformat()

        self._save_state()

        report['new_domains'] = new_domains
        report['total_known_domains'] = len(self.discovered_domains)

        logger.info(
            f"\nDiscovery cycle complete: "
            f"{new_domains} new domains, "
            f"{len(self.discovered_domains)} total known"
        )

        return report


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='PhishTrace SERP Mining Scanner — Proactive phishing discovery'
    )
    parser.add_argument('--max-queries', type=int, default=50,
                        help='Maximum number of queries to submit')
    parser.add_argument('--results-per-query', type=int, default=30,
                        help='Max results per query per engine')
    parser.add_argument('--threshold', type=float, default=0.25,
                        help='Phishing score threshold for candidates')
    parser.add_argument('--delay', type=float, default=3.0,
                        help='Delay between queries (seconds)')
    parser.add_argument('--output-dir', type=str,
                        default=str(Path(__file__).parent.parent.parent / 'dataset'),
                        help='Output directory for results')
    parser.add_argument('--continuous', action='store_true',
                        help='Run in continuous discovery mode')
    parser.add_argument('--dry-run', action='store_true',
                        help='Generate queries without searching')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s [%(name)s] %(levelname)s: %(message)s'
    )
    logging.getLogger('urllib3').setLevel(logging.WARNING)

    import warnings
    import urllib3
    urllib3.disable_warnings()
    warnings.filterwarnings('ignore')

    if args.dry_run:
        print("=" * 60)
        print("SERP Mining — Dry Run (query generation only)")
        print("=" * 60)
        gen = ToxicQueryGenerator()
        queries = gen.generate_queries(max_queries=args.max_queries)
        print(f"\nGenerated {len(queries)} queries:")
        cat_dist = Counter(q.category for q in queries)
        for cat, count in cat_dist.most_common():
            print(f"  {cat}: {count}")
        print(f"\nTop 20 by estimated toxicity:")
        for q in queries[:20]:
            print(f"  [{q.toxicity_score:.2f}] ({q.category}) {q.query}")
        return

    if args.continuous:
        scheduler = ContinuousDiscoveryScheduler(
            max_queries_per_run=args.max_queries,
        )
        report = scheduler.run_discovery_cycle()
    else:
        pipeline = SearchEngineMiningPipeline(
            max_queries=args.max_queries,
            results_per_query=args.results_per_query,
            phishing_score_threshold=args.threshold,
            delay_between_queries=args.delay,
        )
        report = pipeline.run(output_dir=Path(args.output_dir))

    # Print summary
    summary = report.get('summary', {})
    print(f"\n{'='*60}")
    print(f"Mining Complete")
    print(f"{'='*60}")
    print(f"Queries submitted: {summary.get('total_queries', 0)}")
    print(f"Total SERP results: {summary.get('total_results', 0)}")
    print(f"Candidate phishing URLs: {summary.get('unique_candidates', 0)}")
    print(f"Unique suspicious domains: {summary.get('unique_domains', 0)}")

    top_queries = report.get('top_toxic_queries', [])[:10]
    if top_queries:
        print(f"\nTop toxic queries:")
        for tq in top_queries:
            print(
                f"  [{tq['toxicity']:.2f}] {tq['query']} "
                f"({tq['phishing_found']}/{tq['results']} phishing)"
            )


if __name__ == '__main__':
    main()
