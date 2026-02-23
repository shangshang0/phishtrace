#!/usr/bin/env python3
"""
daemon_discovery.py — Background Discovery + Crawl + Detection Pipeline
========================================================================
Continuously:
  1. Runs wild domain discovery (CT logs, URLScan, NOD, URLhaus)
  2. Validates liveness of discovered URLs (HTTP HEAD check)
  3. Crawls live URLs with PhishingCrawler (trace + screenshots)
  4. Generates baseline-compatible files (info.txt + shot.png per trace)
  5. Runs PhishTraceDetector on each trace to classify phishing vs benign
  6. Stores confirmed phishing traces in a separate showcase dataset

Output layout under DATA_ROOT/discovery/:
  urls/
    discovered_YYYYMMDD.jsonl          # raw discovery output (per day)
    validated_YYYYMMDD.jsonl           # liveness-checked URLs
  traces/
    <url_hash>/
      trace.json                       # full interaction trace
      screenshots/                     # per-step screenshots
        step_001_initial.png ...
      info.txt                         # URL (baseline compat)
      shot.png                         # landing screenshot (baseline compat)
  confirmed_phishing/
    <url_hash>/
      trace.json
      screenshots/
      info.txt
      shot.png
      detection.json                   # detection verdict + confidence
  stats.json                           # running statistics

Usage:
    python scripts/daemon_discovery.py                         # single cycle
    python scripts/daemon_discovery.py --loop --interval 3600  # every hour
    python scripts/daemon_discovery.py --crawl-only            # skip discovery, crawl pending
    python scripts/daemon_discovery.py --detect-only           # skip crawl, classify existing
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DATASET_DIR = Path(os.environ.get("PHISHTRACE_DATA",
                                   PROJECT_ROOT / "data")) / "dataset"
DISCOVERY_DIR = DATASET_DIR / "discovery"
DISCOVERY_URLS_DIR = DISCOVERY_DIR / "urls"
DISCOVERY_TRACES_DIR = DISCOVERY_DIR / "traces"
CONFIRMED_PHISHING_DIR = DISCOVERY_DIR / "confirmed_phishing"
STATS_FILE = DISCOVERY_DIR / "stats.json"

# ─── Config ───────────────────────────────────────────────────────────────────
PROXY = os.environ.get("PHISHTRACE_PROXY", "http://127.0.0.1:10809")
MAX_CRAWL_CONCURRENCY = int(os.environ.get("DISCOVERY_CONCURRENCY", "4"))
LIVENESS_WORKERS = int(os.environ.get("DISCOVERY_LIVENESS_WORKERS", "50"))
LIVENESS_TIMEOUT = 8  # seconds
CRAWL_TIMEOUT_MS = 30000
DETECTION_THRESHOLD = 0.5  # confidence threshold for confirmed phishing
MAX_URLS_PER_CYCLE = int(os.environ.get("DISCOVERY_MAX_URLS", "500"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [discovery] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("discovery")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Domain Discovery
# ═══════════════════════════════════════════════════════════════════════════════

def run_discovery() -> List[dict]:
    """Run wild domain discovery and save results."""
    from src.scanner.wild_scanner import WildDomainDiscovery, DiscoveredDomain
    from dataclasses import asdict

    log.info("Phase 1: Running wild domain discovery...")
    scanner = WildDomainDiscovery(use_proxy=bool(PROXY))
    domains = scanner.discover_all()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_file = DISCOVERY_URLS_DIR / f"discovered_{ts}.jsonl"
    scanner.save(domains, out_file)

    log.info(f"Phase 1 complete: {len(domains)} domains discovered")
    return [asdict(d) for d in domains]


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Liveness Validation
# ═══════════════════════════════════════════════════════════════════════════════

def _check_liveness(session: requests.Session, url: str) -> Tuple[bool, int, str]:
    """Quick HTTP HEAD/GET check. Returns (alive, status_code, final_url)."""
    try:
        r = session.head(url, timeout=LIVENESS_TIMEOUT, allow_redirects=True)
        return True, r.status_code, str(r.url)
    except Exception:
        pass
    try:
        r = session.get(url, timeout=LIVENESS_TIMEOUT, allow_redirects=True,
                        stream=True)
        r.close()
        return True, r.status_code, str(r.url)
    except Exception as e:
        return False, 0, str(e)


def validate_liveness(domains: List[dict]) -> List[dict]:
    """Check which discovered domains are currently alive."""
    log.info(f"Phase 2: Validating liveness of {len(domains)} URLs...")

    session = requests.Session()
    if PROXY:
        session.proxies = {"http": PROXY, "https": PROXY}
    session.verify = False
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
    )

    alive: List[dict] = []
    dead = 0

    with ThreadPoolExecutor(max_workers=LIVENESS_WORKERS) as pool:
        futures = {}
        for d in domains[:MAX_URLS_PER_CYCLE]:
            url = d.get("url", f"https://{d['domain']}")
            futures[pool.submit(_check_liveness, session, url)] = d

        for future in as_completed(futures):
            d = futures[future]
            try:
                is_alive, status, final_url = future.result(timeout=LIVENESS_TIMEOUT + 5)
                if is_alive:
                    d["status_code"] = status
                    d["final_url"] = final_url
                    alive.append(d)
                else:
                    dead += 1
            except Exception:
                dead += 1

    # Save validated URLs
    ts = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_file = DISCOVERY_URLS_DIR / f"validated_{ts}.jsonl"
    with open(out_file, "a", encoding="utf-8") as f:
        for d in alive:
            f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")

    log.info(f"Phase 2 complete: {len(alive)} alive, {dead} dead")
    return alive


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Crawl discovered URLs (trace + screenshots + baseline files)
# ═══════════════════════════════════════════════════════════════════════════════

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:16]


def _already_crawled(url: str) -> bool:
    h = _url_hash(url)
    return (DISCOVERY_TRACES_DIR / h / "trace.json").exists()


def _generate_baseline_files(trace_dir: Path, url: str, ss_dir: Path):
    """Generate shot.png and info.txt for PhishIntention/Phishpedia compatibility."""
    try:
        # info.txt — the URL
        info_file = trace_dir / "info.txt"
        if not info_file.exists():
            info_file.write_text(url, encoding="utf-8")

        # shot.png — first screenshot (landing page with brand logo)
        shot_file = trace_dir / "shot.png"
        if not shot_file.exists() and ss_dir.exists():
            pngs = sorted(ss_dir.glob("*.png"))
            if pngs:
                shutil.copy2(pngs[0], shot_file)
    except Exception as e:
        log.debug(f"Baseline file gen failed: {e}")


async def _crawl_single(url: str, trace_dir: Path) -> Optional[dict]:
    """Crawl a single URL, producing trace + screenshots + baseline files."""
    from src.crawler.phishing_crawler import PhishingCrawler
    from dataclasses import asdict

    ss_dir = trace_dir / "screenshots"
    ss_dir.mkdir(parents=True, exist_ok=True)

    try:
        crawler = PhishingCrawler(
            headless=True,
            timeout=CRAWL_TIMEOUT_MS,
            capture_screenshots=True,
            screenshot_dir=str(ss_dir),
        )
        if PROXY:
            crawler.proxy = PROXY

        trace = await asyncio.wait_for(
            crawler.crawl(url, max_depth=2),
            timeout=CRAWL_TIMEOUT_MS / 1000 + 15,
        )

        # Serialize trace
        from dataclasses import asdict as _asdict
        trace_dict = _asdict(trace)
        trace_dict["url"] = url
        trace_dict["crawled_at"] = datetime.now(timezone.utc).isoformat()

        trace_file = trace_dir / "trace.json"
        trace_file.write_text(
            json.dumps(trace_dict, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        # Generate baseline-compatible files
        _generate_baseline_files(trace_dir, url, ss_dir)

        return trace_dict

    except asyncio.TimeoutError:
        log.debug(f"Crawl timeout: {url[:50]}")
        return None
    except Exception as e:
        log.debug(f"Crawl error ({url[:50]}): {e}")
        return None


async def _crawl_batch(urls: List[dict], concurrency: int) -> List[dict]:
    """Crawl a batch of URLs with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    results: List[dict] = []

    async def _limited(d: dict):
        url = d.get("url", f"https://{d['domain']}")
        if _already_crawled(url):
            return
        h = _url_hash(url)
        trace_dir = DISCOVERY_TRACES_DIR / h
        async with sem:
            trace = await _crawl_single(url, trace_dir)
            if trace:
                trace["discovery_source"] = d.get("source", "unknown")
                results.append(trace)

    tasks = [_limited(d) for d in urls]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


def crawl_discovered(domains: List[dict]) -> List[dict]:
    """Synchronous wrapper for crawling discovered URLs."""
    # Filter out already-crawled
    pending = [d for d in domains if not _already_crawled(
        d.get("url", f"https://{d['domain']}")
    )]
    if not pending:
        log.info("Phase 3: No new URLs to crawl")
        return []

    log.info(f"Phase 3: Crawling {len(pending)} new URLs (concurrency={MAX_CRAWL_CONCURRENCY})...")

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            results = pool.submit(
                asyncio.run,
                _crawl_batch(pending, MAX_CRAWL_CONCURRENCY)
            ).result(timeout=3600)
    else:
        results = asyncio.run(_crawl_batch(pending, MAX_CRAWL_CONCURRENCY))

    log.info(f"Phase 3 complete: {len(results)} traces collected")
    gc.collect()
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Phishing Detection + Confirmed Phishing Archive
# ═══════════════════════════════════════════════════════════════════════════════

def detect_and_archive(traces: Optional[List[dict]] = None) -> Dict[str, int]:
    """
    Run PhishTraceDetector on crawled discovery traces.
    Confirmed phishing traces are copied to confirmed_phishing/.
    """
    log.info("Phase 4: Running phishing detection on discovery traces...")

    from src.detector.phish_detector import PhishTraceDetector
    from src.analyzer.graph_builder import InteractionGraphBuilder

    detector = PhishTraceDetector()
    builder = InteractionGraphBuilder()

    stats = {"total": 0, "phishing": 0, "benign": 0, "error": 0}

    # Collect trace dirs to process
    trace_dirs: List[Path] = []
    if traces:
        for t in traces:
            url = t.get("url", "")
            h = _url_hash(url)
            td = DISCOVERY_TRACES_DIR / h
            if (td / "trace.json").exists():
                trace_dirs.append(td)
    else:
        # Process all unclassified traces
        if DISCOVERY_TRACES_DIR.exists():
            for td in DISCOVERY_TRACES_DIR.iterdir():
                if td.is_dir() and (td / "trace.json").exists():
                    # Skip already-classified
                    if not (td / "detection.json").exists():
                        trace_dirs.append(td)

    log.info(f"  Processing {len(trace_dirs)} traces...")

    for td in trace_dirs:
        stats["total"] += 1
        try:
            trace_data = json.loads((td / "trace.json").read_text(encoding="utf-8"))
            url = trace_data.get("url", "")

            # Build graph and extract features
            graph = builder.build_graph_from_dict(trace_data)
            features = builder.extract_features(graph)

            # Classify
            if detector.model is not None:
                feature_vec = detector._features_to_vector(features)
                prediction = detector.model.predict([feature_vec])[0]
                proba = detector.model.predict_proba([feature_vec])[0]
                is_phishing = bool(prediction == 1)
                confidence = float(proba[1]) if len(proba) > 1 else float(prediction)
                method = "ml_model"
            else:
                # Heuristic fallback
                patterns = builder.detect_phishing_patterns(features)
                is_phishing = patterns["risk_score"] > DETECTION_THRESHOLD
                confidence = patterns["risk_score"]
                method = "heuristic"

            verdict = {
                "url": url,
                "is_phishing": is_phishing,
                "confidence": round(confidence, 4),
                "method": method,
                "detected_at": datetime.now(timezone.utc).isoformat(),
                "features": {
                    "num_nodes": features.num_nodes,
                    "num_edges": features.num_edges,
                    "num_forms": features.num_forms,
                    "has_password_input": features.has_password_input,
                    "has_email_input": features.has_email_input,
                    "graph_density": round(features.graph_density, 4),
                },
            }

            # Save detection result alongside trace
            (td / "detection.json").write_text(
                json.dumps(verdict, indent=2, default=str), encoding="utf-8"
            )

            if is_phishing:
                stats["phishing"] += 1
                # Copy to confirmed_phishing archive
                h = td.name
                dest = CONFIRMED_PHISHING_DIR / h
                if not dest.exists():
                    shutil.copytree(td, dest)
                    log.info(f"  PHISHING confirmed: {url[:60]} "
                             f"(conf={confidence:.2f}, method={method})")
            else:
                stats["benign"] += 1

        except Exception as e:
            stats["error"] += 1
            log.debug(f"  Detection error for {td.name}: {e}")

    log.info(f"Phase 4 complete: {stats}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════════
# Full Pipeline Cycle
# ═══════════════════════════════════════════════════════════════════════════════

def _load_stats() -> dict:
    if STATS_FILE.exists():
        return json.loads(STATS_FILE.read_text(encoding="utf-8"))
    return {"cycles": 0, "total_discovered": 0, "total_crawled": 0,
            "total_phishing": 0, "total_benign": 0}


def _save_stats(stats: dict):
    STATS_FILE.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")


def run_full_cycle(skip_discovery: bool = False, skip_crawl: bool = False,
                   skip_detect: bool = False):
    """Run one full discovery → liveness → crawl → detect cycle."""
    log.info("=" * 70)
    log.info("DISCOVERY PIPELINE: Starting full cycle")
    log.info("=" * 70)

    # Ensure directories
    for d in [DISCOVERY_URLS_DIR, DISCOVERY_TRACES_DIR, CONFIRMED_PHISHING_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    global_stats = _load_stats()
    global_stats["cycles"] += 1
    global_stats["last_cycle_start"] = datetime.now(timezone.utc).isoformat()

    # Phase 1: Discovery
    domains = []
    if not skip_discovery:
        domains = run_discovery()
        global_stats["total_discovered"] += len(domains)
    else:
        # Load latest discovered file
        jsonl_files = sorted(DISCOVERY_URLS_DIR.glob("discovered_*.jsonl"), reverse=True)
        if jsonl_files:
            with open(jsonl_files[0], "r", encoding="utf-8") as f:
                domains = [json.loads(line) for line in f if line.strip()]
            log.info(f"Loaded {len(domains)} domains from {jsonl_files[0].name}")

    # Phase 2: Liveness validation
    alive = []
    if domains and not skip_crawl:
        alive = validate_liveness(domains)

    # Phase 3: Crawl
    traces = []
    if alive and not skip_crawl:
        traces = crawl_discovered(alive)
        global_stats["total_crawled"] += len(traces)

    # Phase 4: Detection
    if not skip_detect:
        det_stats = detect_and_archive(traces if traces else None)
        global_stats["total_phishing"] += det_stats.get("phishing", 0)
        global_stats["total_benign"] += det_stats.get("benign", 0)

    elapsed = time.time() - t0
    global_stats["last_cycle_elapsed_s"] = round(elapsed, 1)
    global_stats["last_cycle_end"] = datetime.now(timezone.utc).isoformat()
    _save_stats(global_stats)

    # Summary
    n_confirmed = len(list(CONFIRMED_PHISHING_DIR.iterdir())) if CONFIRMED_PHISHING_DIR.exists() else 0
    log.info(f"Cycle complete in {elapsed:.0f}s")
    log.info(f"  Cumulative stats: {global_stats['total_discovered']} discovered, "
             f"{global_stats['total_crawled']} crawled, "
             f"{n_confirmed} confirmed phishing")
    log.info("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhishTrace Discovery Pipeline — discover, crawl, detect"
    )
    parser.add_argument("--loop", action="store_true",
                        help="Run continuously in a loop")
    parser.add_argument("--interval", type=int, default=3600,
                        help="Seconds between cycles (default: 3600)")
    parser.add_argument("--crawl-only", action="store_true",
                        help="Skip discovery, only crawl pending URLs")
    parser.add_argument("--detect-only", action="store_true",
                        help="Skip discovery+crawl, only run detection")
    parser.add_argument("--max-urls", type=int, default=500,
                        help="Max URLs per cycle (default: 500)")
    args = parser.parse_args()

    global MAX_URLS_PER_CYCLE
    MAX_URLS_PER_CYCLE = args.max_urls

    if args.loop:
        log.info(f"Starting discovery loop (interval={args.interval}s)")
        while True:
            try:
                run_full_cycle(
                    skip_discovery=args.crawl_only or args.detect_only,
                    skip_crawl=args.detect_only,
                )
            except Exception as e:
                log.error(f"Cycle error: {e}", exc_info=True)
            log.info(f"Sleeping {args.interval}s until next cycle...")
            time.sleep(args.interval)
    else:
        run_full_cycle(
            skip_discovery=args.crawl_only or args.detect_only,
            skip_crawl=args.detect_only,
        )


if __name__ == "__main__":
    main()
