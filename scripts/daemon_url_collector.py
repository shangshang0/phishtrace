#!/usr/bin/env python3
"""
daemon_url_collector.py — Background URL Intelligence & Validation Daemon
=========================================================================
Periodically:
  1. Fetches latest URLs from all intel sources (OpenPhish, PhishTank, etc.)
  2. Validates all URLs (HTTP liveness check via requests — lightweight)
  3. Saves validated URLs to dataset/url_dataset_valid_{YYYYMMDD_HHMMSS}/

Memory & Performance Optimizations:
  - Streaming validation with ThreadPoolExecutor (no full-list in memory)
  - Connection pooling via requests.Session (reuse TCP connections)
  - Incremental saves every 10k URLs (crash-safe)
  - Explicit gc.collect() after each cycle
  - Configurable worker count and sleep interval

Usage:
    py -3 scripts/daemon_url_collector.py                      # Run once then exit
    py -3 scripts/daemon_url_collector.py --loop --interval 3600  # Loop every hour
    py -3 scripts/daemon_url_collector.py --validate-only       # Skip intel fetch
"""

import argparse
import gc
import json
import logging
import os
import random
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import urllib3
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
ORIGIN_DIR = DATASET_DIR / "url_dataset_origin"

# ─── Config ───────────────────────────────────────────────────────────────────
PROXY = os.environ.get("PHISHTRACE_PROXY", "http://127.0.0.1:10809")
MAX_WORKERS_HTTP = 300       # HTTP validation concurrency
MAX_WORKERS_DNS = 100        # DNS validation concurrency
VALIDATE_TIMEOUT = 5         # seconds per URL
SAVE_INTERVAL = 10000        # save progress every N URLs
MAX_MEMORY_URLS = 50000      # chunk processing if source > this

UA_LIST = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0",
    "Mozilla/5.0 (X11; Linux x86_64; rv:121.0) Gecko/20100101 Firefox/121.0",
]

SOURCE_CONFIGS = {
    "openphish":  {"label": "phishing", "method": "http"},
    "phishtank":  {"label": "phishing", "method": "http"},
    "urlhaus":    {"label": "phishing", "method": "http"},
    "threatfox":  {"label": "phishing", "method": "http"},
    "urlscan":    {"label": "phishing", "method": "http"},
    "cert_pl":    {"label": "phishing", "method": "http"},
    "tranco":     {"label": "benign",   "method": "dns"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [url-daemon] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("url-daemon")


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP / DNS checking (lightweight — no Playwright)
# ═══════════════════════════════════════════════════════════════════════════════

def make_session(use_proxy: bool = True, pool_size: int = 300) -> requests.Session:
    """Create session with connection pooling."""
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=min(pool_size, 100), pool_maxsize=pool_size, max_retries=0)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.verify = False
    if use_proxy:
        s.proxies = {"http": PROXY, "https": PROXY}
    return s


def check_url_http(session: requests.Session, url: str, timeout: int = 5) -> Tuple[bool, int, str]:
    """GET with stream=True — any HTTP response = alive. Lightweight."""
    try:
        r = session.get(
            url,
            headers={"User-Agent": random.choice(UA_LIST), "Accept": "*/*"},
            timeout=timeout,
            allow_redirects=True,
            stream=True,
        )
        r.close()  # Don't read body — save memory
        return True, r.status_code, "ok"
    except requests.exceptions.SSLError:
        return True, 0, "ssl_error_but_alive"
    except requests.exceptions.TooManyRedirects:
        return True, 0, "too_many_redirects"
    except requests.exceptions.ConnectionError as e:
        err = str(e).lower()
        if "name or service not known" in err or "getaddrinfo failed" in err:
            return False, 0, "dns_failed"
        if "connection refused" in err or "no route to host" in err:
            return False, 0, "conn_refused"
        return False, 0, "conn_error"
    except requests.exceptions.Timeout:
        return False, 0, "timeout"
    except Exception as e:
        return False, 0, type(e).__name__


def check_url_dns(url: str, timeout: int = 5) -> Tuple[bool, int, str]:
    """DNS-only check for benign URLs (Tranco) — very lightweight."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or url
        socket.setdefaulttimeout(timeout)
        socket.getaddrinfo(host, 443)
        return True, 0, "dns_ok"
    except (socket.gaierror, socket.timeout, OSError):
        return False, 0, "dns_failed"


# ═══════════════════════════════════════════════════════════════════════════════
# Intel fetching — uses WildDomainDiscovery for fresh URL sources
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_tranco(top_n: int = 5000):
    """Download Tranco top-1M list and save top_n entries as benign URL pool.

    Result goes to url_dataset_origin/tranco/urls.json — the format expected
    by daemon_trace_crawler.py ``load_benign_urls()``.
    """
    out = ORIGIN_DIR / "tranco" / "urls.json"
    if out.exists():
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if len(existing.get("urls", [])) >= top_n:
                log.info(f"Tranco already has {len(existing['urls'])} URLs — skipping download")
                return True
        except Exception:
            pass

    log.info(f"Downloading Tranco top-1M list (keeping top {top_n}) ...")
    try:
        # Tranco list is a plain CSV: rank,domain
        url = "https://tranco-list.eu/download/6W2QN/1000000"  # permalink to stable list
        fallback = "https://tranco-list.eu/top-1m.csv.zip"
        resp = requests.get(url, timeout=60)
        if resp.status_code != 200:
            log.info("Tranco permalink failed, trying fallback ...")
            resp = requests.get(fallback, timeout=60)
            resp.raise_for_status()
            import zipfile, io
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                fname = zf.namelist()[0]
                text = zf.open(fname).read().decode("utf-8")
        else:
            text = resp.text

        # Filter out infrastructure/CDN domains that are not browsable websites
        _INFRA_KW = {
            'awsdns', 'akamai', 'cloudfront', 'amazonaws', 'fastly',
            'edgecast', 'limelight', 'gtld-servers', 'root-servers',
            'registrar-servers', 'akadns', 'edgekey', 'edgesuite',
            'azurefd', 'trafficmanager', 'msecnd', 'azureedge',
            'cloudflare-dns', 'in-addr.arpa', 'ip6.arpa',
        }
        _INFRA_TLD = {
            '.arpa', '.internal', '.local', '.test',
        }

        def _is_tranco_browsable(domain: str) -> bool:
            d = domain.lower()
            for kw in _INFRA_KW:
                if kw in d:
                    return False
            for tld in _INFRA_TLD:
                if d.endswith(tld):
                    return False
            # Skip IP-only entries
            import re as _re
            if _re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', d):
                return False
            return True

        urls = []
        skipped_infra = 0
        # Read MORE lines to compensate for filtered infrastructure domains
        for line in text.strip().splitlines()[:top_n * 3]:
            if len(urls) >= top_n:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                rank, domain = int(parts[0]), parts[1].strip()
            else:
                domain = parts[0].strip()
                rank = len(urls) + 1
            if not _is_tranco_browsable(domain):
                skipped_infra += 1
                continue
            urls.append({
                "url": f"https://{domain}",
                "domain": domain,
                "rank": rank,
                "source": "tranco",
            })
        log.info(f"Tranco: filtered out {skipped_infra} infrastructure domains")

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "urls": urls,
            "total": len(urls),
            "fetched": datetime.now().isoformat(),
            "source": "tranco-list.eu",
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        log.info(f"Tranco: saved {len(urls)} benign domains to {out}")
        return True
    except Exception as e:
        log.error(f"Tranco download failed: {e}")
        return False


def fetch_intel_sources():
    """Fetch latest URLs from OSINT discovery sources.

    Also saves per-source files to url_dataset_origin/{source}/urls.json
    so that daemon_trace_crawler can load them by source name.
    """
    try:
        from src.scanner.wild_scanner import WildDomainDiscovery
        proxy = os.environ.get("PHISHTRACE_PROXY", "")
        scanner = WildDomainDiscovery(
            use_proxy=bool(proxy),
            proxy_url=proxy or None,
            keyword_filter=False,
        )
        domains = scanner.discover_all()
        if domains:
            # Save aggregate
            out = ORIGIN_DIR / "discovery" / "urls.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            urls = [{"url": d.url, "domain": d.domain, "source": d.source} for d in domains]
            with open(out, "w") as f:
                json.dump({"urls": urls, "fetched": datetime.now().isoformat()}, f)

            # Save per-source (for trace crawler)
            by_source: Dict[str, list] = {}
            for d in domains:
                by_source.setdefault(d.source, []).append(
                    {"url": d.url, "domain": d.domain, "source": d.source}
                )
            for src, src_urls in by_source.items():
                # Map source names to expected directory names
                src_dir_name = {
                    "openphish": "openphish",
                    "phishtank": "phishtank",
                    "crt.sh": "cert_pl",
                    "urlscan.io": "urlscan",
                    "urlhaus": "urlhaus",
                    "whoisds-nod": "threatfox",
                }.get(src, src.replace(".", "_"))
                src_out = ORIGIN_DIR / src_dir_name / "urls.json"
                src_out.parent.mkdir(parents=True, exist_ok=True)
                # Merge with existing
                existing = []
                if src_out.exists():
                    try:
                        existing = json.loads(src_out.read_text())["urls"]
                    except Exception:
                        pass
                existing_urls = {u["url"] for u in existing}
                new = [u for u in src_urls if u["url"] not in existing_urls]
                merged = existing + new
                src_out.write_text(json.dumps(
                    {"urls": merged, "fetched": datetime.now().isoformat()},
                    ensure_ascii=False
                ))
                if new:
                    log.info(f"  {src_dir_name}: {len(new)} new URLs (total {len(merged)})")

            log.info(f"Fetched {len(domains)} URLs from discovery sources")
        return True
    except Exception as e:
        log.error(f"Intel fetch error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Validation engine
# ═══════════════════════════════════════════════════════════════════════════════

def load_source_urls(source: str) -> List[dict]:
    """Load URLs from origin dataset for a source."""
    fpath = ORIGIN_DIR / source / "urls.json"
    if not fpath.exists():
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("urls", [])


def validate_source(
    source: str,
    cfg: dict,
    out_dir: Path,
    session: requests.Session,
    workers: int,
    timeout: int,
) -> dict:
    """Validate all URLs from a single source. Memory-optimized with incremental saves."""

    urls_data = load_source_urls(source)
    if not urls_data:
        return {"source": source, "total": 0, "valid": 0, "invalid": 0}

    total = len(urls_data)
    is_dns = cfg["method"] == "dns"
    w = min(workers, MAX_WORKERS_DNS) if is_dns else min(workers, MAX_WORKERS_HTTP)

    # Resume check
    result_file = out_dir / source / "urls.json"
    if result_file.exists():
        try:
            ex = json.loads(result_file.read_text(encoding="utf-8"))
            if ex.get("completed"):
                log.info(f"[{source}] Already completed ({ex['valid_count']} valid). Skipping.")
                return {"source": source, "total": total, "valid": ex["valid_count"],
                        "invalid": ex["invalid_count"], "skipped": True}
        except Exception:
            pass

    log.info(f"[{source}] Validating {total} URLs with {w} workers ({'DNS' if is_dns else 'HTTP'})")

    valid = []
    invalid_count = 0
    reasons = {}
    checked = 0
    t0 = time.time()

    def _check(item):
        if is_dns:
            return item, check_url_dns(item["url"], timeout)
        return item, check_url_http(session, item["url"], timeout)

    # Process in chunks to limit memory
    chunk_size = min(MAX_MEMORY_URLS, total)
    for chunk_start in range(0, total, chunk_size):
        chunk = urls_data[chunk_start : chunk_start + chunk_size]

        with ThreadPoolExecutor(max_workers=w) as pool:
            futs = {pool.submit(_check, it): it for it in chunk}
            for f in as_completed(futs):
                try:
                    item, (ok, code, reason) = f.result()
                except Exception:
                    item = futs[f]
                    ok, code, reason = False, 0, "executor_error"

                checked += 1

                if ok:
                    rec = {
                        "url": item["url"],
                        "domain": item.get("domain", ""),
                        "source": source,
                        "validated_at": datetime.now(timezone.utc).isoformat(),
                    }
                    if code > 0:
                        rec["http_status"] = code
                    # Copy extra fields (rank, etc.)
                    for k in ("rank", "tags", "first_seen"):
                        if k in item:
                            rec[k] = item[k]
                    valid.append(rec)
                else:
                    invalid_count += 1

                reasons[reason] = reasons.get(reason, 0) + 1

                if checked % 2000 == 0 or checked == total:
                    el = time.time() - t0
                    rate = checked / max(0.1, el)
                    eta = int((total - checked) / max(0.01, rate))
                    log.info(f"[{source}] {checked}/{total} — {len(valid)} valid, "
                             f"{invalid_count} invalid — {rate:.0f}/s, ETA {eta}s")

                # Incremental save
                if checked % SAVE_INTERVAL == 0:
                    _save_validation(out_dir, source, cfg, valid, invalid_count,
                                     total, reasons, completed=False)

        # Free chunk memory
        del chunk
        gc.collect()

    elapsed = time.time() - t0
    _save_validation(out_dir, source, cfg, valid, invalid_count, total, reasons, completed=True)

    vr = f"{len(valid)}/{total}" if total else "0/0"
    log.info(f"[{source}] DONE: {vr} valid in {elapsed:.0f}s ({total / max(0.1, elapsed):.0f}/s)")

    # Free memory
    del valid, urls_data
    gc.collect()

    return {"source": source, "total": total, "valid": len(valid) if 'valid' in dir() else 0,
            "invalid": invalid_count}


def _save_validation(out_dir, source, cfg, valid, invalid_count, total, reasons, completed):
    """Save validation results to disk."""
    (out_dir / source).mkdir(parents=True, exist_ok=True)
    data = {
        "source": source,
        "label": cfg["label"],
        "total": total,
        "valid_count": len(valid),
        "invalid_count": invalid_count,
        "valid_rate": f"{len(valid)/total*100:.1f}%" if total else "0%",
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "error_reasons": dict(sorted(reasons.items(), key=lambda x: -x[1])[:15]),
        "urls": valid,
    }
    (out_dir / source / "urls.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main daemon loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_cycle(skip_fetch: bool = False, workers: int = 300, timeout: int = 5):
    """Run one full cycle: fetch intel → validate all → save."""
    cycle_start = datetime.now()
    ts = cycle_start.strftime("%Y%m%d_%H%M%S")
    out_dir = DATASET_DIR / f"url_dataset_valid_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"{'=' * 60}")
    log.info(f"URL Collection Cycle — {ts}")
    log.info(f"Output: {out_dir}")
    log.info(f"{'=' * 60}")

    # Step 1: Fetch latest intel (optional)
    if not skip_fetch:
        fetch_tranco()         # Ensure benign URL pool exists
        fetch_intel_sources()  # Phishing intelligence sources
        gc.collect()

    # Step 2: Validate all sources
    session = make_session(use_proxy=True, pool_size=workers)

    # Quick proxy test
    try:
        r = session.get("https://httpbin.org/ip", timeout=10)
        log.info(f"Proxy OK: {r.json().get('origin', '?')}")
    except Exception as e:
        log.warning(f"Proxy test failed: {e} — continuing anyway")

    results = []
    # Process HTTP sources first (faster), then DNS
    order = sorted(
        SOURCE_CONFIGS.keys(),
        key=lambda s: (0 if SOURCE_CONFIGS[s]["method"] == "http" else 1),
    )

    for source in order:
        cfg = SOURCE_CONFIGS[source]
        r = validate_source(source, cfg, out_dir, session, workers, timeout)
        results.append(r)
        gc.collect()

    session.close()

    # Summary
    tv = sum(r.get("total", 0) for r in results)
    vv = sum(r.get("valid", 0) for r in results)
    elapsed = (datetime.now() - cycle_start).total_seconds()

    summary = {
        "validated_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(out_dir),
        "total_urls": tv,
        "total_valid": vv,
        "valid_rate": f"{vv/tv*100:.1f}%" if tv else "0%",
        "elapsed_seconds": round(elapsed, 1),
        "sources": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Create symlink/pointer to latest
    latest_link = DATASET_DIR / "url_dataset_valid_latest"
    try:
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        # On Windows, write a pointer file instead of symlink
        latest_link.write_text(str(out_dir), encoding="utf-8")
    except Exception:
        pass

    log.info(f"Cycle complete: {vv}/{tv} valid URLs in {elapsed:.0f}s")
    log.info(f"Saved to: {out_dir}")

    gc.collect()
    return out_dir


def main():
    ap = argparse.ArgumentParser(description="URL Intelligence & Validation Daemon")
    ap.add_argument("--loop", action="store_true", help="Run continuously in a loop")
    ap.add_argument("--interval", type=int, default=3600,
                    help="Seconds between cycles (default: 3600 = 1 hour)")
    ap.add_argument("--validate-only", action="store_true", help="Skip intel fetch, only validate")
    ap.add_argument("--workers", type=int, default=300, help="Validation concurrency")
    ap.add_argument("--timeout", type=int, default=5, help="Per-URL timeout in seconds")
    args = ap.parse_args()

    if args.loop:
        log.info(f"Starting URL daemon loop (interval={args.interval}s)")
        while True:
            try:
                run_cycle(skip_fetch=args.validate_only,
                          workers=args.workers, timeout=args.timeout)
            except Exception as e:
                log.error(f"Cycle failed: {e}")
            log.info(f"Sleeping {args.interval}s until next cycle...")
            time.sleep(args.interval)
    else:
        run_cycle(skip_fetch=args.validate_only,
                  workers=args.workers, timeout=args.timeout)


if __name__ == "__main__":
    main()
