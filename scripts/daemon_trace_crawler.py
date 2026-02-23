#!/usr/bin/env python3
"""
daemon_trace_crawler.py — Background Trace Crawler Daemon
=========================================================
Continuously crawls validated URLs using Playwright to generate interaction
traces + screenshots. Picks the LATEST url_dataset_valid_* directory.

Memory & Performance Optimizations:
  - Bounded Playwright concurrency (default: 6)
  - Chunked URL processing (200 per batch, gc between batches)
  - Resume support: skips already-crawled URLs by hash
  - Single browser instance shared across chunk (reopened per chunk)
  - Explicit resource cleanup and gc.collect()
  - Progress persistence (stats file updated incrementally)

Usage:
    py -3 scripts/daemon_trace_crawler.py --target 2000            # 2000 phishing + 2000 benign
    py -3 scripts/daemon_trace_crawler.py --target 2000 --concurrency 4
    py -3 scripts/daemon_trace_crawler.py --phishing-only --target 500
    py -3 scripts/daemon_trace_crawler.py --loop --interval 1800   # Re-check for new URLs every 30 min
"""

import argparse
import asyncio
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import asdict

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
RAW_CRAWL_DIR = DATASET_DIR / "raw_crawl"
ORIGIN_DIR = DATASET_DIR / "url_dataset_origin"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ─── Config ───────────────────────────────────────────────────────────────────
PROXY = os.environ.get("PHISHTRACE_PROXY", "http://127.0.0.1:10809")
CHUNK_SIZE = 200          # URLs per async batch (limits memory)
MAX_CONCURRENCY = 8       # Hard cap even if user requests more
DEFAULT_TIMEOUT_MS = 30000
CHECKPOINT_INTERVAL = 2000  # Create checkpoint every N new OK traces
CHECKPOINT_DIR = RAW_CRAWL_DIR / "checkpoints"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [trace-daemon] %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("trace-daemon")


# ═══════════════════════════════════════════════════════════════════════════════
# URL loading
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_valid_dir() -> Optional[Path]:
    """Find the latest url_dataset_valid_* directory, or fall back to pointer file."""
    # Check for pointer file first
    pointer = DATASET_DIR / "url_dataset_valid_latest"
    if pointer.exists() and pointer.is_file():
        p = Path(pointer.read_text(encoding="utf-8").strip())
        if p.exists():
            return p

    # Scan for directories matching the pattern
    candidates = sorted(
        [d for d in DATASET_DIR.iterdir()
         if d.is_dir() and d.name.startswith("url_dataset_valid_")],
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_phishing_urls(valid_dir: Path) -> List[dict]:
    """Load all validated phishing URLs from dataset, deduplicated."""
    sources = ["openphish", "phishtank", "urlscan", "urlhaus", "cert_pl", "threatfox"]
    all_urls = []
    for src in sources:
        fpath = valid_dir / src / "urls.json"
        if not fpath.exists():
            continue
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        urls = data.get("urls", [])
        for u in urls:
            u["source"] = src
            u["label"] = "phishing"
        all_urls.extend(urls)
        log.info(f"  {src}: {len(urls)} validated phishing URLs")

    # Deduplicate
    seen: Set[str] = set()
    unique = []
    for u in all_urls:
        url = u["url"]
        if url not in seen:
            seen.add(url)
            unique.append(u)

    random.seed(42)
    random.shuffle(unique)
    log.info(f"  Phishing pool: {len(unique)} unique URLs")
    return unique


def load_benign_urls() -> List[dict]:
    """Load benign URLs from Tranco (origin dataset)."""
    fpath = ORIGIN_DIR / "tranco" / "urls.json"
    if not fpath.exists():
        log.error(f"Tranco file not found: {fpath}")
        return []
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    urls = data.get("urls", [])
    urls.sort(key=lambda x: x.get("rank", 99999))
    for u in urls:
        u["label"] = "benign"
        u["source"] = "tranco"
    log.info(f"  Benign pool: {len(urls)} Tranco URLs")
    return urls


# ═══════════════════════════════════════════════════════════════════════════════
# Crawl helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_existing_hashes(out_dir: Path) -> Tuple[Set[str], int]:
    """Scan output_dir for already-crawled URL hashes. Returns (set_of_hashes, count_ok)."""
    hashes: Set[str] = set()
    ok = 0
    if not out_dir.exists():
        return hashes, ok
    for d in out_dir.iterdir():
        if not d.is_dir():
            continue
        hashes.add(d.name)
        tf = d / "trace.json"
        if tf.exists():
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    # Only read first 200 bytes to check success flag — saves memory
                    head = f.read(300)
                if '"success": true' in head or '"success":true' in head:
                    ok += 1
            except Exception:
                pass
    return hashes, ok


async def crawl_single(url_info: dict, output_dir: Path,
                       timeout_ms: int = 30000) -> Optional[dict]:
    """Crawl a single URL. Returns wrapped trace dict or None."""
    from src.crawler.phishing_crawler import PhishingCrawler

    url = url_info["url"]
    label = url_info["label"]
    source = url_info.get("source", "unknown")
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]

    trace_dir = output_dir / url_hash
    trace_file = trace_dir / "trace.json"

    # Skip if already crawled successfully
    if trace_file.exists():
        try:
            with open(trace_file, "r", encoding="utf-8") as f:
                head = f.read(300)
            if '"success": true' in head or '"success":true' in head:
                return {"success": True, "cached": True}
        except Exception:
            pass

    try:
        ss_dir = trace_dir / "screenshots"
        crawler = PhishingCrawler(
            headless=True,
            timeout=timeout_ms,
            capture_screenshots=True,
            screenshot_dir=str(ss_dir),
            proxy=PROXY,
        )

        trace = await asyncio.wait_for(
            crawler.crawl(url, max_depth=3),
            timeout=timeout_ms / 1000 + 20,
        )

        if not trace or not trace.events:
            return None

        # Serialize
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
            "url": url,
            "hash": url_hash,
            "source": source,
            "label": label,
            "success": True,
            "domain": url_info.get("domain", ""),
            "crawled_at": datetime.now(timezone.utc).isoformat(),
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
                "console_logs": (trace.console_logs or [])[:100],
            },
            "screenshots": [
                e.screenshot_path for e in trace.events
                if getattr(e, 'screenshot_path', None)
            ],
            "events": events_ser,
            "network_requests": net_reqs_ser,
            "redirects": getattr(trace, "redirects", []),
        }

        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_file.write_text(
            json.dumps(wrapped, indent=2, default=str, ensure_ascii=False),
            encoding="utf-8",
        )

        # Generate shot.png + info.txt for baseline compatibility
        _generate_baseline_files(trace_dir, url, ss_dir)

        # Free large objects immediately
        del trace, events_ser, net_reqs_ser
        return wrapped

    except asyncio.TimeoutError:
        return None
    except Exception as e:
        log.debug(f"  Crawl error ({url[:40]}): {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Baseline compatibility: shot.png + info.txt
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_baseline_files(trace_dir: Path, url: str, ss_dir: Path):
    """Generate shot.png and info.txt in trace_dir for PhishIntention/Phishpedia.
    Picks the FIRST screenshot (initial page with brand logo visible).
    Visual baselines need the landing page, not the post-submit redirect."""
    try:
        # Write info.txt (URL)
        info_file = trace_dir / "info.txt"
        if not info_file.exists():
            info_file.write_text(url, encoding="utf-8")

        # Copy first screenshot as shot.png (landing page = brand logo visible)
        shot_file = trace_dir / "shot.png"
        if not shot_file.exists() and ss_dir.exists():
            pngs = sorted(ss_dir.glob("*.png"))
            if pngs:
                shutil.copy2(pngs[0], shot_file)  # first = landing page with logo
    except Exception as e:
        log.debug(f"Baseline file generation failed for {trace_dir.name}: {e}")


def retrofit_baseline_files():
    """Retrofit shot.png + info.txt for all existing traces that lack them."""
    created = 0
    skipped = 0
    for label in ["phishing", "benign"]:
        cat_dir = RAW_CRAWL_DIR / label
        if not cat_dir.exists():
            continue
        for trace_dir in cat_dir.iterdir():
            if not trace_dir.is_dir():
                continue
            trace_file = trace_dir / "trace.json"
            if not trace_file.exists():
                continue

            shot_exists = (trace_dir / "shot.png").exists()
            info_exists = (trace_dir / "info.txt").exists()
            if shot_exists and info_exists:
                skipped += 1
                continue

            # Read URL from trace.json
            try:
                with open(trace_file, "r", encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                url = data.get("url", "")
                if not url:
                    continue
            except Exception:
                continue

            ss_dir = trace_dir / "screenshots"
            _generate_baseline_files(trace_dir, url, ss_dir)

            if (trace_dir / "shot.png").exists():
                created += 1

    log.info(f"Retrofit complete: {created} new shot.png+info.txt created, {skipped} already existed")
    return created, skipped


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpoint system
# ═══════════════════════════════════════════════════════════════════════════════

def _save_checkpoint(label: str, new_count: int):
    """Save a checkpoint manifest listing all trace dirs at this point."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Collect all current traces
    manifest = {"timestamp": ts, "categories": {}}
    for cat in ["phishing", "benign"]:
        cat_dir = RAW_CRAWL_DIR / cat
        if not cat_dir.exists():
            manifest["categories"][cat] = []
            continue
        hashes = sorted([
            d.name for d in cat_dir.iterdir()
            if d.is_dir() and (d / "trace.json").exists()
        ])
        manifest["categories"][cat] = hashes

    n_p = len(manifest["categories"].get("phishing", []))
    n_b = len(manifest["categories"].get("benign", []))
    manifest["counts"] = {"phishing": n_p, "benign": n_b, "total": n_p + n_b}
    manifest["trigger"] = {"label": label, "new_ok_at_trigger": new_count}

    ckpt_file = CHECKPOINT_DIR / f"ckpt_{ts}_p{n_p}_b{n_b}.json"
    ckpt_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"Checkpoint saved: {ckpt_file.name} (phishing={n_p}, benign={n_b}, total={n_p+n_b})")
    return ckpt_file


async def crawl_category(
    urls: List[dict],
    output_dir: Path,
    concurrency: int,
    timeout_ms: int,
    label: str,
    target_ok: int,
):
    """Crawl a category (phishing/benign) with chunked processing and early stop."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get already-crawled info for resume
    existing_hashes, pre_ok = get_existing_hashes(output_dir)
    log.info(f"[{label}] Pre-existing: {pre_ok} ok / {len(existing_hashes)} total dirs")

    if target_ok > 0 and pre_ok >= target_ok:
        log.info(f"[{label}] Target {target_ok} already reached. Done.")
        return {"label": label, "ok": pre_ok, "new": 0, "fail": 0, "skipped": True}

    # Filter out already-crawled URLs
    remaining = []
    for u in urls:
        h = hashlib.md5(u["url"].encode()).hexdigest()[:12]
        if h not in existing_hashes:
            remaining.append(u)

    log.info(f"[{label}] {len(remaining)} new URLs to crawl (target: {target_ok}, have: {pre_ok})")

    sem = asyncio.Semaphore(concurrency)
    counters = {"ok": pre_ok, "new_ok": 0, "fail": 0, "processed": 0,
                "last_checkpoint_ok": 0}
    start = time.time()
    total_remaining = len(remaining)

    async def _crawl(url_info):
        if target_ok > 0 and counters["ok"] >= target_ok:
            return
        async with sem:
            if target_ok > 0 and counters["ok"] >= target_ok:
                return
            result = await crawl_single(url_info, output_dir, timeout_ms)
            if result and result.get("success"):
                if not result.get("cached"):
                    counters["ok"] += 1
                    counters["new_ok"] += 1
            else:
                counters["fail"] += 1
            counters["processed"] += 1

            # Checkpoint every CHECKPOINT_INTERVAL new OK traces
            if (counters["new_ok"] - counters["last_checkpoint_ok"]) >= CHECKPOINT_INTERVAL:
                _save_checkpoint(label, counters["new_ok"])
                counters["last_checkpoint_ok"] = counters["new_ok"]

            done = counters["processed"]
            if done % 50 == 0 or done == total_remaining or \
               (target_ok > 0 and counters["ok"] >= target_ok):
                el = time.time() - start
                rate = done / max(1, el)
                need = max(0, target_ok - counters["ok"]) if target_ok > 0 else total_remaining - done
                ok_frac = counters["new_ok"] / max(1, done)
                if target_ok > 0 and ok_frac > 0:
                    est_urls = int(need / ok_frac)
                    eta_s = int(est_urls / max(0.01, rate))
                else:
                    eta_s = int(need / max(0.01, rate))
                h, m = divmod(eta_s // 60, 60)
                eta_str = f"{h}h{m}m" if h > 0 else f"{m}m{eta_s%60}s"
                t_str = f"/{target_ok}" if target_ok > 0 else ""
                log.info(f"  [{label}] done={done}/{total_remaining} OK={counters['ok']}{t_str} "
                         f"new_ok={counters['new_ok']} fail={counters['fail']} "
                         f"{rate:.1f}/s ETA {eta_str}")

    # Process in chunks
    for chunk_start in range(0, total_remaining, CHUNK_SIZE):
        if target_ok > 0 and counters["ok"] >= target_ok:
            break
        chunk = remaining[chunk_start : chunk_start + CHUNK_SIZE]
        tasks = [_crawl(u) for u in chunk]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Memory cleanup between chunks
        del tasks, chunk
        gc.collect()
        log.debug(f"  [{label}] Chunk done, gc.collect()")

    elapsed = time.time() - start
    log.info(f"[{label}] DONE: {counters['ok']} ok ({counters['new_ok']} new, "
             f"{counters['fail']} fail) in {elapsed:.0f}s")

    return {
        "label": label,
        "ok": counters["ok"],
        "new": counters["new_ok"],
        "fail": counters["fail"],
        "pre_existing": pre_ok,
        "elapsed_s": round(elapsed, 1),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

async def run_cycle(
    target: int = 2000,
    concurrency: int = 6,
    timeout_ms: int = 30000,
    phishing_only: bool = False,
    benign_only: bool = False,
):
    """One crawl cycle: load latest validated URLs → crawl → save traces."""
    cycle_start = datetime.now()
    log.info(f"{'=' * 60}")
    log.info(f"Trace Crawl Cycle — {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Target: {target} per category, concurrency={concurrency}")
    log.info(f"{'=' * 60}")

    # Find latest validated URL dataset
    valid_dir = find_latest_valid_dir()
    if not valid_dir:
        log.error("No validated URL dataset found. Run daemon_url_collector.py first.")
        return

    log.info(f"Using validated URLs from: {valid_dir}")

    phishing_raw = RAW_CRAWL_DIR / "phishing"
    benign_raw = RAW_CRAWL_DIR / "benign"
    stats = {}

    if not benign_only:
        phishing_urls = load_phishing_urls(valid_dir)
        stats["phishing"] = await crawl_category(
            phishing_urls, phishing_raw, concurrency, timeout_ms, "phishing", target,
        )
        del phishing_urls
        gc.collect()

    if not phishing_only:
        benign_urls = load_benign_urls()
        stats["benign"] = await crawl_category(
            benign_urls, benign_raw, concurrency, timeout_ms, "benign", target,
        )
        del benign_urls
        gc.collect()

    # Save stats
    stats_file = RAW_CRAWL_DIR / "crawl_stats.json"
    stats["completed_at"] = datetime.now(timezone.utc).isoformat()
    stats["valid_url_dir"] = str(valid_dir)
    stats_file.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    elapsed = (datetime.now() - cycle_start).total_seconds()
    log.info(f"Cycle complete in {elapsed:.0f}s")
    for cat, s in stats.items():
        if isinstance(s, dict) and "ok" in s:
            log.info(f"  {cat}: {s['ok']} ok ({s.get('new', '?')} new)")

    gc.collect()


def main():
    ap = argparse.ArgumentParser(description="Background Trace Crawler Daemon")
    ap.add_argument("--target", type=int, default=999999,
                    help="Target valid traces per category (default: 999999 = all)")
    ap.add_argument("--concurrency", type=int, default=6,
                    help="Playwright concurrency (default: 6)")
    ap.add_argument("--timeout", type=int, default=30000,
                    help="Per-URL timeout in ms (default: 30000)")
    ap.add_argument("--phishing-only", action="store_true")
    ap.add_argument("--benign-only", action="store_true")
    ap.add_argument("--loop", action="store_true",
                    help="Run continuously, re-checking for new URLs")
    ap.add_argument("--interval", type=int, default=1800,
                    help="Seconds between loop cycles (default: 1800)")
    ap.add_argument("--retrofit", action="store_true",
                    help="Retrofit shot.png + info.txt for existing traces, then exit")
    ap.add_argument("--checkpoint-interval", type=int, default=2000,
                    help="Save checkpoint every N new OK traces (default: 2000)")
    args = ap.parse_args()

    # Retrofit mode: just generate missing shot.png + info.txt
    if args.retrofit:
        log.info("Retrofitting shot.png + info.txt for existing traces...")
        created, skipped = retrofit_baseline_files()
        log.info(f"Done. Created: {created}, Already existed: {skipped}")
        return

    global CHECKPOINT_INTERVAL
    CHECKPOINT_INTERVAL = args.checkpoint_interval

    conc = min(args.concurrency, MAX_CONCURRENCY)

    if args.loop:
        log.info(f"Starting trace crawler loop (interval={args.interval}s)")
        while True:
            try:
                asyncio.run(run_cycle(
                    target=args.target,
                    concurrency=conc,
                    timeout_ms=args.timeout,
                    phishing_only=args.phishing_only,
                    benign_only=args.benign_only,
                ))
            except Exception as e:
                log.error(f"Cycle error: {e}")
                traceback.print_exc()
            log.info(f"Sleeping {args.interval}s...")
            time.sleep(args.interval)
    else:
        asyncio.run(run_cycle(
            target=args.target,
            concurrency=conc,
            timeout_ms=args.timeout,
            phishing_only=args.phishing_only,
            benign_only=args.benign_only,
        ))


if __name__ == "__main__":
    main()
