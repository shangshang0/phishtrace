"""
Pipeline configuration — single source of truth.
All paths and tunables are defined here and driven by environment variables.
"""
import os
from pathlib import Path

# ─── Mode ──────────────────────────────────────────────────────────────────────
MODE = os.environ.get("PHISHTRACE_MODE", "micro")  # "micro" | "full"
IS_MICRO = MODE == "micro"

# ─── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # /app inside container
DATA_DIR = Path(os.environ.get("PHISHTRACE_DATA", PROJECT_ROOT / "data"))

DATASET_DIR   = DATA_DIR / "dataset"
TRACES_DIR    = DATASET_DIR / "traces"
RAW_CRAWL_DIR = DATASET_DIR / "raw_crawl"
URL_ORIGIN_DIR = DATASET_DIR / "url_dataset_origin"
DISCOVERY_DIR = DATASET_DIR / "discovery"
DISCOVERY_TRACES_DIR = DISCOVERY_DIR / "traces"
CONFIRMED_PHISHING_DIR = DISCOVERY_DIR / "confirmed_phishing"

RESULTS_DIR   = DATA_DIR / "results" / "intersection_comparison"
MAPPED_DIR    = DATA_DIR / "results" / "mapped"  # prepare_intersection_results output
LATEX_DIR     = PROJECT_ROOT / "iccs2026latex"
OUTPUT_DIR    = DATA_DIR / "output"
LOG_DIR       = DATA_DIR / "logs"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"

# ─── Network ───────────────────────────────────────────────────────────────────
PROXY = os.environ.get("PHISHTRACE_PROXY", "")  # empty = no proxy

# ─── Scheduling ────────────────────────────────────────────────────────────────
EXPERIMENT_INTERVAL_S = int(os.environ.get("EXPERIMENT_INTERVAL", 43200))  # 12h
CRAWL_URL_INTERVAL_S  = int(os.environ.get("CRAWL_URL_INTERVAL", 3600))   # 1h
CRAWL_TRACE_INTERVAL_S = int(os.environ.get("CRAWL_TRACE_INTERVAL", 1800)) # 30m
DISCOVERY_INTERVAL_S  = int(os.environ.get("DISCOVERY_INTERVAL", 3600))    # 1h

# ─── Micro mode limits ────────────────────────────────────────────────────────
MICRO_PHISHING_LIMIT = int(os.environ.get("MICRO_PHISHING_LIMIT", 20))
MICRO_BENIGN_LIMIT   = int(os.environ.get("MICRO_BENIGN_LIMIT", 20))
MICRO_URL_LIMIT      = int(os.environ.get("MICRO_URL_LIMIT", 50))

# ─── Concurrency ──────────────────────────────────────────────────────────────
CRAWL_CONCURRENCY = int(os.environ.get("CRAWL_CONCURRENCY", 4 if IS_MICRO else 6))

def ensure_dirs():
    """Create all required directories."""
    for d in [DATASET_DIR, TRACES_DIR, TRACES_DIR / "phishing", TRACES_DIR / "benign",
              RAW_CRAWL_DIR, URL_ORIGIN_DIR, RESULTS_DIR, MAPPED_DIR,
              DISCOVERY_DIR, DISCOVERY_DIR / "urls", DISCOVERY_TRACES_DIR,
              CONFIRMED_PHISHING_DIR,
              OUTPUT_DIR, LOG_DIR, CHECKPOINT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

def describe():
    """Print current configuration."""
    print(f"  Mode:           {MODE}")
    print(f"  Data dir:       {DATA_DIR}")
    print(f"  Dataset dir:    {DATASET_DIR}")
    print(f"  Traces dir:     {TRACES_DIR}")
    print(f"  Results dir:    {RESULTS_DIR}")
    print(f"  Mapped dir:     {MAPPED_DIR}")
    print(f"  Output dir:     {OUTPUT_DIR}")
    print(f"  Proxy:          {PROXY or '(none)'}")
    print(f"  Exp interval:   {EXPERIMENT_INTERVAL_S}s ({EXPERIMENT_INTERVAL_S/3600:.1f}h)")
    if IS_MICRO:
        print(f"  Micro limits:   {MICRO_PHISHING_LIMIT} phishing + {MICRO_BENIGN_LIMIT} benign traces")
