#!/usr/bin/env python3
"""
PhishTrace Pipeline Orchestrator
=================================
Master process that coordinates:
  1. Continuous crawling daemons (URL collection + trace crawling)
  2. Scheduled experiment runs (every 12h after last completion)
  3. Results → LaTeX macros → PDF compilation

Usage:
    python -m pipeline.orchestrator          # default: micro mode
    PHISHTRACE_MODE=full python -m pipeline.orchestrator

Designed to run inside Docker with /data mounted to the host.
"""

import json
import os
import signal
import subprocess
import sys
import time
import threading
import logging
from datetime import datetime
from pathlib import Path

# Ensure project root on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.config import (
    MODE, IS_MICRO, DATA_DIR, DATASET_DIR, TRACES_DIR,
    RAW_CRAWL_DIR, DISCOVERY_TRACES_DIR,
    RESULTS_DIR, MAPPED_DIR, LATEX_DIR, OUTPUT_DIR, LOG_DIR,
    CHECKPOINT_DIR, PROXY, EXPERIMENT_INTERVAL_S,
    CRAWL_URL_INTERVAL_S, CRAWL_TRACE_INTERVAL_S,
    DISCOVERY_INTERVAL_S, DISCOVERY_DIR, CONFIRMED_PHISHING_DIR,
    MICRO_PHISHING_LIMIT, MICRO_BENIGN_LIMIT, MICRO_URL_LIMIT,
    CRAWL_CONCURRENCY, ensure_dirs, describe,
)

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [orchestrator] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrator")

# Global flag — set True when SIGTERM/SIGINT received
_shutdown = threading.Event()


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0: Promote real crawled traces into the experiment dataset
# ═══════════════════════════════════════════════════════════════════════════════

def promote_crawled_traces():
    """Copy successful traces from raw_crawl/ → dataset/traces/ for experiments.

    The trace crawler (daemon_trace_crawler.py) saves Playwright-crawled traces
    to raw_crawl/{phishing,benign}/{hash}/trace.json.  Experiments read from
    dataset/traces/{phishing,benign}/*.json.  This function bridges the two.

    Only traces with "success": true are promoted.  Already-promoted hashes
    (based on filename) are skipped.
    """
    import shutil

    raw_dir = RAW_CRAWL_DIR
    traces_dir = TRACES_DIR

    promoted = {"phishing": 0, "benign": 0}
    skipped = {"phishing": 0, "benign": 0}

    for label in ("phishing", "benign"):
        src_dir = raw_dir / label
        dst_dir = traces_dir / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            continue

        # Existing hashes in traces dir (avoid re-promoting)
        existing_hashes = {f.stem.replace("trace_", "")
                          for f in dst_dir.glob("trace_*.json")}

        for subdir in sorted(src_dir.iterdir()):
            if not subdir.is_dir():
                continue
            trace_file = subdir / "trace.json"
            if not trace_file.exists():
                continue

            url_hash = subdir.name
            if url_hash in existing_hashes:
                skipped[label] += 1
                continue

            # Check success flag
            try:
                data = json.loads(trace_file.read_text(encoding="utf-8"))
                if not data.get("success"):
                    continue
                # Ensure the label field is set correctly
                data["label"] = label
            except Exception:
                continue

            # Write to traces dir
            dst_file = dst_dir / f"trace_{url_hash}.json"
            dst_file.write_text(
                json.dumps(data, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            promoted[label] += 1

    # Also promote discovery traces if they exist
    disc_dir = DISCOVERY_TRACES_DIR
    if disc_dir.exists():
        dst_dir = traces_dir / "phishing"
        existing_hashes = {f.stem.replace("trace_", "")
                          for f in dst_dir.glob("trace_*.json")}
        for tf in sorted(disc_dir.glob("*.json")):
            url_hash = tf.stem.replace("trace_", "")
            if url_hash in existing_hashes:
                continue
            try:
                data = json.loads(tf.read_text(encoding="utf-8"))
                if not data.get("success"):
                    continue
                data["label"] = "phishing"
                dst_file = dst_dir / f"trace_{url_hash}.json"
                dst_file.write_text(
                    json.dumps(data, indent=2, default=str, ensure_ascii=False),
                    encoding="utf-8",
                )
                promoted["phishing"] += 1
            except Exception:
                continue

    n_phish = len(list((traces_dir / "phishing").glob("*.json")))
    n_benign = len(list((traces_dir / "benign").glob("*.json")))
    log.info(f"Promoted {promoted['phishing']} phishing + {promoted['benign']} benign "
             f"new traces (skipped {skipped['phishing']}+{skipped['benign']} existing)")
    log.info(f"Dataset now has {n_phish} phishing + {n_benign} benign = "
             f"{n_phish + n_benign} total traces")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Crawling daemons (background threads)
# ═══════════════════════════════════════════════════════════════════════════════

def _run_daemon(name: str, cmd: list[str], log_file: Path):
    """Run a subprocess daemon, restarting on crash."""
    while not _shutdown.is_set():
        log.info(f"[{name}] Starting: {' '.join(cmd)}")
        with open(log_file, "a") as lf:
            try:
                proc = subprocess.Popen(
                    cmd, stdout=lf, stderr=subprocess.STDOUT,
                    cwd=str(PROJECT_ROOT),
                    env={**os.environ,
                         "PHISHTRACE_PROXY": PROXY,
                         "PYTHONPATH": str(PROJECT_ROOT),
                         "PYTHONUNBUFFERED": "1"},
                )
                while proc.poll() is None:
                    if _shutdown.is_set():
                        proc.terminate()
                        proc.wait(timeout=10)
                        return
                    time.sleep(5)
                log.warning(f"[{name}] Exited with code {proc.returncode}")
            except Exception as e:
                log.error(f"[{name}] Error: {e}")
        if not _shutdown.is_set():
            log.info(f"[{name}] Restarting in 30s...")
            _shutdown.wait(30)


def start_crawl_daemons():
    """Launch continuous URL collection, trace crawling, and discovery daemons."""
    threads = []

    # URL collector
    url_cmd = [
        sys.executable, "scripts/daemon_url_collector.py",
        "--loop", "--interval", str(CRAWL_URL_INTERVAL_S),
    ]
    t1 = threading.Thread(
        target=_run_daemon,
        args=("url-collector", url_cmd, LOG_DIR / "url_collector.log"),
        daemon=True,
    )
    threads.append(t1)

    # Trace crawler
    trace_cmd = [
        sys.executable, "scripts/daemon_trace_crawler.py",
        "--loop", "--interval", str(CRAWL_TRACE_INTERVAL_S),
        "--concurrency", str(CRAWL_CONCURRENCY),
    ]
    if IS_MICRO:
        trace_cmd.extend(["--target", str(MICRO_PHISHING_LIMIT)])
    t2 = threading.Thread(
        target=_run_daemon,
        args=("trace-crawler", trace_cmd, LOG_DIR / "trace_crawler.log"),
        daemon=True,
    )
    threads.append(t2)

    # Discovery daemon (wild domain discovery + crawl + detection)
    discovery_cmd = [
        sys.executable, "scripts/daemon_discovery.py",
        "--loop", "--interval", str(DISCOVERY_INTERVAL_S),
    ]
    if IS_MICRO:
        discovery_cmd.extend(["--max-urls", "30"])
    t3 = threading.Thread(
        target=_run_daemon,
        args=("discovery", discovery_cmd, LOG_DIR / "discovery.log"),
        daemon=True,
    )
    threads.append(t3)

    for t in threads:
        t.start()
    log.info(f"Started {len(threads)} daemon(s): url-collector, trace-crawler, discovery")
    return threads


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Run experiments
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_dataset_paths():
    """
    Many experiment scripts expect dataset at PROJECT_ROOT/dataset/traces/.
    If our data lives elsewhere (e.g., /data/dataset), create a symlink.
    """
    expected = PROJECT_ROOT / "dataset"
    actual = DATASET_DIR
    if expected.resolve() != actual.resolve():
        if expected.is_symlink():
            expected.unlink()
        elif expected.is_dir():
            # A real directory exists (e.g., from baked-in dataset/SERP/).
            # Move its contents into the volume-backed actual dir, then
            # replace the directory with a symlink.
            import shutil
            for item in expected.iterdir():
                dest = actual / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            shutil.rmtree(str(expected))
            log.info(f"Merged {expected} contents into {actual}")
        expected.symlink_to(actual)
        log.info(f"Symlinked {expected} → {actual}")


def _patch_results_paths():
    """
    Experiments write to experiments/intersection_comparison/ and experiments/results/.
    Symlink them to DATA_DIR so results persist on the host volume.
    """
    for subdir, target in [
        ("experiments/intersection_comparison", RESULTS_DIR),
        ("experiments/results", MAPPED_DIR),
    ]:
        src = PROJECT_ROOT / subdir
        if src.resolve() != target.resolve():
            if src.is_symlink():
                src.unlink()
            elif src.is_dir():
                # Move existing contents to target then replace with symlink
                import shutil
                for f in src.iterdir():
                    dest = target / f.name
                    if not dest.exists():
                        shutil.move(str(f), str(dest))
                shutil.rmtree(str(src))
            src.symlink_to(target)
            log.info(f"Symlinked {src} → {target}")


def run_experiments():
    """Run all intersection experiments using the master runner."""
    _patch_dataset_paths()
    _patch_results_paths()

    # Count available traces
    n_phish = len(list((TRACES_DIR / "phishing").glob("*.json")))
    n_benign = len(list((TRACES_DIR / "benign").glob("*.json")))
    log.info(f"Running experiments on {n_phish} phishing + {n_benign} benign traces")

    if n_phish == 0 or n_benign == 0:
        log.warning("No traces available — skipping experiments")
        return False

    # Save experiment checkpoint (timestamp for data consistency)
    checkpoint = {
        "start_time": _ts(),
        "mode": MODE,
        "n_phishing": n_phish,
        "n_benign": n_benign,
    }
    ckpt_file = CHECKPOINT_DIR / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    ckpt_file.write_text(json.dumps(checkpoint, indent=2))
    log.info(f"Checkpoint: {ckpt_file}")

    # Run the master experiment script
    cmd = [sys.executable, "scripts/run_all_intersection_experiments.py"]
    log.info(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT), "PYTHONUNBUFFERED": "1"},
        capture_output=False,
    )

    if result.returncode != 0:
        log.error(f"Experiments exited with code {result.returncode}")
        return False

    # Update checkpoint
    checkpoint["end_time"] = _ts()
    ckpt_file.write_text(json.dumps(checkpoint, indent=2))
    log.info("All experiments completed successfully")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Formal verification (Z3 + CTL)
# ═══════════════════════════════════════════════════════════════════════════════

def run_verification():
    """Run formal verification suite."""
    log.info("Running formal verification...")
    cmds = [
        [sys.executable, "verification/verify_all.py"],
        [sys.executable, "verification/formal_verification.py"],
    ]
    for cmd in cmds:
        log.info(f"  {' '.join(cmd)}")
        r = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
            capture_output=False,
        )
        if r.returncode != 0:
            log.warning(f"  Verification script exited {r.returncode}")
    log.info("Verification complete")


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Results → LaTeX → PDF
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_results():
    """Convert intersection_comparison/ → experiments/results/ (mapped format)."""
    log.info("Preparing results (intersection → mapped)...")
    r = subprocess.run(
        [sys.executable, "scripts/prepare_intersection_results.py"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=False,
    )
    if r.returncode != 0:
        log.error(f"prepare_intersection_results exited {r.returncode}")
        return False
    return True


def generate_latex_data():
    """Generate LaTeX macros from experiment results."""
    log.info("Generating LaTeX data...")
    r = subprocess.run(
        [sys.executable, "scripts/generate_latex_data.py"],
        cwd=str(PROJECT_ROOT),
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)},
        capture_output=False,
    )
    if r.returncode != 0:
        log.error(f"generate_latex_data exited {r.returncode}")
        return False
    return True


def compile_pdf():
    """Compile LaTeX paper to PDF."""
    log.info("Compiling PDF...")
    latex_dir = LATEX_DIR

    for step in ["pdflatex", "bibtex", "pdflatex", "pdflatex"]:
        if step == "bibtex":
            cmd = ["bibtex", "paper"]
        else:
            cmd = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "paper.tex"]

        r = subprocess.run(
            cmd,
            cwd=str(latex_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if r.returncode != 0 and step != "bibtex":
            log.error(f"LaTeX compilation failed at {step}")
            # Save error log
            err_file = LOG_DIR / "latex_errors.log"
            err_file.write_text(r.stdout + "\n" + r.stderr)
            log.error(f"See {err_file}")
            return False

    # Copy PDF to output directory
    pdf_src = latex_dir / "paper.pdf"
    if pdf_src.exists():
        import shutil
        pdf_dst = OUTPUT_DIR / "paper.pdf"
        shutil.copy2(pdf_src, pdf_dst)
        log.info(f"PDF generated: {pdf_dst} ({pdf_dst.stat().st_size:,} bytes)")

        # Also copy generated_data.tex for inspection
        gen_src = latex_dir / "generated_data.tex"
        if gen_src.exists():
            shutil.copy2(gen_src, OUTPUT_DIR / "generated_data.tex")
        return True
    else:
        log.error("paper.pdf not found after compilation")
        return False


def full_build_pipeline():
    """Run the complete build: experiments → results → LaTeX → PDF."""
    log.info("=" * 70)
    log.info("STARTING FULL BUILD PIPELINE")
    log.info("=" * 70)

    t0 = time.time()

    # Promote any newly crawled traces before running experiments
    promote_crawled_traces()

    # Experiments
    ok = run_experiments()
    if not ok:
        log.warning("Experiment run had issues — continuing with available results")

    # Verification
    run_verification()

    # Results mapping
    ok = prepare_results()
    if not ok:
        log.warning("Results preparation had issues — continuing with available data")

    # LaTeX macros
    ok = generate_latex_data()
    if not ok:
        log.warning("LaTeX data generation had issues — attempting PDF compilation anyway")

    # PDF compilation
    ok = compile_pdf()
    if not ok:
        log.error("PDF compilation failed")
        return False

    elapsed = time.time() - t0
    log.info(f"Full pipeline completed in {elapsed:.1f}s ({elapsed/60:.1f}m)")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Handle signals
    def _handle_signal(sig, frame):
        log.info(f"Received signal {sig} — shutting down gracefully...")
        _shutdown.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Init
    log.info("╔══════════════════════════════════════════════════════════════╗")
    log.info("║          PhishTrace Pipeline Orchestrator                   ║")
    log.info("╚══════════════════════════════════════════════════════════════╝")
    log.info(f"Configuration:")
    describe()

    ensure_dirs()
    _patch_dataset_paths()
    _patch_results_paths()

    # ─── Promote real crawled traces into experiment dataset ───
    promote_crawled_traces()

    # ─── Start crawling daemons (background) ───
    crawl_threads = start_crawl_daemons()

    # ─── Initial build ───
    log.info("Running initial experiment + PDF build...")
    full_build_pipeline()

    # ─── Scheduled loop ───
    log.info(f"Entering scheduled loop: rebuild every {EXPERIMENT_INTERVAL_S}s "
             f"({EXPERIMENT_INTERVAL_S/3600:.1f}h)")

    while not _shutdown.is_set():
        log.info(f"Next build in {EXPERIMENT_INTERVAL_S}s. Waiting...")
        _shutdown.wait(EXPERIMENT_INTERVAL_S)
        if _shutdown.is_set():
            break
        log.info("Scheduled rebuild triggered")
        full_build_pipeline()

    log.info("Orchestrator shutting down.")


if __name__ == "__main__":
    main()
