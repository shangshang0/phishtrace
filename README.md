# PhishTrace

**Detecting Credential-Theft Websites via Stacked Multi-View Ensemble on Enhanced Interaction Trace Graphs**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](Dockerfile)

PhishTrace detects credential-theft websites by *interacting* with them rather than merely inspecting static artifacts. A Playwright-driven crawler discovers forms and buttons, fills credential fields, submits them, and follows post-submission flows. Each crawl produces an **Enhanced Interaction Trace Graph (EITG)** encoded across six feature views. A **Stacked Multi-View Ensemble (SMVE)** fuses per-view classifiers through a meta-learner, achieving state-of-the-art detection performance.

> **Paper:** *PhishTrace: Detecting Credential-Theft Websites via Stacked Multi-View Ensemble on Enhanced Interaction Trace Graphs* — ICCS 2026

---

## One-Click Deployment (Docker)

The Docker pipeline runs the **complete reproducible workflow**: data promotion → 11 experiments → formal verification (6 Z3 theorems) → results mapping → LaTeX macro generation → PDF compilation.

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) ≥ 20.10
- [Docker Compose](https://docs.docker.com/compose/) ≥ 2.0
- ~8 GB RAM, ~4 GB disk space

### Step 1: Build

```bash
git clone https://github.com/shangshang0/phishtrace.git
cd phishtrace
docker build -t phishtrace:latest .
```

### Step 2: Run

```bash
# Micro mode — quick pipeline validation (~5 min)
docker compose up phishtrace-micro

# Full mode — complete experiments + paper regeneration (~1.5 hours)
docker compose up phishtrace-full
```

### Step 3: Collect Results

All outputs appear in `docker-phishtrace/` on the host:

```
docker-phishtrace/
├── output/
│   ├── paper.pdf                  # Compiled paper (15 pages, LNCS format)
│   └── generated_data.tex         # 825+ auto-generated LaTeX macros
├── results/
│   └── mapped/                    # Experiment JSON results
│       ├── eitg_results.json          # Core EITG-SMVE performance
│       ├── gnn_comparison_results.json
│       ├── adversarial_results.json
│       ├── url_adversarial_v3.json
│       ├── results.json               # Baseline comparisons
│       └── ...                        # 12 result files total
├── logs/                          # Pipeline / daemon logs
└── dataset/                       # Crawled interaction traces
```

> **Zero hardcoded numbers** — every metric in the paper is a LaTeX macro auto-generated from experiment JSON output.

---

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Run tests
make test

# Run all experiments
make experiments

# Generate LaTeX macros from results
python scripts/generate_latex_data.py

# Compile paper
make paper
```

### Baseline Setup

To include PhishIntention and Phishpedia baselines:

```bash
python scripts/setup_baselines.py
```

---

## Project Structure

```
phishtrace/
├── src/                         # Core library
│   ├── crawler/                     # Deep interaction crawler (Playwright)
│   ├── analyzer/                    # EITG construction + feature extraction
│   ├── detector/                    # SMVE classifier pipeline
│   ├── scanner/                     # Multi-source domain discovery
│   ├── validator/                   # Trace quality validation
│   └── utils/                       # Shared utilities
├── experiments/                 # 11 reproducible experiment modules
│   ├── enhanced_itg_detector.py     # EITG multi-view detection
│   ├── gnn_comparison.py            # GCN / GAT / GraphSAGE comparison
│   ├── comprehensive_detector.py    # AGFL stacked detection
│   ├── adversarial_eval.py          # Adversarial robustness (graph attacks)
│   ├── url_adversarial_eval_v3.py   # URL-camouflage robustness
│   ├── invariance_and_decoy_test.py # Behavioral invariance test
│   ├── trace_interpretability.py    # Trace interpretability analysis
│   ├── depth_ablation.py            # Trace-depth ablation study
│   ├── compute_baseline_stddevs.py  # Baseline std deviations + Wilcoxon
│   ├── reviewer_response_experiments.py
│   └── baselines/                   # Baseline implementations
├── scripts/                     # Pipeline scripts
│   ├── run_all_intersection_experiments.py  # Master experiment runner
│   ├── generate_latex_data.py       # Results → 825+ LaTeX macros
│   ├── prepare_intersection_results.py
│   ├── daemon_url_collector.py      # URL collection daemon
│   ├── daemon_trace_crawler.py      # Trace crawling daemon
│   └── setup_baselines.py
├── verification/                # Formal verification (6 theorems)
│   ├── verify_all.py               # Master verification runner
│   ├── formal_verification.py      # Z3-based theorem proving
│   ├── ctl_checker.py              # CTL model checking
│   └── z3_invariants.py            # SMT invariant checking
├── pipeline/                    # Docker orchestration
│   ├── orchestrator.py              # Master pipeline controller
│   └── config.py                    # Central configuration
├── tests/                       # Test suite
├── iccs2026latex/               # Paper source (Springer LNCS)
├── Dockerfile                   # Multi-stage build (Python+Playwright+TeX+Z3)
├── docker-compose.yml           # Micro + Full mode services
├── entrypoint.sh
├── Makefile
├── requirements.txt
└── LICENSE                      # Apache 2.0
```

## Pipeline Architecture

```
orchestrator.py
├── promote_crawled_traces()          # Bridge raw_crawl/ → dataset/traces/
├── start_crawl_daemons()             # 3 background daemons
│   ├── daemon_url_collector.py       #   Threat intel URL feeds
│   ├── daemon_trace_crawler.py       #   Playwright interaction crawling
│   └── daemon_discovery.py           #   Wild domain discovery
└── full_build_pipeline() [loop]      # Scheduled rebuild (default: 12h)
    ├── run_experiments()             #   11 experiments
    ├── run_verification()            #   6 Z3 theorems + CTL checking
    ├── prepare_results()             #   Map results to standard format
    ├── generate_latex_data()         #   JSON → 825+ LaTeX macros
    └── compile_pdf()                 #   pdflatex + bibtex
```

## Makefile Targets

| Target | Description |
|--------|-------------|
| `make install` | Install Python dependencies + Playwright |
| `make test` | Run test suite |
| `make experiments` | Run all 11 experiments |
| `make paper` | Compile LaTeX paper |
| `make docker-micro` | Run micro mode in Docker |
| `make docker-full` | Run full mode in Docker |
| `make clean` | Remove build artifacts |

## Citation

```bibtex
@inproceedings{phishtrace2026,
  title     = {PhishTrace: Detecting Credential-Theft Websites via Stacked 
               Multi-View Ensemble on Enhanced Interaction Trace Graphs},
  author    = {Shang, Shang},
  booktitle = {International Conference on Computational Science (ICCS)},
  year      = {2026},
  publisher = {Springer}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
