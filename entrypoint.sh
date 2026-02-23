#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          PhishTrace Pipeline — Docker Entrypoint            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Mode:     ${PHISHTRACE_MODE:-micro}"
echo "Data dir: ${PHISHTRACE_DATA:-/data}"
echo "Proxy:    ${PHISHTRACE_PROXY:-(none)}"
echo ""

# Ensure data directories exist
mkdir -p /data/{dataset/traces/phishing,dataset/traces/benign,dataset/raw_crawl,dataset/url_dataset_origin}
mkdir -p /data/{results/intersection_comparison,results/mapped,output,logs,checkpoints}

# Run the orchestrator
exec python -m pipeline.orchestrator "$@"
