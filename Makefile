# PhishTrace — Phishing Detection via Interaction Trace Graphs
# Makefile for development, testing, and paper compilation

.PHONY: help install test clean paper paper-clean experiments lint setup-baselines docker-micro docker-full

PYTHON ?= python3

help: ## Show this help
	@echo "PhishTrace - Phishing Detection via Interaction Trace Graphs"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	pip install -r requirements.txt
	playwright install chromium
	playwright install-deps chromium

setup-baselines: ## Clone baseline repos (PhishIntention, Phishpedia)
	$(PYTHON) scripts/setup_baselines.py

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

experiments: ## Run all paper experiments
	$(PYTHON) scripts/run_all_intersection_experiments.py

paper: ## Compile LaTeX paper (ICCS 2026)
	cd iccs2026latex && pdflatex -interaction=nonstopmode paper.tex && bibtex paper && pdflatex -interaction=nonstopmode paper.tex && pdflatex -interaction=nonstopmode paper.tex

paper-clean: ## Clean LaTeX build artifacts
	cd iccs2026latex && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz paper.pdf

docker-micro: ## Run micro mode pipeline in Docker
	docker compose up phishtrace-micro

docker-full: ## Run full mode pipeline in Docker
	docker compose up phishtrace-full

clean: ## Clean generated files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage

lint: ## Run linting
	$(PYTHON) -m flake8 src/ experiments/ --max-line-length=120 --ignore=E501,W503
