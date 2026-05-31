# eval harness — locale evaluation against the running translator-api.
# Run from this directory (`cd eval`).

# MetricX pins transformers==4.30.2 / sentencepiece==0.1.99 — those have no
# prebuilt wheels for Python 3.13+, so we need 3.10 or 3.11. Override with:
#   make setup PYTHON_BIN=python3.11
PYTHON_BIN  ?= $(shell command -v python3.11 2>/dev/null || command -v python3.10 2>/dev/null)

VENV         := .venv
PYTHON       := $(VENV)/bin/python
PIP          := $(VENV)/bin/pip
METRICX_DIR  := .metricx
METRICX_REPO := https://github.com/google-research/metricx.git
SETUP_STAMP  := .setup-complete

.PHONY: help setup calibrate run run-skip-judge clean-results clean

help:
	@echo "eval harness — usage:"
	@echo "  make setup           Create venv, install deps, install MetricX"
	@echo "  make calibrate       Run harness on calibration.csv (validated locales)"
	@echo "  make run             Run harness on candidates.csv (full incl. MetricX)"
	@echo "  make run-skip-judge  Run harness on candidates.csv (Phase 1 only, no MetricX)"
	@echo "  make clean-results   Delete results/results-*.csv and translations-*.csv"
	@echo "  make clean           clean-results PLUS remove venv, .metricx, caches"
	@echo
	@echo "Note: MetricX model (~1.2 GB) downloads from HuggingFace on first run."

setup: $(SETUP_STAMP)

$(SETUP_STAMP): requirements.txt
	@if [ -z "$(PYTHON_BIN)" ]; then \
		echo "ERROR: no compatible Python found (need 3.10 or 3.11)."; \
		echo "  MetricX pins old transformers/sentencepiece without wheels for 3.12+."; \
		echo "  Install via: brew install python@3.11"; \
		echo "  Then re-run: make setup"; \
		echo "  Or override:  make setup PYTHON_BIN=/path/to/python3.11"; \
		exit 1; \
	fi
	@echo "▸ creating venv at $(VENV) using $(PYTHON_BIN)…"
	$(PYTHON_BIN) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@if [ ! -d "$(METRICX_DIR)" ]; then \
		echo "▸ cloning MetricX into $(METRICX_DIR)…"; \
		git clone --depth 1 $(METRICX_REPO) $(METRICX_DIR); \
	else \
		echo "▸ $(METRICX_DIR) already present, skipping clone"; \
	fi
	@echo "▸ patching predict.py to disable MPS device on Apple Silicon…"
	@if ! grep -q "use_mps_device=False" $(METRICX_DIR)/metricx23/predict.py; then \
		sed -i.bak '/dataloader_pin_memory=False,/a\'$$'\n''      use_mps_device=False,' $(METRICX_DIR)/metricx23/predict.py; \
		echo "  ✓ patched"; \
	else \
		echo "  ✓ already patched"; \
	fi
	@echo "▸ installing MetricX runtime deps (transformers, sentencepiece, …)…"
	$(PIP) install -r $(METRICX_DIR)/requirements.txt
	@echo "▸ installing harness deps…"
	$(PIP) install -r requirements.txt
	@touch $@
	@echo
	@echo "✓ setup complete."
	@echo "  MetricX is run via PYTHONPATH (no pip install — the repo isn't packaged)."
	@echo "  Activate the venv with: source $(VENV)/bin/activate"

calibrate: $(SETUP_STAMP)
	$(PYTHON) harness.py --candidates calibration.csv

run: $(SETUP_STAMP)
	$(PYTHON) harness.py --candidates candidates.csv

run-skip-judge: $(SETUP_STAMP)
	$(PYTHON) harness.py --candidates candidates.csv --skip-judge

clean-results:
	@rm -f results/results-*.csv results/translations-*.csv
	@echo "✓ results cleared"

clean: clean-results
	@rm -rf $(VENV) $(METRICX_DIR) $(SETUP_STAMP) __pycache__ *.pyc
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@echo "✓ venv, MetricX, and caches removed"
