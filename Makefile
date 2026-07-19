PYTHON ?= python
IMAGE ?= genome-firewall:local
AMRFINDER_BIN ?= amrfinder

.PHONY: help install install-dev run test lint format-check check docker-build docker-run amrfinder-check

help:
	@echo "Genome Firewall commands"
	@echo "  make install          Install runtime dependencies and package"
	@echo "  make install-dev      Install development and test tools"
	@echo "  make run              Start the Streamlit app"
	@echo "  make check            Run lint and tests"
	@echo "  make docker-build     Build the transparent demo image"
	@echo "  make docker-run       Run the demo image on port 8501"
	@echo "  make amrfinder-check  Show the configured AMRFinderPlus version"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install --no-deps -e .

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

run:
	$(PYTHON) -m streamlit run app.py

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

format-check:
	$(PYTHON) -m ruff format --check .

check: lint format-check test

docker-build:
	docker build --tag $(IMAGE) .

docker-run:
	docker run --rm --publish 8501:8501 $(IMAGE)

amrfinder-check:
	$(AMRFINDER_BIN) --version
