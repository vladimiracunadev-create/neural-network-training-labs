.PHONY: install dev mlops validate doctor test test-network lint format catalog quality benchmark leaderboard docs security sources sources-refresh clean

install:
	python -m pip install -e ".[full]"

dev:
	python -m pip install -e ".[full,dev]"

mlops:
	python -m pip install -e ".[mlops,data-versioning]"

validate:
	python -m neural_labs.cli validate
	python scripts/validate_repository.py

doctor:
	python -m neural_labs.cli doctor

test:
	pytest -m "not network and not slow" --cov=neural_labs --cov-report=term-missing -q

test-network:
	RUN_NETWORK_TESTS=1 pytest -m network -q

lint:
	python -m ruff check src scripts tests

typecheck:
	python -m mypy src/neural_labs

format:
	python -m ruff format src scripts tests
	python -m ruff check --fix src scripts tests

catalog:
	python -m neural_labs.cli catalog

quality:
	python -m neural_labs.cli quality --lab 00_numpy_neuron --quick

benchmark:
	python -m neural_labs.cli benchmark --lab 00_numpy_neuron --quick --seeds 41 42 43

leaderboard:
	python -m neural_labs.cli leaderboard

docs:
	mkdocs serve

security:
	pip-audit
	bandit -c pyproject.toml -r src

sources:
	python scripts/verify-sources

sources-refresh:
	python scripts/refresh-sources

clean:
	python scripts/clean_runs.py --runs --processed
