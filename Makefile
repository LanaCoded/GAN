.PHONY: help install lint format typecheck test test-fast train smoke report clean all

help:
	@echo "Available targets:"
	@echo "  install    - Install package in development mode with dependencies"
	@echo "  lint       - Run ruff linting"
	@echo "  format     - Format code with black"
	@echo "  typecheck  - Run mypy type checking"
	@echo "  test       - Run full test suite with coverage"
	@echo "  test-fast  - Run fast tests (skip slow tests)"
	@echo "  train      - Run full training with default config"
	@echo "  smoke      - Run smoke training with reduced config"
	@echo "  report     - Generate report from last training run"
	@echo "  clean      - Remove generated files and directories"
	@echo "  all        - Run lint + typecheck + test"
	@echo "  help       - Show this help message"

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests

format:
	ruff check src tests --fix
	black src tests

typecheck:
	mypy src

test:
	pytest --cov=gan_mnist --cov-report=term-missing --cov-report=html:outputs/reports/coverage

test-fast:
	pytest -m "not slow"

train:
	python -m gan_mnist train --config configs/default.yaml

smoke:
	python -m gan_mnist train --config configs/smoke.yaml

report:
	python -m gan_mnist report --run-dir outputs/runs/$$(ls -td outputs/runs/* | head -1)

clean:
	rm -rf outputs/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf __pycache__ src/**/__pycache__ tests/**/__pycache__
	rm -f .coverage

all: lint typecheck test
