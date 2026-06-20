.PHONY: help ci test lint format format-check typecheck compile sync lock-check fastapi clean

help:                ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "  %-18s %s\n", $$1, $$2}'

ci: lock-check sync format-check lint typecheck compile test  ## Full quality gate

test:                ## Run all tests
	uv run pytest

lint:                ## Ruff linter (Rust)
	uv run ruff check .

format:              ## Ruff format (in-place)
	uv run ruff format .

format-check:        ## Ruff format check (CI-safe — no modifications)
	uv run ruff format --check .

typecheck:           ## ty type check
	uv run ty check src

compile:             ## Compile src/ + tests/ — catches import + syntax errors
	uv run python -m compileall -q src

sync:                ## Install dev dependencies via uv
	uv sync --dev

lock-check:          ## Validate uv.lock matches pyproject.toml
	uv lock --check

fastapi:             ## Start FastAPI dev server (--reload, port 8000)
	uv run uvicorn src.main:app --reload --port 8000

clean:               ## Remove __pycache__ + .pytest_cache + .ruff_cache
	@find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	@rm -rf .pytest_cache .ruff_cache dist build
	@echo "Cleaned."

run-calibration:     ## Run the DSPy judge calibration script
	uv run python src/scripts/01_calibrate_judge.py
