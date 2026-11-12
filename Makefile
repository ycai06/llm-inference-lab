.PHONY: install check test smoke-test doctor smoke server-smoke server-baseline baseline report
export UV_CACHE_DIR := $(CURDIR)/.cache/uv
install:
	UV_CACHE_DIR=.cache/uv uv sync --locked
check:
	uv run --frozen ruff check .
	uv run --frozen ruff format --check .
	uv run --frozen mypy backend
	uv run --frozen pytest
test:
	uv run --frozen pytest
smoke-test:
	uv run --frozen pytest tests/integration -q
doctor:
	uv run --frozen liol doctor --binary "$$LIOL_LLAMA_BENCH"
smoke:
	uv run --frozen liol run configs/smoke.yaml
server-smoke:
	uv run --frozen liol run configs/server-smoke.yaml
server-baseline:
	uv run --frozen liol run configs/server-baseline.yaml
baseline:
	uv run --frozen liol run configs/baseline.yaml
report:
	uv run --frozen liol report $(ID) --output reports/experiment.md
