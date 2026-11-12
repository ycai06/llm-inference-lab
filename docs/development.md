# Development status

## v0.1.0

The initial release provides a local CLI for reproducible llama.cpp synthetic prefill and decode
microbenchmarks on CPU and Apple Metal. It includes a typed backend interface, managed process
groups, strict YAML matrices, model fingerprinting, individual trial storage, SQLite migrations,
CSV/JSON exports, descriptive statistics, bootstrap intervals and Matplotlib reports.

The architecture separates configuration, orchestration, engine integration, process management,
metrics and persistence. A per-user lock serializes benchmark workers. Timeouts and interrupts
terminate owned processes and retain failure states. Model files are checked for changes before
and after each case. Unsupported dimensions and inconsistent engine results are rejected.

## Validation

- 141 automated tests passed on macOS arm64 with Python 3.13.14.
- Ruff lint/format checks and strict mypy checks passed for the backend.
- Integration tests cover CLI execution, persistence, exports, report generation, failed
  processes, malformed output, timeout, cancellation, cleanup and user-facing errors.
- Unit tests cover configuration, matrix bounds, model inspection, timing consistency, backend
  controls, process groups, locks, migrations, statistics and runner failure paths.
- Four real-model experiments recorded 70 measured repetitions across 14 cases with the existing
  Gemma GGUF and llama.cpp b12182. The records include CPU/Metal verification and microbatch tuning.
- The wheel was installed in a clean temporary environment and its SQLite migration verified.
  Wheel/source archives exclude weights, caches, databases and raw local results.
- Published graphs were visually inspected and can be regenerated from the compact sample data.

Run `make check` for automated validation and `make smoke-test` for model-free CLI integration
checks. Real-model validation uses `make smoke` after configuring model and engine paths.

## Current boundaries

Only synthetic warm microbenchmarks are implemented. TTFT, request and inter-token latency,
cold-load instrumentation, concurrent serving, GPU utilization, power and GPU-memory metrics
are unavailable. Decode begins without a populated prompt context. Process RSS and CPU time
include loading and warmup and are sampled rather than inference-region measurements.

The verified environment is documented in [environment.md](environment.md). Original engine
CMake flags and the existing model's source revision/license remain unverified. Only one
compatible quantization is available. Battery power and unknown thermal conditions limit
performance conclusions. The microbatch tuning experiment did not validate a speedup.

## Roadmap

1. Add a managed localhost llama-server backend with deterministic text fixtures and actual
   model-tokenizer counts, explicit sampling and prefix-cache policies, and token-event TTFT.
2. Add independent alternating/interleaved sessions and compare controlled conditions across
   prompt lengths, generation lengths, CPU thread counts and quantizations.
3. Add a localhost FastAPI service with a serialized worker and lifecycle cleanup.
4. Add a React dashboard backed by real stored results, with compatible-comparison checks.
5. Measure concurrent serving separately from single-request workloads.
6. Use supported profiling tools to identify a specific bottleneck before implementing and
   evaluating an original code optimization.

See [limitations.md](limitations.md), [the technical report](../reports/technical-report.md),
and [contribution ideas](contributing-ideas.md) for further context.
