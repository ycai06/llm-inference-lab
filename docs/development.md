# Development status

## v0.2.0

Warm text-request mode adds a managed authenticated localhost llama-server, actual model
tokenization of deterministic fixtures, full-request warmups, monotonic streaming TTFT and
end-to-end request latency. It preserves native prompt/decode timings and actual output counts,
checks disabled prefix reuse, and withholds token intervals when transport/Unicode events
cannot support them. Request samples have a typed schema and a separate SQLite table.

The original synthetic benchmark path remains available with its original timing definitions.

The initial release provides a local CLI for reproducible llama.cpp synthetic prefill and decode
microbenchmarks on CPU and Apple Metal. It includes a typed backend interface, managed process
groups, strict YAML matrices, model fingerprinting, individual trial storage, SQLite migrations,
CSV/JSON exports, descriptive statistics, bootstrap intervals and Matplotlib reports.

The architecture separates configuration, orchestration, engine integration, process management,
metrics and persistence. A per-user lock serializes benchmark workers. Timeouts and interrupts
terminate owned processes and retain failure states. Model files are checked for changes before
and after each case. Unsupported dimensions and inconsistent engine results are rejected.

## Validation

- 287 automated tests passed on macOS arm64 with Python 3.13.14.
- Ruff lint/format checks and strict mypy checks passed for the backend.
- Integration tests cover CLI execution, persistence, exports, report generation, failed
  processes, malformed output, timeout, cancellation, cleanup and user-facing errors.
- Unit tests cover configuration, matrix bounds, model inspection, timing consistency, backend
  controls, process groups, locks, migrations, statistics and runner failure paths.
- Four real-model experiments recorded 70 measured repetitions across 14 cases with the existing
  Gemma GGUF and llama.cpp b12182. The records include CPU/Metal verification and microbatch tuning.
- Warm text-request verification recorded 15 measured requests across exact 128/512/1024-token
  fixture prefixes, each after a full warmup. All generated exactly 16 tokens with zero cache reuse.
  A real in-flight cancellation retained partial evidence, returned exit 130 and reaped its server.
- Stream regression tests include HTTP/1.0 close-delimited bodies and HTTP/1.1 keep-alive chunked
  bodies, checking that cancellation interrupts reads without racing the HTTP response decoder.
- The wheel was installed in a clean temporary environment and its SQLite migration verified.
  Wheel/source archives exclude weights, caches, databases and raw local results.
- Published graphs were visually inspected and can be regenerated from the compact sample data.

Run `make check` for automated validation and `make smoke-test` for model-free CLI integration
checks. Real-model validation uses `make smoke` or `make server-smoke` after configuring paths.

## Current boundaries

Synthetic warm microbenchmarks and warm native text completion are implemented. Cold-load
instrumentation, concurrent serving, GPU utilization, power and GPU-memory metrics remain
unavailable. Synthetic decode begins without a populated prompt context. Request mode uses an
explicit tokenized prefix and fixed context; token intervals require complete stream evidence.
Process RSS and CPU time include loading and warmup and are sampled rather than inference-region
measurements.

The verified environment is documented in [environment.md](environment.md). Original engine
CMake flags and the existing model's source revision/license remain unverified. Only one
compatible quantization is available. Battery power and unknown thermal conditions limit
performance conclusions. The microbatch tuning experiment did not validate a speedup.

## Roadmap

1. Add independent alternating/interleaved sessions and compare controlled conditions across
   prompt lengths, generation lengths, CPU thread counts and quantizations.
2. Add a localhost FastAPI service with a serialized worker and lifecycle cleanup.
3. Add a React dashboard backed by real stored results, with compatible-comparison checks.
4. Measure concurrent serving separately from single-request workloads.
5. Use supported profiling tools to identify a specific bottleneck before implementing and
   evaluating an original code optimization.

See [limitations.md](limitations.md), [the technical report](../reports/technical-report.md),
and [contribution ideas](contributing-ideas.md) for further context.
