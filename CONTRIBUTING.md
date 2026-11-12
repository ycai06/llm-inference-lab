# Contributing

LIOL prioritizes reproducible measurements and small, understandable systems components.

- Keep model weights and external engine installations outside the repository. Never modify
  them as part of a benchmark.
- Preserve raw measurements and provenance. Test-only engines and data belong under `tests/`.
- Label llama-bench workloads as synthetic. Keep process startup time separate from inference
  timing, and leave unavailable measurements null. Process RSS is not GPU VRAM.
- Use type hints, explicit backend interfaces and strict configuration validation. Reject
  unsupported settings rather than silently changing the requested workload.
- Keep dependencies small and update `uv.lock` with dependency changes.
- Add focused tests for changed behavior. Run `make check` before submitting changes.
- Follow [the measurement methodology](docs/methodology.md) when reporting performance.

## Local development

```sh
make install
make check
make smoke-test
```

The automated tests do not require model weights. To test actual inference, configure an existing
GGUF and compatible llama-bench as described in the README, then run `make smoke`.

Describe the problem, implementation, validation and measurement limits in a pull request.
Performance changes should include controlled before/after measurements. Configuration tuning
and original code optimizations should be identified separately.
