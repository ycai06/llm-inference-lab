# Verified environment

Inspected locally on November 12, 2026. Hardware identifiers such as serial numbers are excluded.

| Item | Observation |
|---|---|
| OS | macOS 27.0.1, build 26A434; Darwin 27.0.0 |
| Architecture | arm64 |
| Machine | MacBook Pro, Mac14,9 |
| SoC | Apple M2 Pro |
| CPU | 10 physical/logical cores: 6 performance + 4 efficiency |
| Physical memory | 32 GiB (34,359,738,368 bytes), unified memory |
| GPU | Apple M2 Pro, 16 cores; Metal supported |
| Available disk at inspection | Approximately 154 GiB; varies with activity |
| System Python | 3.13.12 |
| Isolated project Python | 3.13.14, selected by uv |
| CMake | Not present on PATH; no engine build attempted |
| Compiler | Apple clang 21.0.0, arm64-apple-darwin27.0.0 |
| Git | 2.50.1 (Apple Git-155) |
| uv | 0.11.29 |
| Node / npm | 26.5.0 / 11.17.0; not required for the CLI |

## Existing engine

The reference runs use an existing llama.cpp macOS arm64 release installation. The engine
is configured through `LIOL_LLAMA_BENCH`; LIOL does not depend on an installation directory layout.

- Binary: `llama-bench`, also accompanied by `llama-cli`, `llama-server`, `llama-tokenize`.
- Reported version: `0.9.0-dev`, build `12182`, commit prefix `fd27b2034`.
- Exact upstream commit, resolved from the official release:
  [`fd27b2034905f8dadd8bd6d2cc44595f4929d391`](https://github.com/ggml-org/llama.cpp/commit/fd27b2034905f8dadd8bd6d2cc44595f4929d391).
- Compiler reported by binary: AppleClang 21.0.0.21000101 for Darwin arm64.
- Existing runtime manifest points to the official macOS arm64 b12182 release archive,
  with recorded archive SHA-256 `71cfee075e0c54852e225dafdc8dc79c32b26e8d45f052369b7d4f78d2ccf1a6`.
  This is pre-existing provenance; the archive was not downloaded or independently rehashed.
- Compiled/registered backends observed: MTL and Accelerate BLAS. Each experiment captures
  live help, version, device output and binary SHA-256.
- Full original CMake configuration is **unavailable**: this is an existing release installation,
  not a local source checkout. Do not infer exact build flags. Dynamically linked libraries
  are not individually fingerprinted in v0.1.0.

With local GPU access, `--list-devices` reports `MTL0: Apple M2 Pro`, with a
recommended working set around 25 GiB. This is a device limit, **not measured GPU memory use**.
Restricted process environments can hide Metal devices. Run Metal benchmarks in a local session
with GPU access. The adapter rejects Metal configurations when discovery cannot find an MTL device.

## Existing weights

The reference experiments use one existing GGUF:

| Item | Observation |
|---|---|
| Filename | `gemma-4-12b-it-Q3_K_M.gguf` |
| File bytes | 5,678,047,424 (~5.29 GiB) |
| SHA-256 | `dbf58f4c413b82b27106867eb42f7087e4172a172f6f201e2e11598a6582e504` |
| GGUF | v3; 667 tensors; architecture `gemma4` |
| Recorded quantization | file type 12; engine reports Q3_K - Medium |
| Recorded size | 12B; engine reports 11,907,350,576 parameters |
| Model metadata | Gemma-4-12B-It, quantized by Unsloth, 48 blocks |
| Model training context metadata | 262,144 tokens; **not allocated for these tests** |

The metadata contains a Google base-model link and both `general.license=apache-2.0`
and a Gemma-specific license link. The source revision and license have **not been independently
verified**. This repository neither redistributes nor downloads these weights. Hashing identifies
the exact existing file; header inspection alone does not validate all tensor contents.

MLX-format weights are not interchangeable with GGUF and are outside this backend.

## Recommended first experiments

The observed file size and actual smoke execution show this GGUF is suitable for small CPU
and Metal workloads. Begin with `smoke.yaml` or `quick-baseline.yaml`, then use `baseline.yaml`
for 128/512/1024 prefill tokens and 64/128/256 decode tokens. The latter takes substantially
longer on CPU. Start with six threads (performance-core count), and measure alternatives rather
than assuming all ten cores help. Use 512 batch / 128 microbatch initially.

CPU configurations disable layer, KV and operation offload and explicitly choose device `none`.
Metal configurations choose a discovered MTL device and request layer offload. Selected settings
and observed engine metadata are saved separately.

Power observations changed from AC to battery during inspection. Actual experiment metadata
records the power snapshot. Thermal pressure, watts and GPU utilization could not be measured
reliably and remain unavailable. Close competing workloads and keep power conditions stable
for publication-grade comparisons. No quantization comparison is currently possible with only
one compatible GGUF. The documented experiments reuse the existing weights.
