# Measurement methodology

## Supported workload

`synthetic_warm` runs **one synthetic workload per managed llama-bench invocation**.
Prefill uses `-p N -n 0`; decode uses `-p 0 -n N`. They measure different workloads
and are never pooled. In particular, decode begins at empty context, so it does not represent
generation after a 128/512/1024-token natural-language prompt.

The pinned upstream [benchmark implementation](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/tools/llama-bench/llama-bench.cpp)
uses synthetic vocabulary token IDs. Counts describe exact computational work, not character
estimates or tokenizer-validated prose. There is no sampled text, temperature, EOS stopping,
chat template, request transport or prompt fixture in this mode. The engine's implementation
and recorded commit define the synthetic workload; its RNG state is not exported by LIOL.

The context is derived by llama-bench from token workload and depth. Depth is required to be
zero in v0.1.0. Arbitrary context size and natural-text fixtures are not implemented;
unsupported configuration keys fail validation. The adapter requires the inspected flag syntax
and fails explicitly for older incompatible binaries rather than guessing.

## Time sources and units

Each upstream `samples_ns` entry is retained separately. Tokens/second is computed as
`tokens * 1e9 / duration_ns` and cross-checked against upstream `samples_ts` (allowing printed
rounding). Model loading and process startup are outside these timing samples.
Prefill and decode throughput have their own cases and units. `duration_ns / 1e6` gives milliseconds
for the **whole synthetic workload**, not end-user request latency or inter-token latency.

The supervisor separately measures wall-clock duration with `perf_counter_ns`; this includes
process startup, model loading, warmup and all repetitions. It must never be presented as
request latency or cold-load latency. Cold-load mode requires separate engine instrumentation.

TTFT, mean/median inter-token latency, concurrent throughput, GPU memory, GPU utilization,
thermal pressure and power are null/unavailable. In a later server mode, TTFT must begin at
request dispatch and stop at the first generated-token event, ignoring headers and metadata.
Network chunks can contain multiple tokens; chunk timing must not be passed off as device
inter-token timing. See the [official server interface](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/tools/server/README.md).

## Warmup, cache and ordering

Upstream native warmup remains enabled for every case. In the inspected source, it performs
prompt work and one decode token. It is not a full repeated decode warmup. The engine clears
KV memory before measured repetitions; LIOL does not enable prefix reuse. Model weight pages
may already be in the OS filesystem cache across processes; this is intentional for throughput.

The runner defaults to five measured repetitions. Cases are shuffled using a recorded seed;
repetitions within each case are contiguous. This reduces some ordering bias but does not
replace randomized interleaved independent sessions. Multiple LIOL workers are prevented by
a per-user process lock on macOS/Linux, including when different database files are selected.
Other programs can still contend for CPU, memory or GPU.

## Resource measurements

psutil samples the owned root process every 20 ms, across startup, warmup and measured work.
`sampled_peak_rss_bytes` is the maximum **observed process RSS**, not the kernel's exact high-water
mark, system memory use, or GPU VRAM. Apple unified-memory allocations can appear in host RSS;
this does not identify their placement. Fast processes may have no resource samples.
`sampled_cpu_seconds` is the last observed cumulative user+system CPU time and may miss the tail.
These measurements do not isolate a single repetition or just the inference region.

## Statistics and interpretation

Aggregate only repetitions from the same case, model checksum, engine and machine conditions.
Report arithmetic mean, median and sample standard deviation. Standard deviation is null
for n=1. Empirical interpolated p95 is withheld below n=20. At n>=5, report a deterministic
percentile bootstrap interval for the median (2,000 resamples, seed 42).

Five sequential samples provide weak evidence. Bootstrap intervals describe the recorded
within-process observations and do not account for thermal drift, power changes, autocorrelation,
different sessions or background load. They are not a population guarantee. Report a negative
or inconclusive result honestly. Compare configurations changing only the intended tuning factor;
do not combine different model revisions or token workloads into a speedup.

## Persistence and failure handling

SQLite schema migrations preserve experiment configuration, model SHA-256, available GGUF
metadata, environment, live binary help/version/device reads, exact argv, raw stdout/stderr,
exit code, wall time and normalized samples. Model stat information is checked before and
after each case to detect changes after hashing. This cannot prevent an external malicious
writer from replacing bytes and restoring all stat fields; models should remain immutable.

Timeouts and cancellation terminate the process group and reap the child. Failed, timed-out,
canceled and malformed-output cases contain no accepted timing samples. Queued cases are
canceled on interrupt. Validation failure during preflight produces no running experiment.
SIGKILL, machine shutdown and interpreter crashes cannot run cleanup; interrupted database
states require manual inspection in v0.1.0. JSON/CSV exports preserve failure rows.

The schema is version 1 (`backend/storage/migrations/001_initial.sql`). New migrations must
advance `PRAGMA user_version`. A newer unsupported database version is rejected.
