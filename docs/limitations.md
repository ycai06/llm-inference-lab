# Scope and remaining work

The CLI supports synthetic microbenchmarks and managed-server warm text requests with persistence,
exports, resource samples, statistics, reports and tests.

- No FastAPI endpoints, React dashboard, concurrency or cold-load instrumentation yet.
- Synthetic context size is derived by llama-bench. Request mode uses a fixed explicit context.
- Request mode measures native completion over fixture token prefixes, without chat templates
  or an answer-quality evaluation. Token prefixes may end partway through an instruction.
- Prefix reuse is disabled; cache-hit and concurrent workloads remain separate future experiments.
- Streaming token intervals are withheld for incomplete token coverage or coalesced receive events.
  TTFT is client-observed arrival timing and can include Unicode/transport buffering.
- Only one quantization was available for the reference experiments. A controlled quantization
  comparison requires compatible weights with recorded source revisions, licenses and checksums.
- Engine build options are only partially known from the existing release. The full upstream
  commit is documented, binary hash recorded, and dynamic library hashes are not yet included.
- Model source revision and license are not independently verified. Weights are never redistributed.
- Process resources include loading/warmup and are sampled. They are not inference-only resources,
  GPU VRAM, exact peak RSS, total system RAM, power or thermal measurements.
- Native decode benchmark operates without a populated prompt context. It does not establish
  latency, answer quality, EOS behavior or chat performance.
- Repetition bootstrap intervals omit between-session variation. Five observations are suitable
  for smoke checks and exploratory tuning, not strong tail or population claims.
- Raw local results include configured filesystem paths and should be reviewed before sharing.
  Public reports use portable labels and only compact measured data.
- The lock coordinates LIOL instances, not unrelated local inference frontends. It is advisory.
- Graceful interrupts/timeouts are handled; a hard kill or host shutdown can leave a running row.
- There is no automatic model/engine download, privileged profiling, upstream modification or CUDA.

Planned work: independent blocked/interleaved trials; FastAPI validation/lifecycle tests;
then a live-data React dashboard and broader length, thread, quantization and concurrency baselines.
Original C++/Python optimization
must follow profiling evidence and have a regression test and controlled before/after measurement.
