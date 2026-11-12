# Scope and remaining work

The v0.1.0 release provides a local CLI with synthetic microbenchmarks, persistence, exports,
resource samples, statistics, reports and tests.

- No FastAPI endpoints, React dashboard, HTTP streaming, concurrency,
  cold-load instrumentation or tokenizer-validated fixture experiments yet.
- No arbitrary context-window control: upstream llama-bench derives context from workload size.
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

Planned work: tokenizer-validated deterministic prompt fixtures via a managed server; request
and token-event latency; independent blocked/interleaved trials; FastAPI validation/lifecycle tests;
then a live-data React dashboard and broader length, thread, quantization and concurrency baselines.
Original C++/Python optimization
must follow profiling evidence and have a regression test and controlled before/after measurement.
