# Contribution ideas

These are investigation directions rather than confirmed upstream issues. Review current
documentation, issues and maintainer guidance before proposing a change.

1. Clarify llama-bench documentation around synthetic token workloads, per-repetition timing,
   warmup, context depth and KV behavior.
2. Investigate machine-readable release provenance for build options and linked runtime
   components, checking existing upstream solutions first.
3. Add version-specific regression fixtures for supported upstream JSON formats without
   changing benchmark timing paths.
4. Develop deterministic text-fixture benchmarks using server tokenization and generated-token
   events, including empty events, chunk coalescing, prefix caching and early EOS cases.
5. Collect a minimal reproducible workload and supported Metal trace for a concrete performance
   issue before proposing a dispatch or kernel change.

Useful LIOL extensions include independent session scheduling, explicit compatibility checks
for comparisons, structured model provenance and server-mode latency measurement. Performance
work should retain exact workloads and report negative or inconclusive results as well as gains.
