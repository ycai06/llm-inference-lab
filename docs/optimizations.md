# Measured optimization investigation

The v0.1.0 investigation evaluates **upstream parameter tuning**. It does not include an
original C++ or Python inference optimization. No upstream files were modified.

## Hypothesis and control

Larger microbatches could reduce the number of prefill chunks and improve GPU use. Hold fixed
the model checksum, engine, Metal offload, six threads, 512 logical batch, flash attention `auto`,
512 synthetic prompt tokens, zero depth, native warmup and five measured repetitions. Change
only requested microbatch size among 64, 128, 256 and 512. Use `configs/tuning.yaml`.

The earlier 128-token exploration included microbatch sizes above the prompt length. Those
values cannot establish larger effective microbatch behavior; that experiment is preserved as
[exploratory data](../reports/microbatch-128-exploratory.json), not evidence of a gain. The
512-token follow-up exercises every requested size and preserves a separate experiment ID.

## Observation

| Microbatch | Median tok/s | Sample SD | Within-process median 95% bootstrap CI |
|---|---:|---:|---:|
| 64 | 127.77 | 1.34 | [124.87, 128.15] |
| 128 | 133.10 | 2.83 | [128.13, 133.87] |
| 256 | 113.19 | 4.22 | [111.12, 121.83] |
| 512 | 121.81 | 2.00 | [118.87, 124.33] |

The proposed increase from 128 to 512 was about 8.5% slower by the ratio of sample medians.
No uncertainty interval for that ratio was estimated. Keep 128 as the starting setting;
there is no validated improvement over the original default in this exploration.

The verbose engine logs report 49/49 layers offloaded in these runs. Sampled process RSS
ranged from about 6.07 to 6.11 GB (decimal), including loading and warmup; this does not measure
GPU VRAM or explain kernel behavior. Wall time and workload time remain separate.

## Interpretation and limits

Parameter sweeps are performance evidence, not a kernel profiler. The observations do not
identify a specific kernel, memory-bandwidth ceiling or scheduling bottleneck. Battery power,
unknown thermal pressure, background work and fixed within-session order can affect results.
Order was randomized by seed 42 (256, 128, 512, 64), with contiguous repetitions per case.
Five samples and within-process bootstrap intervals are insufficient for a general tuning rule.

Before changing code, run independent alternating 128/512 sessions on stable AC power, inspect
Metal traces with supported profiling tools, distinguish host dispatch from device work, and
repeat at other prompt lengths. Only then choose an original optimization and compare the same
workload before and after with a regression test. Speculative C++ edits are not justified by
this evidence. See [technical report](../reports/technical-report.md) and
[measured samples](../reports/microbatch-512.json).
