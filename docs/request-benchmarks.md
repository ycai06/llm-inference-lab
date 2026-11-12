# Warm text-request benchmarks

`request_warm` benchmarks native text completions through an owned localhost llama-server.
It measures actual request and generated-token event latency after initialization. It is a
separate operational mode from the synthetic prefill/decode workloads in `synthetic_warm`.

## Setup and execution

Configure `LIOL_MODEL` to an existing compatible GGUF and `LIOL_LLAMA_SERVER` to an existing
llama-server executable. No model or engine is downloaded automatically.

```sh
uv run liol doctor --engine server --binary "$LIOL_LLAMA_SERVER"
uv run liol plan configs/server-smoke.yaml
uv run liol run configs/server-smoke.yaml
uv run liol run configs/server-baseline.yaml
```

The smoke configuration is intentionally small and uses `ignore_eos: true` for a controlled
generation length. The baseline can take substantially longer, especially on CPU. Select
`gpu_layers: [0]` for CPU-only work or a supported Metal offload setting on an Apple GPU.
Close competing workloads and keep power conditions stable when comparing configurations.

Each matrix case owns a server with one request slot and a fixed context size. Startup and
model initialization are outside request timing. The runner performs one full unmeasured
warmup request by default, followed by five measured requests. It retains warmup evidence
separately from measured trials. Cases are shuffled by a recorded seed; requests within a
case are sequential. The existing per-user benchmark lock also covers this mode.

## Text fixtures and token counts

Fixture files are version-controlled UTF-8 text under `data/prompts/`. The runner stores the
exact source text and SHA-256 and asks the model's `/tokenize` endpoint for token IDs using
`add_special=true` and `parse_special=false`. Lengths include special tokens returned by this
policy; they are not estimates based on characters or words.

For a configured prompt length N, the runner selects the first N IDs from the tokenized fixture
and sends those IDs directly to native `/completion`. The selected prefix may end partway
through the fixture's discussion. Sending IDs avoids a second implicit text-tokenization or
BOS-insertion step. A fixture that yields fewer than N model tokens is rejected rather than
silently padded or repeated. The full fixture token count and selected prompt IDs are recorded.

Requests use native completion without a chat template, message wrappers, tools or multimodal
input. Their measurements therefore describe this explicit raw-completion workload rather than
an interactive chat application's entire processing path.

## Sampling and cache policy

Requests explicitly set a seed, temperature zero, generation target, streaming token IDs,
slot zero and `cache_prompt=false`. Prefix reuse is disabled in configuration and at server
startup. Every returned timing must report `timings.cache_n=0`; unexpected reuse fails validation.
The top-level `tokens_cached` field is not used as the reused-prefix count because the inspected
server includes the final slot state in that field.

`ignore_eos` is recorded explicitly. When false, EOS may end generation before the target;
observed token counts and `early_stop` are retained. The smoke setting forces continued
generation for repeatable workload lengths and should not be interpreted as ordinary chat
stopping behavior. Different stopping policies should be compared separately.

## Metric definitions

| Metric | Definition |
|---|---|
| Request latency, ms | Client monotonic time from dispatch through the terminal completion event; excludes server startup, tokenization and warmup |
| TTFT, ms | Dispatch to receipt of the first event containing a generated token ID; headers, null messages, empty metadata, progress and pings do not count |
| Native prefill, tokens/s | The server's prompt-processing rate for newly processed prompt tokens |
| Native decode, tokens/s | The server's decode timing, using `predicted_n - 1` token steps because the first generated token comes from prompt logits |
| Mean/median observed token interval, ms | Spacing between complete, individually observable generated-token arrivals, available only when event coverage is adequate |
| Generated tokens | Authoritative terminal generated-token count, checked against server timings; not an estimate from text length or SSE message count |
| Early stop | Observed generated count below the requested target |

Native throughput is retained separately from client-observed latency. A one-token completion
has no positive native decode interval rate, so decode throughput is unavailable. Zero-token
completion can be retained with unavailable TTFT and decode metrics. No timing is synthesized
to fill a missing measurement.

The client timestamps received transport chunks. Multiple SSE events can arrive together, and
one event can contain multiple IDs. Incomplete UTF-8 output can also cause event IDs to omit
tokens counted by the terminal response. These cases do not invalidate authoritative final
counts, but they make exact token-interval reconstruction unreliable. The runner records a
warning and leaves interval metrics unavailable when coverage is incomplete, IDs are grouped,
or arrivals are coalesced. TTFT is the observable arrival of a generated-token event, not an
exact timestamp of the device's first-token computation.

## Storage, reports and failure handling

SQLite schema version 2 adds a separate `request_samples` table. Existing synthetic rows remain
unchanged and use their original table. Request JSON exports retain each repetition's request
payload, terminal response, output, warnings, event-relative arrival times, observed counts and
metrics. Case records retain the fixture/tokenization evidence and managed-server provenance.
API credentials are omitted from persisted request payloads and process metadata.

CSV exports use request-specific units and fields. Reports aggregate within each case and
keep request latency, TTFT, native prefill and native decode distinct. Missing metrics remain
explicitly unavailable. Mean, median and sample SD are reported where applicable; median
bootstrap intervals require five measurements and p95 requires twenty. These intervals do not
capture between-session drift, thermal changes or correlation among requests within a process.

Timeouts and interrupts terminate the owned process group. A failed, timed-out or canceled
case has no accepted aggregate samples; successfully completed observations from an interrupted
case remain in its partial-evidence metadata. This prevents a partial case from being silently
treated as a completed experiment. Model changes before or during execution reject measurements.

Process RSS and cumulative CPU time include initialization and warmup. RSS on a unified-memory
Mac is not measured GPU VRAM. GPU utilization, power and thermal metrics remain unavailable.
The server binds to localhost, uses a private runtime credential and is shut down when the
case ends. This mode does not add a public API service, dashboard or concurrent serving test.

The protocol and timing definitions were checked against the inspected llama.cpp build and
the pinned [server documentation](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/tools/server/README.md)
and [timing definitions](https://github.com/ggml-org/llama.cpp/blob/fd27b2034905f8dadd8bd6d2cc44595f4929d391/tools/server/server-common.h).
