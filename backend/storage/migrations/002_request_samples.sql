CREATE TABLE request_samples (
    case_id TEXT NOT NULL REFERENCES cases(id),
    trial_index INTEGER NOT NULL CHECK (trial_index >= 0),
    request_latency_ms REAL NOT NULL CHECK (request_latency_ms > 0),
    ttft_ms REAL CHECK (ttft_ms >= 0),
    prompt_tokens INTEGER NOT NULL CHECK (prompt_tokens > 0),
    prompt_processed_tokens INTEGER NOT NULL CHECK (prompt_processed_tokens >= 0),
    generated_tokens INTEGER NOT NULL CHECK (generated_tokens >= 0),
    target_generated_tokens INTEGER NOT NULL CHECK (target_generated_tokens > 0),
    prefill_tokens_per_second REAL CHECK (prefill_tokens_per_second > 0),
    decode_tokens_per_second REAL CHECK (decode_tokens_per_second > 0),
    mean_inter_token_latency_ms REAL CHECK (mean_inter_token_latency_ms >= 0),
    median_inter_token_latency_ms REAL CHECK (median_inter_token_latency_ms >= 0),
    cache_reused_tokens INTEGER NOT NULL CHECK (cache_reused_tokens >= 0),
    details_json TEXT NOT NULL,
    PRIMARY KEY (case_id, trial_index)
);
