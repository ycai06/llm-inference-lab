CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('running', 'completed', 'failed', 'canceled')),
    created_at TEXT NOT NULL,
    finished_at TEXT,
    config_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    error TEXT
);

CREATE TABLE cases (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(id),
    position INTEGER NOT NULL CHECK (position >= 0),
    config_json TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'timeout', 'canceled')),
    process_json TEXT,
    error TEXT,
    UNIQUE (experiment_id, position)
);

CREATE TABLE samples (
    case_id TEXT NOT NULL REFERENCES cases(id),
    trial_index INTEGER NOT NULL CHECK (trial_index >= 0),
    duration_ns INTEGER NOT NULL CHECK (duration_ns > 0),
    tokens INTEGER NOT NULL CHECK (tokens > 0),
    tokens_per_second REAL NOT NULL CHECK (tokens_per_second > 0),
    PRIMARY KEY (case_id, trial_index)
);

CREATE INDEX cases_experiment ON cases(experiment_id, position);
