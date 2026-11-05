// store/schema.rs
// SQLite schema version 2: tables, indexes, and analysis views
// Applied once to an empty database

pub const APPLICATION_ID: i64 = 0x4132454e;
pub const VERSION: i64 = 2;

pub const TABLES: &str = "
CREATE TABLE studies(
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    definition TEXT NOT NULL,
    judge_config TEXT);
CREATE TABLE cases(
    study TEXT NOT NULL REFERENCES studies(id),
    id TEXT NOT NULL,
    hash TEXT NOT NULL,
    category TEXT NOT NULL,
    split TEXT NOT NULL,
    tags TEXT NOT NULL,
    definition TEXT NOT NULL,
    PRIMARY KEY(study, id));
CREATE TABLE model_loads(
    id INTEGER PRIMARY KEY,
    study TEXT NOT NULL REFERENCES studies(id),
    model TEXT NOT NULL,
    context INTEGER NOT NULL,
    load_ms REAL,
    peak_bytes INTEGER,
    template_sha256 TEXT,
    template TEXT,
    server_version TEXT,
    props TEXT,
    error TEXT);
CREATE TABLE runs(
    id INTEGER PRIMARY KEY,
    study TEXT NOT NULL REFERENCES studies(id),
    job TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    state TEXT NOT NULL,
    phase TEXT NOT NULL,
    model TEXT NOT NULL,
    context INTEGER NOT NULL,
    variant TEXT NOT NULL,
    variant_config TEXT NOT NULL,
    mode TEXT NOT NULL,
    repeat_index INTEGER NOT NULL,
    case_id TEXT NOT NULL,
    category TEXT NOT NULL,
    split TEXT NOT NULL,
    load_id INTEGER REFERENCES model_loads(id),
    trace TEXT NOT NULL,
    status TEXT,
    reason TEXT,
    failure_class TEXT,
    answer TEXT,
    evidence TEXT,
    final_state TEXT,
    retrieval_success INTEGER,
    model_calls INTEGER,
    tool_invocations INTEGER,
    valid_tool_calls INTEGER,
    argument_errors INTEGER,
    repairs INTEGER,
    prompt_tokens INTEGER,
    generated_tokens INTEGER,
    cached_tokens INTEGER,
    prefill_ms REAL,
    decode_ms REAL,
    http_ms REAL,
    template_ms REAL,
    tokenize_ms REAL,
    parse_ms REAL,
    validate_ms REAL,
    tool_ms REAL,
    tool_setup_ms REAL,
    wall_ms REAL,
    harness_other_ms REAL,
    harness_peak_bytes INTEGER,
    inference_peak_bytes INTEGER,
    combined_peak_bytes INTEGER,
    available_min_bytes INTEGER,
    cpu_time_ms INTEGER,
    minor_faults INTEGER,
    major_faults INTEGER,
    peak_temperature_c REAL,
    throttled TEXT,
    energy_j REAL,
    power_scope TEXT,
    sample_count INTEGER,
    memory_exceeded INTEGER,
    UNIQUE(study, job, attempt));
CREATE INDEX runs_condition ON runs(study, model, variant, context, category);
CREATE TABLE turns(
    run_id INTEGER NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    visible_tools INTEGER,
    template_ms REAL,
    tokenize_ms REAL,
    http_ms REAL,
    prefill_ms REAL,
    decode_ms REAL,
    prompt_tokens INTEGER,
    cached_tokens INTEGER,
    generated_tokens INTEGER,
    parse_ms REAL,
    parse_status TEXT NOT NULL,
    parse_error TEXT,
    raw_text TEXT,
    PRIMARY KEY(run_id, idx)) WITHOUT ROWID;
CREATE TABLE tool_calls(
    run_id INTEGER NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    turn INTEGER NOT NULL,
    tool TEXT NOT NULL,
    arguments TEXT NOT NULL,
    valid INTEGER NOT NULL,
    ok INTEGER NOT NULL,
    validate_ms REAL,
    ms REAL,
    output_bytes INTEGER,
    error TEXT,
    PRIMARY KEY(run_id, idx)) WITHOUT ROWID;
CREATE TABLE checks(
    run_id INTEGER NOT NULL REFERENCES runs(id),
    idx INTEGER NOT NULL,
    kind TEXT NOT NULL,
    label TEXT NOT NULL,
    passed INTEGER NOT NULL,
    failure_class TEXT,
    PRIMARY KEY(run_id, idx)) WITHOUT ROWID;
CREATE TABLE samples(
    run_id INTEGER REFERENCES runs(id),
    load_id INTEGER REFERENCES model_loads(id),
    idx INTEGER NOT NULL,
    t_ms REAL NOT NULL,
    harness_bytes INTEGER,
    inference_bytes INTEGER,
    available_bytes INTEGER,
    cpu_ms INTEGER,
    temperature_c REAL,
    throttled TEXT,
    power_w REAL,
    minor_faults INTEGER,
    major_faults INTEGER,
    CHECK((run_id IS NULL) != (load_id IS NULL)));
CREATE INDEX samples_run ON samples(run_id, idx);
CREATE INDEX samples_load ON samples(load_id, idx);
CREATE TABLE judgments(
    cache_key TEXT PRIMARY KEY,
    rubric TEXT NOT NULL,
    config TEXT NOT NULL,
    packet TEXT NOT NULL,
    response TEXT NOT NULL);
CREATE TABLE run_judgments(
    run_id INTEGER PRIMARY KEY REFERENCES runs(id),
    cache_key TEXT NOT NULL REFERENCES judgments(cache_key));
";

/// Views count only the latest attempt of each job; a judged `pass` counts as success.
pub const VIEWS: &str = "
CREATE VIEW v_run_summary AS
SELECT r.*,
    json_extract(j.response, '$.verdict') AS verdict,
    coalesce(json_extract(j.response, '$.provisional'), 0) AS provisional,
    (r.status = 'awaiting_judge' AND (j.cache_key IS NULL
        OR json_extract(j.response, '$.verdict') = 'unscored'
        OR coalesce(json_extract(j.response, '$.provisional'), 0))) AS pending,
    (r.status = 'success' OR (r.status = 'awaiting_judge'
        AND json_extract(j.response, '$.verdict') = 'pass'
        AND NOT coalesce(json_extract(j.response, '$.provisional'), 0))) AS passed
FROM runs r
LEFT JOIN run_judgments rj ON rj.run_id = r.id
LEFT JOIN judgments j ON j.cache_key = rj.cache_key
WHERE r.attempt = (SELECT max(n.attempt) FROM runs n WHERE n.study = r.study AND n.job = r.job);

CREATE VIEW v_category_success AS
SELECT study, model, variant, context, category,
    count(*) AS attempts,
    sum(NOT pending) AS scored,
    sum(passed AND NOT pending) AS passes,
    sum(pending) AS pending
FROM v_run_summary
WHERE phase = 'measured' AND state = 'complete'
GROUP BY study, model, variant, context, category;

CREATE VIEW v_failures AS
SELECT study, model, variant, context, failure_class, count(*) AS attempts
FROM v_run_summary
WHERE phase = 'measured' AND state = 'complete' AND failure_class IS NOT NULL
GROUP BY study, model, variant, context, failure_class;

CREATE VIEW v_resources AS
SELECT study, model, variant, context,
    count(*) AS attempts,
    max(combined_peak_bytes) AS peak_bytes,
    min(available_min_bytes) AS available_min_bytes,
    max(peak_temperature_c) AS peak_temperature_c,
    avg(wall_ms) AS mean_wall_ms,
    avg(cpu_time_ms) AS mean_cpu_ms,
    sum(memory_exceeded) AS memory_exceeded,
    sum(energy_j) AS energy_j
FROM v_run_summary
WHERE phase = 'measured' AND state = 'complete'
GROUP BY study, model, variant, context;
";
