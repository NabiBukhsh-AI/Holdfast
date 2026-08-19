-- SC-GUARD schema. TASK-022, spec 19.1.
--
-- The registry is an EVENT LOG, not a mutable document. Spec 19.1 is explicit about why a
-- single JSONB registry column on sessions would be simpler and wrong: it makes point in time
-- reconstruction impossible (FR-082), makes concurrent appends a read-modify-write race, and
-- makes per constraint auditing impossible.
--
-- Two invariants are enforced by the DATABASE rather than by application code, because
-- application code is exactly what fails during an incident:
--   ck_supersede_status    a superseded row must point at what superseded it
--   uq_session_normalized  two workers racing on the same text produce one row, not two

BEGIN;

CREATE TABLE IF NOT EXISTS sessions (
    session_id        TEXT PRIMARY KEY,
    tenant_id         TEXT        NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    registry_version  INTEGER     NOT NULL DEFAULT 0,
    extractor_model   TEXT        NOT NULL,
    prompt_hash       TEXT        NOT NULL,
    schema_version    INTEGER     NOT NULL DEFAULT 1,
    expires_at        TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_tenant_activity ON sessions (tenant_id, last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_expiry          ON sessions (expires_at);

DO $do$ BEGIN
    CREATE TYPE sc_category AS ENUM ('action','information','process','preference','output','other');
EXCEPTION WHEN duplicate_object THEN NULL; END $do$;

DO $do$ BEGIN
    CREATE TYPE sc_status AS ENUM ('active','superseded','revoked','evicted');
EXCEPTION WHEN duplicate_object THEN NULL; END $do$;

CREATE TABLE IF NOT EXISTS session_constraints (
    constraint_id     TEXT        NOT NULL,
    session_id        TEXT        NOT NULL,
    tenant_id         TEXT        NOT NULL,
    seq               INTEGER     NOT NULL,          -- monotonic within session
    canonical_text    TEXT        NOT NULL,
    evidence_span     TEXT,                          -- verbatim user text; may contain PII
    evidence_span_enc BYTEA,                         -- optional field level encryption
    category          sc_category NOT NULL DEFAULT 'other',
    status            sc_status   NOT NULL DEFAULT 'active',
    superseded_by     TEXT,
    source_turn_index INTEGER     NOT NULL,
    token_count       INTEGER     NOT NULL,
    pinned            BOOLEAN     NOT NULL DEFAULT FALSE,
    extractor_model   TEXT        NOT NULL,
    prompt_hash       TEXT        NOT NULL,
    normalized_text   TEXT        NOT NULL,          -- for exact duplicate detection
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    status_changed_at TIMESTAMPTZ,

    PRIMARY KEY (session_id, constraint_id),
    CONSTRAINT uq_session_seq         UNIQUE (session_id, seq),
    CONSTRAINT uq_session_normalized  UNIQUE (session_id, normalized_text),
    CONSTRAINT ck_supersede_status    CHECK (
        (status = 'superseded' AND superseded_by IS NOT NULL) OR
        (status <> 'superseded' AND superseded_by IS NULL)
    ),
    CONSTRAINT ck_token_count_positive CHECK (token_count > 0),
    -- Exactly one of the plaintext and encrypted evidence columns is populated. Spec 19.1
    -- leaves this to application code plus a scheduled audit query; asserting it here means a
    -- regulated tenant cannot silently accumulate plaintext PII.
    CONSTRAINT ck_evidence_exclusive CHECK (
        evidence_span IS NULL OR evidence_span_enc IS NULL
    )
) PARTITION BY HASH (session_id);

-- Growth is driven by session count, so hash partitioning distributes writes evenly and keeps
-- each partition active index small.
DO $do$
DECLARE i INTEGER;
BEGIN
    FOR i IN 0..7 LOOP
        EXECUTE format(
            'CREATE TABLE IF NOT EXISTS session_constraints_p%s PARTITION OF session_constraints '
            'FOR VALUES WITH (MODULUS 8, REMAINDER %s)', i, i
        );
    END LOOP;
END $do$;

-- Assembly reads only active rows. The active set is bounded by budget; the total set is not.
CREATE INDEX IF NOT EXISTS idx_sc_session_active
    ON session_constraints (session_id, seq) WHERE status = 'active';
CREATE INDEX IF NOT EXISTS idx_sc_tenant_created  ON session_constraints (tenant_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_sc_category_active ON session_constraints (category) WHERE status = 'active';

-- Append only is enforced here, not merely intended. Rewriting a constraint text after the
-- fact would let the registry lie about what the user asked for, which is unauditable.
CREATE OR REPLACE FUNCTION forbid_constraint_text_update() RETURNS TRIGGER AS $fn$
BEGIN
    IF NEW.canonical_text IS DISTINCT FROM OLD.canonical_text
       OR NEW.evidence_span IS DISTINCT FROM OLD.evidence_span
       OR NEW.normalized_text IS DISTINCT FROM OLD.normalized_text
       OR NEW.source_turn_index IS DISTINCT FROM OLD.source_turn_index
       OR NEW.created_at IS DISTINCT FROM OLD.created_at THEN
        RAISE EXCEPTION
            'session_constraints is append only: constraint % text and provenance are immutable (FR-080). Tombstone and supersede instead.',
            OLD.constraint_id;
    END IF;
    RETURN NEW;
END $fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_forbid_constraint_text_update ON session_constraints;
CREATE TRIGGER trg_forbid_constraint_text_update
    BEFORE UPDATE ON session_constraints
    FOR EACH ROW EXECUTE FUNCTION forbid_constraint_text_update();

DO $do$ BEGIN
    CREATE TYPE job_status AS ENUM ('queued','running','succeeded','failed','parse_error');
EXCEPTION WHEN duplicate_object THEN NULL; END $do$;

CREATE TABLE IF NOT EXISTS extraction_jobs (
    job_id            TEXT        PRIMARY KEY,
    session_id        TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    turn_index        INTEGER     NOT NULL,
    content_hash      TEXT        NOT NULL,
    status            job_status  NOT NULL DEFAULT 'queued',
    attempts          SMALLINT    NOT NULL DEFAULT 0,
    n_extracted       SMALLINT,
    raw_response      TEXT,                          -- retained for audit and reparse
    error_detail      TEXT,
    enqueued_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    claimed_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    latency_ms        INTEGER,

    CONSTRAINT uq_job_idempotency UNIQUE (session_id, turn_index, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_jobs_pending ON extraction_jobs (session_id)
    WHERE status IN ('queued','running');

CREATE TABLE IF NOT EXISTS audit_events (
    event_id        BIGSERIAL,
    session_id      TEXT        NOT NULL,
    tenant_id       TEXT        NOT NULL,
    event_type      TEXT        NOT NULL,
    constraint_id   TEXT,
    turn_index      INTEGER,
    payload         JSONB       NOT NULL,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, occurred_at)
) PARTITION BY RANGE (occurred_at);

CREATE INDEX IF NOT EXISTS idx_audit_session_time ON audit_events (session_id, occurred_at);
CREATE INDEX IF NOT EXISTS idx_audit_type_time    ON audit_events (event_type, occurred_at DESC);
CREATE INDEX IF NOT EXISTS idx_audit_payload      ON audit_events USING GIN (payload);

CREATE TABLE IF NOT EXISTS assemblies (
    assembly_id          TEXT        PRIMARY KEY,
    session_id           TEXT        NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    compaction_index     INTEGER     NOT NULL,
    registry_version     INTEGER     NOT NULL,
    active_count         INTEGER     NOT NULL,
    injected_count       INTEGER     NOT NULL,
    evicted_count        INTEGER     NOT NULL,
    registry_tokens      INTEGER     NOT NULL,
    summary_tokens       INTEGER     NOT NULL,
    registry_incomplete  BOOLEAN     NOT NULL,
    drain_wait_ms        INTEGER     NOT NULL,
    assembly_ms          INTEGER     NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT uq_assembly UNIQUE (session_id, compaction_index)
);
-- The incident response query: show me every assembly that ran without a complete registry.
CREATE INDEX IF NOT EXISTS idx_assemblies_incomplete ON assemblies (created_at DESC)
    WHERE registry_incomplete = TRUE;

COMMIT;
