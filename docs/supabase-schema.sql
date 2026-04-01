-- =============================================================================
-- ManaSim — Supabase Memory Schema
-- =============================================================================
-- Apply this script once to your Supabase project before using the Supabase
-- memory backend.
--
-- Prerequisites:
--   • pgvector extension must be available (included in Supabase by default)
--   • Run as a superuser or the postgres role in the SQL Editor
--
-- Usage:
--   psql "$DATABASE_URL" -f docs/supabase-schema.sql
--   — or —
--   Paste into Supabase SQL Editor and click Run.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;   -- used by full-text GIN indexes


-- ---------------------------------------------------------------------------
-- Graphs
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS graphs (
    graph_id    TEXT        PRIMARY KEY,
    name        TEXT        NOT NULL DEFAULT '',
    description TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  graphs              IS 'One row per simulation knowledge graph.';
COMMENT ON COLUMN graphs.graph_id    IS 'Caller-supplied identifier (e.g. simulation UUID).';


-- ---------------------------------------------------------------------------
-- Episodes
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS episodes (
    uuid        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id    TEXT        NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,
    content     TEXT        NOT NULL,
    status      TEXT        NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending', 'processing', 'complete', 'error')),
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

COMMENT ON TABLE  episodes          IS 'Raw text chunks ingested into a graph. Processing extracts nodes and edges.';
COMMENT ON COLUMN episodes.status   IS 'pending → processing → complete | error';
COMMENT ON COLUMN episodes.error    IS 'Populated when status = ''error''.';

CREATE INDEX IF NOT EXISTS episodes_graph_id_status_idx
    ON episodes (graph_id, status);


-- ---------------------------------------------------------------------------
-- Nodes  (entities)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS nodes (
    uuid        UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id    TEXT        NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,

    -- Core fields
    name        TEXT        NOT NULL,
    labels      TEXT[]      NOT NULL DEFAULT '{}',   -- e.g. ['Person', 'Agent']
    summary     TEXT        NOT NULL DEFAULT '',
    attributes  JSONB       NOT NULL DEFAULT '{}',

    -- Vector embedding of  name + summary  (dimension must match your model)
    -- text-embedding-3-small → 1536 dimensions
    embedding   VECTOR(1536),

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- A node is uniquely identified by its name within a graph
    UNIQUE (graph_id, name)
);

COMMENT ON TABLE  nodes            IS 'Entities extracted from episodes.';
COMMENT ON COLUMN nodes.labels     IS 'Type tags for the entity, e.g. Person, Organisation, Location.';
COMMENT ON COLUMN nodes.attributes IS 'Arbitrary key-value metadata as JSON.';
COMMENT ON COLUMN nodes.embedding  IS 'Vector embedding used for semantic search.';

CREATE INDEX IF NOT EXISTS nodes_graph_id_idx
    ON nodes (graph_id);

CREATE INDEX IF NOT EXISTS nodes_labels_idx
    ON nodes USING GIN (labels);

-- Full-text search index on name + summary
CREATE INDEX IF NOT EXISTS nodes_fts_idx
    ON nodes USING GIN (to_tsvector('english', name || ' ' || summary));

-- ANN index for fast approximate nearest-neighbour search (IVFFlat)
-- Tune lists = sqrt(row_count) after the table has data.
CREATE INDEX IF NOT EXISTS nodes_embedding_idx
    ON nodes USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- ---------------------------------------------------------------------------
-- Edges  (relationships / facts)
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS edges (
    uuid             UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    graph_id         TEXT        NOT NULL REFERENCES graphs(graph_id) ON DELETE CASCADE,

    -- The human-readable name of the relationship (e.g. "works_for")
    name             TEXT        NOT NULL,

    -- The natural-language statement of the fact
    fact             TEXT        NOT NULL DEFAULT '',

    source_node_uuid UUID        REFERENCES nodes(uuid) ON DELETE SET NULL,
    target_node_uuid UUID        REFERENCES nodes(uuid) ON DELETE SET NULL,

    attributes       JSONB       NOT NULL DEFAULT '{}',

    -- Temporal validity
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_at         TIMESTAMPTZ,          -- when the relationship became true
    invalid_at       TIMESTAMPTZ,          -- when the relationship stopped being true
    expired_at       TIMESTAMPTZ,          -- soft-delete timestamp

    -- Vector embedding of  name + fact  (same model/dimension as nodes)
    embedding        VECTOR(1536)
);

COMMENT ON TABLE  edges                 IS 'Directed relationships between entities.';
COMMENT ON COLUMN edges.name            IS 'Relationship type label, e.g. works_for, mentioned_in.';
COMMENT ON COLUMN edges.fact            IS 'Natural-language statement of the relationship.';
COMMENT ON COLUMN edges.valid_at        IS 'When the relationship became true in simulation time.';
COMMENT ON COLUMN edges.invalid_at      IS 'When the relationship stopped being true. NULL = still active.';
COMMENT ON COLUMN edges.expired_at      IS 'Soft-delete marker. NULL = record is live.';
COMMENT ON COLUMN edges.embedding       IS 'Vector embedding used for semantic search.';

CREATE INDEX IF NOT EXISTS edges_graph_id_idx
    ON edges (graph_id);

CREATE INDEX IF NOT EXISTS edges_source_node_idx
    ON edges (source_node_uuid);

CREATE INDEX IF NOT EXISTS edges_target_node_idx
    ON edges (target_node_uuid);

-- Partial index: only active (non-expired, non-invalidated) edges
CREATE INDEX IF NOT EXISTS edges_active_idx
    ON edges (graph_id, valid_at)
    WHERE expired_at IS NULL AND invalid_at IS NULL;

-- Full-text search index on name + fact
CREATE INDEX IF NOT EXISTS edges_fts_idx
    ON edges USING GIN (to_tsvector('english', name || ' ' || fact));

-- ANN index for approximate nearest-neighbour search
CREATE INDEX IF NOT EXISTS edges_embedding_idx
    ON edges USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);


-- ---------------------------------------------------------------------------
-- Helper: auto-update updated_at on nodes and episodes
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS nodes_updated_at    ON nodes;
CREATE TRIGGER nodes_updated_at
    BEFORE UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

DROP TRIGGER IF EXISTS episodes_updated_at ON episodes;
CREATE TRIGGER episodes_updated_at
    BEFORE UPDATE ON episodes
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();


-- ---------------------------------------------------------------------------
-- Semantic search function: match_edges
-- ---------------------------------------------------------------------------
-- Returns edges ranked by a blend of vector similarity and full-text score.
-- Call this from the Supabase client or directly via SQL.
--
-- Parameters:
--   p_graph_id       — restrict to this graph
--   p_embedding      — query vector (same model used at ingestion)
--   p_query_text     — raw query string for full-text ranking
--   p_match_count    — maximum rows to return
--   p_include_expired — when false, edges with expired_at or invalid_at set
--                       are excluded (default: false)
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION match_edges(
    p_graph_id        TEXT,
    p_embedding       VECTOR(1536),
    p_query_text      TEXT,
    p_match_count     INT     DEFAULT 10,
    p_include_expired BOOLEAN DEFAULT FALSE
)
RETURNS TABLE (
    uuid             UUID,
    graph_id         TEXT,
    name             TEXT,
    fact             TEXT,
    source_node_uuid UUID,
    target_node_uuid UUID,
    attributes       JSONB,
    created_at       TIMESTAMPTZ,
    valid_at         TIMESTAMPTZ,
    invalid_at       TIMESTAMPTZ,
    expired_at       TIMESTAMPTZ,
    similarity       FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT
        e.uuid,
        e.graph_id,
        e.name,
        e.fact,
        e.source_node_uuid,
        e.target_node_uuid,
        e.attributes,
        e.created_at,
        e.valid_at,
        e.invalid_at,
        e.expired_at,
        -- Blend vector cosine similarity (0-1) and normalised ts_rank (0-1)
        (
            (1 - (e.embedding <=> p_embedding)) * 0.7
            +
            ts_rank(
                to_tsvector('english', e.name || ' ' || e.fact),
                plainto_tsquery('english', p_query_text)
            ) * 0.3
        ) AS similarity
    FROM edges e
    WHERE
        e.graph_id = p_graph_id
        AND e.embedding IS NOT NULL
        AND (p_include_expired OR (e.expired_at IS NULL AND e.invalid_at IS NULL))
    ORDER BY similarity DESC
    LIMIT p_match_count;
$$;

COMMENT ON FUNCTION match_edges IS
    'Semantic + keyword hybrid search over edges for a given graph.';


-- ---------------------------------------------------------------------------
-- Semantic search function: match_nodes
-- ---------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION match_nodes(
    p_graph_id     TEXT,
    p_embedding    VECTOR(1536),
    p_query_text   TEXT,
    p_match_count  INT DEFAULT 10
)
RETURNS TABLE (
    uuid       UUID,
    graph_id   TEXT,
    name       TEXT,
    labels     TEXT[],
    summary    TEXT,
    attributes JSONB,
    created_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ,
    similarity FLOAT
)
LANGUAGE sql STABLE AS $$
    SELECT
        n.uuid,
        n.graph_id,
        n.name,
        n.labels,
        n.summary,
        n.attributes,
        n.created_at,
        n.updated_at,
        (
            (1 - (n.embedding <=> p_embedding)) * 0.7
            +
            ts_rank(
                to_tsvector('english', n.name || ' ' || n.summary),
                plainto_tsquery('english', p_query_text)
            ) * 0.3
        ) AS similarity
    FROM nodes n
    WHERE
        n.graph_id = p_graph_id
        AND n.embedding IS NOT NULL
    ORDER BY similarity DESC
    LIMIT p_match_count;
$$;

COMMENT ON FUNCTION match_nodes IS
    'Semantic + keyword hybrid search over nodes for a given graph.';


-- ---------------------------------------------------------------------------
-- Row-Level Security (optional but recommended)
-- ---------------------------------------------------------------------------
-- Uncomment the block below if you want each simulation to be isolated at
-- the database level using the graph_id claim in a Supabase JWT.

-- ALTER TABLE graphs   ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE episodes ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE nodes    ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE edges    ENABLE ROW LEVEL SECURITY;
--
-- CREATE POLICY "graph owner can read"
--     ON graphs FOR SELECT
--     USING (graph_id = current_setting('app.graph_id', TRUE));
--
-- -- Repeat similar policies for episodes, nodes, edges as needed.


-- ---------------------------------------------------------------------------
-- Done
-- ---------------------------------------------------------------------------

-- Verify:
-- SELECT tablename FROM pg_tables WHERE schemaname = 'public'
--     AND tablename IN ('graphs', 'episodes', 'nodes', 'edges');
