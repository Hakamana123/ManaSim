# Memory Layer

ManaSim uses a pluggable memory backend to store and query the knowledge
graph built from simulation documents.  All graph operations go through a
single abstract interface; swapping backends requires only a new Python
module and an environment variable change.

---

## Architecture

```
ManaSim services
      │
      ▼
MemoryBackend (abstract)          ← services/memory/base.py
      │
      ├── SupabaseMemoryBackend   ← services/memory/supabase.py  (stub — implement this)
      └── YourCustomBackend       ← services/memory/<name>.py
```

The active backend is selected at startup via the `MEMORY_BACKEND`
environment variable (default: `supabase`).

---

## Selecting a backend

In your `.env`:

```env
MEMORY_BACKEND=supabase
```

ManaSim will import `services/memory/<MEMORY_BACKEND>.py` and instantiate
the class named `MemoryBackendImpl`.

---

## Implementing a new backend

1. Copy `services/memory/supabase.py` to `services/memory/<yourname>.py`.
2. Implement every method that currently raises `NotImplementedError`.
3. Set `MEMORY_BACKEND=<yourname>` in your environment.

The interface is defined in `services/memory/base.py`.  Each method has a
docstring that describes exactly what it must do.

### Method summary

| Method | Description |
|---|---|
| `create_graph(graph_id, name, description)` | Create a new graph |
| `delete_graph(graph_id)` | Delete a graph and all its data |
| `add_episode(graph_id, text)` | Ingest a text chunk; returns episode UUID |
| `get_episode_status(episode_uuid)` | Poll episode processing status |
| `upsert_nodes(graph_id, nodes)` | Insert or update nodes |
| `upsert_edges(graph_id, edges)` | Insert or update edges |
| `search_edges(graph_id, query, limit)` | Semantic search over edges |
| `search_nodes(graph_id, query, limit)` | Semantic search over nodes |
| `fetch_all_nodes(graph_id, max_items)` | Return all nodes (paginated internally) |
| `fetch_all_edges(graph_id)` | Return all edges |
| `get_node(node_uuid)` | Single-node lookup |
| `get_node_edges(node_uuid)` | All edges for a node |

---

## Implementing the Supabase backend

### Prerequisites

- A [Supabase](https://supabase.com) project with the `pgvector` extension enabled.
- The `supabase-py` client: `pip install supabase`.
- Environment variables:
  ```env
  SUPABASE_URL=https://<project-ref>.supabase.co
  SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
  ```

### Schema setup

Apply `docs/supabase-schema.sql` once to your Supabase project:

```bash
psql "$DATABASE_URL" -f docs/supabase-schema.sql
```

Or paste it into the Supabase SQL Editor.

### Episode ingestion and extraction

The `add_episode` method must:

1. Insert a row into the `episodes` table (`status = 'pending'`).
2. Trigger an extraction pipeline that:
   - Sends the text to an LLM for Named Entity Recognition + Relation Extraction.
   - Writes the resulting entities as nodes (`upsert_nodes`).
   - Writes the resulting relations as edges (`upsert_edges`).
   - Updates the episode row to `status = 'complete'`.

The extraction pipeline can be implemented as:
- A [Supabase Edge Function](https://supabase.com/docs/guides/functions) triggered by a database webhook on the `episodes` table.
- A background worker process that polls for `status = 'pending'` episodes.
- A synchronous call within `add_episode` itself (simplest, but blocks).

### Semantic search

`search_edges` and `search_nodes` should use the `match_edges` and
`match_nodes` PostgreSQL functions defined in `docs/supabase-schema.sql`.

These functions combine:
- **pgvector** cosine similarity on pre-computed embeddings.
- **tsvector** full-text search for keyword recall.

To generate embeddings, use the same model at ingestion and query time
(e.g. `text-embedding-3-small` via the OpenAI API).

---

## Data types

Backends exchange data using these dataclasses (all defined in `base.py`):

| Type | Fields |
|---|---|
| `NodeData` | uuid, name, labels, summary, attributes |
| `EdgeData` | uuid, name, fact, source_node_uuid, target_node_uuid, attributes, created_at, valid_at, invalid_at, expired_at |
| `EpisodeStatus` | uuid, status, error |
| `SearchResults` | facts, edges, nodes, query, total_count |

Higher-level types (`EntityNode`, `FilteredEntities`) are assembled from
`NodeData`/`EdgeData` by `services/entity_reader.py` and are not returned
directly by the backend.

---

## Temporal edges

Edges support four timestamp fields that track validity over simulation time:

| Field | Meaning |
|---|---|
| `created_at` | When the edge was first recorded |
| `valid_at` | When the relationship became true in simulation time |
| `invalid_at` | When the relationship stopped being true |
| `expired_at` | When the record was soft-deleted |

An edge where `invalid_at` or `expired_at` is set is treated as historical.
The `PanoramaSearch` tool surfaces both active and historical facts.
