"""
Supabase memory backend stub.

This file is intentionally unimplemented. To contribute this backend:

  1. Read docs/memory-layer.md for the full interface contract.
  2. Apply docs/supabase-schema.sql to your Supabase project.
  3. Add SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY to your .env file.
  4. Install the supabase-py client:  pip install supabase
  5. Implement each method below, removing the NotImplementedError.

Every method includes a docstring describing exactly what the
implementation must do and which SQL tables/functions it should use.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import (
    EdgeData,
    EpisodeStatus,
    MemoryBackend,
    NodeData,
    SearchResults,
)


class MemoryBackendImpl(MemoryBackend):
    """Supabase-backed knowledge graph store."""

    def __init__(self) -> None:
        # TODO: Initialize the supabase-py client here.
        #
        # Required environment variables:
        #   SUPABASE_URL              — your project REST URL
        #   SUPABASE_SERVICE_ROLE_KEY — service role key (not the anon key)
        #
        # Example:
        #   import os
        #   from supabase import create_client
        #   self.client = create_client(
        #       os.environ["SUPABASE_URL"],
        #       os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        #   )
        raise NotImplementedError(
            "SupabaseMemoryBackend is not yet implemented. "
            "See docs/memory-layer.md and docs/supabase-schema.sql."
        )

    # ------------------------------------------------------------------
    # Graph lifecycle
    # ------------------------------------------------------------------

    def create_graph(self, graph_id: str, name: str, description: str = "") -> str:
        """
        Insert a row into the `graphs` table.

        SQL:
            INSERT INTO graphs (graph_id, name, description, created_at)
            VALUES (%s, %s, %s, NOW())
            ON CONFLICT (graph_id) DO NOTHING;

        Returns:
            graph_id (same as the input argument)
        """
        raise NotImplementedError

    def delete_graph(self, graph_id: str) -> None:
        """
        Delete the graph row and cascade-delete all associated data.

        The schema uses ON DELETE CASCADE on nodes, edges, and episodes so
        a single delete on graphs removes everything:

        SQL:
            DELETE FROM graphs WHERE graph_id = %s;
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Episode ingestion
    # ------------------------------------------------------------------

    def add_episode(self, graph_id: str, text: str) -> str:
        """
        Store a raw text episode and enqueue entity/relationship extraction.

        Recommended approach:
          1. INSERT a row into `episodes` with status='pending'.
          2. Trigger a Supabase Edge Function (or pg_net webhook) that:
               a. Calls an LLM for named entity recognition + relation extraction.
               b. Writes the resulting nodes/edges via upsert_nodes/upsert_edges.
               c. Updates the episode row to status='complete'.

        Returns:
            The UUID of the newly created episode row.
        """
        raise NotImplementedError

    def get_episode_status(self, episode_uuid: str) -> EpisodeStatus:
        """
        Poll the processing status of an episode.

        SQL:
            SELECT uuid, status, error FROM episodes WHERE uuid = %s;

        Map the DB status string to one of: "pending", "processing",
        "complete", "failed".

        Returns:
            EpisodeStatus
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Structured writes
    # ------------------------------------------------------------------

    def upsert_nodes(self, graph_id: str, nodes: List[Dict[str, Any]]) -> None:
        """
        Insert or update a batch of nodes.

        SQL (conceptual):
            INSERT INTO nodes (uuid, graph_id, name, labels, summary, attributes)
            VALUES (...)
            ON CONFLICT (graph_id, name, labels)
            DO UPDATE SET
                summary    = EXCLUDED.summary,
                attributes = EXCLUDED.attributes;

        Each input dict contains:
            name       (str)
            labels     (list[str])
            summary    (str)
            attributes (dict)

        Generate a UUID for new nodes. Store labels as a Postgres text[].
        """
        raise NotImplementedError

    def upsert_edges(self, graph_id: str, edges: List[Dict[str, Any]]) -> None:
        """
        Insert or update a batch of directed edges.

        SQL (conceptual):
            INSERT INTO edges (uuid, graph_id, name, fact,
                               source_node_uuid, target_node_uuid,
                               attributes, created_at, valid_at,
                               invalid_at, expired_at)
            VALUES (...)
            ON CONFLICT (graph_id, source_node_uuid, target_node_uuid, name)
            DO UPDATE SET
                fact       = EXCLUDED.fact,
                attributes = EXCLUDED.attributes,
                valid_at   = EXCLUDED.valid_at,
                invalid_at = EXCLUDED.invalid_at;

        Each input dict contains:
            name             (str)
            fact             (str)
            source_node_uuid (str)
            target_node_uuid (str)
        Optional:
            attributes  (dict)
            created_at  (str ISO-8601)
            valid_at    (str ISO-8601)
            invalid_at  (str ISO-8601)
            expired_at  (str ISO-8601)
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_edges(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """
        Hybrid semantic + keyword search over edges (relationship facts).

        Recommended: call the `match_edges` Postgres function defined in
        docs/supabase-schema.sql, which combines pgvector cosine similarity
        with tsvector full-text search.

        Steps:
          1. Generate a query embedding using the same model used at ingestion
             (e.g. text-embedding-3-small via OpenAI API).
          2. Call match_edges(graph_id, query_embedding, query_text, limit).
          3. Convert results to EdgeData objects.

        Returns:
            SearchResults with facts (list[str]) and edges (list[EdgeData]).
        """
        raise NotImplementedError

    def search_nodes(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """
        Hybrid semantic + keyword search over nodes (entities).

        Use the `match_nodes` Postgres function from docs/supabase-schema.sql.

        Returns:
            SearchResults with nodes (list[NodeData]).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Bulk reads
    # ------------------------------------------------------------------

    def fetch_all_nodes(self, graph_id: str, max_items: int = 2000) -> List[NodeData]:
        """
        Return all nodes in the graph up to max_items.

        SQL:
            SELECT uuid, name, labels, summary, attributes
            FROM nodes
            WHERE graph_id = %s
            LIMIT %s;

        Convert each row to a NodeData object.
        """
        raise NotImplementedError

    def fetch_all_edges(self, graph_id: str) -> List[EdgeData]:
        """
        Return all edges in the graph.

        SQL:
            SELECT uuid, name, fact, source_node_uuid, target_node_uuid,
                   attributes, created_at, valid_at, invalid_at, expired_at
            FROM edges
            WHERE graph_id = %s;

        Convert each row to an EdgeData object.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Point reads
    # ------------------------------------------------------------------

    def get_node(self, node_uuid: str) -> Optional[NodeData]:
        """
        Fetch a single node by UUID.

        SQL:
            SELECT uuid, name, labels, summary, attributes
            FROM nodes
            WHERE uuid = %s;

        Returns:
            NodeData if found, None otherwise.
        """
        raise NotImplementedError

    def get_node_edges(self, node_uuid: str) -> List[EdgeData]:
        """
        Return all edges connected to a given node (both directions).

        SQL:
            SELECT uuid, name, fact, source_node_uuid, target_node_uuid,
                   attributes, created_at, valid_at, invalid_at, expired_at
            FROM edges
            WHERE source_node_uuid = %s OR target_node_uuid = %s;

        Returns:
            List of EdgeData (incoming and outgoing).
        """
        raise NotImplementedError
