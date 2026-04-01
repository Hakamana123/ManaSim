"""
Memory layer abstraction for ManaSim.

Every storage backend must implement MemoryBackend.
Select a backend at runtime via the MEMORY_BACKEND environment variable.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ---------------------------------------------------------------------------
# Primitive data types returned by the backend
# ---------------------------------------------------------------------------

@dataclass
class NodeData:
    """A node (entity) in the knowledge graph."""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]


@dataclass
class EdgeData:
    """A directed relationship between two nodes."""

    uuid: str
    name: str                   # relationship type label
    fact: str                   # natural-language fact sentence
    source_node_uuid: str
    target_node_uuid: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class EpisodeStatus:
    """Processing status of a submitted text episode."""

    uuid: str
    status: str   # "pending" | "processing" | "complete" | "failed"
    error: Optional[str] = None

    @property
    def is_complete(self) -> bool:
        return self.status == "complete"

    @property
    def is_failed(self) -> bool:
        return self.status == "failed"


@dataclass
class SearchResults:
    """Unified container for graph search results."""

    facts: List[str]
    edges: List[EdgeData]
    nodes: List[NodeData]
    query: str
    total_count: int


# ---------------------------------------------------------------------------
# Higher-level entity types used throughout the application
# ---------------------------------------------------------------------------

@dataclass
class EntityNode:
    """
    A graph node enriched with its neighbouring edges and nodes.
    Used throughout the application to represent simulation entities.
    """

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    related_edges: List[Dict[str, Any]] = field(default_factory=list)
    related_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
            "related_edges": self.related_edges,
            "related_nodes": self.related_nodes,
        }

    def get_entity_type(self) -> Optional[str]:
        """Return the primary custom label, excluding generic 'Entity'/'Node'."""
        for label in self.labels:
            if label not in ("Entity", "Node"):
                return label
        return None


@dataclass
class FilteredEntities:
    """Result of filtering graph nodes to known entity types."""

    entities: List[EntityNode]
    entity_types: Set[str]
    total_count: int
    filtered_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": [e.to_dict() for e in self.entities],
            "entity_types": list(self.entity_types),
            "total_count": self.total_count,
            "filtered_count": self.filtered_count,
        }


# ---------------------------------------------------------------------------
# Abstract backend interface
# ---------------------------------------------------------------------------

class MemoryBackend(ABC):
    """
    Abstract memory backend.

    Implement this class to add a new storage backend (Supabase, Neo4j, …).
    All methods that previously called Zep Cloud APIs map to methods here.
    """

    # ------------------------------------------------------------------
    # Graph lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def create_graph(self, graph_id: str, name: str, description: str = "") -> str:
        """
        Create a new knowledge graph.

        Args:
            graph_id: Caller-chosen stable identifier for this graph.
            name: Human-readable graph name.
            description: Optional description.

        Returns:
            The graph_id that was created (same as the input).
        """

    @abstractmethod
    def delete_graph(self, graph_id: str) -> None:
        """
        Permanently delete a graph and all its nodes and edges.

        Args:
            graph_id: The graph to delete.
        """

    # ------------------------------------------------------------------
    # Episode ingestion (unstructured text → graph)
    # ------------------------------------------------------------------

    @abstractmethod
    def add_episode(self, graph_id: str, text: str) -> str:
        """
        Submit a plain-text episode for extraction and ingestion into the graph.

        The backend is responsible for parsing entities and relationships from
        the text and persisting them as nodes and edges.

        Args:
            graph_id: Target graph.
            text: Natural-language text to ingest.

        Returns:
            An episode UUID that can be polled with get_episode_status().
        """

    @abstractmethod
    def get_episode_status(self, episode_uuid: str) -> EpisodeStatus:
        """
        Poll the processing status of a previously submitted episode.

        Args:
            episode_uuid: UUID returned by add_episode().

        Returns:
            EpisodeStatus with current status and any error message.
        """

    # ------------------------------------------------------------------
    # Structured writes
    # ------------------------------------------------------------------

    @abstractmethod
    def upsert_nodes(self, graph_id: str, nodes: List[Dict[str, Any]]) -> None:
        """
        Insert or update a batch of nodes in the graph.

        Each dict must contain at minimum: name, labels, summary, attributes.
        Existing nodes with the same identity should be updated, not duplicated.

        Args:
            graph_id: Target graph.
            nodes: List of node attribute dicts.
        """

    @abstractmethod
    def upsert_edges(self, graph_id: str, edges: List[Dict[str, Any]]) -> None:
        """
        Insert or update a batch of directed edges in the graph.

        Each dict must contain: name, fact, source_node_uuid, target_node_uuid.
        Optional fields: attributes, created_at, valid_at, invalid_at, expired_at.

        Args:
            graph_id: Target graph.
            edges: List of edge attribute dicts.
        """

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    @abstractmethod
    def search_edges(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """
        Semantic / hybrid search over graph edges (relationship facts).

        Args:
            graph_id: Graph to search.
            query: Natural-language search query.
            limit: Maximum results to return.

        Returns:
            SearchResults with matched facts and edge objects.
        """

    @abstractmethod
    def search_nodes(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
    ) -> SearchResults:
        """
        Semantic / hybrid search over graph nodes (entities).

        Args:
            graph_id: Graph to search.
            query: Natural-language search query.
            limit: Maximum results to return.

        Returns:
            SearchResults with matched node objects.
        """

    # ------------------------------------------------------------------
    # Bulk reads
    # ------------------------------------------------------------------

    @abstractmethod
    def fetch_all_nodes(self, graph_id: str, max_items: int = 2000) -> List[NodeData]:
        """
        Return every node in the graph, handling pagination internally.

        Args:
            graph_id: Source graph.
            max_items: Hard cap on the number of nodes returned.

        Returns:
            List of NodeData objects.
        """

    @abstractmethod
    def fetch_all_edges(self, graph_id: str) -> List[EdgeData]:
        """
        Return every edge in the graph, handling pagination internally.

        Args:
            graph_id: Source graph.

        Returns:
            List of EdgeData objects.
        """

    # ------------------------------------------------------------------
    # Point reads
    # ------------------------------------------------------------------

    @abstractmethod
    def get_node(self, node_uuid: str) -> Optional[NodeData]:
        """
        Fetch a single node by its UUID.

        Args:
            node_uuid: The node's UUID.

        Returns:
            NodeData if found, None otherwise.
        """

    @abstractmethod
    def get_node_edges(self, node_uuid: str) -> List[EdgeData]:
        """
        Return all edges that connect to or from a given node (both directions).

        Args:
            node_uuid: The node's UUID.

        Returns:
            List of EdgeData (incoming and outgoing).
        """
