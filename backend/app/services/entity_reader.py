"""
Entity reader service.

Reads nodes from the configured memory backend and filters them to the
entity types defined in the simulation ontology.

Replaces the former zep_entity_reader.py.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from .memory import get_memory_backend
from .memory.base import EntityNode, FilteredEntities, MemoryBackend
from ..utils.logger import get_logger

logger = get_logger('manasim.entity_reader')


class EntityReader:
    """
    Reads and filters graph entities using the configured memory backend.

    Responsibilities:
      1. Fetch all nodes from the graph.
      2. Filter to nodes that have custom entity labels.
      3. Optionally enrich each entity with related edges and neighbours.
    """

    def __init__(self, backend: Optional[MemoryBackend] = None) -> None:
        self.backend = backend or get_memory_backend()

    # ------------------------------------------------------------------
    # Raw data fetchers
    # ------------------------------------------------------------------

    def get_all_nodes(self, graph_id: str) -> List[Dict[str, Any]]:
        """Return all nodes in the graph as plain dicts."""
        logger.info(f"Fetching all nodes for graph {graph_id}...")
        nodes = self.backend.fetch_all_nodes(graph_id)
        result = [
            {
                "uuid": n.uuid,
                "name": n.name,
                "labels": n.labels,
                "summary": n.summary,
                "attributes": n.attributes,
            }
            for n in nodes
        ]
        logger.info(f"Fetched {len(result)} nodes")
        return result

    def get_all_edges(self, graph_id: str) -> List[Dict[str, Any]]:
        """Return all edges in the graph as plain dicts."""
        logger.info(f"Fetching all edges for graph {graph_id}...")
        edges = self.backend.fetch_all_edges(graph_id)
        result = [
            {
                "uuid": e.uuid,
                "name": e.name,
                "fact": e.fact,
                "source_node_uuid": e.source_node_uuid,
                "target_node_uuid": e.target_node_uuid,
                "attributes": e.attributes,
            }
            for e in edges
        ]
        logger.info(f"Fetched {len(result)} edges")
        return result

    def get_node_edges(self, node_uuid: str) -> List[Dict[str, Any]]:
        """Return all edges for a specific node as plain dicts."""
        try:
            edges = self.backend.get_node_edges(node_uuid)
            return [
                {
                    "uuid": e.uuid,
                    "name": e.name,
                    "fact": e.fact,
                    "source_node_uuid": e.source_node_uuid,
                    "target_node_uuid": e.target_node_uuid,
                    "attributes": e.attributes,
                }
                for e in edges
            ]
        except Exception as exc:
            logger.warning(f"Failed to fetch edges for node {node_uuid}: {exc}")
            return []

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_defined_entities(
        self,
        graph_id: str,
        defined_entity_types: Optional[List[str]] = None,
        enrich_with_edges: bool = True,
    ) -> FilteredEntities:
        """
        Filter graph nodes to those with custom (non-generic) entity labels.

        Filtering logic:
          - Nodes whose labels contain only "Entity" or "Node" are skipped.
          - Nodes with any additional label qualify as typed entities.
          - If defined_entity_types is provided, only matching labels are kept.

        Args:
            graph_id: The graph to read.
            defined_entity_types: Optional allowlist of entity type labels.
            enrich_with_edges: Populate related_edges / related_nodes when True.

        Returns:
            FilteredEntities
        """
        logger.info(f"Filtering entities in graph {graph_id}...")

        all_nodes = self.get_all_nodes(graph_id)
        total_count = len(all_nodes)

        all_edges = self.get_all_edges(graph_id) if enrich_with_edges else []
        node_map = {n["uuid"]: n for n in all_nodes}

        filtered_entities: List[EntityNode] = []
        entity_types_found: Set[str] = set()

        for node in all_nodes:
            labels = node.get("labels", [])
            custom_labels = [lbl for lbl in labels if lbl not in ("Entity", "Node")]

            if not custom_labels:
                continue

            if defined_entity_types:
                matching = [lbl for lbl in custom_labels if lbl in defined_entity_types]
                if not matching:
                    continue
                entity_type = matching[0]
            else:
                entity_type = custom_labels[0]

            entity_types_found.add(entity_type)

            entity = EntityNode(
                uuid=node["uuid"],
                name=node["name"],
                labels=labels,
                summary=node["summary"],
                attributes=node["attributes"],
            )

            if enrich_with_edges:
                related_edges: List[Dict[str, Any]] = []
                related_node_uuids: Set[str] = set()

                for edge in all_edges:
                    if edge["source_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "outgoing",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "target_node_uuid": edge["target_node_uuid"],
                        })
                        related_node_uuids.add(edge["target_node_uuid"])
                    elif edge["target_node_uuid"] == node["uuid"]:
                        related_edges.append({
                            "direction": "incoming",
                            "edge_name": edge["name"],
                            "fact": edge["fact"],
                            "source_node_uuid": edge["source_node_uuid"],
                        })
                        related_node_uuids.add(edge["source_node_uuid"])

                entity.related_edges = related_edges
                entity.related_nodes = [
                    {
                        "uuid": node_map[uid]["uuid"],
                        "name": node_map[uid]["name"],
                        "labels": node_map[uid]["labels"],
                        "summary": node_map[uid].get("summary", ""),
                    }
                    for uid in related_node_uuids
                    if uid in node_map
                ]

            filtered_entities.append(entity)

        logger.info(
            f"Filter complete: {total_count} total nodes, "
            f"{len(filtered_entities)} entities kept, "
            f"types: {entity_types_found}"
        )

        return FilteredEntities(
            entities=filtered_entities,
            entity_types=entity_types_found,
            total_count=total_count,
            filtered_count=len(filtered_entities),
        )

    # ------------------------------------------------------------------
    # Point reads
    # ------------------------------------------------------------------

    def get_entity_with_context(
        self,
        graph_id: str,
        entity_uuid: str,
    ) -> Optional[EntityNode]:
        """Fetch a single entity with its edges and neighbouring nodes."""
        try:
            node = self.backend.get_node(entity_uuid)
            if node is None:
                return None

            raw_edges = self.get_node_edges(entity_uuid)
            all_nodes = self.get_all_nodes(graph_id)
            node_map = {n["uuid"]: n for n in all_nodes}

            related_edges: List[Dict[str, Any]] = []
            related_node_uuids: Set[str] = set()

            for edge in raw_edges:
                if edge["source_node_uuid"] == entity_uuid:
                    related_edges.append({
                        "direction": "outgoing",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "target_node_uuid": edge["target_node_uuid"],
                    })
                    related_node_uuids.add(edge["target_node_uuid"])
                else:
                    related_edges.append({
                        "direction": "incoming",
                        "edge_name": edge["name"],
                        "fact": edge["fact"],
                        "source_node_uuid": edge["source_node_uuid"],
                    })
                    related_node_uuids.add(edge["source_node_uuid"])

            related_nodes = [
                {
                    "uuid": node_map[uid]["uuid"],
                    "name": node_map[uid]["name"],
                    "labels": node_map[uid]["labels"],
                    "summary": node_map[uid].get("summary", ""),
                }
                for uid in related_node_uuids
                if uid in node_map
            ]

            return EntityNode(
                uuid=node.uuid,
                name=node.name,
                labels=node.labels,
                summary=node.summary,
                attributes=node.attributes,
                related_edges=related_edges,
                related_nodes=related_nodes,
            )

        except Exception as exc:
            logger.error(f"Failed to fetch entity {entity_uuid}: {exc}")
            return None

    def get_entities_by_type(
        self,
        graph_id: str,
        entity_type: str,
        enrich_with_edges: bool = True,
    ) -> List[EntityNode]:
        """Return all entities of a specific type."""
        result = self.filter_defined_entities(
            graph_id=graph_id,
            defined_entity_types=[entity_type],
            enrich_with_edges=enrich_with_edges,
        )
        return result.entities
