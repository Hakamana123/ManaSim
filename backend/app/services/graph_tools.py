"""
Graph retrieval tools for the Report Agent.

Provides semantic search, entity lookup, and higher-level analysis
operations over the configured memory backend.

Replaces the former zep_tools.py.
"""

from __future__ import annotations

import time
import json
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

from .memory import get_memory_backend
from .memory.base import EdgeData, MemoryBackend, NodeData
from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient

logger = get_logger('manasim.graph_tools')


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """Search result returned by quick_search / search_graph."""

    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count,
        }

    def to_text(self) -> str:
        parts = [f"Search query: {self.query}", f"Found {self.total_count} relevant results"]
        if self.facts:
            parts.append("\n### Relevant facts:")
            for i, fact in enumerate(self.facts, 1):
                parts.append(f"{i}. {fact}")
        return "\n".join(parts)


@dataclass
class NodeInfo:
    """Node information used internally by graph tools."""

    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]

    @classmethod
    def from_node_data(cls, nd: NodeData) -> "NodeInfo":
        return cls(
            uuid=nd.uuid,
            name=nd.name,
            labels=nd.labels,
            summary=nd.summary,
            attributes=nd.attributes,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes,
        }

    def to_text(self) -> str:
        entity_type = next(
            (lbl for lbl in self.labels if lbl not in ("Entity", "Node")),
            "unknown type",
        )
        return f"Entity: {self.name} (type: {entity_type})\nSummary: {self.summary}"


@dataclass
class EdgeInfo:
    """Edge information used internally by graph tools."""

    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None

    @classmethod
    def from_edge_data(cls, ed: EdgeData) -> "EdgeInfo":
        return cls(
            uuid=ed.uuid,
            name=ed.name,
            fact=ed.fact,
            source_node_uuid=ed.source_node_uuid,
            target_node_uuid=ed.target_node_uuid,
            created_at=ed.created_at,
            valid_at=ed.valid_at,
            invalid_at=ed.invalid_at,
            expired_at=ed.expired_at,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at,
        }

    def to_text(self, include_temporal: bool = False) -> str:
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        text = f"Relation: {source} --[{self.name}]--> {target}\nFact: {self.fact}"
        if include_temporal:
            valid_at = self.valid_at or "unknown"
            invalid_at = self.invalid_at or "present"
            text += f"\nValidity: {valid_at} - {invalid_at}"
            if self.expired_at:
                text += f" (expired: {self.expired_at})"
        return text

    @property
    def is_expired(self) -> bool:
        return self.expired_at is not None

    @property
    def is_invalid(self) -> bool:
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """Deep-insight retrieval result (InsightForge tool)."""

    query: str
    simulation_requirement: str
    sub_queries: List[str]
    semantic_facts: List[str] = field(default_factory=list)
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)
    relationship_chains: List[str] = field(default_factory=list)
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships,
        }

    def to_text(self) -> str:
        parts = [
            "## Deep Insight Analysis",
            f"Query: {self.query}",
            f"Simulation context: {self.simulation_requirement}",
            f"\n### Statistics",
            f"- Relevant facts: {self.total_facts}",
            f"- Entities involved: {self.total_entities}",
            f"- Relationship chains: {self.total_relationships}",
        ]
        if self.sub_queries:
            parts.append("\n### Sub-questions analysed")
            for i, sq in enumerate(self.sub_queries, 1):
                parts.append(f"{i}. {sq}")
        if self.semantic_facts:
            parts.append("\n### Key facts (cite these in the report)")
            for i, fact in enumerate(self.semantic_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.entity_insights:
            parts.append("\n### Core entities")
            for entity in self.entity_insights:
                parts.append(
                    f"- **{entity.get('name', 'unknown')}** ({entity.get('type', 'entity')})"
                )
                if entity.get("summary"):
                    parts.append(f'  Summary: "{entity.get("summary")}"')
        if self.relationship_chains:
            parts.append("\n### Relationship chains")
            for chain in self.relationship_chains:
                parts.append(f"- {chain}")
        return "\n".join(parts)


@dataclass
class PanoramaResult:
    """Breadth-first search result (PanoramaSearch tool)."""

    query: str
    all_nodes: List[NodeInfo] = field(default_factory=list)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    active_facts: List[str] = field(default_factory=list)
    historical_facts: List[str] = field(default_factory=list)
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count,
        }

    def to_text(self) -> str:
        parts = [
            "## Panorama Search Results",
            f"Query: {self.query}",
            f"\n### Statistics",
            f"- Total nodes: {self.total_nodes}",
            f"- Total edges: {self.total_edges}",
            f"- Current active facts: {self.active_count}",
            f"- Historical/expired facts: {self.historical_count}",
        ]
        if self.active_facts:
            parts.append("\n### Active facts (simulation output)")
            for i, fact in enumerate(self.active_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.historical_facts:
            parts.append("\n### Historical/expired facts (evolution record)")
            for i, fact in enumerate(self.historical_facts, 1):
                parts.append(f'{i}. "{fact}"')
        if self.all_nodes:
            parts.append("\n### Entities involved")
            for node in self.all_nodes:
                entity_type = next(
                    (lbl for lbl in node.labels if lbl not in ("Entity", "Node")),
                    "entity",
                )
                parts.append(f"- **{node.name}** ({entity_type})")
        return "\n".join(parts)


@dataclass
class AgentInterview:
    """Single agent interview record."""

    agent_name: str
    agent_role: str
    agent_bio: str
    question: str
    response: str
    key_quotes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes,
        }

    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        text += f"_Bio: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Key quotes:**\n"
            for quote in self.key_quotes:
                # Clean various quote characters
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Skip content that contains question numbers
                skip = any(f'question{d}' in clean_quote.lower() for d in '123456789')
                if skip:
                    continue
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """Multi-agent interview result."""

    interview_topic: str
    interview_questions: List[str]
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    interviews: List[AgentInterview] = field(default_factory=list)
    selection_reasoning: str = ""
    summary: str = ""
    total_agents: int = 0
    interviewed_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count,
        }

    def to_text(self) -> str:
        parts = [
            "## Deep Interview Report",
            f"**Topic:** {self.interview_topic}",
            f"**Interviewees:** {self.interviewed_count} / {self.total_agents} simulation agents",
            "\n### Agent selection rationale",
            self.selection_reasoning or "(automatic selection)",
            "\n---",
            "\n### Interview transcripts",
        ]
        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                parts.append(interview.to_text())
                parts.append("\n---")
        else:
            parts.append("(no interview records)\n\n---")
        parts.append("\n### Summary and key insights")
        parts.append(self.summary or "(no summary)")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main service class
# ---------------------------------------------------------------------------

class GraphToolsService:
    """
    Graph retrieval tools for the Report Agent.

    Core retrieval tools:
      1. insight_forge       — deep insight retrieval: auto-generates sub-questions
      2. panorama_search     — breadth-first search including expired facts
      3. quick_search        — fast single-query search
      4. interview_agents    — interview live simulation agents via the API

    Base tools:
      - search_graph         — semantic search (edges or nodes)
      - get_all_nodes        — fetch all nodes
      - get_all_edges        — fetch all edges with temporal metadata
      - get_node_detail      — single-node lookup
      - get_node_edges       — edges connected to a node
      - get_entities_by_type — filter nodes by label
      - get_entity_summary   — entity relationship summary
      - get_graph_statistics — overall graph statistics
      - get_simulation_context — context relevant to a simulation requirement
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2.0

    def __init__(
        self,
        backend: Optional[MemoryBackend] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.backend = backend or get_memory_backend()
        self._llm_client = llm_client

    @property
    def llm(self) -> LLMClient:
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_graph(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """
        Semantic / hybrid search over graph edges or nodes.

        Tries the backend's search method first; falls back to local
        keyword matching if the backend raises an exception.

        Args:
            graph_id: Graph to search.
            query: Natural-language search query.
            limit: Maximum results to return.
            scope: "edges" or "nodes".

        Returns:
            SearchResult
        """
        logger.info(f"Graph search: graph_id={graph_id}, query={query[:50]}...")

        try:
            if scope == "nodes":
                results = self.backend.search_nodes(graph_id, query, limit)
            else:
                results = self.backend.search_edges(graph_id, query, limit)

            facts = results.facts
            edges = [
                {
                    "uuid": e.uuid,
                    "name": e.name,
                    "fact": e.fact,
                    "source_node_uuid": e.source_node_uuid,
                    "target_node_uuid": e.target_node_uuid,
                }
                for e in results.edges
            ]
            nodes = [
                {
                    "uuid": n.uuid,
                    "name": n.name,
                    "labels": n.labels,
                    "summary": n.summary,
                }
                for n in results.nodes
            ]

            logger.info(f"Search complete: found {len(facts)} facts")
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts),
            )

        except NotImplementedError:
            logger.warning("Backend search not implemented, falling back to local search")
            return self._local_search(graph_id, query, limit, scope)
        except Exception as exc:
            logger.warning(f"Backend search failed, falling back to local search: {exc}")
            return self._local_search(graph_id, query, limit, scope)

    def _local_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10,
        scope: str = "edges",
    ) -> SearchResult:
        """Local keyword-match fallback when the backend search is unavailable."""
        logger.info(f"Local search: query={query[:30]}...")

        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(',', ' ').replace('，', ' ').split()
            if len(w.strip()) > 1
        ]

        def match_score(text: str) -> int:
            if not text:
                return 0
            text_lower = text.lower()
            if query_lower in text_lower:
                return 100
            return sum(10 for kw in keywords if kw in text_lower)

        facts: List[str] = []
        edges_result: List[Dict[str, Any]] = []
        nodes_result: List[Dict[str, Any]] = []

        try:
            if scope in ("edges", "both"):
                all_edges = self.get_all_edges(graph_id)
                scored = sorted(
                    [(match_score(e.fact) + match_score(e.name), e) for e in all_edges],
                    key=lambda x: x[0],
                    reverse=True,
                )
                for score, edge in scored[:limit]:
                    if score > 0:
                        if edge.fact:
                            facts.append(edge.fact)
                        edges_result.append({
                            "uuid": edge.uuid,
                            "name": edge.name,
                            "fact": edge.fact,
                            "source_node_uuid": edge.source_node_uuid,
                            "target_node_uuid": edge.target_node_uuid,
                        })

            if scope in ("nodes", "both"):
                all_nodes = self.get_all_nodes(graph_id)
                scored = sorted(
                    [(match_score(n.name) + match_score(n.summary), n) for n in all_nodes],
                    key=lambda x: x[0],
                    reverse=True,
                )
                for score, node in scored[:limit]:
                    if score > 0:
                        nodes_result.append({
                            "uuid": node.uuid,
                            "name": node.name,
                            "labels": node.labels,
                            "summary": node.summary,
                        })
                        if node.summary:
                            facts.append(f"[{node.name}]: {node.summary}")

            logger.info(f"Local search complete: found {len(facts)} facts")

        except Exception as exc:
            logger.error(f"Local search failed: {exc}")

        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts),
        )

    # ------------------------------------------------------------------
    # Bulk reads
    # ------------------------------------------------------------------

    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """Fetch all nodes in the graph."""
        logger.info(f"Fetching all nodes for graph {graph_id}...")
        nodes = self.backend.fetch_all_nodes(graph_id)
        result = [NodeInfo.from_node_data(n) for n in nodes]
        logger.info(f"Fetched {len(result)} nodes")
        return result

    def get_all_edges(
        self, graph_id: str, include_temporal: bool = True
    ) -> List[EdgeInfo]:
        """Fetch all edges in the graph, including temporal metadata."""
        logger.info(f"Fetching all edges for graph {graph_id}...")
        edges = self.backend.fetch_all_edges(graph_id)
        result = [EdgeInfo.from_edge_data(e) for e in edges]
        logger.info(f"Fetched {len(result)} edges")
        return result

    # ------------------------------------------------------------------
    # Point reads
    # ------------------------------------------------------------------

    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """Fetch a single node by UUID."""
        logger.info(f"Fetching node detail: {node_uuid[:8]}...")
        try:
            node = self.backend.get_node(node_uuid)
            if node is None:
                return None
            return NodeInfo.from_node_data(node)
        except Exception as exc:
            logger.error(f"Failed to fetch node detail: {exc}")
            return None

    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """Return all edges connected to a given node."""
        logger.info(f"Fetching edges for node {node_uuid[:8]}...")
        try:
            edges = self.backend.get_node_edges(node_uuid)
            return [EdgeInfo.from_edge_data(e) for e in edges]
        except Exception as exc:
            logger.warning(f"Failed to fetch node edges: {exc}")
            return []

    # ------------------------------------------------------------------
    # Entity helpers
    # ------------------------------------------------------------------

    def get_entities_by_type(self, graph_id: str, entity_type: str) -> List[NodeInfo]:
        """Return all nodes whose labels include entity_type."""
        logger.info(f"Fetching entities of type '{entity_type}'...")
        all_nodes = self.get_all_nodes(graph_id)
        filtered = [n for n in all_nodes if entity_type in n.labels]
        logger.info(f"Found {len(filtered)} entities of type '{entity_type}'")
        return filtered

    def get_entity_summary(self, graph_id: str, entity_name: str) -> Dict[str, Any]:
        """Return a relationship summary for a named entity."""
        logger.info(f"Building entity summary for '{entity_name}'...")
        search_result = self.search_graph(graph_id=graph_id, query=entity_name, limit=20)
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = next(
            (n for n in all_nodes if n.name.lower() == entity_name.lower()), None
        )
        related_edges: List[EdgeInfo] = []
        if entity_node:
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges),
        }

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """Return overall statistics for a graph."""
        logger.info(f"Computing statistics for graph {graph_id}...")
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        entity_types: Dict[str, int] = {}
        for node in nodes:
            for label in node.labels:
                if label not in ("Entity", "Node"):
                    entity_types[label] = entity_types.get(label, 0) + 1
        relation_types: Dict[str, int] = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def get_simulation_context(
        self, graph_id: str, simulation_requirement: str, limit: int = 30
    ) -> Dict[str, Any]:
        """Return context relevant to a simulation requirement."""
        logger.info(f"Building simulation context: {simulation_requirement[:50]}...")
        search_result = self.search_graph(
            graph_id=graph_id, query=simulation_requirement, limit=limit
        )
        stats = self.get_graph_statistics(graph_id)
        all_nodes = self.get_all_nodes(graph_id)
        entities = [
            {
                "name": n.name,
                "type": next(
                    (lbl for lbl in n.labels if lbl not in ("Entity", "Node")), "entity"
                ),
                "summary": n.summary,
            }
            for n in all_nodes
            if any(lbl not in ("Entity", "Node") for lbl in n.labels)
        ]
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],
            "total_entities": len(entities),
        }

    # ------------------------------------------------------------------
    # Core retrieval tools (used by Report Agent)
    # ------------------------------------------------------------------

    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5,
    ) -> InsightForgeResult:
        """
        Deep insight retrieval.

        1. Decompose the query into sub-questions using the LLM.
        2. Run semantic search for each sub-question.
        3. Collect related entities and build relationship chains.
        4. Return a consolidated InsightForgeResult.
        """
        logger.info(f"InsightForge: {query[:50]}...")

        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[],
        )

        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries,
        )
        result.sub_queries = sub_queries
        logger.info(f"Generated {len(sub_queries)} sub-questions")

        all_facts: List[str] = []
        all_edges: List[Dict[str, Any]] = []
        seen_facts: set = set()

        for sub_query in sub_queries:
            sr = self.search_graph(graph_id=graph_id, query=sub_query, limit=15, scope="edges")
            for fact in sr.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            all_edges.extend(sr.edges)

        main_sr = self.search_graph(graph_id=graph_id, query=query, limit=20, scope="edges")
        for fact in main_sr.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)

        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)

        # Collect entity UUIDs from matched edges
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                src = edge_data.get('source_node_uuid', '')
                tgt = edge_data.get('target_node_uuid', '')
                if src:
                    entity_uuids.add(src)
                if tgt:
                    entity_uuids.add(tgt)

        entity_insights: List[Dict[str, Any]] = []
        node_map: Dict[str, NodeInfo] = {}

        for uuid in entity_uuids:
            if not uuid:
                continue
            try:
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next(
                        (lbl for lbl in node.labels if lbl not in ("Entity", "Node")),
                        "entity",
                    )
                    related_facts = [f for f in all_facts if node.name.lower() in f.lower()]
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts,
                    })
            except Exception as exc:
                logger.debug(f"Failed to fetch node {uuid}: {exc}")
                continue

        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)

        relationship_chains: List[str] = []
        _empty_node = NodeInfo('', '', [], '', {})
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                src_uuid = edge_data.get('source_node_uuid', '')
                tgt_uuid = edge_data.get('target_node_uuid', '')
                rel_name = edge_data.get('name', '')
                src_name = node_map.get(src_uuid, _empty_node).name or src_uuid[:8]
                tgt_name = node_map.get(tgt_uuid, _empty_node).name or tgt_uuid[:8]
                chain = f"{src_name} --[{rel_name}]--> {tgt_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)

        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)

        logger.info(
            f"InsightForge complete: {result.total_facts} facts, "
            f"{result.total_entities} entities, {result.total_relationships} chains"
        )
        return result

    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5,
    ) -> List[str]:
        """Use the LLM to decompose a query into sub-questions."""
        system_prompt = (
            "You are an expert query analyst. Decompose the given question into "
            "independent sub-questions that can each be answered by searching the "
            "simulation graph separately. Cover different dimensions: who, what, "
            "why, how, when, where. Return JSON: {\"sub_queries\": [\"...\", ...]}"
        )
        user_prompt = (
            f"Simulation context:\n{simulation_requirement}\n\n"
            + (f"Report context: {report_context[:500]}\n\n" if report_context else "")
            + f"Decompose this question into {max_queries} sub-questions:\n{query}\n\n"
            + "Return JSON."
        )
        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
            sub_queries = response.get("sub_queries", [])
            return [str(sq) for sq in sub_queries[:max_queries]]
        except Exception as exc:
            logger.warning(f"Sub-query generation failed: {exc}, using defaults")
            return [
                query,
                f"Main participants in: {query}",
                f"Causes and effects of: {query}",
                f"How did it unfold: {query}",
            ][:max_queries]

    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50,
    ) -> PanoramaResult:
        """
        Breadth-first search returning all relevant content including history.

        Fetches every node and edge, classifies facts as active or historical,
        and ranks them by relevance to the query.
        """
        logger.info(f"PanoramaSearch: {query[:50]}...")

        result = PanoramaResult(query=query)

        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)

        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)

        active_facts: List[str] = []
        historical_facts: List[str] = []
        _empty = NodeInfo('', '', [], '', {})

        for edge in all_edges:
            if not edge.fact:
                continue
            source_name = node_map.get(edge.source_node_uuid, _empty).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, _empty).name or edge.target_node_uuid[:8]

            if edge.is_expired or edge.is_invalid:
                valid_at = edge.valid_at or "unknown"
                invalid_at = edge.invalid_at or edge.expired_at or "unknown"
                historical_facts.append(f"[{valid_at} - {invalid_at}] {edge.fact}")
            else:
                active_facts.append(edge.fact)

        query_lower = query.lower()
        keywords = [
            w.strip()
            for w in query_lower.replace(',', ' ').replace('，', ' ').split()
            if len(w.strip()) > 1
        ]

        def relevance(fact: str) -> int:
            fl = fact.lower()
            score = 100 if query_lower in fl else 0
            score += sum(10 for kw in keywords if kw in fl)
            return score

        active_facts.sort(key=relevance, reverse=True)
        historical_facts.sort(key=relevance, reverse=True)

        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)

        logger.info(
            f"PanoramaSearch complete: {result.active_count} active, "
            f"{result.historical_count} historical"
        )
        return result

    def quick_search(self, graph_id: str, query: str, limit: int = 10) -> SearchResult:
        """Fast single-query search over edges."""
        logger.info(f"QuickSearch: {query[:50]}...")
        result = self.search_graph(graph_id=graph_id, query=query, limit=limit, scope="edges")
        logger.info(f"QuickSearch complete: {result.total_count} results")
        return result

    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: Optional[List[str]] = None,
    ) -> InterviewResult:
        """
        Interview live simulation agents via the OASIS interview API.

        1. Read agent profiles from disk to learn all available agents.
        2. Use the LLM to select the most relevant agents.
        3. Generate interview questions with the LLM.
        4. Call /api/simulation/interview/batch for both platforms.
        5. Consolidate results.

        NOTE: The simulation environment (OASIS) must be running.
        """
        logger.info(f"InterviewAgents: {interview_requirement[:50]}...")

        import os
        import requests
        from ..config import Config

        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or [],
        )

        # Locate the simulation data directory
        from .simulation_runner import SimulationRunner
        sim_dir = os.path.join(SimulationRunner.RUN_STATE_DIR, simulation_id)

        # Load Reddit profiles (preferred) or Twitter profiles
        profiles: List[Dict[str, Any]] = []
        reddit_path = os.path.join(
            os.path.dirname(__file__), '../../uploads/simulations', simulation_id,
            'reddit_profiles.json'
        )
        twitter_path = os.path.join(
            os.path.dirname(__file__), '../../uploads/simulations', simulation_id,
            'twitter_profiles.csv'
        )

        if os.path.exists(reddit_path):
            try:
                with open(reddit_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
            except Exception as exc:
                logger.warning(f"Failed to read reddit profiles: {exc}")

        result.total_agents = len(profiles)
        if not profiles:
            logger.warning("No agent profiles found — cannot conduct interviews")
            return result

        # Select agents using LLM
        selected = self._select_interview_agents(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents,
        )
        result.selected_agents = selected.get("agents", [])
        result.selection_reasoning = selected.get("reasoning", "")

        # Generate questions if none supplied
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=result.selected_agents,
            )

        # Conduct interviews via the simulation API
        interviews: List[AgentInterview] = []
        base_url = f"http://127.0.0.1:{Config.PORT if hasattr(Config, 'PORT') else 5000}"

        for agent_info in result.selected_agents:
            agent_id = agent_info.get("user_id")
            agent_name = agent_info.get("name", "Unknown")
            agent_role = agent_info.get("entity_type", "Agent")
            agent_bio = agent_info.get("bio", "")
            question = result.interview_questions[0] if result.interview_questions else interview_requirement

            try:
                resp = requests.post(
                    f"{base_url}/api/simulation/{simulation_id}/interview",
                    json={"agent_id": agent_id, "message": question},
                    timeout=60,
                )
                if resp.ok:
                    resp_data = resp.json()
                    response_text = resp_data.get("data", {}).get("response", "")
                    interviews.append(AgentInterview(
                        agent_name=agent_name,
                        agent_role=agent_role,
                        agent_bio=agent_bio,
                        question=question,
                        response=response_text,
                    ))
            except Exception as exc:
                logger.warning(f"Interview failed for agent {agent_name}: {exc}")

        result.interviews = interviews
        result.interviewed_count = len(interviews)

        if interviews:
            result.summary = self._summarise_interviews(
                interviews=interviews,
                interview_requirement=interview_requirement,
            )

        logger.info(f"InterviewAgents complete: {result.interviewed_count} interviews")
        return result

    def _select_interview_agents(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int,
    ) -> Dict[str, Any]:
        """Use the LLM to select the most relevant agents for an interview."""
        profiles_summary = [
            {
                "user_id": p.get("user_id"),
                "name": p.get("name"),
                "bio": (p.get("bio") or "")[:200],
                "entity_type": p.get("source_entity_type", ""),
            }
            for p in profiles[:50]
        ]
        system_prompt = (
            "You are a research assistant selecting interview subjects. "
            "Choose agents whose backgrounds are most relevant to the interview topic. "
            "Return JSON: {\"agents\": [{\"user_id\": ..., \"name\": ..., "
            "\"entity_type\": ..., \"bio\": ...}], \"reasoning\": \"...\"}"
        )
        user_prompt = (
            f"Simulation context: {simulation_requirement}\n\n"
            f"Interview topic: {interview_requirement}\n\n"
            f"Available agents (max {max_agents} to select):\n"
            + json.dumps(profiles_summary, ensure_ascii=False)
        )
        try:
            return self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
            )
        except Exception as exc:
            logger.warning(f"Agent selection failed: {exc}")
            return {"agents": profiles[:max_agents], "reasoning": "automatic selection"}

    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate interview questions with the LLM."""
        system_prompt = (
            "You are a journalist generating interview questions. "
            "Return JSON: {\"questions\": [\"...\", ...]}"
        )
        user_prompt = (
            f"Simulation context: {simulation_requirement}\n\n"
            f"Interview topic: {interview_requirement}\n\n"
            f"Interviewees: {[a.get('name') for a in selected_agents]}\n\n"
            "Generate 3 focused interview questions."
        )
        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )
            return response.get("questions", [interview_requirement])
        except Exception as exc:
            logger.warning(f"Question generation failed: {exc}")
            return [interview_requirement]

    def _summarise_interviews(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str,
    ) -> str:
        """Summarise all interview responses with the LLM."""
        transcripts = "\n\n".join(
            f"{iv.agent_name} ({iv.agent_role}): {iv.response[:500]}"
            for iv in interviews
        )
        system_prompt = (
            "Summarise the following interview responses into a concise paragraph "
            "highlighting the key viewpoints and any consensus or disagreement."
        )
        user_prompt = (
            f"Interview topic: {interview_requirement}\n\nTranscripts:\n{transcripts}"
        )
        try:
            return self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.5,
            )
        except Exception as exc:
            logger.warning(f"Interview summary failed: {exc}")
            return ""


# Backwards-compatible alias used by existing imports
ZepToolsService = GraphToolsService
