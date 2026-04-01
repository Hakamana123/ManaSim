"""
Graph builder service.

Constructs a knowledge graph from raw text by chunking the input,
submitting episodes to the configured memory backend, and polling
until each episode is processed.
"""

import os
import uuid
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from .memory import get_memory_backend
from .memory.base import MemoryBackend
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """图谱信息"""
    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


class GraphBuilderService:
    """
    Graph builder service.

    Constructs a knowledge graph from raw text using the configured
    memory backend.
    """

    def __init__(self, backend: Optional[MemoryBackend] = None, **_kwargs: Any) -> None:
        # Accept and ignore legacy keyword arguments (e.g. api_key) so existing
        # call sites don't need to be updated simultaneously.
        self.backend = backend or get_memory_backend()
        self.task_manager = TaskManager()
    
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 3
    ) -> str:
        """
        异步构建图谱
        
        Args:
            text: 输入文本
            ontology: 本体定义（来自接口1的输出）
            graph_name: 图谱名称
            chunk_size: 文本块大小
            chunk_overlap: 块重叠大小
            batch_size: 每批发送的块数量
            
        Returns:
            任务ID
        """
        # 创建任务
        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            }
        )
        
        # 在后台线程中执行构建
        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(task_id, text, ontology, graph_name, chunk_size, chunk_overlap, batch_size)
        )
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int
    ):
        """图谱构建工作线程"""
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message="开始构建图谱..."
            )
            
            # 1. Create graph
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=f"Graph created: {graph_id}",
            )

            # 2. Ontology is defined in the backend schema (e.g. SQL migrations).
            #    No runtime call needed here.
            self.task_manager.update_task(
                task_id,
                progress=15,
                message="Skipping ontology step (handled by backend schema)",
            )

            # 3. Chunk text
            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            total_chunks = len(chunks)
            self.task_manager.update_task(
                task_id,
                progress=20,
                message=f"Text split into {total_chunks} chunks",
            )
            
            # 4. Send chunks as episodes
            episode_uuids = self.add_text_batches(
                graph_id, chunks, batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.4),  # 20-60 %
                    message=msg,
                )
            )

            # 5. Wait for the backend to process the episodes
            self.task_manager.update_task(
                task_id,
                progress=60,
                message="Waiting for backend to process episodes...",
            )

            self._wait_for_episodes(
                episode_uuids,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 0.3),  # 60-90 %
                    message=msg,
                )
            )

            # 6. Fetch final graph info
            self.task_manager.update_task(
                task_id,
                progress=90,
                message="Fetching graph info...",
            )
            
            graph_info = self._get_graph_info(graph_id)
            
            # 完成
            self.task_manager.complete_task(task_id, {
                "graph_id": graph_id,
                "graph_info": graph_info.to_dict(),
                "chunks_processed": total_chunks,
            })
            
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)
    
    def create_graph(self, name: str) -> str:
        """Create a new knowledge graph and return its ID."""
        graph_id = f"mirofish_{uuid.uuid4().hex[:16]}"
        self.backend.create_graph(
            graph_id=graph_id,
            name=name,
            description="ManaSim Social Simulation Graph",
        )
        return graph_id

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """
        No-op: ontology is defined in the backend schema (e.g. SQL migrations).

        The ontology dict generated by OntologyGenerator is preserved here
        for reference but does not need to be applied via a runtime API call.
        For Supabase, apply docs/supabase-schema.sql once during project setup.
        """
        # ontology dict available for callers that need to inspect it
        _ = ontology  # intentionally unused
    
    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 3,
        progress_callback: Optional[Callable] = None
    ) -> List[str]:
        """分批添加文本到图谱，返回所有 episode 的 uuid 列表"""
        episode_uuids = []
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks + batch_size - 1) // batch_size

            if progress_callback:
                progress = (i + len(batch_chunks)) / total_chunks
                progress_callback(
                    f"Sending batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...",
                    progress,
                )

            try:
                for chunk in batch_chunks:
                    ep_uuid = self.backend.add_episode(graph_id, chunk)
                    if ep_uuid:
                        episode_uuids.append(ep_uuid)
                time.sleep(1)  # avoid overwhelming the backend

            except Exception as exc:
                if progress_callback:
                    progress_callback(f"Batch {batch_num} failed: {exc}", 0)
                raise

        return episode_uuids
    
    def _wait_for_episodes(
        self,
        episode_uuids: List[str],
        progress_callback: Optional[Callable] = None,
        timeout: int = 600
    ):
        """Poll episode processing until all are complete or timeout is reached."""
        if not episode_uuids:
            if progress_callback:
                progress_callback("No episodes to wait for", 1.0)
            return

        start_time = time.time()
        pending = set(episode_uuids)
        completed_count = 0
        total_episodes = len(episode_uuids)

        if progress_callback:
            progress_callback(f"Waiting for {total_episodes} episodes to process...", 0)

        while pending:
            if time.time() - start_time > timeout:
                if progress_callback:
                    progress_callback(
                        f"Timeout: {completed_count}/{total_episodes} episodes completed",
                        completed_count / total_episodes,
                    )
                break

            for ep_uuid in list(pending):
                try:
                    status = self.backend.get_episode_status(ep_uuid)
                    if status.is_complete or status.is_failed:
                        pending.discard(ep_uuid)
                        completed_count += 1
                except Exception:
                    pass  # ignore transient errors and retry

            elapsed = int(time.time() - start_time)
            if progress_callback:
                progress_callback(
                    f"Processing... {completed_count}/{total_episodes} done, "
                    f"{len(pending)} pending ({elapsed}s)",
                    completed_count / total_episodes if total_episodes > 0 else 0,
                )

            if pending:
                time.sleep(3)

        if progress_callback:
            progress_callback(f"Done: {completed_count}/{total_episodes}", 1.0)
    
    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        """Fetch node/edge counts and entity types for a graph."""
        nodes = self.backend.fetch_all_nodes(graph_id)
        edges = self.backend.fetch_all_edges(graph_id)

        # 统计实体类型
        entity_types = set()
        for node in nodes:
            if node.labels:
                for label in node.labels:
                    if label not in ["Entity", "Node"]:
                        entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types)
        )
    
    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """Return all nodes and edges for a graph as serialisable dicts."""
        nodes = self.backend.fetch_all_nodes(graph_id)
        edges = self.backend.fetch_all_edges(graph_id)

        node_map = {n.uuid: n.name or "" for n in nodes}

        nodes_data = [
            {
                "uuid": n.uuid,
                "name": n.name,
                "labels": n.labels or [],
                "summary": n.summary or "",
                "attributes": n.attributes or {},
                "created_at": None,  # not available in NodeData
            }
            for n in nodes
        ]

        edges_data = []
        for edge in edges:
            edges_data.append({
                "uuid": edge.uuid,
                "name": edge.name or "",
                "fact": edge.fact or "",
                "fact_type": edge.name or "",
                "source_node_uuid": edge.source_node_uuid,
                "target_node_uuid": edge.target_node_uuid,
                "source_node_name": node_map.get(edge.source_node_uuid, ""),
                "target_node_name": node_map.get(edge.target_node_uuid, ""),
                "attributes": edge.attributes or {},
                "created_at": edge.created_at,
                "valid_at": edge.valid_at,
                "invalid_at": edge.invalid_at,
                "expired_at": edge.expired_at,
                "episodes": [],
            })
        
        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }
    
    def delete_graph(self, graph_id: str) -> None:
        """Delete a graph and all its data."""
        self.backend.delete_graph(graph_id)

