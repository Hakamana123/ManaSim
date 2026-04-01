"""
Memory updater service.

Monitors simulation action logs and streams agent activities to the
configured memory backend as natural-language episodes.

Replaces the former zep_graph_memory_updater.py.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import datetime
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

from .memory import get_memory_backend
from .memory.base import MemoryBackend
from ..utils.logger import get_logger

logger = get_logger('manasim.memory_updater')


# ---------------------------------------------------------------------------
# Agent activity record
# ---------------------------------------------------------------------------

@dataclass
class AgentActivity:
    """A single agent action recorded during a simulation round."""

    platform: str
    agent_id: int
    agent_name: str
    action_type: str
    action_args: Dict[str, Any]
    round_num: int
    timestamp: str

    def to_episode_text(self) -> str:
        """Convert the activity to a natural-language sentence for ingestion."""
        action_descriptions = {
            "CREATE_POST":      self._describe_create_post,
            "LIKE_POST":        self._describe_like_post,
            "DISLIKE_POST":     self._describe_dislike_post,
            "REPOST":           self._describe_repost,
            "QUOTE_POST":       self._describe_quote_post,
            "FOLLOW":           self._describe_follow,
            "CREATE_COMMENT":   self._describe_create_comment,
            "LIKE_COMMENT":     self._describe_like_comment,
            "DISLIKE_COMMENT":  self._describe_dislike_comment,
            "SEARCH_POSTS":     self._describe_search,
            "SEARCH_USER":      self._describe_search_user,
            "MUTE":             self._describe_mute,
        }
        describe = action_descriptions.get(self.action_type, self._describe_generic)
        return f"{self.agent_name}: {describe()}"

    # ------------------------------------------------------------------
    # Action description helpers
    # ------------------------------------------------------------------

    def _describe_create_post(self) -> str:
        content = self.action_args.get("content", "")
        return f'published a post: "{content}"' if content else "published a post"

    def _describe_like_post(self) -> str:
        content = self.action_args.get("post_content", "")
        author = self.action_args.get("post_author_name", "")
        if content and author:
            return f'liked {author}\'s post: "{content}"'
        if content:
            return f'liked a post: "{content}"'
        if author:
            return f"liked a post by {author}"
        return "liked a post"

    def _describe_dislike_post(self) -> str:
        content = self.action_args.get("post_content", "")
        author = self.action_args.get("post_author_name", "")
        if content and author:
            return f'disliked {author}\'s post: "{content}"'
        if content:
            return f'disliked a post: "{content}"'
        if author:
            return f"disliked a post by {author}"
        return "disliked a post"

    def _describe_repost(self) -> str:
        content = self.action_args.get("original_content", "")
        author = self.action_args.get("original_author_name", "")
        if content and author:
            return f'reposted {author}\'s post: "{content}"'
        if content:
            return f'reposted: "{content}"'
        if author:
            return f"reposted from {author}"
        return "reposted a post"

    def _describe_quote_post(self) -> str:
        content = self.action_args.get("original_content", "")
        author = self.action_args.get("original_author_name", "")
        comment = self.action_args.get("quote_content", "") or self.action_args.get("content", "")
        if content and author:
            base = f'quoted {author}\'s post "{content}"'
        elif content:
            base = f'quoted a post "{content}"'
        elif author:
            base = f"quoted {author}'s post"
        else:
            base = "quoted a post"
        if comment:
            base += f' with comment: "{comment}"'
        return base

    def _describe_follow(self) -> str:
        target = self.action_args.get("target_user_name", "")
        return f"followed {target}" if target else "followed a user"

    def _describe_create_comment(self) -> str:
        content = self.action_args.get("content", "")
        post_content = self.action_args.get("post_content", "")
        post_author = self.action_args.get("post_author_name", "")
        if content:
            if post_content and post_author:
                return f'commented on {post_author}\'s post "{post_content}": "{content}"'
            if post_content:
                return f'commented on "{post_content}": "{content}"'
            if post_author:
                return f'commented on {post_author}\'s post: "{content}"'
            return f'commented: "{content}"'
        return "posted a comment"

    def _describe_like_comment(self) -> str:
        content = self.action_args.get("comment_content", "")
        author = self.action_args.get("comment_author_name", "")
        if content and author:
            return f'liked {author}\'s comment: "{content}"'
        if content:
            return f'liked a comment: "{content}"'
        if author:
            return f"liked a comment by {author}"
        return "liked a comment"

    def _describe_dislike_comment(self) -> str:
        content = self.action_args.get("comment_content", "")
        author = self.action_args.get("comment_author_name", "")
        if content and author:
            return f'disliked {author}\'s comment: "{content}"'
        if content:
            return f'disliked a comment: "{content}"'
        if author:
            return f"disliked a comment by {author}"
        return "disliked a comment"

    def _describe_search(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("keyword", "")
        return f'searched for "{query}"' if query else "performed a search"

    def _describe_search_user(self) -> str:
        query = self.action_args.get("query", "") or self.action_args.get("username", "")
        return f'searched for user "{query}"' if query else "searched for a user"

    def _describe_mute(self) -> str:
        target = self.action_args.get("target_user_name", "")
        return f"muted {target}" if target else "muted a user"

    def _describe_generic(self) -> str:
        return f"performed action {self.action_type}"


# ---------------------------------------------------------------------------
# Memory updater (single graph)
# ---------------------------------------------------------------------------

class MemoryUpdater:
    """
    Streams agent activities to the memory backend as text episodes.

    Accumulates activities per platform until BATCH_SIZE is reached,
    then submits a combined natural-language episode to the backend.
    """

    BATCH_SIZE = 5
    SEND_INTERVAL = 0.5   # seconds between batch submissions
    MAX_RETRIES = 3
    RETRY_DELAY = 2       # seconds

    PLATFORM_DISPLAY = {
        'twitter': 'World-1 (Twitter)',
        'reddit':  'World-2 (Reddit)',
    }

    def __init__(
        self,
        graph_id: str,
        backend: Optional[MemoryBackend] = None,
    ) -> None:
        self.graph_id = graph_id
        self.backend = backend or get_memory_backend()

        self._activity_queue: Queue = Queue()
        self._platform_buffers: Dict[str, List[AgentActivity]] = {
            'twitter': [],
            'reddit':  [],
        }
        self._buffer_lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Statistics
        self._total_activities = 0
        self._total_sent = 0
        self._total_items_sent = 0
        self._failed_count = 0
        self._skipped_count = 0

        logger.info(
            f"MemoryUpdater initialised: graph_id={graph_id}, "
            f"batch_size={self.BATCH_SIZE}"
        )

    def start(self) -> None:
        """Start the background worker thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name=f"MemoryUpdater-{self.graph_id[:8]}",
        )
        self._worker_thread.start()
        logger.info(f"MemoryUpdater started: graph_id={self.graph_id}")

    def stop(self) -> None:
        """Stop the worker and flush remaining activities."""
        self._running = False
        self._flush_remaining()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10)
        logger.info(
            f"MemoryUpdater stopped: graph_id={self.graph_id}, "
            f"activities={self._total_activities}, "
            f"batches_sent={self._total_sent}, "
            f"items_sent={self._total_items_sent}, "
            f"failed={self._failed_count}, "
            f"skipped={self._skipped_count}"
        )

    def add_activity(self, activity: AgentActivity) -> None:
        """Enqueue an agent activity (DO_NOTHING is silently dropped)."""
        if activity.action_type == "DO_NOTHING":
            self._skipped_count += 1
            return
        self._activity_queue.put(activity)
        self._total_activities += 1
        logger.debug(f"Queued activity: {activity.agent_name} — {activity.action_type}")

    def add_activity_from_dict(self, data: Dict[str, Any], platform: str) -> None:
        """Parse a dict from the actions.jsonl log and enqueue it."""
        if "event_type" in data:
            return
        activity = AgentActivity(
            platform=platform,
            agent_id=data.get("agent_id", 0),
            agent_name=data.get("agent_name", ""),
            action_type=data.get("action_type", ""),
            action_args=data.get("action_args", {}),
            round_num=data.get("round", 0),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
        )
        self.add_activity(activity)

    # ------------------------------------------------------------------
    # Internal worker
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        while self._running or not self._activity_queue.empty():
            try:
                try:
                    activity = self._activity_queue.get(timeout=1)
                    platform = activity.platform.lower()
                    with self._buffer_lock:
                        if platform not in self._platform_buffers:
                            self._platform_buffers[platform] = []
                        self._platform_buffers[platform].append(activity)
                        if len(self._platform_buffers[platform]) >= self.BATCH_SIZE:
                            batch = self._platform_buffers[platform][:self.BATCH_SIZE]
                            self._platform_buffers[platform] = \
                                self._platform_buffers[platform][self.BATCH_SIZE:]
                            self._send_batch(batch, platform)
                            time.sleep(self.SEND_INTERVAL)
                except Empty:
                    pass
            except Exception as exc:
                logger.error(f"Worker loop error: {exc}")
                time.sleep(1)

    def _send_batch(self, activities: List[AgentActivity], platform: str) -> None:
        """Submit a batch of activities as a single text episode."""
        if not activities:
            return
        combined_text = "\n".join(a.to_episode_text() for a in activities)
        for attempt in range(self.MAX_RETRIES):
            try:
                self.backend.add_episode(self.graph_id, combined_text)
                self._total_sent += 1
                self._total_items_sent += len(activities)
                display = self.PLATFORM_DISPLAY.get(platform, platform)
                logger.info(
                    f"Sent {len(activities)} {display} activities to graph {self.graph_id}"
                )
                return
            except NotImplementedError:
                # Backend not yet implemented — skip silently to avoid spam
                self._skipped_count += len(activities)
                return
            except Exception as exc:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(
                        f"Batch send attempt {attempt + 1}/{self.MAX_RETRIES} failed: {exc}"
                    )
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"Batch send failed after {self.MAX_RETRIES} attempts: {exc}")
                    self._failed_count += 1

    def _flush_remaining(self) -> None:
        """Drain the queue and send all remaining buffered activities."""
        while not self._activity_queue.empty():
            try:
                activity = self._activity_queue.get_nowait()
                platform = activity.platform.lower()
                with self._buffer_lock:
                    self._platform_buffers.setdefault(platform, []).append(activity)
            except Empty:
                break

        with self._buffer_lock:
            for platform, buffer in self._platform_buffers.items():
                if buffer:
                    display = self.PLATFORM_DISPLAY.get(platform, platform)
                    logger.info(f"Flushing {len(buffer)} remaining {display} activities")
                    self._send_batch(buffer, platform)
            for platform in self._platform_buffers:
                self._platform_buffers[platform] = []

    def get_stats(self) -> Dict[str, Any]:
        with self._buffer_lock:
            buffer_sizes = {p: len(b) for p, b in self._platform_buffers.items()}
        return {
            "graph_id": self.graph_id,
            "batch_size": self.BATCH_SIZE,
            "total_activities": self._total_activities,
            "batches_sent": self._total_sent,
            "items_sent": self._total_items_sent,
            "failed_count": self._failed_count,
            "skipped_count": self._skipped_count,
            "queue_size": self._activity_queue.qsize(),
            "buffer_sizes": buffer_sizes,
            "running": self._running,
        }


# ---------------------------------------------------------------------------
# Manager (multiple simulations)
# ---------------------------------------------------------------------------

class MemoryUpdaterManager:
    """
    Manages one MemoryUpdater per active simulation.
    """

    _updaters: Dict[str, MemoryUpdater] = {}
    _lock = threading.Lock()
    _stop_all_done = False

    @classmethod
    def create_updater(cls, simulation_id: str, graph_id: str) -> MemoryUpdater:
        """Create and start an updater for a simulation."""
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
            updater = MemoryUpdater(graph_id)
            updater.start()
            cls._updaters[simulation_id] = updater
            logger.info(
                f"Memory updater created: simulation_id={simulation_id}, "
                f"graph_id={graph_id}"
            )
            return updater

    @classmethod
    def get_updater(cls, simulation_id: str) -> Optional[MemoryUpdater]:
        return cls._updaters.get(simulation_id)

    @classmethod
    def stop_updater(cls, simulation_id: str) -> None:
        with cls._lock:
            if simulation_id in cls._updaters:
                cls._updaters[simulation_id].stop()
                del cls._updaters[simulation_id]
                logger.info(f"Memory updater stopped: simulation_id={simulation_id}")

    @classmethod
    def stop_all(cls) -> None:
        if cls._stop_all_done:
            return
        cls._stop_all_done = True
        with cls._lock:
            for simulation_id, updater in list(cls._updaters.items()):
                try:
                    updater.stop()
                except Exception as exc:
                    logger.error(
                        f"Failed to stop updater: simulation_id={simulation_id}, error={exc}"
                    )
            cls._updaters.clear()
            logger.info("All memory updaters stopped")

    @classmethod
    def get_all_stats(cls) -> Dict[str, Dict[str, Any]]:
        return {sid: u.get_stats() for sid, u in cls._updaters.items()}


# Backwards-compatible aliases used by existing imports
ZepGraphMemoryUpdater = MemoryUpdater
ZepGraphMemoryManager = MemoryUpdaterManager
