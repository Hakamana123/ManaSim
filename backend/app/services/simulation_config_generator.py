"""
Simulation configuration generator.

Uses an LLM to produce detailed simulation parameters from a requirement
description, document text, and entity list — no manual tuning needed.

Multi-step generation avoids overly long single-call outputs:
  1. Time configuration
  2. Event configuration
  3. Agent configurations (in batches of AGENTS_PER_BATCH)
  4. Environment configuration
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from .memory.base import EntityNode

logger = get_logger('mirofish.simulation_config')


# ---------------------------------------------------------------------------
# Environment type constants
# ---------------------------------------------------------------------------

ENVIRONMENT_CLASSROOM = "classroom"
ENVIRONMENT_ORGANISATION = "organisation"
# Legacy social-media environments (kept for backwards compatibility)
ENVIRONMENT_TWITTER = "twitter"
ENVIRONMENT_REDDIT = "reddit"

# Maps each environment to the underlying OASIS platform used by the runner.
# classroom   → Reddit OASIS platform  (threaded discussion suits classroom)
# organisation → Twitter OASIS platform (broadcast/response suits org comms)
OASIS_PLATFORM_MAP: Dict[str, str] = {
    ENVIRONMENT_CLASSROOM:    "reddit",
    ENVIRONMENT_ORGANISATION: "twitter",
    ENVIRONMENT_TWITTER:      "twitter",
    ENVIRONMENT_REDDIT:       "reddit",
}


# ---------------------------------------------------------------------------
# Per-environment activity schedule defaults
# ---------------------------------------------------------------------------

# Classroom: concentrated during school hours, essentially silent otherwise.
CLASSROOM_SCHEDULE: Dict[str, Any] = {
    "total_simulation_hours": 40,       # ~5 school days
    "minutes_per_round": 30,
    "peak_hours": [9, 10, 11, 13, 14],  # core teaching slots (excl. lunch)
    "off_peak_hours": list(range(0, 7)) + list(range(16, 24)),
    "morning_hours": [7, 8],
    "work_hours": [8, 9, 10, 11, 12, 13, 14, 15],
    "peak_activity_multiplier": 1.8,
    "off_peak_activity_multiplier": 0.02,
    "morning_activity_multiplier": 0.25,
    "work_activity_multiplier": 1.0,
    "reasoning": (
        "Classroom schedule: peak during teaching hours (09:00–15:00); "
        "near-zero outside the school day."
    ),
}

# Organisation: standard 9–17 office hours, lunch dip, minimal evening.
ORGANISATION_SCHEDULE: Dict[str, Any] = {
    "total_simulation_hours": 80,         # ~2 work weeks
    "minutes_per_round": 60,
    "peak_hours": [9, 10, 11, 14, 15, 16],  # morning and afternoon blocks
    "off_peak_hours": list(range(0, 8)) + list(range(20, 24)),
    "morning_hours": [8, 9],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17],
    "peak_activity_multiplier": 1.5,
    "off_peak_activity_multiplier": 0.02,
    "morning_activity_multiplier": 0.2,
    "work_activity_multiplier": 0.9,
    "reasoning": (
        "Organisation schedule: standard 09:00–17:00 office hours; "
        "reduced at lunch (12:00–13:00); minimal evening engagement."
    ),
}

# Legacy social-media schedule (Chinese timezone) — unchanged for compatibility.
SOCIAL_MEDIA_SCHEDULE: Dict[str, Any] = {
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": list(range(9, 19)),
    "peak_activity_multiplier": 1.5,
    "off_peak_activity_multiplier": 0.05,
    "morning_activity_multiplier": 0.4,
    "work_activity_multiplier": 0.7,
    "reasoning": "Social-media schedule: Chinese timezone, peak 19:00–22:00.",
}

ENVIRONMENT_SCHEDULES: Dict[str, Dict[str, Any]] = {
    ENVIRONMENT_CLASSROOM:    CLASSROOM_SCHEDULE,
    ENVIRONMENT_ORGANISATION: ORGANISATION_SCHEDULE,
    ENVIRONMENT_TWITTER:      SOCIAL_MEDIA_SCHEDULE,
    ENVIRONMENT_REDDIT:       SOCIAL_MEDIA_SCHEDULE,
}


# ---------------------------------------------------------------------------
# Per-environment LLM prompt framing
# ---------------------------------------------------------------------------

ENVIRONMENT_FRAMING: Dict[str, Dict[str, str]] = {
    ENVIRONMENT_CLASSROOM: {
        "domain": "educational classroom",
        "activity_noun": "participation",
        "post_noun": "message / question / response",
        "actor_desc": "students, teachers, and support staff",
        "time_note": (
            "Activity is concentrated during school hours (08:00–15:00). "
            "Outside these hours participants are essentially inactive."
        ),
        "agent_guidance": (
            "- Teachers/Professors: activity_level 0.4–0.7, active_hours 8–15, "
            "influence_weight 2.0–3.0, response_delay 10–30 min\n"
            "- Students: activity_level 0.5–0.9, active_hours 8–15, "
            "influence_weight 0.7–1.2, response_delay 1–10 min\n"
            "- Administrators: activity_level 0.1–0.3, active_hours 9–16, "
            "influence_weight 2.5–3.5, response_delay 30–120 min"
        ),
    },
    ENVIRONMENT_ORGANISATION: {
        "domain": "professional organisation",
        "activity_noun": "communication",
        "post_noun": "message / update / decision",
        "actor_desc": "executives, managers, team members, and stakeholders",
        "time_note": (
            "Activity follows standard business hours (09:00–17:00). "
            "Lunch (12:00–13:00) shows reduced activity. Minimal evening engagement."
        ),
        "agent_guidance": (
            "- Executives / Senior leaders: activity_level 0.2–0.4, active_hours 9–17, "
            "influence_weight 3.0–4.0, response_delay 30–120 min\n"
            "- Managers: activity_level 0.4–0.6, active_hours 9–17, "
            "influence_weight 2.0–3.0, response_delay 15–60 min\n"
            "- Team members: activity_level 0.6–0.8, active_hours 9–17, "
            "influence_weight 0.8–1.5, response_delay 5–30 min"
        ),
    },
    ENVIRONMENT_TWITTER: {
        "domain": "social media (Twitter)",
        "activity_noun": "posting",
        "post_noun": "tweet",
        "actor_desc": "social media users",
        "time_note": (
            "Peak hours 19:00–22:00 (Chinese timezone). "
            "Near-zero activity 00:00–05:00."
        ),
        "agent_guidance": (
            "- Official bodies: activity_level 0.1–0.3, active_hours 9–17, "
            "influence_weight 2.5–3.0\n"
            "- Media outlets: activity_level 0.4–0.6, active_hours 7–23, "
            "influence_weight 2.0–2.5\n"
            "- Individuals: activity_level 0.6–0.9, active_hours 18–23, "
            "influence_weight 0.8–1.2"
        ),
    },
    ENVIRONMENT_REDDIT: {
        "domain": "social media (Reddit)",
        "activity_noun": "posting / commenting",
        "post_noun": "post / comment",
        "actor_desc": "community members",
        "time_note": (
            "Peak hours 19:00–22:00 (Chinese timezone). "
            "Near-zero activity 00:00–05:00."
        ),
        "agent_guidance": (
            "- Official bodies: activity_level 0.1–0.3, active_hours 9–17, "
            "influence_weight 2.5–3.0\n"
            "- Media outlets: activity_level 0.4–0.6, active_hours 7–23, "
            "influence_weight 2.0–2.5\n"
            "- Individuals: activity_level 0.6–0.9, active_hours 18–23, "
            "influence_weight 0.8–1.2"
        ),
    },
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class AgentActivityConfig:
    """Activity configuration for a single agent."""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str

    # Overall activity level (0.0–1.0)
    activity_level: float = 0.5

    # Expected interactions per simulated hour
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0

    # Hours during which the agent is active (0–23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))

    # How quickly the agent reacts to events (simulated minutes)
    response_delay_min: int = 5
    response_delay_max: int = 60

    # Sentiment tendency (-1.0 negative → +1.0 positive)
    sentiment_bias: float = 0.0

    # Stance on the simulation topic
    stance: str = "neutral"  # supportive | opposing | neutral | observer

    # Probability weight that this agent's messages are seen by others
    influence_weight: float = 1.0


@dataclass
class TimeSimulationConfig:
    """Time and pacing configuration for the simulation."""

    # Total simulated duration in hours
    total_simulation_hours: int = 72

    # How many simulated minutes each simulation round represents
    minutes_per_round: int = 60

    # Range of agents activated per simulated hour
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20

    # Peak activity hours and multiplier
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5

    # Low-activity hours and multiplier
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05

    # Morning ramp-up
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4

    # Core working / school hours
    work_hours: List[int] = field(
        default_factory=lambda: list(range(9, 19))
    )
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Scenario events that seed or steer the simulation."""

    # Posts that are injected at the start of the simulation
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)

    # Events scheduled to fire at specific simulated times
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)

    # Key topic keywords used to focus agent attention
    hot_topics: List[str] = field(default_factory=list)

    # High-level narrative direction for the simulation
    narrative_direction: str = ""


@dataclass
class EnvironmentConfig:
    """
    Domain-environment configuration.

    Replaces the old PlatformConfig. The ``environment_type`` field identifies
    the domain (classroom / organisation / twitter / reddit); the ``platform``
    field records which OASIS platform backs this environment and is written
    to the saved JSON so that simulation scripts can read it directly.
    """
    environment_type: str   # classroom | organisation | twitter | reddit
    platform: str = ""      # oasis platform: "twitter" or "reddit"

    # Content-feed algorithm weights
    recency_weight: float = 0.4
    popularity_weight: float = 0.3
    relevance_weight: float = 0.3

    # Interaction threshold before content is pushed virally
    viral_threshold: int = 10

    # Degree to which similar-view agents cluster together
    echo_chamber_strength: float = 0.5


# Backwards-compatible alias so that any code still importing PlatformConfig works.
PlatformConfig = EnvironmentConfig


@dataclass
class SimulationParameters:
    """Complete simulation parameter configuration."""

    # Identity
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str

    # Domain environment
    environment_type: str = ENVIRONMENT_REDDIT

    # Time pacing
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)

    # Per-agent activity settings
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)

    # Scenario events
    event_config: EventConfig = field(default_factory=EventConfig)

    # Primary environment configuration (new)
    environment_config: Optional[EnvironmentConfig] = None

    # Legacy fields — populated for backwards compatibility with old scripts
    # that read twitter_config / reddit_config directly from the JSON.
    twitter_config: Optional[EnvironmentConfig] = None
    reddit_config: Optional[EnvironmentConfig] = None

    # LLM details stored for reference / reproducibility
    llm_model: str = ""
    llm_base_url: str = ""

    # Generation metadata
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "environment_type": self.environment_type,
            "time_config": asdict(self.time_config),
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "environment_config": (
                asdict(self.environment_config) if self.environment_config else None
            ),
            # Legacy keys kept so old simulation scripts can still read them
            "twitter_config": (
                asdict(self.twitter_config) if self.twitter_config else None
            ),
            "reddit_config": (
                asdict(self.reddit_config) if self.reddit_config else None
            ),
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class SimulationConfigGenerator:
    """
    Intelligent simulation configuration generator.

    Analyses the simulation requirement, document text, and entity list via an
    LLM and produces a fully-populated SimulationParameters object.

    Generation is split into multiple LLM calls to avoid token-limit failures:
      1. Time configuration
      2. Event configuration
      3. Agent configurations (AGENTS_PER_BATCH entities per call)
      4. Environment configuration (rule-based, no LLM call needed)
    """

    MAX_CONTEXT_LENGTH = 50000
    AGENTS_PER_BATCH = 15

    TIME_CONFIG_CONTEXT_LENGTH = 10000
    EVENT_CONFIG_CONTEXT_LENGTH = 8000
    ENTITY_SUMMARY_LENGTH = 300
    AGENT_SUMMARY_LENGTH = 300
    ENTITIES_PER_TYPE_DISPLAY = 20

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME

        if not self.api_key:
            raise ValueError("LLM_API_KEY is not configured.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        environment_type: str = ENVIRONMENT_REDDIT,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        # Legacy parameters — kept for backwards compatibility, deprecated.
        enable_twitter: Optional[bool] = None,
        enable_reddit: Optional[bool] = None,
    ) -> SimulationParameters:
        """
        Generate a complete SimulationParameters object.

        Args:
            simulation_id:          Unique ID for this simulation run.
            project_id:             ID of the parent project.
            graph_id:               Memory graph ID.
            simulation_requirement: Human-readable description of what to simulate.
            document_text:          Raw source document (used as LLM context).
            entities:               Filtered entity list from the memory graph.
            environment_type:       Domain environment — one of "classroom",
                                    "organisation", "twitter", "reddit".
            progress_callback:      Optional fn(current_step, total_steps, message).
            enable_twitter:         Deprecated. Use environment_type="twitter".
            enable_reddit:          Deprecated. Use environment_type="reddit".

        Returns:
            SimulationParameters
        """
        # Handle legacy enable_twitter / enable_reddit params
        if enable_twitter is not None or enable_reddit is not None:
            logger.warning(
                "enable_twitter / enable_reddit are deprecated. "
                "Use environment_type instead."
            )
            if enable_twitter and not enable_reddit:
                environment_type = ENVIRONMENT_TWITTER
            else:
                environment_type = ENVIRONMENT_REDDIT

        if environment_type not in ENVIRONMENT_SCHEDULES:
            logger.warning(
                "Unknown environment_type '%s'; falling back to '%s'.",
                environment_type, ENVIRONMENT_REDDIT,
            )
            environment_type = ENVIRONMENT_REDDIT

        logger.info(
            "Generating simulation config: simulation_id=%s, entities=%d, "
            "environment_type=%s",
            simulation_id, len(entities), environment_type,
        )

        num_batches = math.ceil(max(len(entities), 1) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # time + event + N agent batches + env
        current_step = 0

        def report(step: int, message: str) -> None:
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info("[%d/%d] %s", step, total_steps, message)

        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities,
        )

        framing = ENVIRONMENT_FRAMING.get(environment_type, ENVIRONMENT_FRAMING[ENVIRONMENT_REDDIT])
        reasoning_parts: List[str] = []

        # Step 1 — time config
        report(1, "Generating time configuration...")
        time_config_result = self._generate_time_config(
            context, len(entities), environment_type, framing
        )
        time_config = self._parse_time_config(
            time_config_result, len(entities), environment_type
        )
        reasoning_parts.append(
            f"time_config: {time_config_result.get('reasoning', 'ok')}"
        )

        # Step 2 — event config
        report(2, "Generating event configuration...")
        event_config_result = self._generate_event_config(
            context, simulation_requirement, entities, framing
        )
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(
            f"event_config: {event_config_result.get('reasoning', 'ok')}"
        )

        # Steps 3…N — agent configs in batches
        all_agent_configs: List[AgentActivityConfig] = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch = entities[start_idx:end_idx]

            report(
                3 + batch_idx,
                f"Generating agent configs ({start_idx + 1}–{end_idx} "
                f"of {len(entities)})...",
            )

            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement,
                environment_type=environment_type,
                framing=framing,
            )
            all_agent_configs.extend(batch_configs)

        reasoning_parts.append(
            f"agent_configs: generated {len(all_agent_configs)}"
        )

        # Assign poster agents to initial posts
        logger.info("Assigning poster agents to initial posts...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned = sum(
            1 for p in event_config.initial_posts if p.get("poster_agent_id") is not None
        )
        reasoning_parts.append(f"initial_posts_assigned: {assigned}")

        # Final step — environment config (rule-based)
        report(total_steps, "Building environment configuration...")
        env_cfg = self._build_environment_config(environment_type)

        # Legacy fields: populate twitter_config / reddit_config so that any
        # old code or old simulation scripts that read those keys still work.
        twitter_config = None
        reddit_config = None
        oasis_platform = OASIS_PLATFORM_MAP.get(environment_type, "reddit")
        if oasis_platform == "twitter":
            twitter_config = env_cfg
        else:
            reddit_config = env_cfg

        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            environment_type=environment_type,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            environment_config=env_cfg,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url or "",
            generation_reasoning=" | ".join(reasoning_parts),
        )

        logger.info(
            "Config generation complete: %d agent configs, environment=%s",
            len(params.agent_configs), environment_type,
        )
        return params

    # ------------------------------------------------------------------
    # Environment config (rule-based)
    # ------------------------------------------------------------------

    def _build_environment_config(self, environment_type: str) -> EnvironmentConfig:
        """Build EnvironmentConfig from environment_type without an LLM call."""
        oasis_platform = OASIS_PLATFORM_MAP.get(environment_type, "reddit")

        presets: Dict[str, Any] = {
            ENVIRONMENT_CLASSROOM: dict(
                recency_weight=0.5,
                popularity_weight=0.2,
                relevance_weight=0.3,
                viral_threshold=5,
                echo_chamber_strength=0.4,
            ),
            ENVIRONMENT_ORGANISATION: dict(
                recency_weight=0.4,
                popularity_weight=0.2,
                relevance_weight=0.4,
                viral_threshold=8,
                echo_chamber_strength=0.5,
            ),
            ENVIRONMENT_TWITTER: dict(
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5,
            ),
            ENVIRONMENT_REDDIT: dict(
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6,
            ),
        }

        p = presets.get(environment_type, presets[ENVIRONMENT_REDDIT])
        return EnvironmentConfig(
            environment_type=environment_type,
            platform=oasis_platform,
            **p,
        )

    # ------------------------------------------------------------------
    # Context building
    # ------------------------------------------------------------------

    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
    ) -> str:
        entity_summary = self._summarize_entities(entities)

        context_parts = [
            f"## Simulation requirement\n{simulation_requirement}",
            f"\n## Entities ({len(entities)} total)\n{entity_summary}",
        ]

        current_len = sum(len(p) for p in context_parts)
        remaining = self.MAX_CONTEXT_LENGTH - current_len - 500

        if remaining > 0 and document_text:
            doc = document_text[:remaining]
            if len(document_text) > remaining:
                doc += "\n...(document truncated)"
            context_parts.append(f"\n## Source document\n{doc}")

        return "\n".join(context_parts)

    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        lines: List[str] = []
        by_type: Dict[str, List[EntityNode]] = {}

        for e in entities:
            t = e.get_entity_type() or "Unknown"
            by_type.setdefault(t, []).append(e)

        for entity_type, group in by_type.items():
            lines.append(f"\n### {entity_type} ({len(group)})")
            for e in group[: self.ENTITIES_PER_TYPE_DISPLAY]:
                preview = e.summary[: self.ENTITY_SUMMARY_LENGTH]
                if len(e.summary) > self.ENTITY_SUMMARY_LENGTH:
                    preview += "..."
                lines.append(f"- {e.name}: {preview}")
            if len(group) > self.ENTITIES_PER_TYPE_DISPLAY:
                lines.append(
                    f"  ... and {len(group) - self.ENTITIES_PER_TYPE_DISPLAY} more"
                )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Time configuration
    # ------------------------------------------------------------------

    def _generate_time_config(
        self,
        context: str,
        num_entities: int,
        environment_type: str,
        framing: Dict[str, str],
    ) -> Dict[str, Any]:
        schedule = ENVIRONMENT_SCHEDULES.get(environment_type, SOCIAL_MEDIA_SCHEDULE)
        ctx = context[: self.TIME_CONFIG_CONTEXT_LENGTH]
        max_agents = max(1, int(num_entities * 0.9))

        prompt = f"""\
Based on the simulation context below, generate a time configuration for a
{framing['domain']} simulation.

{ctx}

## Timing note
{framing['time_note']}

## Guidelines
- Adjust total_simulation_hours to fit the scenario duration.
- Set peak_hours to the periods when {framing['actor_desc']} are most active.
- Keep off_peak_hours to periods of near-zero activity.
- agents_per_hour_min and agents_per_hour_max must be between 1 and {max_agents}.

## Suggested defaults for {environment_type}
- total_simulation_hours: {schedule['total_simulation_hours']}
- minutes_per_round: {schedule['minutes_per_round']}
- peak_hours: {schedule['peak_hours']}
- off_peak_hours: {schedule['off_peak_hours']}

Return ONLY a JSON object (no markdown):
{{
    "total_simulation_hours": <int>,
    "minutes_per_round": <int>,
    "agents_per_hour_min": <int 1–{max_agents}>,
    "agents_per_hour_max": <int 1–{max_agents}>,
    "peak_hours": [<int>, ...],
    "off_peak_hours": [<int>, ...],
    "morning_hours": [<int>, ...],
    "work_hours": [<int>, ...],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = (
            f"You are an expert simulation designer specialising in "
            f"{framing['domain']} environments. Return pure JSON only."
        )

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as exc:
            logger.warning("Time config LLM call failed: %s — using defaults.", exc)
            return self._get_default_time_config(num_entities, environment_type)

    def _get_default_time_config(
        self, num_entities: int, environment_type: str = ENVIRONMENT_REDDIT
    ) -> Dict[str, Any]:
        schedule = ENVIRONMENT_SCHEDULES.get(environment_type, SOCIAL_MEDIA_SCHEDULE)
        return {
            **schedule,
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
        }

    def _parse_time_config(
        self,
        result: Dict[str, Any],
        num_entities: int,
        environment_type: str = ENVIRONMENT_REDDIT,
    ) -> TimeSimulationConfig:
        schedule = ENVIRONMENT_SCHEDULES.get(environment_type, SOCIAL_MEDIA_SCHEDULE)

        min_agents = result.get("agents_per_hour_min", max(1, num_entities // 15))
        max_agents = result.get("agents_per_hour_max", max(5, num_entities // 5))

        if min_agents > num_entities:
            logger.warning(
                "agents_per_hour_min (%d) exceeds entity count (%d); clamping.",
                min_agents, num_entities,
            )
            min_agents = max(1, num_entities // 10)

        if max_agents > num_entities:
            logger.warning(
                "agents_per_hour_max (%d) exceeds entity count (%d); clamping.",
                max_agents, num_entities,
            )
            max_agents = max(min_agents + 1, num_entities // 2)

        if min_agents >= max_agents:
            min_agents = max(1, max_agents // 2)
            logger.warning(
                "agents_per_hour_min >= max; corrected to %d.", min_agents
            )

        return TimeSimulationConfig(
            total_simulation_hours=result.get(
                "total_simulation_hours", schedule["total_simulation_hours"]
            ),
            minutes_per_round=result.get(
                "minutes_per_round", schedule["minutes_per_round"]
            ),
            agents_per_hour_min=min_agents,
            agents_per_hour_max=max_agents,
            peak_hours=result.get("peak_hours", schedule["peak_hours"]),
            peak_activity_multiplier=schedule["peak_activity_multiplier"],
            off_peak_hours=result.get("off_peak_hours", schedule["off_peak_hours"]),
            off_peak_activity_multiplier=schedule["off_peak_activity_multiplier"],
            morning_hours=result.get("morning_hours", schedule["morning_hours"]),
            morning_activity_multiplier=schedule["morning_activity_multiplier"],
            work_hours=result.get("work_hours", schedule["work_hours"]),
            work_activity_multiplier=schedule["work_activity_multiplier"],
        )

    # ------------------------------------------------------------------
    # Event configuration
    # ------------------------------------------------------------------

    def _generate_event_config(
        self,
        context: str,
        simulation_requirement: str,
        entities: List[EntityNode],
        framing: Dict[str, str],
    ) -> Dict[str, Any]:
        # Build a type→example-names index for the LLM
        type_examples: Dict[str, List[str]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in type_examples:
                type_examples[t] = []
            if len(type_examples[t]) < 3:
                type_examples[t].append(e.name)

        type_info = "\n".join(
            f"- {t}: {', '.join(examples)}"
            for t, examples in type_examples.items()
        )

        ctx = context[: self.EVENT_CONFIG_CONTEXT_LENGTH]

        prompt = f"""\
Based on the simulation context below, generate an event configuration for a
{framing['domain']} simulation.

## Simulation requirement
{simulation_requirement}

{ctx}

## Available entity types and examples
{type_info}

## Task
Generate an event configuration that:
- Identifies key topics relevant to the {framing['domain']} scenario.
- Describes the expected direction of participant {framing['activity_noun']}.
- Designs 2–5 opening {framing['post_noun']}s that seed the simulation.
  Each opening message MUST include a "poster_type" chosen from the available
  entity types above so it can be assigned to the right agent.

Return ONLY a JSON object (no markdown):
{{
    "hot_topics": ["<topic>", ...],
    "narrative_direction": "<description of how the simulation will unfold>",
    "initial_posts": [
        {{"content": "<opening message>", "poster_type": "<entity type from list above>"}},
        ...
    ],
    "reasoning": "<brief explanation>"
}}"""

        system_prompt = (
            f"You are a simulation scenario designer specialising in "
            f"{framing['domain']} environments. Return pure JSON only. "
            "poster_type must exactly match one of the available entity types."
        )

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as exc:
            logger.warning("Event config LLM call failed: %s — using defaults.", exc)
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Default (LLM unavailable)",
            }

    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", ""),
        )

    # ------------------------------------------------------------------
    # Initial post → agent assignment
    # ------------------------------------------------------------------

    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig],
    ) -> EventConfig:
        """Match each initial post's poster_type to a concrete agent_id."""
        if not event_config.initial_posts:
            return event_config

        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            t = agent.entity_type.lower()
            agents_by_type.setdefault(t, []).append(agent)

        # Fuzzy-match aliases so LLM type variations still resolve
        type_aliases: Dict[str, List[str]] = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher", "faculty"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
            "manager": ["manager", "official", "person"],
            "executive": ["executive", "official", "person"],
        }

        used_indices: Dict[str, int] = {}
        updated_posts: List[Dict[str, Any]] = []

        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            matched_id: Optional[int] = None

            # Direct match
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # Alias match
                for alias_key, aliases in type_aliases.items():
                    if poster_type == alias_key or poster_type in aliases:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_id is not None:
                        break

            # Fallback: highest-influence agent
            if matched_id is None:
                logger.warning(
                    "No agent found for poster_type '%s'; "
                    "falling back to highest-influence agent.",
                    poster_type,
                )
                if agent_configs:
                    matched_id = max(
                        agent_configs, key=lambda a: a.influence_weight
                    ).agent_id
                else:
                    matched_id = 0

            updated_posts.append({
                "content": post.get("content", ""),
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_id,
            })
            logger.info(
                "Initial post assigned: poster_type='%s' → agent_id=%d",
                poster_type, matched_id,
            )

        event_config.initial_posts = updated_posts
        return event_config

    # ------------------------------------------------------------------
    # Agent configuration (batched)
    # ------------------------------------------------------------------

    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str,
        environment_type: str,
        framing: Dict[str, str],
    ) -> List[AgentActivityConfig]:
        schedule = ENVIRONMENT_SCHEDULES.get(environment_type, SOCIAL_MEDIA_SCHEDULE)

        entity_list = [
            {
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": (e.summary or "")[: self.AGENT_SUMMARY_LENGTH],
            }
            for i, e in enumerate(entities)
        ]

        prompt = f"""\
Generate activity configurations for the agents listed below in a
{framing['domain']} simulation.

## Simulation requirement
{simulation_requirement}

## Agents
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Activity guidance for this environment
{framing['agent_guidance']}

## Timing note
{framing['time_note']}
Active hours should fall within the domain's active window.
Peak hours for this environment: {schedule['peak_hours']}.

Return ONLY a JSON object (no markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <must match input>,
            "activity_level": <float 0.0–1.0>,
            "posts_per_hour": <float>,
            "comments_per_hour": <float>,
            "active_hours": [<int 0–23>, ...],
            "response_delay_min": <int minutes>,
            "response_delay_max": <int minutes>,
            "sentiment_bias": <float -1.0–1.0>,
            "stance": "<supportive|opposing|neutral|observer>",
            "influence_weight": <float>
        }},
        ...
    ]
}}"""

        system_prompt = (
            f"You are a behavioural analyst for {framing['domain']} simulations. "
            "Return pure JSON only."
        )

        llm_configs: Dict[int, Dict[str, Any]] = {}
        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {
                cfg["agent_id"]: cfg
                for cfg in result.get("agent_configs", [])
            }
        except Exception as exc:
            logger.warning(
                "Agent config batch LLM call failed: %s — using rule-based fallback.",
                exc,
            )

        configs: List[AgentActivityConfig] = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id) or self._rule_based_agent_config(
                entity, environment_type
            )

            configs.append(AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 17))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0),
            ))

        return configs

    def _rule_based_agent_config(
        self, entity: EntityNode, environment_type: str
    ) -> Dict[str, Any]:
        """Produce a sensible default config without an LLM call."""
        etype = (entity.get_entity_type() or "Unknown").lower()

        if environment_type == ENVIRONMENT_CLASSROOM:
            if etype in ("professor", "teacher", "faculty"):
                return dict(
                    activity_level=0.6, posts_per_hour=0.4, comments_per_hour=0.8,
                    active_hours=list(range(8, 16)), response_delay_min=5,
                    response_delay_max=20, sentiment_bias=0.0,
                    stance="neutral", influence_weight=2.5,
                )
            elif etype in ("university", "governmentagency", "organization"):
                return dict(
                    activity_level=0.2, posts_per_hour=0.1, comments_per_hour=0.1,
                    active_hours=list(range(9, 16)), response_delay_min=30,
                    response_delay_max=120, sentiment_bias=0.0,
                    stance="neutral", influence_weight=3.0,
                )
            else:  # student / default
                return dict(
                    activity_level=0.75, posts_per_hour=0.5, comments_per_hour=1.5,
                    active_hours=list(range(8, 16)), response_delay_min=1,
                    response_delay_max=10, sentiment_bias=0.0,
                    stance="neutral", influence_weight=0.9,
                )

        elif environment_type == ENVIRONMENT_ORGANISATION:
            if etype in ("executive", "director", "ceo", "official"):
                return dict(
                    activity_level=0.3, posts_per_hour=0.2, comments_per_hour=0.3,
                    active_hours=list(range(9, 18)), response_delay_min=30,
                    response_delay_max=120, sentiment_bias=0.0,
                    stance="neutral", influence_weight=3.5,
                )
            elif etype in ("manager", "professor", "expert"):
                return dict(
                    activity_level=0.5, posts_per_hour=0.4, comments_per_hour=0.6,
                    active_hours=list(range(9, 18)), response_delay_min=15,
                    response_delay_max=60, sentiment_bias=0.0,
                    stance="neutral", influence_weight=2.0,
                )
            elif etype in ("university", "governmentagency", "organization", "ngo"):
                return dict(
                    activity_level=0.2, posts_per_hour=0.1, comments_per_hour=0.1,
                    active_hours=list(range(9, 17)), response_delay_min=60,
                    response_delay_max=240, sentiment_bias=0.0,
                    stance="neutral", influence_weight=3.0,
                )
            else:  # team member / default
                return dict(
                    activity_level=0.65, posts_per_hour=0.5, comments_per_hour=1.0,
                    active_hours=list(range(9, 18)), response_delay_min=5,
                    response_delay_max=30, sentiment_bias=0.0,
                    stance="neutral", influence_weight=1.0,
                )

        else:
            # Legacy social-media fallback (original Chinese-timezone rules)
            if etype in ("university", "governmentagency", "ngo"):
                return dict(
                    activity_level=0.2, posts_per_hour=0.1, comments_per_hour=0.05,
                    active_hours=list(range(9, 18)), response_delay_min=60,
                    response_delay_max=240, sentiment_bias=0.0,
                    stance="neutral", influence_weight=3.0,
                )
            elif etype == "mediaoutlet":
                return dict(
                    activity_level=0.5, posts_per_hour=0.8, comments_per_hour=0.3,
                    active_hours=list(range(7, 24)), response_delay_min=5,
                    response_delay_max=30, sentiment_bias=0.0,
                    stance="observer", influence_weight=2.5,
                )
            elif etype in ("professor", "expert", "official"):
                return dict(
                    activity_level=0.4, posts_per_hour=0.3, comments_per_hour=0.5,
                    active_hours=list(range(8, 22)), response_delay_min=15,
                    response_delay_max=90, sentiment_bias=0.0,
                    stance="neutral", influence_weight=2.0,
                )
            elif etype == "student":
                return dict(
                    activity_level=0.8, posts_per_hour=0.6, comments_per_hour=1.5,
                    active_hours=[8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                    response_delay_min=1, response_delay_max=15,
                    sentiment_bias=0.0, stance="neutral", influence_weight=0.8,
                )
            elif etype == "alumni":
                return dict(
                    activity_level=0.6, posts_per_hour=0.4, comments_per_hour=0.8,
                    active_hours=[12, 13, 19, 20, 21, 22, 23],
                    response_delay_min=5, response_delay_max=30,
                    sentiment_bias=0.0, stance="neutral", influence_weight=1.0,
                )
            else:
                return dict(
                    activity_level=0.7, posts_per_hour=0.5, comments_per_hour=1.2,
                    active_hours=[9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],
                    response_delay_min=2, response_delay_max=20,
                    sentiment_bias=0.0, stance="neutral", influence_weight=1.0,
                )

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _call_llm_with_retry(
        self, prompt: str, system_prompt: str
    ) -> Dict[str, Any]:
        """Call the LLM up to 3 times with decreasing temperature."""
        import re
        import time

        max_attempts = 3
        last_error: Optional[Exception] = None

        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - attempt * 0.1,
                )

                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    logger.warning(
                        "LLM output truncated (attempt %d).", attempt + 1
                    )
                    content = self._fix_truncated_json(content)

                try:
                    return json.loads(content)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "JSON parse failed (attempt %d): %s",
                        attempt + 1, str(exc)[:80],
                    )
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    last_error = exc

            except Exception as exc:
                logger.warning(
                    "LLM call failed (attempt %d): %s", attempt + 1, str(exc)[:80]
                )
                last_error = exc
                time.sleep(2 * (attempt + 1))

        raise last_error or RuntimeError("LLM call failed after all retries.")

    def _fix_truncated_json(self, content: str) -> str:
        content = content.strip()
        open_braces = content.count("{") - content.count("}")
        open_brackets = content.count("[") - content.count("]")
        if content and content[-1] not in '",}]':
            content += '"'
        content += "]" * open_brackets
        content += "}" * open_braces
        return content

    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        import re

        content = self._fix_truncated_json(content)
        match = re.search(r"\{[\s\S]*\}", content)
        if not match:
            return None

        json_str = match.group()

        def _clean_string(m: re.Match) -> str:
            s = m.group(0)
            s = s.replace("\n", " ").replace("\r", " ")
            s = re.sub(r"\s+", " ", s)
            return s

        json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', _clean_string, json_str)

        try:
            return json.loads(json_str)
        except Exception:
            json_str = re.sub(r"[\x00-\x1f\x7f-\x9f]", " ", json_str)
            json_str = re.sub(r"\s+", " ", json_str)
            try:
                return json.loads(json_str)
            except Exception:
                return None
