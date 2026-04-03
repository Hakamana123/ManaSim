"""
Simulation manager.

Orchestrates the full preparation pipeline for a ManaSim simulation:
  1. Read and filter entities from the memory graph.
  2. Generate OASIS agent profiles — either from research pipeline output
     (ProfileBridge) or directly from graph entities (OasisProfileGenerator).
  3. LLM-generate simulation configuration parameters.
  4. Save all artefacts to the simulation directory.

Supports domain environments:
  - "classroom"    — uses Reddit OASIS platform (threaded discussion)
  - "organisation" — uses Twitter OASIS platform (broadcast / response)
  - "twitter"      — legacy social-media (Twitter OASIS)
  - "reddit"       — legacy social-media (Reddit OASIS)
"""

import os
import json
import traceback as tb
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.logger import get_logger
from .entity_reader import EntityReader as ZepEntityReader
from .memory.base import FilteredEntities
from .oasis_profile_generator import OasisProfileGenerator, OasisAgentProfile
from .simulation_config_generator import (
    SimulationConfigGenerator,
    SimulationParameters,
    ENVIRONMENT_CLASSROOM,
    ENVIRONMENT_ORGANISATION,
    ENVIRONMENT_TWITTER,
    ENVIRONMENT_REDDIT,
    OASIS_PLATFORM_MAP,
)

logger = get_logger('mirofish.simulation')


# ---------------------------------------------------------------------------
# Environment type helpers
# ---------------------------------------------------------------------------

# All recognised environment type strings
VALID_ENVIRONMENTS = {
    ENVIRONMENT_CLASSROOM,
    ENVIRONMENT_ORGANISATION,
    ENVIRONMENT_TWITTER,
    ENVIRONMENT_REDDIT,
}


def _derive_environment_from_legacy(
    enable_twitter: bool, enable_reddit: bool
) -> str:
    """Convert old boolean flags to an environment_type string."""
    if enable_twitter and not enable_reddit:
        return ENVIRONMENT_TWITTER
    return ENVIRONMENT_REDDIT


def _profile_file_for_environment(environment_type: str):
    """
    Return (filename, oasis_format) for profile serialisation.

    oasis_format is the string passed to OasisProfileGenerator.save_profiles()
    (and eventually to oasis_profile_generator.py step 7).

    Both new domain environments use JSON so that the new simulation scripts
    can load them via generate_reddit_agent_graph / generate_twitter_agent_graph
    with a JSON-aware wrapper (implemented in step 7).
    """
    mapping = {
        ENVIRONMENT_CLASSROOM:    ("classroom_profiles.json",    "classroom"),
        ENVIRONMENT_ORGANISATION: ("organisation_profiles.json", "organisation"),
        ENVIRONMENT_TWITTER:      ("twitter_profiles.csv",       "twitter"),
        ENVIRONMENT_REDDIT:       ("reddit_profiles.json",       "reddit"),
    }
    return mapping.get(environment_type, ("reddit_profiles.json", "reddit"))


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SimulationStatus(str, Enum):
    CREATED   = "created"
    PREPARING = "preparing"
    READY     = "ready"
    RUNNING   = "running"
    PAUSED    = "paused"
    STOPPED   = "stopped"
    COMPLETED = "completed"
    FAILED    = "failed"


class EnvironmentType(str, Enum):
    """Domain environment for the simulation."""
    CLASSROOM    = ENVIRONMENT_CLASSROOM
    ORGANISATION = ENVIRONMENT_ORGANISATION
    TWITTER      = ENVIRONMENT_TWITTER   # legacy
    REDDIT       = ENVIRONMENT_REDDIT    # legacy


# Backwards-compatible alias
PlatformType = EnvironmentType


# ---------------------------------------------------------------------------
# SimulationState
# ---------------------------------------------------------------------------

@dataclass
class SimulationState:
    """Persistent state for a single simulation run."""

    simulation_id: str
    project_id:    str
    graph_id:      str

    # Domain environment
    environment_type: str = ENVIRONMENT_REDDIT

    # Lifecycle status
    status: SimulationStatus = SimulationStatus.CREATED

    # Preparation summary
    entities_count: int = 0
    profiles_count: int = 0
    entity_types:   List[str] = field(default_factory=list)

    # Config generation
    config_generated:  bool = False
    config_reasoning:  str  = ""

    # Runtime progress
    current_round:      int = 0
    environment_status: str = "not_started"

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    # Error detail
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Full state dict (persisted to state.json and used internally)."""
        oasis_platform = OASIS_PLATFORM_MAP.get(self.environment_type, "reddit")
        return {
            "simulation_id":    self.simulation_id,
            "project_id":       self.project_id,
            "graph_id":         self.graph_id,
            "environment_type": self.environment_type,
            "status":           self.status.value,
            "entities_count":   self.entities_count,
            "profiles_count":   self.profiles_count,
            "entity_types":     self.entity_types,
            "config_generated": self.config_generated,
            "config_reasoning": self.config_reasoning,
            "current_round":    self.current_round,
            "environment_status": self.environment_status,
            # Legacy keys kept so that any frontend / runner code that still
            # reads enable_twitter / enable_reddit / twitter_status continues
            # to receive sensible values.
            "enable_twitter":   oasis_platform == "twitter",
            "enable_reddit":    oasis_platform == "reddit",
            "twitter_status":   self.environment_status if oasis_platform == "twitter" else "not_started",
            "reddit_status":    self.environment_status if oasis_platform == "reddit"  else "not_started",
            "created_at":       self.created_at,
            "updated_at":       self.updated_at,
            "error":            self.error,
        }

    def to_simple_dict(self) -> Dict[str, Any]:
        """Slim dict returned by the API."""
        return {
            "simulation_id":    self.simulation_id,
            "project_id":       self.project_id,
            "graph_id":         self.graph_id,
            "environment_type": self.environment_type,
            "status":           self.status.value,
            "entities_count":   self.entities_count,
            "profiles_count":   self.profiles_count,
            "entity_types":     self.entity_types,
            "config_generated": self.config_generated,
            "error":            self.error,
        }


# ---------------------------------------------------------------------------
# SimulationManager
# ---------------------------------------------------------------------------

class SimulationManager:
    """
    Manages the preparation lifecycle for OASIS simulations.

    Core responsibilities:
      1. Read and filter entities from the Zep memory graph.
      2. Generate OASIS agent profiles (from research output or graph entities).
      3. LLM-generate simulation configuration.
      4. Write artefacts to the simulation directory.
    """

    SIMULATION_DATA_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )

    def __init__(self):
        os.makedirs(self.SIMULATION_DATA_DIR, exist_ok=True)
        self._simulations: Dict[str, SimulationState] = {}

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------

    def _get_simulation_dir(self, simulation_id: str) -> str:
        sim_dir = os.path.join(self.SIMULATION_DATA_DIR, simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        return sim_dir

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_simulation_state(self, state: SimulationState) -> None:
        sim_dir = self._get_simulation_dir(state.simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        state.updated_at = datetime.now().isoformat()
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        self._simulations[state.simulation_id] = state

    def _load_simulation_state(self, simulation_id: str) -> Optional[SimulationState]:
        if simulation_id in self._simulations:
            return self._simulations[simulation_id]

        sim_dir = self._get_simulation_dir(simulation_id)
        state_file = os.path.join(sim_dir, "state.json")
        if not os.path.exists(state_file):
            return None

        with open(state_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Derive environment_type — prefer explicit field, fall back to
        # legacy enable_twitter / enable_reddit booleans.
        environment_type = data.get("environment_type")
        if not environment_type or environment_type not in VALID_ENVIRONMENTS:
            environment_type = _derive_environment_from_legacy(
                data.get("enable_twitter", False),
                data.get("enable_reddit", True),
            )

        # Derive environment_status from legacy twitter_status / reddit_status
        oasis_platform = OASIS_PLATFORM_MAP.get(environment_type, "reddit")
        if oasis_platform == "twitter":
            legacy_status = data.get("twitter_status", "not_started")
        else:
            legacy_status = data.get("reddit_status", "not_started")
        environment_status = data.get("environment_status", legacy_status)

        state = SimulationState(
            simulation_id=simulation_id,
            project_id=data.get("project_id", ""),
            graph_id=data.get("graph_id", ""),
            environment_type=environment_type,
            status=SimulationStatus(data.get("status", "created")),
            entities_count=data.get("entities_count", 0),
            profiles_count=data.get("profiles_count", 0),
            entity_types=data.get("entity_types", []),
            config_generated=data.get("config_generated", False),
            config_reasoning=data.get("config_reasoning", ""),
            current_round=data.get("current_round", 0),
            environment_status=environment_status,
            created_at=data.get("created_at", datetime.now().isoformat()),
            updated_at=data.get("updated_at", datetime.now().isoformat()),
            error=data.get("error"),
        )

        self._simulations[simulation_id] = state
        return state

    # ------------------------------------------------------------------
    # Public: create
    # ------------------------------------------------------------------

    def create_simulation(
        self,
        project_id:       str,
        graph_id:         str,
        environment_type: str = ENVIRONMENT_REDDIT,
        # Legacy params — kept for backwards compatibility, deprecated.
        enable_twitter:   Optional[bool] = None,
        enable_reddit:    Optional[bool] = None,
    ) -> SimulationState:
        """
        Create a new simulation record.

        Args:
            project_id:       Parent project ID.
            graph_id:         Memory graph ID.
            environment_type: Domain environment — "classroom", "organisation",
                              "twitter", or "reddit".
            enable_twitter:   Deprecated. Use environment_type="twitter".
            enable_reddit:    Deprecated. Use environment_type="reddit".

        Returns:
            SimulationState
        """
        # Handle legacy boolean flags
        if enable_twitter is not None or enable_reddit is not None:
            logger.warning(
                "enable_twitter / enable_reddit are deprecated in "
                "create_simulation(); use environment_type instead."
            )
            environment_type = _derive_environment_from_legacy(
                bool(enable_twitter), bool(enable_reddit)
            )

        if environment_type not in VALID_ENVIRONMENTS:
            logger.warning(
                "Unknown environment_type '%s'; falling back to '%s'.",
                environment_type, ENVIRONMENT_REDDIT,
            )
            environment_type = ENVIRONMENT_REDDIT

        import uuid
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"

        state = SimulationState(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            environment_type=environment_type,
            status=SimulationStatus.CREATED,
        )

        self._save_simulation_state(state)
        logger.info(
            "Simulation created: id=%s project=%s graph=%s environment=%s",
            simulation_id, project_id, graph_id, environment_type,
        )
        return state

    # ------------------------------------------------------------------
    # Public: prepare
    # ------------------------------------------------------------------

    def prepare_simulation(
        self,
        simulation_id:         str,
        simulation_requirement: str,
        document_text:          str,
        defined_entity_types:  Optional[List[str]] = None,
        use_llm_for_profiles:  bool = True,
        progress_callback:     Optional[callable] = None,
        parallel_profile_count: int = 3,
        # Research pipeline integration (Session 3 → Session 4 wiring)
        research_output=None,   # Optional[ResearchOutput] from research/schemas.py
        agents_per_segment: int = 5,
    ) -> SimulationState:
        """
        Prepare the simulation environment end-to-end.

        Steps:
          1. Read and filter entities from the memory graph.
          2. Generate agent profiles:
               - If research_output is provided, use ProfileBridge (research pipeline).
               - Otherwise, use OasisProfileGenerator (entity-based).
          3. LLM-generate simulation configuration.
          4. Save profile file and configuration to the simulation directory.

        Args:
            simulation_id:          ID of a simulation created via create_simulation().
            simulation_requirement: Human-readable scenario description.
            document_text:          Source document text (LLM context).
            defined_entity_types:   Optional entity-type filter for the graph.
            use_llm_for_profiles:   Whether to use LLM for entity-based profiles.
            progress_callback:      Optional fn(stage, pct, message, **kwargs).
            parallel_profile_count: Parallel workers for entity-based generation.
            research_output:        Optional ResearchOutput from the research agent.
                                    When supplied, ProfileBridge is used instead of
                                    OasisProfileGenerator.
            agents_per_segment:     Agents to generate per research segment
                                    (only used when research_output is provided).

        Returns:
            SimulationState
        """
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")

        try:
            state.status = SimulationStatus.PREPARING
            self._save_simulation_state(state)

            sim_dir = self._get_simulation_dir(simulation_id)
            environment_type = state.environment_type
            oasis_platform = OASIS_PLATFORM_MAP.get(environment_type, "reddit")

            # ---- Stage 1: read entities from memory graph ----
            if progress_callback:
                progress_callback("reading", 0, "Connecting to memory graph...")

            reader = ZepEntityReader()

            if progress_callback:
                progress_callback("reading", 30, "Reading entity nodes...")

            filtered = reader.filter_defined_entities(
                graph_id=state.graph_id,
                defined_entity_types=defined_entity_types,
                enrich_with_edges=True,
            )

            state.entities_count = filtered.filtered_count
            state.entity_types   = list(filtered.entity_types)

            if progress_callback:
                progress_callback(
                    "reading", 100,
                    f"Done — {filtered.filtered_count} entities found.",
                    current=filtered.filtered_count,
                    total=filtered.filtered_count,
                )

            if filtered.filtered_count == 0:
                state.status = SimulationStatus.FAILED
                state.error  = (
                    "No entities found in the memory graph. "
                    "Please check that the graph has been built correctly."
                )
                self._save_simulation_state(state)
                return state

            # ---- Stage 2: generate agent profiles ----
            profile_filename, profile_format = _profile_file_for_environment(
                environment_type
            )
            profile_path     = os.path.join(sim_dir, profile_filename)
            total_entities   = len(filtered.entities)

            if progress_callback:
                progress_callback(
                    "generating_profiles", 0, "Starting profile generation...",
                    current=0, total=total_entities,
                )

            if research_output is not None:
                # -- Research-pipeline path --
                profiles = self._generate_profiles_from_research(
                    research_output=research_output,
                    agents_per_segment=agents_per_segment,
                    output_platform=oasis_platform,
                    progress_callback=progress_callback,
                )
            else:
                # -- Entity-based path --
                profiles = self._generate_profiles_from_entities(
                    filtered=filtered,
                    graph_id=state.graph_id,
                    use_llm=use_llm_for_profiles,
                    parallel_count=parallel_profile_count,
                    realtime_output_path=profile_path,
                    output_platform=profile_format,
                    progress_callback=progress_callback,
                )

            state.profiles_count = len(profiles)

            # Save the final profile file (entity path already streams to disk;
            # research path needs an explicit save here).
            if progress_callback:
                progress_callback(
                    "generating_profiles", 95, "Saving profile file...",
                    current=total_entities, total=total_entities,
                )

            generator = OasisProfileGenerator(graph_id=state.graph_id)
            generator.save_profiles(
                profiles=profiles,
                file_path=profile_path,
                platform=profile_format,
            )

            if progress_callback:
                progress_callback(
                    "generating_profiles", 100,
                    f"Done — {len(profiles)} profiles saved.",
                    current=len(profiles), total=len(profiles),
                )

            # ---- Stage 3: generate simulation configuration ----
            if progress_callback:
                progress_callback(
                    "generating_config", 0, "Analysing simulation requirement...",
                    current=0, total=3,
                )

            config_gen = SimulationConfigGenerator()

            if progress_callback:
                progress_callback(
                    "generating_config", 30, "Calling LLM to generate config...",
                    current=1, total=3,
                )

            sim_params = config_gen.generate_config(
                simulation_id=simulation_id,
                project_id=state.project_id,
                graph_id=state.graph_id,
                simulation_requirement=simulation_requirement,
                document_text=document_text,
                entities=filtered.entities,
                environment_type=environment_type,
            )

            if progress_callback:
                progress_callback(
                    "generating_config", 70, "Saving configuration file...",
                    current=2, total=3,
                )

            config_path = os.path.join(sim_dir, "simulation_config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write(sim_params.to_json())

            state.config_generated = True
            state.config_reasoning = sim_params.generation_reasoning

            if progress_callback:
                progress_callback(
                    "generating_config", 100, "Configuration ready.",
                    current=3, total=3,
                )

            # Runner scripts live in backend/scripts/ — no copy needed.
            state.status = SimulationStatus.READY
            self._save_simulation_state(state)

            logger.info(
                "Simulation ready: id=%s entities=%d profiles=%d environment=%s",
                simulation_id, state.entities_count,
                state.profiles_count, environment_type,
            )
            return state

        except Exception as exc:
            logger.error(
                "Simulation preparation failed: id=%s error=%s\n%s",
                simulation_id, exc, tb.format_exc(),
            )
            state.status = SimulationStatus.FAILED
            state.error  = str(exc)
            self._save_simulation_state(state)
            raise

    # ------------------------------------------------------------------
    # Profile generation helpers
    # ------------------------------------------------------------------

    def _generate_profiles_from_research(
        self,
        research_output,
        agents_per_segment: int,
        output_platform:    str,
        progress_callback:  Optional[callable],
    ) -> List[OasisAgentProfile]:
        """Generate profiles from a ResearchOutput via ProfileBridge."""
        from .research.profile_bridge import ProfileBridge

        if progress_callback:
            progress_callback(
                "generating_profiles", 10,
                "Using research pipeline (ProfileBridge)...",
            )

        bridge   = ProfileBridge()
        profiles = bridge.generate(
            research=research_output,
            agents_per_segment=agents_per_segment,
            output_platform=output_platform,
            parallel=True,
        )

        logger.info(
            "ProfileBridge generated %d profiles from %d segments.",
            len(profiles), len(research_output.human_segments),
        )
        return profiles

    def _generate_profiles_from_entities(
        self,
        filtered,
        graph_id:             str,
        use_llm:              bool,
        parallel_count:       int,
        realtime_output_path: str,
        output_platform:      str,
        progress_callback:    Optional[callable],
    ) -> List[OasisAgentProfile]:
        """Generate profiles from graph entities via OasisProfileGenerator."""
        generator = OasisProfileGenerator(graph_id=graph_id)
        total     = len(filtered.entities)

        def _on_progress(current: int, total: int, msg: str) -> None:
            if progress_callback:
                progress_callback(
                    "generating_profiles",
                    int(current / total * 100) if total else 0,
                    msg,
                    current=current, total=total, item_name=msg,
                )

        return generator.generate_profiles_from_entities(
            entities=filtered.entities,
            use_llm=use_llm,
            progress_callback=_on_progress,
            graph_id=graph_id,
            parallel_count=parallel_count,
            realtime_output_path=realtime_output_path,
            output_platform=output_platform,
        )

    # ------------------------------------------------------------------
    # Public: query methods
    # ------------------------------------------------------------------

    def get_simulation(self, simulation_id: str) -> Optional[SimulationState]:
        return self._load_simulation_state(simulation_id)

    def list_simulations(
        self, project_id: Optional[str] = None
    ) -> List[SimulationState]:
        simulations: List[SimulationState] = []

        if not os.path.exists(self.SIMULATION_DATA_DIR):
            return simulations

        for sim_id in os.listdir(self.SIMULATION_DATA_DIR):
            sim_path = os.path.join(self.SIMULATION_DATA_DIR, sim_id)
            if sim_id.startswith('.') or not os.path.isdir(sim_path):
                continue
            state = self._load_simulation_state(sim_id)
            if state and (project_id is None or state.project_id == project_id):
                simulations.append(state)

        return simulations

    def get_profiles(
        self,
        simulation_id:    str,
        # platform is accepted for backwards compat but ignored in favour of
        # the environment_type stored on the state.
        platform:         str = "reddit",
    ) -> List[Dict[str, Any]]:
        """Return the agent profiles saved for this simulation."""
        state = self._load_simulation_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")

        profile_filename, _ = _profile_file_for_environment(state.environment_type)
        profile_path = os.path.join(
            self._get_simulation_dir(simulation_id), profile_filename
        )

        if not os.path.exists(profile_path):
            # Backwards compat: try the legacy filename the caller requested
            legacy_path = os.path.join(
                self._get_simulation_dir(simulation_id),
                f"{platform}_profiles.json",
            )
            if os.path.exists(legacy_path):
                profile_path = legacy_path
            else:
                return []

        with open(profile_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_simulation_config(
        self, simulation_id: str
    ) -> Optional[Dict[str, Any]]:
        """Return the saved simulation_config.json as a dict."""
        config_path = os.path.join(
            self._get_simulation_dir(simulation_id), "simulation_config.json"
        )
        if not os.path.exists(config_path):
            return None
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_run_instructions(self, simulation_id: str) -> Dict[str, str]:
        """Return CLI commands and paths needed to launch the simulation."""
        state      = self._load_simulation_state(simulation_id)
        env_type   = state.environment_type if state else ENVIRONMENT_REDDIT
        sim_dir    = self._get_simulation_dir(simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")
        scripts_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '../../scripts')
        )

        # Map environment to the script that runs it
        script_map = {
            ENVIRONMENT_CLASSROOM:    "run_classroom_simulation.py",
            ENVIRONMENT_ORGANISATION: "run_organisation_simulation.py",
            ENVIRONMENT_TWITTER:      "run_twitter_simulation.py",
            ENVIRONMENT_REDDIT:       "run_reddit_simulation.py",
        }
        primary_script = script_map.get(env_type, "run_reddit_simulation.py")
        primary_cmd    = (
            f"python {scripts_dir}/{primary_script} --config {config_path}"
        )

        return {
            "simulation_dir": sim_dir,
            "scripts_dir":    scripts_dir,
            "config_file":    config_path,
            "environment_type": env_type,
            "commands": {
                env_type: primary_cmd,
                # Legacy keys so that any frontend code still referencing
                # "twitter" or "reddit" command keys gets a sensible value.
                "twitter": (
                    f"python {scripts_dir}/run_twitter_simulation.py "
                    f"--config {config_path}"
                ),
                "reddit": (
                    f"python {scripts_dir}/run_reddit_simulation.py "
                    f"--config {config_path}"
                ),
            },
            "instructions": (
                f"1. Activate the conda environment: conda activate MiroFish\n"
                f"2. Run the simulation (scripts are in {scripts_dir}):\n"
                f"   {primary_cmd}"
            ),
        }
