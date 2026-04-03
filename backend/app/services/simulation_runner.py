"""
Simulation runner.

Launches simulation scripts as background subprocesses, tails their action
logs in a monitor thread, and exposes real-time status + history queries.

Supports domain environments:
  classroom    → run_classroom_simulation.py
  organisation → run_organisation_simulation.py
  twitter      → run_twitter_simulation.py  (legacy)
  reddit       → run_reddit_simulation.py   (legacy)
  parallel     → run_parallel_simulation.py (legacy dual-platform)
"""

import os
import sys
import json
import time
import asyncio
import threading
import subprocess
import signal
import atexit
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from queue import Queue

from ..config import Config
from ..utils.logger import get_logger
from .memory_updater import MemoryUpdaterManager as ZepGraphMemoryManager
from .simulation_ipc import SimulationIPCClient, CommandType, IPCResponse
from .simulation_config_generator import OASIS_PLATFORM_MAP

logger = get_logger('mirofish.simulation_runner')

_cleanup_registered = False
IS_WINDOWS = sys.platform == 'win32'

# Maps environment / legacy platform names to their runner script.
SCRIPT_MAP: Dict[str, str] = {
    "classroom":    "run_classroom_simulation.py",
    "organisation": "run_organisation_simulation.py",
    "twitter":      "run_twitter_simulation.py",
    "reddit":       "run_reddit_simulation.py",
    "parallel":     "run_parallel_simulation.py",  # legacy dual-platform
}


# ---------------------------------------------------------------------------
# Enums / dataclasses
# ---------------------------------------------------------------------------

class RunnerStatus(str, Enum):
    IDLE      = "idle"
    STARTING  = "starting"
    RUNNING   = "running"
    PAUSED    = "paused"
    STOPPING  = "stopping"
    STOPPED   = "stopped"
    COMPLETED = "completed"
    FAILED    = "failed"


@dataclass
class AgentAction:
    """Record of a single agent action observed in the simulation log."""
    round_num:   int
    timestamp:   str
    platform:    str   # environment name: classroom / organisation / twitter / reddit
    agent_id:    int
    agent_name:  str
    action_type: str
    action_args: Dict[str, Any] = field(default_factory=dict)
    result:      Optional[str] = None
    success:     bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num":   self.round_num,
            "timestamp":   self.timestamp,
            "platform":    self.platform,
            "agent_id":    self.agent_id,
            "agent_name":  self.agent_name,
            "action_type": self.action_type,
            "action_args": self.action_args,
            "result":      self.result,
            "success":     self.success,
        }


@dataclass
class RoundSummary:
    """Per-round action totals."""
    round_num:      int
    start_time:     str
    end_time:       Optional[str] = None
    simulated_hour: int = 0
    env_actions:    int = 0   # replaces twitter_actions / reddit_actions
    active_agents:  List[int] = field(default_factory=list)
    actions:        List[AgentAction] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_num":      self.round_num,
            "start_time":     self.start_time,
            "end_time":       self.end_time,
            "simulated_hour": self.simulated_hour,
            "env_actions":    self.env_actions,
            # Legacy keys for frontend backwards compat
            "twitter_actions": self.env_actions,
            "reddit_actions":  self.env_actions,
            "active_agents":  self.active_agents,
            "actions_count":  len(self.actions),
            "actions":        [a.to_dict() for a in self.actions],
        }


@dataclass
class SimulationRunState:
    """Real-time run state for one simulation."""
    simulation_id:  str
    runner_status:  RunnerStatus = RunnerStatus.IDLE
    environment_type: str = "reddit"

    # Overall progress
    current_round:          int = 0
    total_rounds:           int = 0
    simulated_hours:        int = 0
    total_simulation_hours: int = 0

    # Single-environment counters (replace twitter_* / reddit_* pairs)
    env_current_round:   int  = 0
    env_simulated_hours: int  = 0
    env_running:         bool = False
    env_actions_count:   int  = 0
    env_completed:       bool = False

    # Round summaries and recent action feed
    rounds:            List[RoundSummary] = field(default_factory=list)
    recent_actions:    List[AgentAction]  = field(default_factory=list)
    max_recent_actions: int = 50

    # Timestamps
    started_at:   Optional[str] = None
    updated_at:   str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    # Error detail and process handle
    error:       Optional[str] = None
    process_pid: Optional[int] = None

    def add_action(self, action: AgentAction) -> None:
        self.recent_actions.insert(0, action)
        if len(self.recent_actions) > self.max_recent_actions:
            self.recent_actions = self.recent_actions[: self.max_recent_actions]
        self.env_actions_count += 1
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        oasis_platform = OASIS_PLATFORM_MAP.get(self.environment_type, "reddit")
        return {
            "simulation_id":          self.simulation_id,
            "runner_status":          self.runner_status.value,
            "environment_type":       self.environment_type,
            "current_round":          self.current_round,
            "total_rounds":           self.total_rounds,
            "simulated_hours":        self.simulated_hours,
            "total_simulation_hours": self.total_simulation_hours,
            "progress_percent": round(
                self.current_round / max(self.total_rounds, 1) * 100, 1
            ),
            # New environment-agnostic fields
            "env_running":         self.env_running,
            "env_completed":       self.env_completed,
            "env_actions_count":   self.env_actions_count,
            "env_current_round":   self.env_current_round,
            "env_simulated_hours": self.env_simulated_hours,
            "total_actions_count": self.env_actions_count,
            # Legacy keys — frontend code that still reads twitter_* / reddit_*
            # receives values mapped from the single environment.
            "twitter_running":       self.env_running if oasis_platform == "twitter" else False,
            "reddit_running":        self.env_running if oasis_platform == "reddit"  else False,
            "twitter_completed":     self.env_completed if oasis_platform == "twitter" else False,
            "reddit_completed":      self.env_completed if oasis_platform == "reddit"  else False,
            "twitter_actions_count": self.env_actions_count if oasis_platform == "twitter" else 0,
            "reddit_actions_count":  self.env_actions_count if oasis_platform == "reddit"  else 0,
            "twitter_current_round": self.env_current_round if oasis_platform == "twitter" else 0,
            "reddit_current_round":  self.env_current_round if oasis_platform == "reddit"  else 0,
            "twitter_simulated_hours": self.env_simulated_hours if oasis_platform == "twitter" else 0,
            "reddit_simulated_hours":  self.env_simulated_hours if oasis_platform == "reddit"  else 0,
            "started_at":   self.started_at,
            "updated_at":   self.updated_at,
            "completed_at": self.completed_at,
            "error":        self.error,
            "process_pid":  self.process_pid,
        }

    def to_detail_dict(self) -> Dict[str, Any]:
        result = self.to_dict()
        result["recent_actions"] = [a.to_dict() for a in self.recent_actions]
        result["rounds_count"]   = len(self.rounds)
        return result


# ---------------------------------------------------------------------------
# SimulationRunner
# ---------------------------------------------------------------------------

class SimulationRunner:
    """
    Manages simulation subprocesses and their live action logs.

    Responsibilities:
      1. Launch the correct simulation script for the environment type.
      2. Tail action log files in a background monitor thread.
      3. Expose run-state, action history, and timeline queries.
      4. Handle pause / stop / resume and cross-platform process cleanup.
    """

    RUN_STATE_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../uploads/simulations'
    )
    SCRIPTS_DIR = os.path.join(
        os.path.dirname(__file__),
        '../../scripts'
    )

    # Class-level shared state (one entry per simulation_id)
    _run_states:     Dict[str, SimulationRunState]    = {}
    _processes:      Dict[str, subprocess.Popen]       = {}
    _action_queues:  Dict[str, Queue]                  = {}
    _monitor_threads: Dict[str, threading.Thread]      = {}
    _stdout_files:   Dict[str, Any]                    = {}
    _stderr_files:   Dict[str, Any]                    = {}
    _graph_memory_enabled: Dict[str, bool]             = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _get_env_type_from_config(cls, simulation_id: str) -> str:
        """Read environment_type from simulation_config.json; fall back to 'reddit'."""
        config_path = os.path.join(
            cls.RUN_STATE_DIR, simulation_id, "simulation_config.json"
        )
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get("environment_type", "reddit")
        except Exception:
            return "reddit"

    @classmethod
    def _env_log_path(cls, sim_dir: str, env_type: str) -> str:
        """Return the primary actions.jsonl path for this environment."""
        return os.path.join(sim_dir, env_type, "actions.jsonl")

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    @classmethod
    def get_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        if simulation_id in cls._run_states:
            return cls._run_states[simulation_id]
        state = cls._load_run_state(simulation_id)
        if state:
            cls._run_states[simulation_id] = state
        return state

    @classmethod
    def _load_run_state(cls, simulation_id: str) -> Optional[SimulationRunState]:
        state_file = os.path.join(
            cls.RUN_STATE_DIR, simulation_id, "run_state.json"
        )
        if not os.path.exists(state_file):
            return None

        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            env_type = data.get(
                "environment_type",
                cls._get_env_type_from_config(simulation_id),
            )
            oasis_platform = OASIS_PLATFORM_MAP.get(env_type, "reddit")

            # Load new-style env_* fields, falling back to legacy twitter_*/reddit_*
            if oasis_platform == "twitter":
                legacy_running   = data.get("twitter_running",       False)
                legacy_completed = data.get("twitter_completed",     False)
                legacy_actions   = data.get("twitter_actions_count", 0)
                legacy_round     = data.get("twitter_current_round", 0)
                legacy_hours     = data.get("twitter_simulated_hours", 0)
            else:
                legacy_running   = data.get("reddit_running",        False)
                legacy_completed = data.get("reddit_completed",      False)
                legacy_actions   = data.get("reddit_actions_count",  0)
                legacy_round     = data.get("reddit_current_round",  0)
                legacy_hours     = data.get("reddit_simulated_hours", 0)

            state = SimulationRunState(
                simulation_id=simulation_id,
                runner_status=RunnerStatus(data.get("runner_status", "idle")),
                environment_type=env_type,
                current_round=data.get("current_round", 0),
                total_rounds=data.get("total_rounds", 0),
                simulated_hours=data.get("simulated_hours", 0),
                total_simulation_hours=data.get("total_simulation_hours", 0),
                env_current_round=data.get("env_current_round", legacy_round),
                env_simulated_hours=data.get("env_simulated_hours", legacy_hours),
                env_running=data.get("env_running", legacy_running),
                env_actions_count=data.get("env_actions_count", legacy_actions),
                env_completed=data.get("env_completed", legacy_completed),
                started_at=data.get("started_at"),
                updated_at=data.get("updated_at", datetime.now().isoformat()),
                completed_at=data.get("completed_at"),
                error=data.get("error"),
                process_pid=data.get("process_pid"),
            )

            for a in data.get("recent_actions", []):
                state.recent_actions.append(AgentAction(
                    round_num=a.get("round_num", 0),
                    timestamp=a.get("timestamp", ""),
                    platform=a.get("platform", env_type),
                    agent_id=a.get("agent_id", 0),
                    agent_name=a.get("agent_name", ""),
                    action_type=a.get("action_type", ""),
                    action_args=a.get("action_args", {}),
                    result=a.get("result"),
                    success=a.get("success", True),
                ))

            return state

        except Exception as exc:
            logger.error("Failed to load run state: %s", exc)
            return None

    @classmethod
    def _save_run_state(cls, state: SimulationRunState) -> None:
        sim_dir = os.path.join(cls.RUN_STATE_DIR, state.simulation_id)
        os.makedirs(sim_dir, exist_ok=True)
        state_file = os.path.join(sim_dir, "run_state.json")
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state.to_detail_dict(), f, ensure_ascii=False, indent=2)
        cls._run_states[state.simulation_id] = state

    # ------------------------------------------------------------------
    # Start
    # ------------------------------------------------------------------

    @classmethod
    def start_simulation(
        cls,
        simulation_id:              str,
        platform:                   str  = None,   # explicit override; None = auto from config
        max_rounds:                 int  = None,
        enable_graph_memory_update: bool = False,
        graph_id:                   str  = None,
    ) -> SimulationRunState:
        """
        Launch a simulation script as a background subprocess.

        Args:
            simulation_id:              The simulation to run.
            platform:                   Override the script selection.  Accepts
                                        "classroom", "organisation", "twitter",
                                        "reddit", or "parallel".  When None the
                                        environment_type stored in the saved
                                        simulation_config.json is used.
            max_rounds:                 Cap the number of simulation rounds.
            enable_graph_memory_update: Stream agent actions back to the memory graph.
            graph_id:                   Required when enable_graph_memory_update=True.

        Returns:
            SimulationRunState
        """
        existing = cls.get_run_state(simulation_id)
        if existing and existing.runner_status in (
            RunnerStatus.RUNNING, RunnerStatus.STARTING
        ):
            raise ValueError(f"Simulation is already running: {simulation_id}")

        sim_dir     = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        config_path = os.path.join(sim_dir, "simulation_config.json")

        if not os.path.exists(config_path):
            raise ValueError(
                "Simulation config not found. Call /prepare first."
            )

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Resolve the target environment / platform
        env_type = config.get("environment_type", "reddit")
        if platform is None:
            platform = env_type

        # Determine the script to run
        script_name = SCRIPT_MAP.get(platform)
        if script_name is None:
            logger.warning(
                "Unknown platform '%s'; falling back to run_reddit_simulation.py.",
                platform,
            )
            script_name = "run_reddit_simulation.py"

        script_path = os.path.join(cls.SCRIPTS_DIR, script_name)
        if not os.path.exists(script_path):
            raise ValueError(f"Script not found: {script_path}")

        # Compute total rounds from config
        time_cfg      = config.get("time_config", {})
        total_hours   = time_cfg.get("total_simulation_hours", 72)
        mins_per_round = time_cfg.get("minutes_per_round", 60)
        total_rounds  = int(total_hours * 60 / mins_per_round)

        if max_rounds and max_rounds > 0:
            original = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original:
                logger.info(
                    "Round count capped: %d → %d (max_rounds=%d)",
                    original, total_rounds, max_rounds,
                )

        state = SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.STARTING,
            environment_type=env_type,
            total_rounds=total_rounds,
            total_simulation_hours=total_hours,
            started_at=datetime.now().isoformat(),
            env_running=True,
        )
        cls._save_run_state(state)

        # Optional graph memory update
        if enable_graph_memory_update:
            if not graph_id:
                raise ValueError(
                    "graph_id is required when enable_graph_memory_update=True"
                )
            try:
                ZepGraphMemoryManager.create_updater(simulation_id, graph_id)
                cls._graph_memory_enabled[simulation_id] = True
                logger.info(
                    "Graph memory update enabled: simulation=%s graph=%s",
                    simulation_id, graph_id,
                )
            except Exception as exc:
                logger.error("Failed to create graph memory updater: %s", exc)
                cls._graph_memory_enabled[simulation_id] = False
        else:
            cls._graph_memory_enabled[simulation_id] = False

        cls._action_queues[simulation_id] = Queue()

        try:
            cmd = [sys.executable, script_path, "--config", config_path]
            if max_rounds and max_rounds > 0:
                cmd.extend(["--max-rounds", str(max_rounds)])

            main_log_path = os.path.join(sim_dir, "simulation.log")
            main_log_file = open(main_log_path, 'w', encoding='utf-8')

            env = os.environ.copy()
            env['PYTHONUTF8']        = '1'
            env['PYTHONIOENCODING']  = 'utf-8'

            process = subprocess.Popen(
                cmd,
                cwd=sim_dir,
                stdout=main_log_file,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                bufsize=1,
                env=env,
                start_new_session=True,
            )

            cls._stdout_files[simulation_id] = main_log_file
            cls._stderr_files[simulation_id] = None

            state.process_pid    = process.pid
            state.runner_status  = RunnerStatus.RUNNING
            cls._processes[simulation_id] = process
            cls._save_run_state(state)

            monitor = threading.Thread(
                target=cls._monitor_simulation,
                args=(simulation_id,),
                daemon=True,
            )
            monitor.start()
            cls._monitor_threads[simulation_id] = monitor

            logger.info(
                "Simulation started: id=%s pid=%d environment=%s script=%s",
                simulation_id, process.pid, env_type, script_name,
            )

        except Exception as exc:
            state.runner_status = RunnerStatus.FAILED
            state.error         = str(exc)
            cls._save_run_state(state)
            raise

        return state

    # ------------------------------------------------------------------
    # Monitor thread
    # ------------------------------------------------------------------

    @classmethod
    def _monitor_simulation(cls, simulation_id: str) -> None:
        """Tail action log files and update run state until the process exits."""
        sim_dir  = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        env_type = cls._get_env_type_from_config(simulation_id)

        # Primary log path for the current environment
        primary_log = cls._env_log_path(sim_dir, env_type)

        # Backwards-compat: also watch legacy twitter/ and reddit/ dirs for
        # simulations that were created before the domain environment model.
        legacy_logs: Dict[str, str] = {}
        for legacy_env in ("twitter", "reddit", "classroom", "organisation"):
            if legacy_env != env_type:
                p = cls._env_log_path(sim_dir, legacy_env)
                legacy_logs[legacy_env] = p

        process = cls._processes.get(simulation_id)
        state   = cls.get_run_state(simulation_id)
        if not process or not state:
            return

        positions: Dict[str, int] = {primary_log: 0}
        for p in legacy_logs.values():
            positions[p] = 0

        try:
            while process.poll() is None:
                # Primary environment log
                if os.path.exists(primary_log):
                    positions[primary_log] = cls._read_action_log(
                        primary_log, positions[primary_log], state, env_type
                    )

                # Legacy logs (backwards compat for old parallel simulations)
                for leg_env, leg_path in legacy_logs.items():
                    if os.path.exists(leg_path):
                        positions[leg_path] = cls._read_action_log(
                            leg_path, positions[leg_path], state, leg_env
                        )

                cls._save_run_state(state)
                time.sleep(2)

            # Final drain after process exits
            if os.path.exists(primary_log):
                cls._read_action_log(
                    primary_log, positions[primary_log], state, env_type
                )
            for leg_env, leg_path in legacy_logs.items():
                if os.path.exists(leg_path):
                    cls._read_action_log(
                        leg_path, positions[leg_path], state, leg_env
                    )

            exit_code = process.returncode
            if exit_code == 0:
                state.runner_status = RunnerStatus.COMPLETED
                state.completed_at  = datetime.now().isoformat()
                logger.info("Simulation completed: %s", simulation_id)
            else:
                state.runner_status = RunnerStatus.FAILED
                # Read tail of main log for the error
                main_log = os.path.join(sim_dir, "simulation.log")
                error_tail = ""
                try:
                    if os.path.exists(main_log):
                        with open(main_log, 'r', encoding='utf-8') as f:
                            error_tail = f.read()[-2000:]
                except Exception:
                    pass
                state.error = f"Process exit code {exit_code}. {error_tail}"
                logger.error("Simulation failed: %s error=%s", simulation_id, state.error)

            state.env_running = False
            cls._save_run_state(state)

        except Exception as exc:
            logger.error("Monitor thread error: %s %s", simulation_id, exc)
            state.runner_status = RunnerStatus.FAILED
            state.error         = str(exc)
            cls._save_run_state(state)

        finally:
            if cls._graph_memory_enabled.get(simulation_id, False):
                try:
                    ZepGraphMemoryManager.stop_updater(simulation_id)
                    logger.info(
                        "Graph memory update stopped: %s", simulation_id
                    )
                except Exception as exc:
                    logger.error("Failed to stop graph memory updater: %s", exc)
                cls._graph_memory_enabled.pop(simulation_id, None)

            cls._processes.pop(simulation_id, None)
            cls._action_queues.pop(simulation_id, None)

            fh = cls._stdout_files.pop(simulation_id, None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
            fh = cls._stderr_files.pop(simulation_id, None)
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Action log reader
    # ------------------------------------------------------------------

    @classmethod
    def _read_action_log(
        cls,
        log_path:  str,
        position:  int,
        state:     SimulationRunState,
        env_label: str,
    ) -> int:
        """
        Read new lines from an actions.jsonl file, update run state.

        Args:
            log_path:  Path to the actions.jsonl file.
            position:  Byte offset of the last read.
            state:     Run state to update in-place.
            env_label: Environment label to stamp on each AgentAction.

        Returns:
            Updated byte offset.
        """
        graph_memory_enabled = cls._graph_memory_enabled.get(
            state.simulation_id, False
        )
        graph_updater = None
        if graph_memory_enabled:
            graph_updater = ZepGraphMemoryManager.get_updater(state.simulation_id)

        try:
            with open(log_path, 'r', encoding='utf-8') as f:
                f.seek(position)
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)

                        if "event_type" in data:
                            event_type = data["event_type"]

                            if event_type == "simulation_end":
                                state.env_completed = True
                                state.env_running   = False
                                logger.info(
                                    "Simulation end event: %s env=%s "
                                    "total_rounds=%s total_actions=%s",
                                    state.simulation_id, env_label,
                                    data.get("total_rounds"),
                                    data.get("total_actions"),
                                )
                                if cls._check_env_completed(state):
                                    state.runner_status = RunnerStatus.COMPLETED
                                    state.completed_at  = datetime.now().isoformat()
                                    logger.info(
                                        "All environments complete: %s",
                                        state.simulation_id,
                                    )

                            elif event_type == "round_end":
                                round_num      = data.get("round", 0)
                                sim_hours      = data.get("simulated_hours", 0)
                                if round_num > state.env_current_round:
                                    state.env_current_round = round_num
                                state.env_simulated_hours = sim_hours
                                if round_num > state.current_round:
                                    state.current_round = round_num
                                state.simulated_hours = sim_hours

                            continue

                        action = AgentAction(
                            round_num=data.get("round", 0),
                            timestamp=data.get(
                                "timestamp", datetime.now().isoformat()
                            ),
                            platform=env_label,
                            agent_id=data.get("agent_id", 0),
                            agent_name=data.get("agent_name", ""),
                            action_type=data.get("action_type", ""),
                            action_args=data.get("action_args", {}),
                            result=data.get("result"),
                            success=data.get("success", True),
                        )
                        state.add_action(action)

                        if action.round_num and action.round_num > state.current_round:
                            state.current_round = action.round_num

                        if graph_updater:
                            graph_updater.add_activity_from_dict(
                                data, env_label
                            )

                    except json.JSONDecodeError:
                        pass

                return f.tell()

        except Exception as exc:
            logger.warning("Failed to read action log %s: %s", log_path, exc)
            return position

    @classmethod
    def _check_env_completed(cls, state: SimulationRunState) -> bool:
        """Return True when the simulation environment has signalled completion."""
        return state.env_completed

    # ------------------------------------------------------------------
    # Process management
    # ------------------------------------------------------------------

    @classmethod
    def _terminate_process(
        cls,
        process:       subprocess.Popen,
        simulation_id: str,
        timeout:       int = 10,
    ) -> None:
        if IS_WINDOWS:
            logger.info(
                "Terminating process tree (Windows): sim=%s pid=%d",
                simulation_id, process.pid,
            )
            try:
                subprocess.run(
                    ['taskkill', '/PID', str(process.pid), '/T'],
                    capture_output=True, timeout=5,
                )
                try:
                    process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning(
                        "Process did not respond; forcing kill: %s", simulation_id
                    )
                    subprocess.run(
                        ['taskkill', '/F', '/PID', str(process.pid), '/T'],
                        capture_output=True, timeout=5,
                    )
                    process.wait(timeout=5)
            except Exception as exc:
                logger.warning("taskkill failed, falling back to terminate: %s", exc)
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        else:
            pgid = os.getpgid(process.pid)
            logger.info(
                "Terminating process group (Unix): sim=%s pgid=%d",
                simulation_id, pgid,
            )
            os.killpg(pgid, signal.SIGTERM)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Process group did not respond to SIGTERM; sending SIGKILL: %s",
                    simulation_id,
                )
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=5)

    @classmethod
    def stop_simulation(cls, simulation_id: str) -> SimulationRunState:
        """Stop a running simulation."""
        state = cls.get_run_state(simulation_id)
        if not state:
            raise ValueError(f"Simulation not found: {simulation_id}")

        if state.runner_status not in (RunnerStatus.RUNNING, RunnerStatus.PAUSED):
            raise ValueError(
                f"Simulation is not running: {simulation_id} "
                f"status={state.runner_status}"
            )

        state.runner_status = RunnerStatus.STOPPING
        cls._save_run_state(state)

        process = cls._processes.get(simulation_id)
        if process and process.poll() is None:
            try:
                cls._terminate_process(process, simulation_id)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.error(
                    "Failed to terminate process group: %s %s", simulation_id, exc
                )
                try:
                    process.terminate()
                    process.wait(timeout=5)
                except Exception:
                    process.kill()

        state.runner_status = RunnerStatus.STOPPED
        state.env_running   = False
        state.completed_at  = datetime.now().isoformat()
        cls._save_run_state(state)

        if cls._graph_memory_enabled.get(simulation_id, False):
            try:
                ZepGraphMemoryManager.stop_updater(simulation_id)
                logger.info(
                    "Graph memory update stopped: %s", simulation_id
                )
            except Exception as exc:
                logger.error("Failed to stop graph memory updater: %s", exc)
            cls._graph_memory_enabled.pop(simulation_id, None)

        logger.info("Simulation stopped: %s", simulation_id)
        return state

    # ------------------------------------------------------------------
    # Action queries
    # ------------------------------------------------------------------

    @classmethod
    def _read_actions_from_file(
        cls,
        file_path:        str,
        default_platform: Optional[str] = None,
        platform_filter:  Optional[str] = None,
        agent_id:         Optional[int] = None,
        round_num:        Optional[int] = None,
    ) -> List[AgentAction]:
        if not os.path.exists(file_path):
            return []

        actions: List[AgentAction] = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "event_type" in data or "agent_id" not in data:
                        continue

                    record_platform = (
                        data.get("platform") or default_platform or ""
                    )

                    if platform_filter and record_platform != platform_filter:
                        continue
                    if agent_id is not None and data.get("agent_id") != agent_id:
                        continue
                    if round_num is not None and data.get("round") != round_num:
                        continue

                    actions.append(AgentAction(
                        round_num=data.get("round", 0),
                        timestamp=data.get("timestamp", ""),
                        platform=record_platform,
                        agent_id=data.get("agent_id", 0),
                        agent_name=data.get("agent_name", ""),
                        action_type=data.get("action_type", ""),
                        action_args=data.get("action_args", {}),
                        result=data.get("result"),
                        success=data.get("success", True),
                    ))
                except json.JSONDecodeError:
                    continue

        return actions

    @classmethod
    def get_all_actions(
        cls,
        simulation_id: str,
        platform:      Optional[str] = None,
        agent_id:      Optional[int] = None,
        round_num:     Optional[int] = None,
    ) -> List[AgentAction]:
        """Return the complete action history, newest first."""
        sim_dir  = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        env_type = cls._get_env_type_from_config(simulation_id)
        actions: List[AgentAction] = []

        # Primary environment log
        primary_log = cls._env_log_path(sim_dir, env_type)
        if not platform or platform == env_type:
            actions.extend(cls._read_actions_from_file(
                primary_log, default_platform=env_type,
                platform_filter=platform, agent_id=agent_id, round_num=round_num,
            ))

        # Backwards-compat: check legacy twitter/ reddit/ classroom/ organisation/ dirs
        for legacy_env in ("twitter", "reddit", "classroom", "organisation"):
            if legacy_env == env_type:
                continue
            leg_log = cls._env_log_path(sim_dir, legacy_env)
            if os.path.exists(leg_log):
                if not platform or platform == legacy_env:
                    actions.extend(cls._read_actions_from_file(
                        leg_log, default_platform=legacy_env,
                        platform_filter=platform, agent_id=agent_id,
                        round_num=round_num,
                    ))

        # Fallback to old single-file format
        if not actions:
            old_log = os.path.join(sim_dir, "actions.jsonl")
            actions = cls._read_actions_from_file(
                old_log, default_platform=None,
                platform_filter=platform, agent_id=agent_id, round_num=round_num,
            )

        actions.sort(key=lambda x: x.timestamp, reverse=True)
        return actions

    @classmethod
    def get_actions(
        cls,
        simulation_id: str,
        limit:     int = 100,
        offset:    int = 0,
        platform:  Optional[str] = None,
        agent_id:  Optional[int] = None,
        round_num: Optional[int] = None,
    ) -> List[AgentAction]:
        """Paginated action history."""
        all_actions = cls.get_all_actions(
            simulation_id=simulation_id,
            platform=platform, agent_id=agent_id, round_num=round_num,
        )
        return all_actions[offset: offset + limit]

    @classmethod
    def get_timeline(
        cls,
        simulation_id: str,
        start_round:   int = 0,
        end_round:     Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Per-round action summary."""
        actions = cls.get_actions(simulation_id, limit=10000)
        env_type = cls._get_env_type_from_config(simulation_id)

        rounds: Dict[int, Dict[str, Any]] = {}
        for action in actions:
            rn = action.round_num
            if rn < start_round:
                continue
            if end_round is not None and rn > end_round:
                continue

            if rn not in rounds:
                rounds[rn] = {
                    "round_num":         rn,
                    "env_actions":       0,
                    "environment_type":  env_type,
                    "active_agents":     set(),
                    "action_types":      {},
                    "first_action_time": action.timestamp,
                    "last_action_time":  action.timestamp,
                }

            r = rounds[rn]
            r["env_actions"] += 1
            r["active_agents"].add(action.agent_id)
            r["action_types"][action.action_type] = (
                r["action_types"].get(action.action_type, 0) + 1
            )
            r["last_action_time"] = action.timestamp

        result = []
        for rn in sorted(rounds.keys()):
            r = rounds[rn]
            result.append({
                "round_num":           rn,
                "environment_type":    env_type,
                "env_actions":         r["env_actions"],
                "total_actions":       r["env_actions"],
                # Legacy keys
                "twitter_actions":     r["env_actions"],
                "reddit_actions":      r["env_actions"],
                "active_agents_count": len(r["active_agents"]),
                "active_agents":       list(r["active_agents"]),
                "action_types":        r["action_types"],
                "first_action_time":   r["first_action_time"],
                "last_action_time":    r["last_action_time"],
            })

        return result

    @classmethod
    def get_agent_stats(cls, simulation_id: str) -> List[Dict[str, Any]]:
        """Per-agent action statistics."""
        actions = cls.get_actions(simulation_id, limit=10000)
        stats: Dict[int, Dict[str, Any]] = {}

        for action in actions:
            aid = action.agent_id
            if aid not in stats:
                stats[aid] = {
                    "agent_id":          aid,
                    "agent_name":        action.agent_name,
                    "total_actions":     0,
                    "env_actions":       0,
                    # Legacy keys
                    "twitter_actions":   0,
                    "reddit_actions":    0,
                    "action_types":      {},
                    "first_action_time": action.timestamp,
                    "last_action_time":  action.timestamp,
                }

            s = stats[aid]
            s["total_actions"] += 1
            s["env_actions"]   += 1
            # Keep legacy keys consistent
            s["twitter_actions"] += 1
            s["reddit_actions"]  += 1
            s["action_types"][action.action_type] = (
                s["action_types"].get(action.action_type, 0) + 1
            )
            s["last_action_time"] = action.timestamp

        return sorted(stats.values(), key=lambda x: x["total_actions"], reverse=True)

    # ------------------------------------------------------------------
    # Log cleanup
    # ------------------------------------------------------------------

    @classmethod
    def cleanup_simulation_logs(cls, simulation_id: str) -> Dict[str, Any]:
        """
        Delete run artefacts so the simulation can be started fresh.

        Does NOT delete simulation_config.json or profile files.
        """
        import shutil

        sim_dir = os.path.join(cls.RUN_STATE_DIR, simulation_id)
        if not os.path.exists(sim_dir):
            return {"success": True, "message": "Simulation directory not found."}

        cleaned: List[str] = []
        errors:  List[str] = []

        files_to_delete = [
            "run_state.json",
            "simulation.log",
            "stdout.log",
            "stderr.log",
            # All known per-environment database files
            "twitter_simulation.db",
            "reddit_simulation.db",
            "classroom_simulation.db",
            "organisation_simulation.db",
            "env_status.json",
        ]

        # All known action-log directories
        dirs_to_clean = ["twitter", "reddit", "classroom", "organisation"]

        for filename in files_to_delete:
            fp = os.path.join(sim_dir, filename)
            if os.path.exists(fp):
                try:
                    os.remove(fp)
                    cleaned.append(filename)
                except Exception as exc:
                    errors.append(f"Could not delete {filename}: {exc}")

        for dir_name in dirs_to_clean:
            actions_file = os.path.join(sim_dir, dir_name, "actions.jsonl")
            if os.path.exists(actions_file):
                try:
                    os.remove(actions_file)
                    cleaned.append(f"{dir_name}/actions.jsonl")
                except Exception as exc:
                    errors.append(
                        f"Could not delete {dir_name}/actions.jsonl: {exc}"
                    )

        cls._run_states.pop(simulation_id, None)
        logger.info(
            "Cleaned simulation logs: %s deleted=%s", simulation_id, cleaned
        )

        return {
            "success": len(errors) == 0,
            "cleaned_files": cleaned,
            "errors": errors or None,
        }

    # ------------------------------------------------------------------
    # Server shutdown cleanup
    # ------------------------------------------------------------------

    _cleanup_done = False

    @classmethod
    def cleanup_all_simulations(cls) -> None:
        """Terminate all running simulation subprocesses (called on server shutdown)."""
        if cls._cleanup_done:
            return
        cls._cleanup_done = True

        if not cls._processes and not cls._graph_memory_enabled:
            return

        logger.info("Cleaning up all simulation processes...")

        try:
            ZepGraphMemoryManager.stop_all()
        except Exception as exc:
            logger.error("Failed to stop graph memory updaters: %s", exc)
        cls._graph_memory_enabled.clear()

        for simulation_id, process in list(cls._processes.items()):
            try:
                if process.poll() is None:
                    logger.info(
                        "Terminating simulation: %s pid=%d",
                        simulation_id, process.pid,
                    )
                    try:
                        cls._terminate_process(process, simulation_id, timeout=5)
                    except (ProcessLookupError, OSError):
                        try:
                            process.terminate()
                            process.wait(timeout=3)
                        except Exception:
                            process.kill()

                    state = cls.get_run_state(simulation_id)
                    if state:
                        state.runner_status = RunnerStatus.STOPPED
                        state.env_running   = False
                        state.completed_at  = datetime.now().isoformat()
                        state.error         = "Server shutdown — simulation terminated."
                        cls._save_run_state(state)

                    # Also stamp state.json for the frontend
                    try:
                        state_file = os.path.join(
                            cls.RUN_STATE_DIR, simulation_id, "state.json"
                        )
                        if os.path.exists(state_file):
                            with open(state_file, 'r', encoding='utf-8') as f:
                                state_data = json.load(f)
                            state_data['status']     = 'stopped'
                            state_data['updated_at'] = datetime.now().isoformat()
                            with open(state_file, 'w', encoding='utf-8') as f:
                                json.dump(state_data, f, indent=2, ensure_ascii=False)
                    except Exception as exc:
                        logger.warning(
                            "Could not update state.json for %s: %s",
                            simulation_id, exc,
                        )

            except Exception as exc:
                logger.error(
                    "Failed to clean up simulation %s: %s", simulation_id, exc
                )

        for fh in list(cls._stdout_files.values()):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
        cls._stdout_files.clear()

        for fh in list(cls._stderr_files.values()):
            try:
                if fh:
                    fh.close()
            except Exception:
                pass
        cls._stderr_files.clear()

        cls._processes.clear()
        cls._action_queues.clear()
        logger.info("Simulation process cleanup complete.")

    @classmethod
    def register_cleanup(cls) -> None:
        """Register atexit / signal handlers for graceful shutdown."""
        global _cleanup_registered

        if _cleanup_registered:
            return

        is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        is_debug_mode = (
            os.environ.get('FLASK_DEBUG') == '1'
            or os.environ.get('WERKZEUG_RUN_MAIN') is not None
        )

        if is_debug_mode and not is_reloader_process:
            _cleanup_registered = True
            return

        original_sigint  = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        has_sighup = hasattr(signal, 'SIGHUP')
        original_sighup = signal.getsignal(signal.SIGHUP) if has_sighup else None

        def cleanup_handler(signum=None, frame=None) -> None:
            if cls._processes or cls._graph_memory_enabled:
                logger.info("Signal %s received — cleaning up...", signum)
            cls.cleanup_all_simulations()

            if signum == signal.SIGINT and callable(original_sigint):
                original_sigint(signum, frame)
            elif signum == signal.SIGTERM and callable(original_sigterm):
                original_sigterm(signum, frame)
            elif has_sighup and signum == signal.SIGHUP and callable(original_sighup):
                original_sighup(signum, frame)

        signal.signal(signal.SIGINT,  cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)
        if has_sighup:
            signal.signal(signal.SIGHUP, cleanup_handler)

        atexit.register(cls.cleanup_all_simulations)
        _cleanup_registered = True
        logger.info("Simulation cleanup handlers registered.")
