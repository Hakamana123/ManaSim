"""
OASIS Organisation Simulation Script

Reads parameters from a simulation config file and executes an organisation
environment simulation using the Twitter OASIS platform (broadcast/response).
The Twitter platform is used because organisational communication is inherently
broadcast-first — a manager posts an update; team members reply or repost.

Agent profiles are stored as JSON (organisation_profiles.json). The script
converts them to the CSV format required by generate_twitter_agent_graph
before starting the simulation.

Features:
- After the simulation loop completes, the environment stays alive and
  enters command-wait mode so interview commands can be issued via IPC.
- Supports single-agent interview, batch interview, and remote close_env.

Usage:
    python run_organisation_simulation.py --config /path/to/simulation_config.json
    python run_organisation_simulation.py --config /path/to/simulation_config.json --no-wait
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import random
import signal
import sys
import sqlite3
import tempfile
from datetime import datetime
from typing import Dict, Any, List, Optional

# Global variables for signal handling
_shutdown_event = None
_cleanup_done = False

# Add project paths
_scripts_dir = os.path.dirname(os.path.abspath(__file__))
_backend_dir = os.path.abspath(os.path.join(_scripts_dir, '..'))
_project_root = os.path.abspath(os.path.join(_backend_dir, '..'))
sys.path.insert(0, _scripts_dir)
sys.path.insert(0, _backend_dir)

# Load .env from project root (contains LLM_API_KEY and other config)
from dotenv import load_dotenv
_env_file = os.path.join(_project_root, '.env')
if os.path.exists(_env_file):
    load_dotenv(_env_file)
else:
    _backend_env = os.path.join(_backend_dir, '.env')
    if os.path.exists(_backend_env):
        load_dotenv(_backend_env)


import re


class UnicodeFormatter(logging.Formatter):
    """Custom log formatter that converts Unicode escape sequences to readable characters."""

    UNICODE_ESCAPE_PATTERN = re.compile(r'\\u([0-9a-fA-F]{4})')

    def format(self, record):
        result = super().format(record)

        def replace_unicode(match):
            try:
                return chr(int(match.group(1), 16))
            except (ValueError, OverflowError):
                return match.group(0)

        return self.UNICODE_ESCAPE_PATTERN.sub(replace_unicode, result)


class MaxTokensWarningFilter(logging.Filter):
    """Suppress camel-ai warnings about max_tokens (intentionally not set)."""

    def filter(self, record):
        if "max_tokens" in record.getMessage() and "Invalid or missing" in record.getMessage():
            return False
        return True


# Apply filter at module load time, before camel code runs
logging.getLogger().addFilter(MaxTokensWarningFilter())


def setup_oasis_logging(log_dir: str):
    """Configure OASIS loggers to write to fixed-name files in *log_dir*."""
    os.makedirs(log_dir, exist_ok=True)

    for f in os.listdir(log_dir):
        old_log = os.path.join(log_dir, f)
        if os.path.isfile(old_log) and f.endswith('.log'):
            try:
                os.remove(old_log)
            except OSError:
                pass

    formatter = UnicodeFormatter("%(levelname)s - %(asctime)s - %(name)s - %(message)s")

    loggers_config = {
        "social.agent": os.path.join(log_dir, "social.agent.log"),
        "social.twitter": os.path.join(log_dir, "social.twitter.log"),
        "social.rec": os.path.join(log_dir, "social.rec.log"),
        "oasis.env": os.path.join(log_dir, "oasis.env.log"),
        "table": os.path.join(log_dir, "table.log"),
    }

    for logger_name, log_file in loggers_config.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.handlers.clear()
        file_handler = logging.FileHandler(log_file, encoding='utf-8', mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.propagate = False


try:
    from camel.models import ModelFactory
    from camel.types import ModelPlatformType
    import oasis
    from oasis import (
        ActionType,
        LLMAction,
        ManualAction,
        generate_twitter_agent_graph,
    )
except ImportError as e:
    print(f"Error: missing dependency — {e}")
    print("Please install: pip install oasis-ai camel-ai")
    sys.exit(1)


# IPC constants
IPC_COMMANDS_DIR = "ipc_commands"
IPC_RESPONSES_DIR = "ipc_responses"
ENV_STATUS_FILE = "env_status.json"


class CommandType:
    """IPC command type constants."""
    INTERVIEW = "interview"
    BATCH_INTERVIEW = "batch_interview"
    CLOSE_ENV = "close_env"


class IPCHandler:
    """Handles IPC commands for the organisation environment."""

    def __init__(self, simulation_dir: str, env, agent_graph):
        self.simulation_dir = simulation_dir
        self.env = env
        self.agent_graph = agent_graph
        self.commands_dir = os.path.join(simulation_dir, IPC_COMMANDS_DIR)
        self.responses_dir = os.path.join(simulation_dir, IPC_RESPONSES_DIR)
        self.status_file = os.path.join(simulation_dir, ENV_STATUS_FILE)
        self._running = True

        os.makedirs(self.commands_dir, exist_ok=True)
        os.makedirs(self.responses_dir, exist_ok=True)

    def update_status(self, status: str):
        """Write the current environment status to disk."""
        with open(self.status_file, 'w', encoding='utf-8') as f:
            json.dump({
                "status": status,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)

    def poll_command(self) -> Optional[Dict[str, Any]]:
        """Return the oldest pending command file, or None if the queue is empty."""
        if not os.path.exists(self.commands_dir):
            return None

        command_files = []
        for filename in os.listdir(self.commands_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.commands_dir, filename)
                command_files.append((filepath, os.path.getmtime(filepath)))

        command_files.sort(key=lambda x: x[1])

        for filepath, _ in command_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

        return None

    def send_response(self, command_id: str, status: str, result: Dict = None, error: str = None):
        """Write a response file and delete the corresponding command file."""
        response = {
            "command_id": command_id,
            "status": status,
            "result": result,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }

        response_file = os.path.join(self.responses_dir, f"{command_id}.json")
        with open(response_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)

        command_file = os.path.join(self.commands_dir, f"{command_id}.json")
        try:
            os.remove(command_file)
        except OSError:
            pass

    async def handle_interview(self, command_id: str, agent_id: int, prompt: str) -> bool:
        """Execute a single-agent interview action.

        Returns:
            True on success, False on failure.
        """
        try:
            agent = self.agent_graph.get_agent(agent_id)

            interview_action = ManualAction(
                action_type=ActionType.INTERVIEW,
                action_args={"prompt": prompt}
            )

            actions = {agent: interview_action}
            await self.env.step(actions)

            result = self._get_interview_result(agent_id)
            self.send_response(command_id, "completed", result=result)
            print(f"  Interview complete: agent_id={agent_id}")
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"  Interview failed: agent_id={agent_id}, error={error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False

    async def handle_batch_interview(self, command_id: str, interviews: List[Dict]) -> bool:
        """Execute a batch of interview actions in a single env step.

        Args:
            interviews: list of {"agent_id": int, "prompt": str} dicts.
        """
        try:
            actions = {}
            agent_prompts = {}

            for interview in interviews:
                agent_id = interview.get("agent_id")
                prompt = interview.get("prompt", "")

                try:
                    agent = self.agent_graph.get_agent(agent_id)
                    actions[agent] = ManualAction(
                        action_type=ActionType.INTERVIEW,
                        action_args={"prompt": prompt}
                    )
                    agent_prompts[agent_id] = prompt
                except Exception as e:
                    print(f"  Warning: could not get agent {agent_id}: {e}")

            if not actions:
                self.send_response(command_id, "failed", error="No valid agents found")
                return False

            await self.env.step(actions)

            results = {}
            for agent_id in agent_prompts.keys():
                results[agent_id] = self._get_interview_result(agent_id)

            self.send_response(command_id, "completed", result={
                "interviews_count": len(results),
                "results": results
            })
            print(f"  Batch interview complete: {len(results)} agents")
            return True

        except Exception as e:
            error_msg = str(e)
            print(f"  Batch interview failed: {error_msg}")
            self.send_response(command_id, "failed", error=error_msg)
            return False

    def _get_interview_result(self, agent_id: int) -> Dict[str, Any]:
        """Query the organisation DB for the most recent interview response for *agent_id*."""
        db_path = os.path.join(self.simulation_dir, "organisation_simulation.db")

        result: Dict[str, Any] = {
            "agent_id": agent_id,
            "response": None,
            "timestamp": None,
        }

        if not os.path.exists(db_path):
            return result

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT user_id, info, created_at
                FROM trace
                WHERE action = ? AND user_id = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (ActionType.INTERVIEW.value, agent_id),
            )

            row = cursor.fetchone()
            if row:
                _user_id, info_json, created_at = row
                try:
                    info = json.loads(info_json) if info_json else {}
                    result["response"] = info.get("response", info)
                    result["timestamp"] = created_at
                except json.JSONDecodeError:
                    result["response"] = info_json

            conn.close()

        except Exception as e:
            print(f"  Failed to read interview result: {e}")

        return result

    async def process_commands(self) -> bool:
        """Process one pending IPC command.

        Returns:
            True to keep running, False to exit.
        """
        command = self.poll_command()
        if not command:
            return True

        command_id = command.get("command_id")
        command_type = command.get("command_type")
        args = command.get("args", {})

        print(f"\nIPC command received: {command_type}, id={command_id}")

        if command_type == CommandType.INTERVIEW:
            await self.handle_interview(
                command_id,
                args.get("agent_id", 0),
                args.get("prompt", ""),
            )
            return True

        elif command_type == CommandType.BATCH_INTERVIEW:
            await self.handle_batch_interview(
                command_id,
                args.get("interviews", []),
            )
            return True

        elif command_type == CommandType.CLOSE_ENV:
            print("Close-environment command received.")
            self.send_response(command_id, "completed", result={"message": "Environment closing."})
            return False

        else:
            self.send_response(command_id, "failed", error=f"Unknown command type: {command_type}")
            return True


def _json_profiles_to_twitter_csv(json_path: str) -> str:
    """Convert an organisation_profiles.json file to a Twitter-compatible CSV.

    The OASIS ``generate_twitter_agent_graph`` function requires a CSV with
    columns: ``user_id``, ``name``, ``username``, ``user_char``, ``description``.

    This helper reads the JSON profile list (produced by
    ``OasisProfileGenerator._save_reddit_json``) and writes an equivalent CSV
    to a temporary file alongside the source JSON.

    Returns:
        Absolute path to the temporary CSV file.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)

    # Write next to the source JSON so it is easy to inspect
    csv_path = os.path.splitext(json_path)[0] + "_twitter_compat.csv"

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'name', 'username', 'user_char', 'description'])

        for idx, p in enumerate(profiles):
            user_id = p.get("user_id", idx)
            name = p.get("name", f"Agent {user_id}")
            username = p.get("username", f"agent_{user_id:04d}")
            bio = p.get("bio", "")
            persona = p.get("persona", "")

            # user_char: full persona for LLM system prompt
            if persona and persona != bio:
                user_char = f"{bio} {persona}".strip()
            else:
                user_char = bio

            user_char = user_char.replace('\n', ' ').replace('\r', ' ')
            description = bio.replace('\n', ' ').replace('\r', ' ')

            writer.writerow([user_id, name, username, user_char, description])

    return csv_path


class OrganisationSimulationRunner:
    """Runs an organisation environment simulation using the Twitter OASIS platform."""

    # Twitter action types available to organisation agents.
    # The organisation framing is embedded in each agent's persona — the
    # underlying OASIS platform actions (post, like, repost …) map naturally
    # to organisational communication (announce, acknowledge, circulate …).
    AVAILABLE_ACTIONS = [
        ActionType.CREATE_POST,
        ActionType.LIKE_POST,
        ActionType.REPOST,
        ActionType.FOLLOW,
        ActionType.DO_NOTHING,
        ActionType.QUOTE_POST,
    ]

    def __init__(self, config_path: str, wait_for_commands: bool = True):
        """
        Args:
            config_path:        Path to simulation_config.json.
            wait_for_commands:  If True, enter command-wait mode after the
                                simulation loop completes (default True).
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.simulation_dir = os.path.dirname(config_path)
        self.wait_for_commands = wait_for_commands
        self.env = None
        self.agent_graph = None
        self.ipc_handler = None
        self._temp_csv_path: Optional[str] = None

    def _load_config(self) -> Dict[str, Any]:
        """Load simulation_config.json."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_profile_path(self) -> str:
        """Return the path to organisation_profiles.json."""
        return os.path.join(self.simulation_dir, "organisation_profiles.json")

    def _get_db_path(self) -> str:
        return os.path.join(self.simulation_dir, "organisation_simulation.db")

    def _get_twitter_csv_path(self) -> str:
        """Convert organisation_profiles.json to a Twitter-compatible CSV and return its path."""
        json_path = self._get_profile_path()
        csv_path = _json_profiles_to_twitter_csv(json_path)
        self._temp_csv_path = csv_path
        print(f"Profile JSON converted to Twitter CSV: {csv_path}")
        return csv_path

    def _create_model(self):
        """Create the LLM model from environment variables set in .env."""
        llm_api_key = os.environ.get("LLM_API_KEY", "")
        llm_base_url = os.environ.get("LLM_BASE_URL", "")
        llm_model = os.environ.get("LLM_MODEL_NAME", "")

        if not llm_model:
            llm_model = self.config.get("llm_model", "gpt-4o-mini")

        if llm_api_key:
            os.environ["OPENAI_API_KEY"] = llm_api_key

        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "Missing API key. Set LLM_API_KEY in the project root .env file."
            )

        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url

        base_url_preview = (llm_base_url[:40] + "...") if llm_base_url else "default"
        print(f"LLM config: model={llm_model}, base_url={base_url_preview}")

        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )

    def _get_active_agents_for_round(
        self,
        env,
        current_hour: int,
        round_num: int,
    ) -> List:
        """Select agents to activate for this round based on time and config."""
        time_config = self.config.get("time_config", {})
        agent_configs = self.config.get("agent_configs", [])

        base_min = time_config.get("agents_per_hour_min", 5)
        base_max = time_config.get("agents_per_hour_max", 20)

        peak_hours = time_config.get("peak_hours", [9, 10, 11, 14, 15, 16])
        off_peak_hours = time_config.get("off_peak_hours", list(range(0, 8)) + list(range(19, 24)))

        if current_hour in peak_hours:
            multiplier = time_config.get("peak_activity_multiplier", 1.5)
        elif current_hour in off_peak_hours:
            multiplier = time_config.get("off_peak_activity_multiplier", 0.1)
        else:
            multiplier = 1.0

        target_count = int(random.uniform(base_min, base_max) * multiplier)

        candidates = []
        for cfg in agent_configs:
            agent_id = cfg.get("agent_id", 0)
            active_hours = cfg.get("active_hours", list(range(8, 18)))
            activity_level = cfg.get("activity_level", 0.5)

            if current_hour not in active_hours:
                continue

            if random.random() < activity_level:
                candidates.append(agent_id)

        selected_ids = (
            random.sample(candidates, min(target_count, len(candidates)))
            if candidates
            else []
        )

        active_agents = []
        for agent_id in selected_ids:
            try:
                agent = env.agent_graph.get_agent(agent_id)
                active_agents.append((agent_id, agent))
            except Exception:
                pass

        return active_agents

    async def run(self, max_rounds: int = None):
        """Run the organisation simulation.

        Args:
            max_rounds: Optional cap on the number of simulation rounds.
        """
        print("=" * 60)
        print("OASIS Organisation Simulation")
        print(f"Config: {self.config_path}")
        print(f"Simulation ID: {self.config.get('simulation_id', 'unknown')}")
        print(f"Command-wait mode: {'enabled' if self.wait_for_commands else 'disabled'}")
        print("=" * 60)

        time_config = self.config.get("time_config", {})
        total_hours = time_config.get("total_simulation_hours", 80)
        minutes_per_round = time_config.get("minutes_per_round", 60)
        total_rounds = (total_hours * 60) // minutes_per_round

        if max_rounds is not None and max_rounds > 0:
            original_rounds = total_rounds
            total_rounds = min(total_rounds, max_rounds)
            if total_rounds < original_rounds:
                print(f"\nRound count capped: {original_rounds} -> {total_rounds} (max_rounds={max_rounds})")

        print(f"\nSimulation parameters:")
        print(f"  - Total simulated hours : {total_hours}")
        print(f"  - Minutes per round     : {minutes_per_round}")
        print(f"  - Total rounds          : {total_rounds}")
        if max_rounds:
            print(f"  - Max rounds cap        : {max_rounds}")
        print(f"  - Agents                : {len(self.config.get('agent_configs', []))}")

        print("\nInitialising LLM model...")
        model = self._create_model()

        print("Loading agent profiles...")
        json_path = self._get_profile_path()
        if not os.path.exists(json_path):
            print(f"Error: profile file not found: {json_path}")
            return

        # generate_twitter_agent_graph requires CSV format
        profile_csv_path = self._get_twitter_csv_path()

        self.agent_graph = await generate_twitter_agent_graph(
            profile_path=profile_csv_path,
            model=model,
            available_actions=self.AVAILABLE_ACTIONS,
        )

        db_path = self._get_db_path()
        if os.path.exists(db_path):
            os.remove(db_path)
            print(f"Removed stale database: {db_path}")

        print("Creating OASIS environment (Twitter platform)...")
        self.env = oasis.make(
            agent_graph=self.agent_graph,
            platform=oasis.DefaultPlatformType.TWITTER,
            database_path=db_path,
            semaphore=30,
        )

        await self.env.reset()
        print("Environment initialised.\n")

        self.ipc_handler = IPCHandler(self.simulation_dir, self.env, self.agent_graph)
        self.ipc_handler.update_status("running")

        # Post initial seed announcements (e.g. management posts a policy update)
        event_config = self.config.get("event_config", {})
        initial_posts = event_config.get("initial_posts", [])

        if initial_posts:
            print(f"Posting initial announcements ({len(initial_posts)} seed posts)...")
            initial_actions = {}
            for post in initial_posts:
                agent_id = post.get("poster_agent_id", 0)
                content = post.get("content", "")
                try:
                    agent = self.env.agent_graph.get_agent(agent_id)
                    initial_actions[agent] = ManualAction(
                        action_type=ActionType.CREATE_POST,
                        action_args={"content": content},
                    )
                except Exception as e:
                    print(f"  Warning: could not create seed post for agent {agent_id}: {e}")

            if initial_actions:
                await self.env.step(initial_actions)
                print(f"  Posted {len(initial_actions)} seed announcements.")

        # Main simulation loop
        print("\nStarting simulation loop...")
        start_time = datetime.now()

        for round_num in range(total_rounds):
            simulated_minutes = round_num * minutes_per_round
            simulated_hour = (simulated_minutes // 60) % 24
            simulated_day = simulated_minutes // (60 * 24) + 1

            active_agents = self._get_active_agents_for_round(
                self.env, simulated_hour, round_num
            )

            if not active_agents:
                continue

            actions = {agent: LLMAction() for _, agent in active_agents}
            await self.env.step(actions)

            if (round_num + 1) % 10 == 0 or round_num == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = (round_num + 1) / total_rounds * 100
                print(
                    f"  [Day {simulated_day}, {simulated_hour:02d}:00] "
                    f"Round {round_num + 1}/{total_rounds} ({progress:.1f}%) "
                    f"- {len(active_agents)} agents active "
                    f"- elapsed: {elapsed:.1f}s"
                )

        total_elapsed = (datetime.now() - start_time).total_seconds()
        print(f"\nSimulation loop complete.")
        print(f"  - Elapsed: {total_elapsed:.1f}s")
        print(f"  - Database: {db_path}")

        if self.wait_for_commands:
            print("\n" + "=" * 60)
            print("Entering command-wait mode — environment remains alive.")
            print("Supported commands: interview, batch_interview, close_env")
            print("=" * 60)

            self.ipc_handler.update_status("alive")

            try:
                while not _shutdown_event.is_set():
                    should_continue = await self.ipc_handler.process_commands()
                    if not should_continue:
                        break
                    try:
                        await asyncio.wait_for(_shutdown_event.wait(), timeout=0.5)
                        break
                    except asyncio.TimeoutError:
                        pass
            except KeyboardInterrupt:
                print("\nInterrupt received.")
            except asyncio.CancelledError:
                print("\nTask cancelled.")
            except Exception as e:
                print(f"\nCommand processing error: {e}")

            print("\nClosing environment...")

        self.ipc_handler.update_status("stopped")
        await self.env.close()

        print("Environment closed.")
        print("=" * 60)


async def main():
    parser = argparse.ArgumentParser(description='OASIS Organisation Simulation')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to simulation_config.json',
    )
    parser.add_argument(
        '--max-rounds',
        type=int,
        default=None,
        help='Optional cap on the number of simulation rounds',
    )
    parser.add_argument(
        '--no-wait',
        action='store_true',
        default=False,
        help='Close the environment immediately after the simulation loop (skip command-wait mode)',
    )

    args = parser.parse_args()

    global _shutdown_event
    _shutdown_event = asyncio.Event()

    if not os.path.exists(args.config):
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)

    simulation_dir = os.path.dirname(args.config) or "."
    setup_oasis_logging(os.path.join(simulation_dir, "log"))

    runner = OrganisationSimulationRunner(
        config_path=args.config,
        wait_for_commands=not args.no_wait,
    )
    await runner.run(max_rounds=args.max_rounds)


def setup_signal_handlers():
    """Register SIGTERM/SIGINT handlers for graceful shutdown."""

    def signal_handler(signum, frame):
        global _cleanup_done
        sig_name = "SIGTERM" if signum == signal.SIGTERM else "SIGINT"
        print(f"\nReceived {sig_name}, shutting down...")
        if not _cleanup_done:
            _cleanup_done = True
            if _shutdown_event:
                _shutdown_event.set()
        else:
            print("Forcing exit...")
            sys.exit(1)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    setup_signal_handlers()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nSimulation interrupted.")
    except SystemExit:
        pass
    finally:
        print("Simulation process exited.")
