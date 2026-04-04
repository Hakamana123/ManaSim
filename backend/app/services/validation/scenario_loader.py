"""
Scenario loader for the ManaSim validation layer.

Loads and validates benchmark scenario JSON files from the
``validation/scenarios/`` directory (or any directory the caller points to).

Usage
-----
::

    from backend.app.services.validation.scenario_loader import ScenarioLoader

    loader = ScenarioLoader()                          # uses default scenarios dir
    scenario = loader.load("microsoft_transformation") # loads .json by stem
    all_scenarios = loader.load_all()                  # loads every .json in dir

The loader validates each file against the ``BenchmarkScenario`` Pydantic model
on load, raising ``ScenarioValidationError`` if the file does not conform.

File resolution
---------------
``ScenarioLoader`` resolves the scenarios directory in this order:

1. Explicit ``scenarios_dir`` argument passed to ``__init__``.
2. The ``MANASIM_SCENARIOS_DIR`` environment variable.
3. Default: ``<repo-root>/validation/scenarios/`` — located by walking up from
   this file's directory until a directory containing ``validation/scenarios/``
   is found, or falling back to ``<cwd>/validation/scenarios/``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import ValidationError

from .schemas import BenchmarkScenario

logger = logging.getLogger("manasim.validation.scenario_loader")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ScenarioNotFoundError(FileNotFoundError):
    """Raised when a requested scenario file cannot be located."""


class ScenarioValidationError(ValueError):
    """Raised when a scenario file fails Pydantic model validation."""


# ---------------------------------------------------------------------------
# Default directory resolution
# ---------------------------------------------------------------------------

def _find_default_scenarios_dir() -> Path:
    """Walk up from this file to find <repo-root>/validation/scenarios/.

    Falls back to <cwd>/validation/scenarios/ if the directory is not found
    within 6 levels of the module's location.
    """
    env_dir = os.environ.get("MANASIM_SCENARIOS_DIR")
    if env_dir:
        p = Path(env_dir)
        if p.is_dir():
            return p
        logger.warning(
            "MANASIM_SCENARIOS_DIR='%s' is not a directory; ignoring.", env_dir
        )

    # Walk upward looking for validation/scenarios/ relative to this file
    current = Path(__file__).resolve()
    for _ in range(6):
        current = current.parent
        candidate = current / "validation" / "scenarios"
        if candidate.is_dir():
            return candidate

    # Ultimate fallback
    fallback = Path.cwd() / "validation" / "scenarios"
    logger.debug(
        "Could not locate validation/scenarios/ via directory walk; "
        "using fallback path: %s", fallback
    )
    return fallback


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ScenarioLoader:
    """Loads and validates benchmark scenario JSON files.

    Parameters
    ----------
    scenarios_dir:
        Directory containing scenario ``.json`` files.  If ``None``, the
        loader resolves the directory automatically (see module docstring).
    """

    def __init__(self, scenarios_dir: Optional[Path | str] = None) -> None:
        if scenarios_dir is not None:
            self._dir = Path(scenarios_dir)
        else:
            self._dir = _find_default_scenarios_dir()

        logger.debug("ScenarioLoader initialised with dir: %s", self._dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scenarios_dir(self) -> Path:
        """The resolved scenarios directory."""
        return self._dir

    def load(self, scenario_id: str) -> BenchmarkScenario:
        """Load and validate a single scenario by ID.

        Parameters
        ----------
        scenario_id:
            File stem (without ``.json``), e.g. ``"microsoft_transformation"``.
            The loader appends ``.json`` and looks in ``scenarios_dir``.

        Returns
        -------
        BenchmarkScenario
            Validated scenario model.

        Raises
        ------
        ScenarioNotFoundError
            If the file does not exist.
        ScenarioValidationError
            If the JSON does not conform to ``BenchmarkScenario``.
        """
        path = self._resolve_path(scenario_id)
        return self._load_file(path)

    def load_all(self) -> Dict[str, BenchmarkScenario]:
        """Load every ``.json`` file in ``scenarios_dir``.

        Returns
        -------
        dict
            Maps file stem → ``BenchmarkScenario``.  Files that fail
            validation are skipped with a warning logged rather than
            raising an exception — this allows the rest of the suite to
            load even if one file is malformed.
        """
        if not self._dir.is_dir():
            logger.warning(
                "Scenarios directory does not exist: %s — returning empty dict.",
                self._dir,
            )
            return {}

        results: Dict[str, BenchmarkScenario] = {}
        for path in sorted(self._dir.glob("*.json")):
            stem = path.stem
            try:
                results[stem] = self._load_file(path)
                logger.debug("Loaded scenario: %s", stem)
            except ScenarioValidationError as exc:
                logger.warning("Skipping malformed scenario '%s': %s", stem, exc)

        logger.info(
            "load_all: loaded %d scenario(s) from %s", len(results), self._dir
        )
        return results

    def list_available(self) -> List[str]:
        """Return the stems of all ``.json`` files in ``scenarios_dir``.

        Does not validate the files — use :meth:`load` or :meth:`load_all`
        for validated access.
        """
        if not self._dir.is_dir():
            return []
        return sorted(p.stem for p in self._dir.glob("*.json"))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, scenario_id: str) -> Path:
        """Return the absolute path for a scenario ID, raising if absent."""
        # Strip accidental .json suffix so callers can pass either form
        stem = scenario_id.removesuffix(".json")
        path = self._dir / f"{stem}.json"
        if not path.exists():
            raise ScenarioNotFoundError(
                f"Scenario '{scenario_id}' not found at: {path}"
            )
        return path

    def _load_file(self, path: Path) -> BenchmarkScenario:
        """Read, parse, and validate one scenario file.

        Raises
        ------
        ScenarioValidationError
            Wraps any JSON decode error or Pydantic validation error,
            preserving the original message for diagnostics.
        """
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ScenarioValidationError(
                f"Cannot read scenario file '{path}': {exc}"
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ScenarioValidationError(
                f"Invalid JSON in scenario file '{path}': {exc}"
            ) from exc

        try:
            scenario = BenchmarkScenario.model_validate(data)
        except ValidationError as exc:
            raise ScenarioValidationError(
                f"Scenario file '{path}' failed schema validation:\n{exc}"
            ) from exc

        logger.debug(
            "Loaded scenario '%s' (domain=%s, rubric=%s, segments=%d)",
            scenario.scenario_id,
            scenario.domain,
            scenario.rubric,
            len(scenario.segment_outcomes),
        )
        return scenario
