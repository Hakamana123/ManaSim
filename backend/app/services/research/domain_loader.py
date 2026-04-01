"""
Domain config loader.

Discovers and validates domain configuration files from the domains/ directory
at the project root. Domain configs are plain JSON files that match the
DomainConfig schema defined in schemas.py.

Usage:
    loader = DomainLoader()
    config = loader.load("education")        # load by domain_id
    all_ids = loader.list_domain_ids()       # ["education", "organisation", ...]
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

from pydantic import ValidationError

from .schemas import DomainConfig

# The domains/ directory lives two levels above backend/app/services/research/
# i.e.  <project_root>/domains/
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
_DOMAINS_DIR = os.path.join(_PROJECT_ROOT, "domains")


class DomainNotFoundError(FileNotFoundError):
    """Raised when no config file exists for the requested domain_id."""


class DomainValidationError(ValueError):
    """Raised when a domain config file fails Pydantic validation."""


class DomainLoader:
    """
    Loads DomainConfig objects from JSON files in the domains/ directory.

    File convention: one JSON file per domain, filename = domain_id + ".json"
    (e.g.  domains/education.json  →  domain_id "education").

    The loader caches parsed configs for the lifetime of the instance so
    repeated calls within a single research run don't re-read the disk.
    """

    def __init__(self, domains_dir: Optional[str] = None) -> None:
        self._domains_dir = domains_dir or _DOMAINS_DIR
        self._cache: Dict[str, DomainConfig] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, domain_id: str) -> DomainConfig:
        """
        Load and validate the config for *domain_id*.

        Args:
            domain_id: The domain identifier, e.g. "education".
                       Case-insensitive; always normalised to lower-case.

        Returns:
            A validated DomainConfig instance.

        Raises:
            DomainNotFoundError: if no matching .json file exists.
            DomainValidationError: if the file fails schema validation.
        """
        domain_id = domain_id.strip().lower()

        if domain_id in self._cache:
            return self._cache[domain_id]

        path = self._resolve_path(domain_id)
        raw = self._read_json(path)
        config = self._parse(raw, path)

        self._cache[domain_id] = config
        return config

    def list_domain_ids(self) -> List[str]:
        """
        Return the domain_id for every .json file found in domains/.

        The id is derived from the filename stem (lower-cased), not from the
        domain_id field inside the file — so the directory listing is always
        authoritative even if a file has not yet been parsed.
        """
        if not os.path.isdir(self._domains_dir):
            return []

        ids: List[str] = []
        for fname in sorted(os.listdir(self._domains_dir)):
            if fname.endswith(".json"):
                ids.append(fname[:-5].lower())
        return ids

    def list_all(self) -> List[DomainConfig]:
        """
        Load and return configs for all domains found in the directory.

        Configs that fail validation are skipped with a warning printed to
        stderr rather than raising, so one bad file doesn't block the rest.
        """
        configs: List[DomainConfig] = []
        for domain_id in self.list_domain_ids():
            try:
                configs.append(self.load(domain_id))
            except (DomainNotFoundError, DomainValidationError) as exc:
                # Non-fatal: log and continue
                import sys
                print(
                    f"[DomainLoader] WARNING: skipping '{domain_id}': {exc}",
                    file=sys.stderr,
                )
        return configs

    def reload(self, domain_id: str) -> DomainConfig:
        """Force a cache-bypass reload (useful in tests or hot-reload flows)."""
        domain_id = domain_id.strip().lower()
        self._cache.pop(domain_id, None)
        return self.load(domain_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_path(self, domain_id: str) -> str:
        """Return the expected filesystem path for a domain_id."""
        path = os.path.join(self._domains_dir, f"{domain_id}.json")
        if not os.path.isfile(path):
            available = self.list_domain_ids()
            hint = (
                f"  Available domains: {available}"
                if available
                else f"  No domain configs found in '{self._domains_dir}'"
            )
            raise DomainNotFoundError(
                f"No domain config found for '{domain_id}'.\n{hint}"
            )
        return path

    def _read_json(self, path: str) -> dict:
        """Read and JSON-parse a file, raising a clear error on bad JSON."""
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            raise DomainValidationError(
                f"Invalid JSON in '{path}': {exc}"
            ) from exc

    def _parse(self, raw: dict, path: str) -> DomainConfig:
        """Validate raw dict against DomainConfig schema."""
        try:
            return DomainConfig.model_validate(raw)
        except ValidationError as exc:
            # Re-raise with the file path for easy debugging
            raise DomainValidationError(
                f"Schema validation failed for '{path}':\n{exc}"
            ) from exc
