"""
Abstract rubric interface for the ManaSim validation layer.

Every concrete rubric (e.g. EducationRubric, OrganisationRubric) must:

1.  Subclass ``BaseRubric``.
2.  Define a class-level ``NAME`` string — this is the key used to look up
    the rubric in ``RUBRIC_REGISTRY`` and must match the ``rubric`` field in
    every BenchmarkScenario that uses it.
3.  Implement ``definition()`` — returns a ``RubricDefinition`` describing
    the rubric's dimensions, weights, and LLM scoring guidance.
4.  Implement ``build_scoring_prompt()`` — returns the full prompt that the
    scorer sends to the LLM when evaluating one segment.

Registration
------------
Subclasses are registered automatically on class creation via the
``RubricMeta`` metaclass.  No manual registration step is needed — simply
defining a ``BaseRubric`` subclass with a valid ``NAME`` is sufficient.

Usage example
-------------
::

    from backend.app.services.validation.rubrics import RUBRIC_REGISTRY

    rubric = RUBRIC_REGISTRY["organisation"]()
    defn   = rubric.definition()
    prompt = rubric.build_scoring_prompt(
        segment_outcome=outcome,
        simulation_excerpt=text,
        scenario=scenario,
    )
"""

from __future__ import annotations

import abc
import logging
from typing import Dict, List, Optional, Type

from ..schemas import (
    BenchmarkScenario,
    RubricDefinition,
    SegmentOutcome,
)

logger = logging.getLogger("manasim.validation.rubric")

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Maps rubric NAME → rubric class.
#: Populated automatically by RubricMeta when a subclass is defined.
RUBRIC_REGISTRY: Dict[str, Type["BaseRubric"]] = {}


# ---------------------------------------------------------------------------
# Metaclass — handles auto-registration
# ---------------------------------------------------------------------------

class RubricMeta(abc.ABCMeta):
    """Metaclass that registers every concrete BaseRubric subclass by NAME."""

    def __new__(
        mcs,
        name: str,
        bases: tuple,
        namespace: dict,
        **kwargs,
    ) -> "RubricMeta":
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip the abstract base itself and any intermediate abstract classes
        rubric_name: Optional[str] = namespace.get("NAME")
        if rubric_name and not getattr(cls, "__abstractmethods__", None):
            if rubric_name in RUBRIC_REGISTRY:
                logger.warning(
                    "Rubric '%s' is already registered — overwriting with %s.",
                    rubric_name, cls.__qualname__,
                )
            RUBRIC_REGISTRY[rubric_name] = cls
            logger.debug("Registered rubric: '%s' → %s", rubric_name, cls.__qualname__)

        return cls


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseRubric(metaclass=RubricMeta):
    """Abstract base class for all ManaSim validation rubrics.

    Subclasses must set ``NAME`` and implement ``definition()`` and
    ``build_scoring_prompt()``.

    Rubric authors should also document:
    - Which theoretical frameworks the rubric draws on.
    - What each dimension measures and how scores should be interpreted.
    - Any domain-specific pre-processing the rubric expects the scorer to
      perform before calling ``build_scoring_prompt()``.
    """

    #: Unique identifier for this rubric.  Must be set by every subclass.
    #: Used as the key in RUBRIC_REGISTRY and in BenchmarkScenario.rubric.
    NAME: Optional[str] = None

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def definition(self) -> RubricDefinition:
        """Return the full rubric definition including all scoring dimensions.

        Called by the scorer before building prompts so it can:
        - Validate that the scenario's framework mappings reference known
          dimensions.
        - Attach dimension weights to the aggregated fidelity score.
        - Surface dimension metadata in the ValidationResult.

        Returns
        -------
        RubricDefinition
            Complete metadata for this rubric, including all
            ``RubricDimension`` entries with weights and scoring guidance.
        """

    @abc.abstractmethod
    def build_scoring_prompt(
        self,
        segment_outcome: SegmentOutcome,
        simulation_excerpt: str,
        scenario: BenchmarkScenario,
        *,
        dimension_names: Optional[List[str]] = None,
    ) -> str:
        """Build the LLM prompt for scoring one segment against this rubric.

        The scorer calls this once per segment and sends the returned string
        to the LLM.  The LLM must respond with a JSON object conforming to
        the structure described in ``_RESPONSE_SCHEMA_HINT`` below.

        Parameters
        ----------
        segment_outcome:
            The ground-truth ``SegmentOutcome`` from the benchmark scenario.
        simulation_excerpt:
            A text excerpt summarising the simulated behaviour of this
            segment.  Typically drawn from the simulation action log,
            aggregated per segment by the scorer.
        scenario:
            The full ``BenchmarkScenario``, provided for context (artefact
            description, scenario-level outcomes, other segments, etc.).
        dimension_names:
            Optional subset of dimension names to score.  If None, score all
            dimensions defined in ``definition()``.

        Returns
        -------
        str
            A complete LLM prompt (system + user merged, or user-only —
            the scorer will inject a system message separately).
        """

    # ------------------------------------------------------------------
    # Concrete helpers available to all subclasses
    # ------------------------------------------------------------------

    @classmethod
    def get_name(cls) -> str:
        """Return the rubric's registered name, raising if not set."""
        if not cls.NAME:
            raise AttributeError(
                f"{cls.__qualname__} has no NAME attribute. "
                "Set NAME = '<rubric_id>' on the class."
            )
        return cls.NAME

    def dimension_weights(self) -> Dict[str, float]:
        """Return a mapping of dimension name → normalised weight (sums to 1.0).

        Convenience method so the scorer does not need to normalise itself.
        """
        dims = self.definition().dimensions
        if not dims:
            return {}

        total = sum(d.weight for d in dims)
        if total == 0.0:
            equal = 1.0 / len(dims)
            return {d.name: equal for d in dims}

        return {d.name: d.weight / total for d in dims}

    def aggregate_score(self, framework_scores: Dict[str, float]) -> float:
        """Compute a weighted aggregate fidelity score from per-dimension scores.

        Parameters
        ----------
        framework_scores:
            Dict mapping dimension name → raw score (0.0–1.0).

        Returns
        -------
        float
            Weighted aggregate in [0.0, 1.0].  Dimensions absent from
            ``framework_scores`` are treated as 0.0.
        """
        weights = self.dimension_weights()
        if not weights:
            return 0.0

        total = sum(
            framework_scores.get(dim, 0.0) * w
            for dim, w in weights.items()
        )
        return round(max(0.0, min(1.0, total)), 4)

    @staticmethod
    def system_prompt() -> str:
        """Return the system message injected before every scoring prompt.

        The scorer sends this as the ``system`` role message and appends the
        result of ``build_scoring_prompt()`` as the ``user`` role message.

        Subclasses may override this to provide rubric-specific framing, but
        the JSON response contract described here must be preserved.
        """
        return (
            "You are a simulation validation expert for ManaSim. "
            "Your task is to evaluate how faithfully a social simulation "
            "reproduced the documented real-world behaviour of a specific "
            "human segment. "
            "You will be given: (1) the documented ground-truth behaviours "
            "from a benchmark scenario, and (2) a text excerpt summarising "
            "what the simulated agents in that segment actually did. "
            "\n\n"
            "Respond with a single valid JSON object containing exactly these keys:\n"
            "  dimension_scores : object — one key per scoring dimension, "
            "each value a float in [0.0, 1.0]\n"
            "  matched_behaviours  : array of strings — simulated behaviours "
            "that match documented expected behaviours\n"
            "  missed_behaviours   : array of strings — expected behaviours "
            "that were NOT observed in the simulation\n"
            "  unexpected_observed : array of strings — behaviours observed "
            "in simulation that were listed as unexpected in the scenario\n"
            "  explanation         : string — a concise paragraph (3–5 "
            "sentences) explaining the score\n"
            "\n"
            "Do not include markdown, prose, or any text outside the JSON object."
        )


# ---------------------------------------------------------------------------
# Response schema hint (for documentation and scorer validation)
# ---------------------------------------------------------------------------

#: Expected JSON structure in the LLM scorer response.
#: The scorer uses this as a reference when parsing and validating the reply.
_RESPONSE_SCHEMA_HINT: dict = {
    "dimension_scores": {
        "<dimension_name>": "<float 0.0–1.0>",
        "...": "one key per dimension in the rubric",
    },
    "matched_behaviours": ["<string>", "..."],
    "missed_behaviours": ["<string>", "..."],
    "unexpected_observed": ["<string>", "..."],
    "explanation": "<string>",
}
