"""
Validation layer — Pydantic schemas.

These models represent the full data contract for the validation system:

  BenchmarkScenario   — a documented real-world event used as ground truth
  SegmentOutcome      — the expected behaviour of one human segment in a scenario
  RubricDefinition    — metadata describing a scoring rubric and its dimensions
  SegmentScore        — per-segment scoring result produced by the scorer
  ValidationResult    — top-level result returned after scoring a simulation run
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Scenario-side models (ground truth)
# ---------------------------------------------------------------------------


class FrameworkMapping(BaseModel):
    """Maps a segment's documented behaviour to one or more framework constructs.

    Examples
    --------
    Kotter mapping::

        FrameworkMapping(
            framework="kotter",
            stage_or_state="create_urgency",
            notes="Senior leadership communicated the burning platform."
        )

    ADKAR mapping::

        FrameworkMapping(
            framework="adkar",
            stage_or_state="awareness",
            notes="Employees were informed via all-hands meeting."
        )

    AITTTS mapping::

        FrameworkMapping(
            framework="aittts",
            stage_or_state="task_completion",
            notes="Teachers completed mandatory LMS onboarding."
        )
    """

    framework: str = Field(
        ...,
        description="Framework identifier, e.g. 'kotter', 'adkar', 'cynefin', 'aittts'.",
    )
    stage_or_state: str = Field(
        ...,
        description="The specific stage, state, or domain within the framework.",
    )
    notes: Optional[str] = Field(
        None,
        description="Free-text annotation explaining the mapping.",
    )


class SegmentOutcome(BaseModel):
    """Documented real-world outcome for one human segment within a scenario.

    This is the ground truth that the scorer compares simulation output against.
    """

    segment_name: str = Field(
        ...,
        description="Name of the human segment (must match segment names in the simulation).",
    )
    description: str = Field(
        ...,
        description="Brief description of who this segment is.",
    )
    expected_behaviours: List[str] = Field(
        default_factory=list,
        description=(
            "List of behaviours documented in the real-world case for this segment. "
            "Each item is a short, concrete behavioural statement."
        ),
    )
    unexpected_behaviours: List[str] = Field(
        default_factory=list,
        description=(
            "Behaviours that were notably absent or suppressed for this segment. "
            "The scorer treats these as negative evidence if observed in simulation."
        ),
    )
    measurable_outcomes: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Quantitative or qualitative outcomes documented for this segment. "
            "Keys are metric names; values are the documented result. "
            "Example: {'adoption_rate': 0.72, 'engagement_level': 'high'}."
        ),
    )
    framework_mappings: List[FrameworkMapping] = Field(
        default_factory=list,
        description="How this segment's behaviour maps to domain frameworks.",
    )
    source_notes: Optional[str] = Field(
        None,
        description="Citation or reference for the documented outcomes.",
    )


class BenchmarkScenario(BaseModel):
    """A documented real-world event used as ground truth for validation.

    Scenario files are stored as JSON under validation/scenarios/.
    They are pure data — no code or scoring logic.
    """

    scenario_id: str = Field(
        ...,
        description="Unique identifier for this scenario, e.g. 'microsoft_transformation'.",
    )
    title: str = Field(
        ...,
        description="Human-readable title.",
    )
    domain: str = Field(
        ...,
        description="Domain type: 'organisation' or 'education'.",
    )
    rubric: str = Field(
        ...,
        description=(
            "Rubric identifier to use when scoring against this scenario. "
            "Must match a registered rubric name, e.g. 'organisation' or 'education'."
        ),
    )
    description: str = Field(
        ...,
        description="Narrative description of the real-world event.",
    )
    artefact: str = Field(
        ...,
        description=(
            "The specific policy, change, or intervention being simulated. "
            "Example: 'Growth mindset policy introduction under Satya Nadella'."
        ),
    )
    timeframe: Optional[str] = Field(
        None,
        description="Real-world timeframe, e.g. '2014–2018'.",
    )
    sources: List[str] = Field(
        default_factory=list,
        description="List of publicly documented sources (papers, reports, case studies).",
    )
    segment_outcomes: List[SegmentOutcome] = Field(
        default_factory=list,
        description="Per-segment documented outcomes that constitute the ground truth.",
    )
    scenario_level_outcomes: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Aggregate outcomes documented at the scenario level. "
            "Example: {'market_cap_growth': '10x', 'employee_engagement_increase': '0.15'}."
        ),
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata (framework-specific context, tags, etc.).",
    )


# ---------------------------------------------------------------------------
# Rubric-side models (scoring framework metadata)
# ---------------------------------------------------------------------------


class RubricDimension(BaseModel):
    """One scoring dimension within a rubric."""

    name: str = Field(
        ...,
        description="Short identifier for this dimension, e.g. 'procedural_compliance'.",
    )
    display_name: str = Field(
        ...,
        description="Human-readable name shown in reports.",
    )
    description: str = Field(
        ...,
        description="What this dimension measures.",
    )
    weight: float = Field(
        1.0,
        ge=0.0,
        description=(
            "Relative weight of this dimension in the overall rubric score. "
            "Weights across all dimensions in a rubric need not sum to 1.0 — "
            "the scorer normalises them."
        ),
    )
    scoring_guidance: str = Field(
        ...,
        description=(
            "Guidance passed to the LLM scorer explaining how to assign a score "
            "for this dimension. Should be concrete and unambiguous."
        ),
    )


class RubricDefinition(BaseModel):
    """Metadata describing a scoring rubric and its dimensions.

    Returned by BaseRubric.definition() so the scorer can discover
    dimensions without inspecting rubric internals.
    """

    name: str = Field(
        ...,
        description="Rubric identifier, e.g. 'organisation' or 'education'.",
    )
    display_name: str = Field(
        ...,
        description="Human-readable rubric name.",
    )
    applicable_domains: List[str] = Field(
        ...,
        description="Domain(s) this rubric applies to, e.g. ['organisation'].",
    )
    frameworks: List[str] = Field(
        default_factory=list,
        description="Theoretical frameworks this rubric draws on.",
    )
    dimensions: List[RubricDimension] = Field(
        ...,
        description="The scoring dimensions that make up this rubric.",
    )
    description: str = Field(
        "",
        description="Narrative description of the rubric's purpose and approach.",
    )


# ---------------------------------------------------------------------------
# Result-side models (scorer output)
# ---------------------------------------------------------------------------


class SegmentScore(BaseModel):
    """Per-segment scoring result produced by the scorer."""

    segment_name: str = Field(
        ...,
        description="Name of the human segment that was scored.",
    )
    fidelity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Overall fidelity of this segment's simulated behaviour relative to "
            "the documented real-world outcome. 1.0 = perfect match."
        ),
    )
    framework_scores: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Per-rubric-dimension scores. Keys are dimension names; "
            "values are 0.0–1.0 scores for that dimension."
        ),
    )
    explanation: str = Field(
        ...,
        description=(
            "Narrative explanation of why this score was given. "
            "Generated by the LLM scorer."
        ),
    )
    matched_behaviours: List[str] = Field(
        default_factory=list,
        description=(
            "Simulation behaviours that aligned with documented expected behaviours "
            "for this segment."
        ),
    )
    missed_behaviours: List[str] = Field(
        default_factory=list,
        description=(
            "Expected behaviours from the benchmark scenario that were not observed "
            "in the simulation output for this segment."
        ),
    )
    unexpected_observed: List[str] = Field(
        default_factory=list,
        description=(
            "Behaviours observed in simulation that were listed as unexpected "
            "in the benchmark scenario (negative evidence)."
        ),
    )


class ValidationResult(BaseModel):
    """Top-level result returned after scoring a simulation run.

    Produced by ``ValidationScorer.score()`` and can be serialised to JSON
    for storage, display, or downstream analysis.
    """

    scenario_id: str = Field(
        ...,
        description="ID of the benchmark scenario used for validation.",
    )
    simulation_id: str = Field(
        ...,
        description="ID of the simulation run that was scored.",
    )
    domain: str = Field(
        ...,
        description="Domain of the simulation: 'organisation' or 'education'.",
    )
    rubric_used: str = Field(
        ...,
        description="Name of the rubric applied during scoring.",
    )
    overall_fidelity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description=(
            "Aggregate fidelity score across all segments. "
            "Computed as the mean of per-segment fidelity scores."
        ),
    )
    per_segment_scores: List[SegmentScore] = Field(
        default_factory=list,
        description="Detailed scoring breakdown for each segment.",
    )
    summary: str = Field(
        ...,
        description=(
            "Narrative summary of the validation result, written by the LLM scorer. "
            "Covers overall fidelity, standout matches, and notable gaps."
        ),
    )
    scored_at: str = Field(
        ...,
        description="ISO-8601 timestamp when scoring was performed.",
    )
    scorer_model: str = Field(
        "",
        description="LLM model used by the scorer.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional metadata attached during scoring.",
    )
