"""
Education rubric for the ManaSim validation layer.

Scores simulation output against documented real-world outcomes for
education-domain scenarios (e.g. EdTech adoption, AI tool rollouts,
hybrid teaching transitions).

Theoretical framework: AITTTS
------------------------------
AITTTS = AI-Teacher Teaching Task Spectrum.

The spectrum maps 9 teaching task types onto a continuum from AI-led
to Teacher-led, organised into three overlapping zones:

  Zone 1 — Instructional (AI-led end)
      Procedural Tasks
      Knowledge Recall Tasks
      Knowledge Explanation Tasks

  Zone 2 — Cognitive-Analytical (shared AI–teacher)
      Application Tasks
      Analytical Tasks
      Evaluative Tasks

  Zone 3 — Human Centric (Teacher-led end)
      Creative Tasks
      Inspiration Tasks
      Pastoral Care Tasks

The spectrum operates within four constraint layers that bound the
entire space:

  Technical Constraints   — outer boundary (what the technology can do)
  Ethical Constraints     — lower boundary (what is permissible)
  Contextual Constraints  — lower-right boundary (institutional/cultural fit)
  Contestable overlap     — Zone 1/2 and Zone 2/3 boundaries are not hard
                            lines; tasks in overlap regions require
                            negotiation between AI capability and teacher
                            professional judgment.

Scoring dimensions
------------------
Four dimensions are derived directly from the AITTTS structure:

  task_spectrum_positioning (0.30)
      Did teacher agents correctly position AI involvement at the
      appropriate point on the spectrum — AI-led for Zone 1 tasks,
      teacher-led for Zone 3 tasks?

  zone_boundary_fidelity (0.30)
      Did agents respect zone boundaries — particularly resisting
      inappropriate delegation of Zone 3 (Creative, Inspiration,
      Pastoral Care) tasks to AI?

  constraint_adherence (0.20)
      Did agent behaviours reflect awareness of all four constraint
      layers — Technical, Ethical, Contextual, and the contestable
      overlap regions?

  shared_zone_negotiation (0.20)
      In Zone 2 (Application, Analytical, Evaluative), did agents
      exhibit appropriate negotiation between AI support and teacher
      professional judgment rather than defaulting entirely to either?
"""

from __future__ import annotations

from typing import List, Optional

from ..schemas import (
    BenchmarkScenario,
    RubricDefinition,
    RubricDimension,
    SegmentOutcome,
)
from .base import BaseRubric

# ---------------------------------------------------------------------------
# AITTTS reference — the 9 task types by zone
# ---------------------------------------------------------------------------

_ZONE_1_TASKS = ["Procedural Tasks", "Knowledge Recall Tasks", "Knowledge Explanation Tasks"]
_ZONE_2_TASKS = ["Application Tasks", "Analytical Tasks", "Evaluative Tasks"]
_ZONE_3_TASKS = ["Creative Tasks", "Inspiration Tasks", "Pastoral Care Tasks"]
_ALL_TASK_TYPES = _ZONE_1_TASKS + _ZONE_2_TASKS + _ZONE_3_TASKS
_CONSTRAINT_LAYERS = ["Technical", "Ethical", "Contextual", "Contestable overlap"]


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

_TASK_SPECTRUM_POSITIONING = RubricDimension(
    name="task_spectrum_positioning",
    display_name="Task Spectrum Positioning",
    description=(
        "Measures whether simulated teacher agents correctly positioned AI "
        "involvement at the appropriate point on the AITTTS spectrum for each "
        "task type encountered. "
        "Zone 1 tasks (Procedural, Knowledge Recall, Knowledge Explanation) "
        "should be delegated to or heavily supported by AI. "
        "Zone 3 tasks (Creative, Inspiration, Pastoral Care) should be "
        "retained by the teacher with minimal or no AI involvement. "
        "Zone 2 tasks (Application, Analytical, Evaluative) may involve either, "
        "depending on context."
    ),
    weight=0.30,
    scoring_guidance=(
        "Score 1.0 if the simulation excerpt shows teacher agents consistently "
        "routing Zone 1 tasks to AI support and Zone 3 tasks to teacher-led "
        "delivery, with nuanced handling of Zone 2 tasks. "
        "Score 0.5 if positioning is correct for one zone but incorrect or "
        "absent for another (e.g. teachers correctly use AI for procedural tasks "
        "but also delegate pastoral care to it). "
        "Score 0.0 if agents show no awareness of task-type positioning — e.g. "
        "uniform AI delegation across all task types, or uniform rejection of AI "
        "regardless of task type."
    ),
)

_ZONE_BOUNDARY_FIDELITY = RubricDimension(
    name="zone_boundary_fidelity",
    display_name="Zone Boundary Fidelity",
    description=(
        "Measures whether simulated agents respected zone boundaries — "
        "specifically whether teacher agents resisted inappropriate delegation "
        "of Zone 3 tasks (Creative, Inspiration, Pastoral Care) to AI, and "
        "whether students and administrators showed appropriate expectations "
        "about which tasks would be AI-supported versus teacher-led. "
        "Note: Zone 1/2 and Zone 2/3 boundaries are intentionally soft (contestable "
        "overlap); the rubric rewards principled boundary navigation, not rigid "
        "rule-following."
    ),
    weight=0.30,
    scoring_guidance=(
        "Score 1.0 if teacher agents explicitly or implicitly retain Zone 3 tasks "
        "and show principled reasoning when navigating the contestable Zone 2 "
        "overlap (e.g. using AI to draft an analytical framework but applying "
        "their own judgment to the evaluative conclusion). "
        "Score 0.5 if boundary respect is partial — some Zone 3 tasks are "
        "retained but others are inappropriately delegated, or agents navigate "
        "Zone 2 without visible reasoning. "
        "Score 0.0 if agents treat all task types identically with respect to "
        "AI involvement, or if pastoral/inspirational tasks are fully delegated "
        "to AI with no teacher override."
    ),
)

_CONSTRAINT_ADHERENCE = RubricDimension(
    name="constraint_adherence",
    display_name="Constraint Adherence",
    description=(
        "Measures whether agent behaviours reflected awareness of the four "
        "AITTTS constraint layers that bound the entire spectrum: "
        "(1) Technical Constraints — acknowledging what the AI tool can and "
        "cannot do; "
        "(2) Ethical Constraints — flagging data privacy, bias, or fairness "
        "concerns when delegating to AI; "
        "(3) Contextual Constraints — adapting AI use to institutional policy, "
        "student demographics, or cultural context; "
        "(4) Contestable overlap — recognising that boundary tasks require "
        "active professional judgment rather than default delegation."
    ),
    weight=0.20,
    scoring_guidance=(
        "Score 1.0 if the simulation excerpt shows agents actively referencing "
        "or navigating at least three of the four constraint layers (e.g. a "
        "teacher noting an AI tool's technical limitations, raising a data "
        "privacy concern, and checking institutional policy before adoption). "
        "Score 0.5 if agents demonstrate awareness of one or two constraint "
        "layers but ignore others. "
        "Score 0.0 if agents show no constraint awareness — adopting or "
        "rejecting AI use without referencing any technical, ethical, or "
        "contextual considerations."
    ),
)

_SHARED_ZONE_NEGOTIATION = RubricDimension(
    name="shared_zone_negotiation",
    display_name="Shared Zone Negotiation (Zone 2)",
    description=(
        "Measures the quality of negotiation between AI support and teacher "
        "professional judgment in Zone 2 (Cognitive-Analytical) tasks: "
        "Application, Analytical, and Evaluative. "
        "These tasks sit in the shared AI–teacher space; high fidelity requires "
        "agents to show a deliberate decision process — weighing AI capability "
        "against professional judgment rather than defaulting to either pole."
    ),
    weight=0.20,
    scoring_guidance=(
        "Score 1.0 if the simulation shows teacher agents explicitly negotiating "
        "Zone 2 tasks — e.g. using AI to generate analytical options then "
        "applying personal judgment to select or modify them, or consulting peers "
        "about appropriate AI involvement for a given application task. "
        "Score 0.5 if Zone 2 tasks appear in the simulation but are handled "
        "without visible negotiation (AI always wins, or teacher always wins, "
        "regardless of task). "
        "Score 0.0 if Zone 2 tasks are absent from the simulation output, or if "
        "all Zone 2 decisions are made identically with no differentiation by "
        "task type or context."
    ),
)


# ---------------------------------------------------------------------------
# Rubric class
# ---------------------------------------------------------------------------

class EducationRubric(BaseRubric):
    """Education-domain scoring rubric based on the AITTTS framework.

    Applicable to scenarios involving AI tool adoption in teaching contexts,
    LMS rollouts that include AI features, hybrid or remote teaching
    transitions, and related educational technology change events where
    the AI–teacher role boundary is at stake.

    The 9 AITTTS task types are the ground truth for all scoring:
      Zone 1 (AI-led):   Procedural, Knowledge Recall, Knowledge Explanation
      Zone 2 (Shared):   Application, Analytical, Evaluative
      Zone 3 (Teacher):  Creative, Inspiration, Pastoral Care
    """

    NAME = "education"

    # ------------------------------------------------------------------
    # BaseRubric interface
    # ------------------------------------------------------------------

    def definition(self) -> RubricDefinition:
        return RubricDefinition(
            name=self.NAME,
            display_name="Education Rubric (AITTTS — AI-Teacher Teaching Task Spectrum)",
            applicable_domains=["education"],
            frameworks=["AITTTS"],
            dimensions=[
                _TASK_SPECTRUM_POSITIONING,
                _ZONE_BOUNDARY_FIDELITY,
                _CONSTRAINT_ADHERENCE,
                _SHARED_ZONE_NEGOTIATION,
            ],
            description=(
                "Scores education-domain simulations against documented outcomes "
                "using four dimensions derived from the AITTTS spectrum: task "
                "spectrum positioning (correct AI involvement by zone), zone "
                "boundary fidelity (resisting inappropriate AI delegation of "
                "Zone 3 tasks), constraint adherence (Technical, Ethical, "
                "Contextual, and contestable-overlap awareness), and shared zone "
                "negotiation (AI–teacher judgment balance in Zone 2 tasks). "
                "Weights: 0.30 / 0.30 / 0.20 / 0.20."
            ),
        )

    def build_scoring_prompt(
        self,
        segment_outcome: SegmentOutcome,
        simulation_excerpt: str,
        scenario: BenchmarkScenario,
        *,
        dimension_names: Optional[List[str]] = None,
    ) -> str:
        dims = self.definition().dimensions
        if dimension_names:
            dims = [d for d in dims if d.name in dimension_names]

        dim_block = "\n\n".join(
            f"### Dimension: {d.display_name}  (key: '{d.name}', weight: {d.weight})\n"
            f"{d.description}\n\n"
            f"Scoring guidance:\n{d.scoring_guidance}"
            for d in dims
        )

        expected_block = (
            "\n".join(f"  - {b}" for b in segment_outcome.expected_behaviours)
            if segment_outcome.expected_behaviours
            else "  (none documented)"
        )

        unexpected_block = (
            "\n".join(f"  - {b}" for b in segment_outcome.unexpected_behaviours)
            if segment_outcome.unexpected_behaviours
            else "  (none documented)"
        )

        if segment_outcome.measurable_outcomes:
            outcomes_block = "\n".join(
                f"  - {k}: {v}"
                for k, v in segment_outcome.measurable_outcomes.items()
            )
        else:
            outcomes_block = "  (none documented)"

        if segment_outcome.framework_mappings:
            mappings_block = "\n".join(
                f"  - [{m.framework.upper()}] {m.stage_or_state}"
                + (f": {m.notes}" if m.notes else "")
                for m in segment_outcome.framework_mappings
            )
        else:
            mappings_block = "  (none documented)"

        other_segments = [
            s.segment_name
            for s in scenario.segment_outcomes
            if s.segment_name != segment_outcome.segment_name
        ]
        other_segments_str = ", ".join(other_segments) if other_segments else "none"

        dim_keys = ", ".join(f'"{d.name}": float' for d in dims)

        return f"""You are scoring an education-domain simulation against a documented real-world EdTech adoption case using the AITTTS (AI-Teacher Teaching Task Spectrum) framework.

## AITTTS Reference

The spectrum maps 9 teaching task types across three zones:

  Zone 1 — AI-led       : {", ".join(_ZONE_1_TASKS)}
  Zone 2 — Shared       : {", ".join(_ZONE_2_TASKS)}
  Zone 3 — Teacher-led  : {", ".join(_ZONE_3_TASKS)}

Constraint layers (bound the entire spectrum):
  {" | ".join(_CONSTRAINT_LAYERS)}

Zone 1/2 and Zone 2/3 boundaries are contestable overlap regions —
tasks in those regions require principled negotiation between AI capability
and teacher professional judgment.

## Scenario Context

Title      : {scenario.title}
Domain     : {scenario.domain}
Artefact   : {scenario.artefact}
Timeframe  : {scenario.timeframe or "not specified"}
Description: {scenario.description}
Other segments in this scenario: {other_segments_str}

## Segment Being Scored

Segment    : {segment_outcome.segment_name}
Description: {segment_outcome.description}

### Documented Expected Behaviours (ground truth)
{expected_block}

### Documented Unexpected Behaviours (negative evidence if observed in simulation)
{unexpected_block}

### Documented Measurable Outcomes
{outcomes_block}

### AITTTS Framework Mappings for This Segment
{mappings_block}

## Simulation Output

The following is a summary of what the simulated agents in the
"{segment_outcome.segment_name}" segment actually did during the simulation:

---
{simulation_excerpt}
---

## Scoring Dimensions

Score each dimension on a scale of 0.0 (no match) to 1.0 (perfect match).

{dim_block}

## Instructions

Compare the simulation output to the documented ground truth for the
"{segment_outcome.segment_name}" segment.

When identifying matched_behaviours: quote or closely paraphrase simulation
actions that align with documented expected behaviours or correct AITTTS
zone positioning.

When identifying missed_behaviours: list expected behaviours or AITTTS-
grounded behaviours that are entirely absent from the simulation excerpt.

When identifying unexpected_observed: list simulation behaviours that match
documented unexpected behaviours (i.e. behaviours that should NOT have
occurred, such as inappropriate AI delegation of Zone 3 tasks).

For explanation: write 3–5 sentences covering overall fidelity, the most
significant match, and the most significant gap.

Respond with a single JSON object using exactly these keys:
  "dimension_scores"    : {{{dim_keys}}}
  "matched_behaviours"  : [strings]
  "missed_behaviours"   : [strings]
  "unexpected_observed" : [strings]
  "explanation"         : string
"""
