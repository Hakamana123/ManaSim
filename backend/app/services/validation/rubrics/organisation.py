"""
Organisation rubric for the ManaSim validation layer.

Scores simulation output against documented real-world outcomes for
organisation-domain scenarios (e.g. cultural transformations, technology
rollouts, strategic change programmes).

Three theoretical frameworks are used as scoring lenses:

──────────────────────────────────────────────────────────────────────
Framework 1 — Kotter's 8-Step Change Model
──────────────────────────────────────────────────────────────────────
John Kotter's widely-validated model describes organisational change as
an ordered sequence of eight stages:

  1.  Create Urgency            — build a compelling case for change
  2.  Build a Guiding Coalition — assemble a credible change-leadership group
  3.  Form a Strategic Vision   — articulate a clear, inspiring direction
  4.  Enlist a Volunteer Army   — mobilise a broad base of supporters
  5.  Enable Action             — remove structural and cultural barriers
  6.  Generate Short-term Wins  — produce and celebrate early visible results
  7.  Sustain Acceleration      — press on; don't declare victory too early
  8.  Institute Change          — anchor new behaviours in culture/processes

Scoring dimension: kotter_stage_alignment
Asks: did agents enact behaviours consistent with the Kotter stages
documented in the benchmark scenario, in the correct sequence?

──────────────────────────────────────────────────────────────────────
Framework 2 — Prosci ADKAR Model
──────────────────────────────────────────────────────────────────────
ADKAR describes individual change readiness as five sequential states:

  Awareness     — of the need for change
  Desire        — to support and participate in the change
  Knowledge     — of how to change (skills, processes, behaviours)
  Ability       — to demonstrate the new skills/behaviours
  Reinforcement — mechanisms that make the change stick

ADKAR is applied per segment — each human segment has a documented
ADKAR state at key scenario timepoints that simulation behaviour should
reproduce.

Scoring dimension: adkar_state_fidelity
Asks: did per-segment simulated ADKAR states match the documented
states at the timepoints recorded in the benchmark scenario?

──────────────────────────────────────────────────────────────────────
Framework 3 — Cynefin Complexity Framework
──────────────────────────────────────────────────────────────────────
Cynefin (Dave Snowden) classifies problem domains by the relationship
between cause and effect:

  Clear       (formerly Simple)  — cause-effect obvious; best practices apply
  Complicated                    — cause-effect requires expert analysis; good practices
  Complex                        — cause-effect only visible in retrospect; emergent
  Chaotic                        — no perceptible cause-effect; act first, then sense
  Disorder                       — unclear which domain applies

Each benchmark scenario documents the Cynefin domain that characterised
the real-world change event.  The simulation should reproduce dynamics
consistent with that domain — not force a best-practice (Clear) framing
onto an inherently Complex or Chaotic situation.

Scoring dimension: cynefin_domain_fit
Asks: did the overall simulation dynamics (agent decision patterns,
information flow, emergence of norms) reflect the documented Cynefin
domain?

──────────────────────────────────────────────────────────────────────
Dimension weights
──────────────────────────────────────────────────────────────────────
  kotter_stage_alignment  0.35  — highest: stage behaviours are the most
                                  directly verifiable against the scenario
  adkar_state_fidelity    0.40  — highest: per-segment ADKAR state is the
                                  core individual-change validity claim
  cynefin_domain_fit      0.25  — lower: domain-level dynamics are a macro
                                  observation harder to pin to individuals
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
# Framework reference constants
# ---------------------------------------------------------------------------

_KOTTER_STEPS: List[str] = [
    "1. Create Urgency",
    "2. Build a Guiding Coalition",
    "3. Form a Strategic Vision",
    "4. Enlist a Volunteer Army",
    "5. Enable Action (remove barriers)",
    "6. Generate Short-term Wins",
    "7. Sustain Acceleration",
    "8. Institute Change",
]

_ADKAR_STATES: List[str] = [
    "Awareness",
    "Desire",
    "Knowledge",
    "Ability",
    "Reinforcement",
]

_CYNEFIN_DOMAINS: List[str] = [
    "Clear (best practices apply; cause-effect obvious)",
    "Complicated (good practices; expert analysis required)",
    "Complex (emergent practices; probe-sense-respond)",
    "Chaotic (novel practices; act-sense-respond)",
    "Disorder (unclear which domain applies)",
]


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

_KOTTER_STAGE_ALIGNMENT = RubricDimension(
    name="kotter_stage_alignment",
    display_name="Kotter Stage Alignment",
    description=(
        "Measures whether simulated agents enacted behaviours consistent with "
        "the Kotter change stages documented for this segment in the benchmark "
        "scenario. "
        "Leadership agents should demonstrate urgency-creation, coalition-building, "
        "and vision-communication in the documented sequence. "
        "Middle-management agents should show enabling behaviour (removing barriers, "
        "celebrating wins). "
        "Individual-contributor agents should progress from resistance through "
        "enlistment to change adoption as the simulation advances. "
        "Kotter's model is sequential — skipping stages or reversing them is "
        "a fidelity failure."
    ),
    weight=0.35,
    scoring_guidance=(
        "Score 1.0 if the simulation excerpt shows this segment enacting "
        "behaviours that match the Kotter stages documented in the scenario "
        "in the correct sequence (e.g. urgency expressed before coalition "
        "forming; short-term wins celebrated before declaring full adoption). "
        "Score 0.5 if relevant Kotter behaviours are present but out of "
        "sequence, or if only a subset of the documented stages are represented. "
        "Score 0.0 if the excerpt shows no Kotter-stage-grounded behaviour "
        "(e.g. agents jump directly from awareness to full adoption with no "
        "intermediate stages, or leadership never communicates a vision)."
    ),
)

_ADKAR_STATE_FIDELITY = RubricDimension(
    name="adkar_state_fidelity",
    display_name="ADKAR State Fidelity",
    description=(
        "Measures whether each segment's simulated ADKAR state matched the "
        "documented ADKAR state at key scenario timepoints. "
        "ADKAR is a sequential individual-change model: agents must pass through "
        "Awareness before Desire, Desire before Knowledge, and so on. "
        "A segment's documented ADKAR state represents where that group was "
        "in the change journey at a specific point in time — the simulation "
        "should reproduce the same state, with the same barriers and enablers, "
        "rather than advancing too quickly or stalling without cause."
    ),
    weight=0.40,
    scoring_guidance=(
        "Score 1.0 if simulated agent behaviour for this segment matches the "
        "documented ADKAR state — e.g. if the segment is documented at "
        "'Desire', agents express motivation but lack specific knowledge of "
        "how to change, and do not yet demonstrate new behaviours. "
        "Score 0.5 if the state is approximately correct (one step ahead or "
        "behind the documented state) or if state is correct but without the "
        "documented barriers/enablers. "
        "Score 0.0 if agents are multiple ADKAR steps away from the documented "
        "state (e.g. demonstrating full Ability when the scenario documents "
        "them still at Awareness), or if ADKAR progression is absent entirely."
    ),
)

_CYNEFIN_DOMAIN_FIT = RubricDimension(
    name="cynefin_domain_fit",
    display_name="Cynefin Domain Fit",
    description=(
        "Measures whether the overall simulation dynamics — agent decision "
        "patterns, information flow, norm emergence, and response to surprises "
        "— reflected the Cynefin domain documented for the scenario. "
        "A Clear-domain scenario should produce consistent best-practice "
        "adoption. A Complex-domain scenario should produce emergent, "
        "non-linear dynamics where no single agent orchestrates the outcome. "
        "A Chaotic scenario should show rapid, reactive behaviour with little "
        "deliberation. Forcing best-practice (Clear) dynamics onto a Complex "
        "or Chaotic scenario is a domain-fit failure."
    ),
    weight=0.25,
    scoring_guidance=(
        "Score 1.0 if the simulation excerpt shows macro-level dynamics "
        "consistent with the scenario's documented Cynefin domain: "
        "  Clear → uniform adoption of stated best practices, low variance; "
        "  Complicated → expert agents lead analysis, others defer; "
        "  Complex → emergent norms, unexpected coalitions, non-linear outcomes; "
        "  Chaotic → rapid reactive posts, high disagreement, quick pivots. "
        "Score 0.5 if the dynamics are partially consistent — e.g. a Complex "
        "scenario shows some emergence but also over-scripted leadership. "
        "Score 0.0 if the simulation dynamics contradict the documented domain "
        "— e.g. a Complex scenario produces perfectly ordered, sequential "
        "adoption with no emergent behaviour."
    ),
)


# ---------------------------------------------------------------------------
# Rubric class
# ---------------------------------------------------------------------------

class OrganisationRubric(BaseRubric):
    """Organisation-domain scoring rubric using Kotter, ADKAR, and Cynefin.

    Applicable to scenarios involving cultural transformation, strategic
    change programmes, technology adoption at organisational scale, and
    any change event where individual change readiness and organisational
    complexity dynamics are both at stake.
    """

    NAME = "organisation"

    # ------------------------------------------------------------------
    # BaseRubric interface
    # ------------------------------------------------------------------

    def definition(self) -> RubricDefinition:
        return RubricDefinition(
            name=self.NAME,
            display_name="Organisation Rubric (Kotter · ADKAR · Cynefin)",
            applicable_domains=["organisation"],
            frameworks=["Kotter 8-Step", "Prosci ADKAR", "Cynefin"],
            dimensions=[
                _KOTTER_STAGE_ALIGNMENT,
                _ADKAR_STATE_FIDELITY,
                _CYNEFIN_DOMAIN_FIT,
            ],
            description=(
                "Scores organisation-domain simulations using three frameworks: "
                "Kotter stage alignment (did agents enact the correct change-stage "
                "behaviours in sequence?), ADKAR state fidelity (did per-segment "
                "individual change states match documented timepoints?), and "
                "Cynefin domain fit (did macro simulation dynamics reflect the "
                "scenario's complexity domain?). "
                "Weights: 0.35 / 0.40 / 0.25."
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

        # Extract the scenario's documented Cynefin domain from metadata
        cynefin_domain = scenario.metadata.get("cynefin_domain", "not specified")

        other_segments = [
            s.segment_name
            for s in scenario.segment_outcomes
            if s.segment_name != segment_outcome.segment_name
        ]
        other_segments_str = ", ".join(other_segments) if other_segments else "none"

        dim_keys = ", ".join(f'"{d.name}": float' for d in dims)

        return f"""You are scoring an organisation-domain simulation against a documented real-world organisational change case using three frameworks: Kotter's 8-Step Change Model, Prosci ADKAR, and Cynefin.

## Framework Reference

### Kotter's 8-Step Change Model (sequential — order matters)
{chr(10).join(f"  {s}" for s in _KOTTER_STEPS)}

### Prosci ADKAR (sequential individual change states)
  States in order: {" → ".join(_ADKAR_STATES)}
  Each segment has a documented ADKAR state at a specific scenario timepoint.
  Agents should exhibit behaviour consistent with that state — not the states
  before it (already passed) or after it (not yet reached).

### Cynefin Complexity Domains
{chr(10).join(f"  {d}" for d in _CYNEFIN_DOMAINS)}
  Documented domain for this scenario: {cynefin_domain}

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

### Framework Mappings for This Segment
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
actions that align with documented expected behaviours, correct Kotter stage
enactment, correct ADKAR state expression, or correct Cynefin-domain dynamics.

When identifying missed_behaviours: list documented expected behaviours, Kotter
stages, or ADKAR states that are entirely absent from the simulation excerpt.

When identifying unexpected_observed: list simulation behaviours that match
documented unexpected behaviours — e.g. agents skipping Kotter stages, exhibiting
ADKAR states they should not have reached yet, or imposing Clear-domain
best-practice logic onto a Complex-domain scenario.

For explanation: write 3–5 sentences covering overall fidelity across all three
frameworks, the most significant alignment, and the most significant gap.

Respond with a single JSON object using exactly these keys:
  "dimension_scores"    : {{{dim_keys}}}
  "matched_behaviours"  : [strings]
  "missed_behaviours"   : [strings]
  "unexpected_observed" : [strings]
  "explanation"         : string
"""
