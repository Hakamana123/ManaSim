# ManaSim Validation Layer

The validation layer scores simulation output against documented real-world outcomes,
producing a `ValidationResult` with per-segment fidelity scores and a narrative summary.

---

## Architecture overview

```
validation/
  scenarios/              ← benchmark scenario JSON files (ground truth)
    microsoft_transformation.json
    covid_edtech_adoption.json

backend/app/services/validation/
  __init__.py
  schemas.py              ← Pydantic models for the entire data contract
  scenario_loader.py      ← loads and validates scenario JSON files
  scorer.py               ← LLM-powered scoring engine
  rubrics/
    __init__.py           ← triggers auto-registration of all built-in rubrics
    base.py               ← BaseRubric abstract class + RUBRIC_REGISTRY
    education.py          ← AITTTS rubric (NAME="education")
    organisation.py       ← Kotter / ADKAR / Cynefin rubric (NAME="organisation")
```

---

## Scoring pipeline

```
ScenarioLoader.load(scenario_id)
        │
        ▼
ValidationScorer.score(scenario, simulation_id)
        │
        ├── _load_rubric(scenario.rubric)        ← resolves from RUBRIC_REGISTRY
        ├── _resolve_env_type(sim_dir, scenario)  ← reads simulation_config.json
        ├── _load_action_log(sim_dir, env_type)   ← reads actions.jsonl
        ├── _load_profiles(sim_dir, env_type)     ← reads profiles.json
        │
        │   for each SegmentOutcome in scenario:
        ├── _build_segment_excerpt(...)           ← keyword matching → text excerpt
        └── _score_segment_with_llm(...)          ← LLM call → SegmentScore
                │
                └── _generate_summary(...)        ← second LLM call → narrative
                        │
                        ▼
                ValidationResult
```

The scorer never does string matching against simulation output — all judgements
are delegated to the LLM using structured prompts built by the rubric.

---

## Schemas

All models are in `backend/app/services/validation/schemas.py`.

| Model | Purpose |
|---|---|
| `BenchmarkScenario` | Ground-truth container for one real-world event |
| `SegmentOutcome` | Expected/unexpected behaviours + framework mappings for one segment |
| `FrameworkMapping` | Maps a behaviour to a specific stage/state in a framework |
| `RubricDefinition` | Metadata for a rubric (dimensions, weights, applicable domains) |
| `RubricDimension` | One scoring dimension with weight and LLM scoring guidance |
| `SegmentScore` | Per-segment scoring result (fidelity score + dimension breakdown) |
| `ValidationResult` | Top-level result returned by `ValidationScorer.score()` |

---

## Adding a new rubric

1. Create `backend/app/services/validation/rubrics/<domain>.py`.
2. Subclass `BaseRubric` and set `NAME = "<domain>"`.
3. Implement `definition()` — return a `RubricDefinition` with all dimensions.
4. Implement `build_scoring_prompt()` — return the LLM prompt string.
5. Import the module in `rubrics/__init__.py` so `RubricMeta` registers it.

```python
# rubrics/healthcare.py
from .base import BaseRubric

class HealthcareRubric(BaseRubric):
    NAME = "healthcare"

    def definition(self) -> RubricDefinition:
        ...

    def build_scoring_prompt(self, segment_outcome, simulation_excerpt, scenario, *, dimension_names=None) -> str:
        ...
```

```python
# rubrics/__init__.py  — add one line:
from . import healthcare   # noqa: F401  registers NAME="healthcare"
```

No other changes are needed. The scorer resolves rubrics from `RUBRIC_REGISTRY`
by matching `scenario.rubric` to `NAME`.

---

## Adding a new benchmark scenario

Create a JSON file in `validation/scenarios/` with the following required fields:

```jsonc
{
  "scenario_id": "unique_snake_case_id",
  "title": "Human-readable title",
  "domain": "organisation",          // must match a domain the rubric handles
  "rubric": "organisation",          // must match a registered rubric NAME
  "description": "...",
  "artefact": "The specific intervention being simulated",
  "timeframe": "YYYY–YYYY",          // optional
  "sources": ["citation 1", "..."],  // optional
  "segment_outcomes": [
    {
      "segment_name": "Senior Leadership",
      "description": "...",
      "expected_behaviours": ["...", "..."],
      "unexpected_behaviours": ["...", "..."],  // optional
      "measurable_outcomes": { "key": "value" },  // optional
      "framework_mappings": [
        {
          "framework": "kotter",
          "stage_or_state": "1. Create Urgency",
          "notes": "..."
        }
      ]
    }
  ],
  "scenario_level_outcomes": {},  // optional aggregate metrics
  "metadata": {}                  // optional, e.g. cynefin_domain for organisation rubric
}
```

The file is validated against `BenchmarkScenario` on load. The `ScenarioLoader`
raises `ScenarioValidationError` if any required field is missing or the wrong type.

### Organisation scenarios: required metadata

The `organisation` rubric reads `scenario.metadata["cynefin_domain"]` when building
scoring prompts. Always include this key:

```json
"metadata": {
  "cynefin_domain": "Complex (emergent practices; probe-sense-respond)"
}
```

Valid values mirror the `_CYNEFIN_DOMAINS` list in `rubrics/organisation.py`.

---

## Running the scorer

```python
from backend.app.services.validation.scenario_loader import ScenarioLoader
from backend.app.services.validation.scorer import ValidationScorer

loader = ScenarioLoader()
scenario = loader.load("microsoft_transformation")

scorer = ValidationScorer(
    api_key="sk-...",
    model_name="claude-opus-4-6",  # or any OpenAI-compatible model
    simulation_data_dir="/path/to/simulation/data",
)

result = scorer.score(scenario, simulation_id="sim_20240101_abc123")

print(result.overall_fidelity_score)   # float 0.0–1.0
print(result.summary)                  # narrative paragraph
for seg in result.per_segment_scores:
    print(seg.segment_name, seg.fidelity_score, seg.framework_scores)
```

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `MANASIM_SCENARIOS_DIR` | Override scenarios directory path | `<repo-root>/validation/scenarios/` |

---

## Rubric dimension weights

### Organisation rubric (`organisation`)

| Dimension | Key | Weight |
|---|---|---|
| Kotter Stage Alignment | `kotter_stage_alignment` | 0.35 |
| ADKAR State Fidelity | `adkar_state_fidelity` | 0.40 |
| Cynefin Domain Fit | `cynefin_domain_fit` | 0.25 |

### Education rubric (`education`)

| Dimension | Key | Weight |
|---|---|---|
| Task Spectrum Positioning | `task_spectrum_positioning` | 0.30 |
| Zone Boundary Fidelity | `zone_boundary_fidelity` | 0.30 |
| Constraint Adherence | `constraint_adherence` | 0.20 |
| Shared Zone Negotiation | `shared_zone_negotiation` | 0.20 |

Weights are normalised by `BaseRubric.dimension_weights()` before aggregation,
so they do not need to sum to 1.0 in the rubric definition.

---

## Segment matching

The scorer matches simulation agents to benchmark segments using keyword overlap.
Keywords are extracted from the segment's `name` and `description` fields and
matched against agent profile fields: `username`, `name`, `bio`, `persona`,
`profession`. Tokens shorter than 4 characters and common English stop-words are
excluded.

Because keyword matching is approximate, the LLM scorer handles ambiguity: it
receives the full segment description alongside the simulation excerpt and applies
its own contextual judgement when an agent's profile is only partially aligned.

---

## Output format

`ValidationResult` fields:

| Field | Type | Description |
|---|---|---|
| `scenario_id` | str | ID of the benchmark scenario |
| `simulation_id` | str | ID of the simulation run |
| `domain` | str | Domain of the scenario |
| `rubric_used` | str | Rubric name used for scoring |
| `scorer_model` | str | LLM model name used for scoring |
| `overall_fidelity_score` | float | Weighted average across all segments (0.0–1.0) |
| `per_segment_scores` | list[SegmentScore] | Per-segment breakdown |
| `summary` | str | Narrative paragraph summarising fidelity |
| `scored_at` | datetime | Timestamp of scoring run |

`SegmentScore` fields:

| Field | Type | Description |
|---|---|---|
| `segment_name` | str | Segment identifier |
| `fidelity_score` | float | Weighted aggregate for this segment (0.0–1.0) |
| `framework_scores` | dict[str, float] | Per-dimension scores |
| `matched_behaviours` | list[str] | Simulation behaviours that matched ground truth |
| `missed_behaviours` | list[str] | Expected behaviours absent from simulation |
| `unexpected_observed` | list[str] | Simulation behaviours listed as unexpected in scenario |
| `explanation` | str | 3–5 sentence LLM explanation |
