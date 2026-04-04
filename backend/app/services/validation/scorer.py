"""
Validation scorer for ManaSim.

Compares simulation output against a BenchmarkScenario and produces a
ValidationResult with per-segment fidelity scores.

Pipeline
--------
1.  Load the rubric for the scenario's domain from RUBRIC_REGISTRY.
2.  Locate the simulation directory and load two data sources:
      a. The action log — JSONL file at {sim_dir}/{env_type}/actions.jsonl
         Each non-meta line is an agent action record.
      b. The profile file — JSON array of agent profiles; used to link
         agent_id to persona text so the excerpt can be annotated.
3.  For each SegmentOutcome in the scenario, build a text excerpt:
      - Match profiles to the segment using keyword overlap between the
        segment name/description and each profile's username, name,
        bio, and persona.
      - Collect all action records from matched agents.
      - Format into a readable narrative capped at MAX_EXCERPT_CHARS.
4.  Send each (segment_outcome, excerpt) pair to the LLM using the
    rubric's system_prompt() + build_scoring_prompt().
5.  Parse the LLM JSON response into a SegmentScore.
6.  Compute the segment fidelity score as a weighted aggregate of the
    rubric's dimension scores.
7.  After all segments are scored, call the LLM once more to generate
    a narrative ValidationResult summary.
8.  Assemble and return the final ValidationResult.

Action log format (JSONL)
-------------------------
Meta-event lines have an "event_type" key and are skipped::

    {"event_type": "round_end", "round": 5, "simulated_hours": 2.5}
    {"event_type": "simulation_end", "total_rounds": 80, "total_actions": 312}

Agent action lines have no "event_type"::

    {
        "round": 12,
        "timestamp": "2024-01-01T10:30:00",
        "agent_id": 3,
        "agent_name": "teacher_reluctant_234",
        "action_type": "CREATE_POST",
        "action_args": {"content": "I'm struggling with this new system..."},
        "result": "post_id:87",
        "success": true
    }

Profile file format (JSON array)
---------------------------------
::

    [
        {
            "user_id": 3,
            "username": "teacher_reluctant_234",
            "name": "Jane Doe",
            "bio": "...",
            "persona": "...",
            "age": 45,
            "gender": "female",
            "mbti": "INFP",
            "country": "Australia",
            "profession": "Secondary school teacher"
        },
        ...
    ]

Note: ``source_entity_type`` is not persisted to the profile file.
Segment matching is therefore performed via keyword overlap between the
segment description and profile text fields.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from ..oasis_profile_generator import OasisAgentProfile  # noqa: F401 (import chain)
from ...config import Config
from .rubrics import RUBRIC_REGISTRY, BaseRubric
from .schemas import (
    BenchmarkScenario,
    SegmentOutcome,
    SegmentScore,
    ValidationResult,
)

logger = logging.getLogger("manasim.validation.scorer")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Maximum characters per segment excerpt sent to the LLM.
MAX_EXCERPT_CHARS = 4_000

#: Actions whose args contain a "content" key — used to build readable excerpts.
CONTENT_ACTIONS = frozenset({
    "CREATE_POST", "CREATE_COMMENT", "QUOTE_POST", "REPOST",
    "INTERVIEW",  # interview responses are the most semantically rich
})

#: Maximum LLM retries per segment.
MAX_LLM_RETRIES = 2

#: Candidate profile text fields searched during segment matching (in priority order).
_MATCH_FIELDS = ("username", "name", "bio", "persona", "profession")


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class ValidationScorer:
    """Scores a completed simulation run against a BenchmarkScenario.

    Parameters
    ----------
    api_key:
        LLM API key.  Defaults to ``Config.LLM_API_KEY``.
    base_url:
        LLM base URL.  Defaults to ``Config.LLM_BASE_URL``.
    model_name:
        LLM model name.  Defaults to ``Config.LLM_MODEL_NAME``.
    simulation_data_dir:
        Root directory that contains simulation sub-directories.
        Defaults to ``Config.OASIS_SIMULATION_DATA_DIR``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        simulation_data_dir: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or Config.LLM_API_KEY
        self._base_url = base_url or Config.LLM_BASE_URL
        self._model = model_name or Config.LLM_MODEL_NAME
        self._sim_root = (
            simulation_data_dir or Config.OASIS_SIMULATION_DATA_DIR
        )

        if not self._api_key:
            raise ValueError("LLM_API_KEY is not configured — cannot score.")

        self._client = OpenAI(api_key=self._api_key, base_url=self._base_url)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        scenario: BenchmarkScenario,
        simulation_id: str,
        *,
        simulation_dir: Optional[str] = None,
    ) -> ValidationResult:
        """Score a simulation run against *scenario*.

        Parameters
        ----------
        scenario:
            The benchmark scenario providing ground truth.
        simulation_id:
            ID of the simulation run to evaluate.
        simulation_dir:
            Explicit path to the simulation directory.  If omitted, resolved
            as ``{simulation_data_dir}/{simulation_id}``.

        Returns
        -------
        ValidationResult
            Full scoring result including per-segment scores, aggregate
            fidelity, and a narrative summary.
        """
        sim_dir = simulation_dir or os.path.join(self._sim_root, simulation_id)
        if not os.path.isdir(sim_dir):
            raise FileNotFoundError(
                f"Simulation directory not found: {sim_dir}"
            )

        logger.info(
            "Scoring simulation '%s' against scenario '%s' (rubric: %s)",
            simulation_id, scenario.scenario_id, scenario.rubric,
        )

        # 1. Resolve rubric
        rubric = self._load_rubric(scenario.rubric)

        # 2. Load simulation data
        all_actions, profiles = self._load_simulation_data(sim_dir, scenario)

        logger.info(
            "Loaded %d action records; %d agent profiles",
            len(all_actions), len(profiles),
        )

        # 3–6. Score each segment
        per_segment_scores: List[SegmentScore] = []
        for seg_outcome in scenario.segment_outcomes:
            excerpt = self._build_segment_excerpt(
                seg_outcome, all_actions, profiles
            )
            seg_score = self._score_segment_with_llm(
                rubric, seg_outcome, excerpt, scenario
            )
            per_segment_scores.append(seg_score)
            logger.info(
                "Segment '%s' fidelity=%.3f",
                seg_outcome.segment_name, seg_score.fidelity_score,
            )

        # 7. Aggregate fidelity
        overall = (
            sum(s.fidelity_score for s in per_segment_scores) / len(per_segment_scores)
            if per_segment_scores
            else 0.0
        )
        overall = round(overall, 4)

        # 8. Narrative summary
        summary = self._generate_summary(scenario, per_segment_scores, rubric)

        return ValidationResult(
            scenario_id=scenario.scenario_id,
            simulation_id=simulation_id,
            domain=scenario.domain,
            rubric_used=rubric.get_name(),
            overall_fidelity_score=overall,
            per_segment_scores=per_segment_scores,
            summary=summary,
            scored_at=datetime.now().isoformat(),
            scorer_model=self._model,
        )

    # ------------------------------------------------------------------
    # Rubric loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_rubric(rubric_name: str) -> BaseRubric:
        """Resolve *rubric_name* from RUBRIC_REGISTRY and return an instance."""
        cls = RUBRIC_REGISTRY.get(rubric_name)
        if cls is None:
            available = ", ".join(sorted(RUBRIC_REGISTRY.keys()))
            raise ValueError(
                f"Unknown rubric '{rubric_name}'. "
                f"Registered rubrics: {available or '(none)'}"
            )
        return cls()

    # ------------------------------------------------------------------
    # Simulation data loading
    # ------------------------------------------------------------------

    def _load_simulation_data(
        self,
        sim_dir: str,
        scenario: BenchmarkScenario,
    ) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
        """Return (all_action_records, profiles_by_user_id).

        Loads the action log for the environment type declared in the
        simulation config (or falls back to scanning all known env dirs).
        Loads the profile file matching the environment type.
        """
        env_type = self._resolve_env_type(sim_dir, scenario)
        all_actions = self._load_action_log(sim_dir, env_type)
        profiles = self._load_profiles(sim_dir, env_type)
        return all_actions, profiles

    @staticmethod
    def _resolve_env_type(sim_dir: str, scenario: BenchmarkScenario) -> str:
        """Read environment_type from simulation_config.json, or fall back."""
        config_path = os.path.join(sim_dir, "simulation_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                env_type = cfg.get("environment_type") or cfg.get("platform")
                if env_type:
                    return env_type
            except Exception as exc:
                logger.warning("Could not read simulation_config.json: %s", exc)

        # Infer from scenario domain
        domain_map = {"organisation": "organisation", "education": "classroom"}
        return domain_map.get(scenario.domain, "reddit")

    @staticmethod
    def _load_action_log(
        sim_dir: str, env_type: str
    ) -> List[Dict[str, Any]]:
        """Load all agent action records from the JSONL action log.

        Searches the primary path ``{sim_dir}/{env_type}/actions.jsonl``,
        then falls back to all known environment subdirectories.
        """
        candidates = [
            os.path.join(sim_dir, env_type, "actions.jsonl"),
            # Legacy / backwards-compat paths
            os.path.join(sim_dir, "classroom",    "actions.jsonl"),
            os.path.join(sim_dir, "organisation", "actions.jsonl"),
            os.path.join(sim_dir, "reddit",       "actions.jsonl"),
            os.path.join(sim_dir, "twitter",      "actions.jsonl"),
            os.path.join(sim_dir, "actions.jsonl"),           # old single-file format
        ]

        # De-duplicate while preserving order
        seen: set = set()
        ordered: List[str] = []
        for p in candidates:
            if p not in seen:
                seen.add(p)
                ordered.append(p)

        actions: List[Dict[str, Any]] = []
        for path in ordered:
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                            # Skip meta-event lines
                            if "event_type" in record:
                                continue
                            actions.append(record)
                        except json.JSONDecodeError:
                            pass
                logger.debug("Loaded %d actions from %s", len(actions), path)
                break  # Use the first file found
            except Exception as exc:
                logger.warning("Failed to read action log %s: %s", path, exc)

        if not actions:
            logger.warning(
                "No action log found for simulation at '%s' (env_type=%s)",
                sim_dir, env_type,
            )

        return actions

    @staticmethod
    def _load_profiles(
        sim_dir: str, env_type: str
    ) -> Dict[int, Dict[str, Any]]:
        """Load the profile JSON file and return a {user_id: profile_dict} map."""
        filename_map = {
            "classroom":    "classroom_profiles.json",
            "organisation": "organisation_profiles.json",
            "reddit":       "reddit_profiles.json",
            "twitter":      "twitter_profiles.csv",
        }
        candidates = []
        primary = filename_map.get(env_type)
        if primary:
            candidates.append(os.path.join(sim_dir, primary))
        # Fallback: try all JSON profile files
        for fname in filename_map.values():
            if fname.endswith(".json"):
                candidates.append(os.path.join(sim_dir, fname))

        for path in candidates:
            if not os.path.exists(path) or not path.endswith(".json"):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    return {
                        int(p.get("user_id", i)): p
                        for i, p in enumerate(data)
                        if isinstance(p, dict)
                    }
            except Exception as exc:
                logger.warning("Failed to read profile file %s: %s", path, exc)

        logger.warning("No profile file found for simulation at '%s'", sim_dir)
        return {}

    # ------------------------------------------------------------------
    # Segment excerpt construction
    # ------------------------------------------------------------------

    def _build_segment_excerpt(
        self,
        segment_outcome: SegmentOutcome,
        all_actions: List[Dict[str, Any]],
        profiles: Dict[int, Dict[str, Any]],
    ) -> str:
        """Build a readable text excerpt of simulated behaviour for one segment.

        Matching strategy
        -----------------
        Keywords are extracted from the segment name and description.
        Each profile is scored by how many keyword tokens appear in its
        text fields (username, name, bio, persona, profession).
        Profiles scoring above zero are treated as segment members.

        If no profiles match, all agents are included and the excerpt is
        labelled accordingly (the LLM is still asked to interpret relevance).
        """
        if not all_actions:
            return (
                f"No simulation actions were recorded for the "
                f"'{segment_outcome.segment_name}' segment."
            )

        # Build keyword set from segment name + description
        keywords = self._extract_keywords(
            segment_outcome.segment_name + " " + segment_outcome.description
        )

        # Match profiles to this segment
        matched_ids: set = set()
        if profiles:
            for uid, profile in profiles.items():
                profile_text = " ".join(
                    str(profile.get(f, "")) for f in _MATCH_FIELDS
                ).lower()
                if any(kw in profile_text for kw in keywords):
                    matched_ids.add(uid)

        unmatched_fallback = False
        if not matched_ids:
            # No profile match — use all agents and note it
            matched_ids = {r.get("agent_id", -1) for r in all_actions}
            unmatched_fallback = True
            logger.debug(
                "No profile match for segment '%s' — using all %d agents",
                segment_outcome.segment_name, len(matched_ids),
            )

        # Filter and sort actions
        seg_actions = [
            r for r in all_actions
            if r.get("agent_id") in matched_ids
        ]
        seg_actions.sort(key=lambda r: (r.get("round", 0), r.get("timestamp", "")))

        if not seg_actions:
            return (
                f"No actions found in the simulation log for agents "
                f"matching segment '{segment_outcome.segment_name}'."
            )

        # Group by agent
        by_agent: Dict[int, List[Dict[str, Any]]] = {}
        for rec in seg_actions:
            aid = rec.get("agent_id", -1)
            by_agent.setdefault(aid, []).append(rec)

        # Format excerpt
        lines: List[str] = []
        total_content_actions = sum(
            1 for r in seg_actions
            if r.get("action_type", "").upper() in CONTENT_ACTIONS
        )

        header = (
            f"Segment: '{segment_outcome.segment_name}' "
            f"({'all agents — no segment match found' if unmatched_fallback else f'{len(by_agent)} matched agents'})\n"
            f"Total actions: {len(seg_actions)} | "
            f"Content-producing actions: {total_content_actions}\n"
            f"Rounds observed: {seg_actions[0].get('round', '?')}–"
            f"{seg_actions[-1].get('round', '?')}\n"
        )
        lines.append(header)

        char_budget = MAX_EXCERPT_CHARS - len(header)
        truncated = False

        for uid, agent_actions in by_agent.items():
            if char_budget <= 0:
                truncated = True
                break

            profile = profiles.get(uid, {})
            agent_label = (
                profile.get("username")
                or profile.get("name")
                or f"agent_{uid}"
            )
            age = profile.get("age", "")
            gender = profile.get("gender", "")
            mbti = profile.get("mbti", "")
            attr_str = ", ".join(filter(None, [str(age), gender, mbti]))
            agent_header = f"\nAgent {agent_label}" + (f" ({attr_str})" if attr_str else "") + ":\n"

            char_budget -= len(agent_header)
            if char_budget <= 0:
                truncated = True
                break
            lines.append(agent_header)

            for rec in agent_actions:
                action_type = rec.get("action_type", "UNKNOWN").upper()
                round_num = rec.get("round", "?")
                args = rec.get("action_args") or {}
                content = args.get("content", "")

                if content:
                    entry = f'  [Round {round_num}] {action_type}: "{content[:200]}"\n'
                elif action_type not in {"DO_NOTHING"}:
                    result = rec.get("result", "")
                    entry = (
                        f"  [Round {round_num}] {action_type}"
                        + (f" → {result}" if result else "")
                        + "\n"
                    )
                else:
                    # Skip DO_NOTHING to avoid noise
                    continue

                char_budget -= len(entry)
                if char_budget <= 0:
                    truncated = True
                    break
                lines.append(entry)

        if truncated:
            lines.append(
                f"\n[Excerpt truncated at {MAX_EXCERPT_CHARS} characters — "
                f"{len(seg_actions)} total actions recorded]\n"
            )

        return "".join(lines)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract lower-case alpha tokens of length ≥ 4 from *text*."""
        tokens = re.findall(r"[a-zA-Z]{4,}", text.lower())
        # Remove common English stop words that would produce false matches
        stop = {
            "that", "this", "with", "from", "they", "their", "have",
            "been", "were", "also", "will", "more", "into", "when",
            "about", "which", "would", "could", "should", "most",
            "some", "each", "over", "than", "then", "there",
        }
        return [t for t in set(tokens) if t not in stop]

    # ------------------------------------------------------------------
    # LLM scoring — per segment
    # ------------------------------------------------------------------

    def _score_segment_with_llm(
        self,
        rubric: BaseRubric,
        segment_outcome: SegmentOutcome,
        excerpt: str,
        scenario: BenchmarkScenario,
    ) -> SegmentScore:
        """Call the LLM to score one segment; parse the response.

        On JSON parse failure, retries up to MAX_LLM_RETRIES times with
        reduced temperature.  If all retries fail, returns a zero score
        with an error explanation.
        """
        system_msg = rubric.system_prompt()
        user_msg = rubric.build_scoring_prompt(
            segment_outcome=segment_outcome,
            simulation_excerpt=excerpt,
            scenario=scenario,
        )

        last_error: Optional[str] = None
        for attempt in range(MAX_LLM_RETRIES + 1):
            temperature = max(0.0, 0.3 - attempt * 0.1)
            try:
                raw = self._call_llm(
                    system_msg, user_msg, temperature=temperature
                )
                return self._parse_segment_score(
                    raw, segment_outcome.segment_name, rubric
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "Segment scoring attempt %d/%d failed for '%s': %s",
                    attempt + 1, MAX_LLM_RETRIES + 1,
                    segment_outcome.segment_name, exc,
                )
                if attempt < MAX_LLM_RETRIES:
                    time.sleep(1.0 * (attempt + 1))

        # All retries exhausted
        logger.error(
            "Segment '%s' could not be scored after %d attempts: %s",
            segment_outcome.segment_name, MAX_LLM_RETRIES + 1, last_error,
        )
        return SegmentScore(
            segment_name=segment_outcome.segment_name,
            fidelity_score=0.0,
            framework_scores={},
            explanation=(
                f"Scoring failed after {MAX_LLM_RETRIES + 1} attempts. "
                f"Last error: {last_error}"
            ),
            matched_behaviours=[],
            missed_behaviours=[b for b in segment_outcome.expected_behaviours],
            unexpected_observed=[],
        )

    def _parse_segment_score(
        self,
        raw_response: str,
        segment_name: str,
        rubric: BaseRubric,
    ) -> SegmentScore:
        """Parse the LLM JSON response into a SegmentScore.

        The scorer accepts both a bare JSON object and an object wrapped in
        a markdown code fence.
        """
        # Strip markdown fences if present
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw_response)
        content = fence.group(1).strip() if fence else raw_response.strip()

        # Attempt to extract outermost {...}
        brace_match = re.search(r"\{[\s\S]*\}", content)
        if brace_match:
            content = brace_match.group(0)

        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM response for segment '{segment_name}' is not valid JSON. "
                f"Parse error: {exc}. "
                f"Preview: {raw_response[:300]}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(
                f"Expected JSON object, got {type(data).__name__} "
                f"for segment '{segment_name}'."
            )

        # Extract dimension scores
        raw_dim_scores: Dict[str, Any] = data.get("dimension_scores", {})
        if not isinstance(raw_dim_scores, dict):
            raw_dim_scores = {}

        # Clamp all dimension scores to [0, 1]
        framework_scores: Dict[str, float] = {}
        for dim_name, val in raw_dim_scores.items():
            try:
                framework_scores[dim_name] = max(0.0, min(1.0, float(val)))
            except (TypeError, ValueError):
                framework_scores[dim_name] = 0.0

        # Compute weighted aggregate fidelity
        fidelity = rubric.aggregate_score(framework_scores)

        def _as_str_list(key: str) -> List[str]:
            items = data.get(key, [])
            if isinstance(items, list):
                return [str(i) for i in items if i]
            return []

        explanation = str(data.get("explanation", "No explanation provided."))

        return SegmentScore(
            segment_name=segment_name,
            fidelity_score=fidelity,
            framework_scores=framework_scores,
            explanation=explanation,
            matched_behaviours=_as_str_list("matched_behaviours"),
            missed_behaviours=_as_str_list("missed_behaviours"),
            unexpected_observed=_as_str_list("unexpected_observed"),
        )

    # ------------------------------------------------------------------
    # LLM scoring — summary
    # ------------------------------------------------------------------

    def _generate_summary(
        self,
        scenario: BenchmarkScenario,
        segment_scores: List[SegmentScore],
        rubric: BaseRubric,
    ) -> str:
        """Call the LLM to generate a narrative summary of the validation result."""
        if not segment_scores:
            return "No segments were scored."

        overall = sum(s.fidelity_score for s in segment_scores) / len(segment_scores)

        scores_block = "\n".join(
            f"  - {s.segment_name}: {s.fidelity_score:.2f} — {s.explanation}"
            for s in segment_scores
        )
        dim_names = [d.name for d in rubric.definition().dimensions]
        frameworks = ", ".join(rubric.definition().frameworks)

        user_msg = f"""You have scored a ManaSim simulation against the following benchmark scenario.

Scenario : {scenario.title} ({scenario.scenario_id})
Domain   : {scenario.domain}
Artefact : {scenario.artefact}
Rubric   : {rubric.definition().display_name}
Frameworks used: {frameworks}
Scoring dimensions: {", ".join(dim_names)}

Overall fidelity: {overall:.2f} / 1.0

Per-segment results:
{scores_block}

Write a narrative summary (4–6 sentences) that:
1. States the overall fidelity and what it means for simulation validity.
2. Identifies the segment with the highest fidelity and explains why it matched well.
3. Identifies the most significant gap (lowest fidelity segment or missed behaviour).
4. Notes any cross-segment patterns — e.g. all segments failed on one dimension.
5. Gives a one-sentence recommendation for improving simulation fidelity.

Write in plain prose. Do not use bullet points or headings.
"""
        system_msg = (
            "You are a simulation validation expert summarising the results of a "
            "fidelity evaluation for ManaSim, a social simulation engine. "
            "Be concise, specific, and grounded in the scores provided."
        )

        try:
            return self._call_llm(system_msg, user_msg, temperature=0.4)
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            return (
                f"Overall fidelity: {overall:.2f}. "
                f"Scored {len(segment_scores)} segment(s) using the "
                f"{rubric.definition().display_name}. "
                f"(Narrative summary generation failed: {exc})"
            )

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(
        self,
        system_message: str,
        user_message: str,
        *,
        temperature: float = 0.2,
    ) -> str:
        """Send a two-message prompt to the LLM and return the response text."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user",   "content": user_message},
            ],
            temperature=temperature,
            # max_tokens intentionally not set — let the model decide
        )
        return response.choices[0].message.content or ""
