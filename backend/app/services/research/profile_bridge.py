"""
Profile bridge.

Converts the human segment profiles produced by the research synthesiser
into OasisAgentProfile instances that the OASIS simulation engine can consume.

This module does NOT modify oasis_profile_generator.py. It imports the
OasisAgentProfile dataclass from there and populates it via its own LLM
prompting strategy tailored to segment-based persona generation.

For each segment the LLM is asked to generate `count` distinct individuals
who all believably belong to that segment. The segment's behavioral_profile,
persona_hints, and evidence_snippets are all passed as context so the LLM
can produce grounded, varied personas rather than generic ones.

Concurrency
-----------
Segments are processed sequentially by default. Pass parallel=True to the
generate() call to process all segments concurrently via ThreadPoolExecutor.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from openai import OpenAI

from ...config import Config
from ..oasis_profile_generator import OasisAgentProfile
from .schemas import HumanSegment, ResearchOutput

logger = logging.getLogger("manasim.research.profile_bridge")

_SYSTEM_PROMPT = """\
You are a persona designer for ManaSim, a social simulation engine.
Your task is to generate realistic, distinct individual agent profiles
for a given human segment type.

Each agent must:
  - Believably belong to the described segment
  - Be distinct from the other agents in the same batch (different name,
    background, personality, and writing style)
  - Have a detailed "persona" field (3-5 sentences) describing their
    beliefs, communication style, online behaviour, and emotional triggers
  - Have a "bio" field (1-2 sentences) written as a social media profile bio
  - Have a unique "user_name" (lowercase, underscores allowed, no spaces)

Output rules:
  - Output ONLY a valid JSON array. No markdown, no prose, no code fences.
  - Array length must equal exactly the requested count.
"""

_PER_AGENT_SCHEMA = """\
{
  "user_name": "string — unique, lowercase, underscores ok",
  "name": "string — realistic full name",
  "bio": "string — 1-2 sentence social media bio",
  "persona": "string — 3-5 sentence behavioural description",
  "age": integer,
  "gender": "string",
  "mbti": "string — one of the 16 MBTI types",
  "country": "string",
  "profession": "string",
  "interested_topics": ["string"],
  "karma": integer
}"""


class ProfileBridge:
    """
    Generates OasisAgentProfile instances from ResearchOutput segments.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> None:
        self._api_key = api_key or Config.LLM_API_KEY
        self._base_url = base_url or Config.LLM_BASE_URL
        self._model = model_name or Config.LLM_MODEL_NAME

        if not self._api_key:
            raise ValueError(
                "LLM_API_KEY is not configured. Cannot generate profiles."
            )

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        research: ResearchOutput,
        agents_per_segment: Optional[int] = None,
        output_platform: str = "reddit",
        parallel: bool = False,
    ) -> List[OasisAgentProfile]:
        """
        Generate OasisAgentProfile instances for every segment in *research*.

        Args:
            research:           Output of the research synthesiser.
            agents_per_segment: Number of agents to generate per segment.
                                Defaults to the value stored in ResearchOutput
                                metadata, falling back to 5.
            output_platform:    "reddit" or "twitter" — controls which
                                numeric engagement fields are emphasised
                                in the karma/follower hints.
            parallel:           If True, process segments concurrently.

        Returns:
            Flat list of OasisAgentProfile instances, ordered by segment then
            agent index. user_id values are globally unique across all segments.
        """
        count = agents_per_segment or 5
        segments = research.human_segments

        logger.info(
            "ProfileBridge: generating %d agents × %d segments = %d total "
            "(platform=%s, parallel=%s)",
            count, len(segments), count * len(segments),
            output_platform, parallel,
        )

        if parallel:
            return self._generate_parallel(segments, count, output_platform)
        return self._generate_sequential(segments, count, output_platform)

    # ------------------------------------------------------------------
    # Sequential and parallel execution
    # ------------------------------------------------------------------

    def _generate_sequential(
        self,
        segments: List[HumanSegment],
        count: int,
        platform: str,
    ) -> List[OasisAgentProfile]:
        all_profiles: List[OasisAgentProfile] = []
        user_id_offset = 0

        for segment in segments:
            profiles = self._generate_for_segment(
                segment, count, platform, user_id_offset
            )
            all_profiles.extend(profiles)
            user_id_offset += len(profiles)

        return all_profiles

    def _generate_parallel(
        self,
        segments: List[HumanSegment],
        count: int,
        platform: str,
    ) -> List[OasisAgentProfile]:
        # Preserve segment order in the output
        results: dict[int, List[OasisAgentProfile]] = {}

        with ThreadPoolExecutor(max_workers=len(segments)) as pool:
            futures = {
                pool.submit(
                    self._generate_for_segment, seg, count, platform, idx * count
                ): idx
                for idx, seg in enumerate(segments)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:
                    logger.error(
                        "ProfileBridge: segment %d generation failed: %s",
                        idx, exc,
                    )
                    results[idx] = []

        all_profiles: List[OasisAgentProfile] = []
        for idx in sorted(results):
            all_profiles.extend(results[idx])
        return all_profiles

    # ------------------------------------------------------------------
    # Per-segment generation
    # ------------------------------------------------------------------

    def _generate_for_segment(
        self,
        segment: HumanSegment,
        count: int,
        platform: str,
        user_id_offset: int,
    ) -> List[OasisAgentProfile]:
        """Generate *count* OasisAgentProfile instances for one segment."""
        logger.info(
            "ProfileBridge: generating %d agents for segment '%s' ...",
            count, segment.name,
        )

        try:
            prompt = self._build_prompt(segment, count, platform)
            raw_list = self._call_llm(prompt)
            profiles = self._parse_profiles(
                raw_list, segment, user_id_offset
            )
            logger.info(
                "ProfileBridge: generated %d profiles for '%s'",
                len(profiles), segment.name,
            )
            return profiles
        except Exception as exc:
            logger.error(
                "ProfileBridge: failed to generate profiles for segment "
                "'%s': %s", segment.name, exc,
            )
            return []

    def _build_prompt(
        self, segment: HumanSegment, count: int, platform: str
    ) -> str:
        bp = segment.behavioral_profile
        ph = segment.persona_hints

        platform_note = (
            "This is a Reddit-style simulation. "
            "karma should reflect engagement level (range 100-10000)."
            if platform == "reddit"
            else
            "This is a Twitter-style simulation. "
            "karma should reflect follower influence (range 50-5000)."
        )

        evidence_block = ""
        if segment.evidence_snippets:
            evidence_block = "\n## Supporting evidence\n" + "\n".join(
                f"  - {s}" for s in segment.evidence_snippets
            )

        mbti_hint = (
            f"Lean toward these MBTI types: {', '.join(ph.likely_mbti)}"
            if ph.likely_mbti else ""
        )
        profession_hint = (
            f"Example professions: {', '.join(ph.example_professions)}"
            if ph.example_professions else ""
        )
        country_hint = (
            f"Likely countries: {', '.join(ph.likely_countries)}"
            if ph.likely_countries else ""
        )
        topics_hint = (
            f"Typical interested topics: {', '.join(ph.interested_topics)}"
            if ph.interested_topics else ""
        )

        concerns = ", ".join(bp.key_concerns) if bp.key_concerns else "general domain concerns"
        behaviors = "\n".join(
            f"  - {b}" for b in bp.typical_behaviors
        ) if bp.typical_behaviors else "  - General stakeholder behaviour"

        return f"""\
## Segment: {segment.name}
{segment.description}

## Behavioral profile
- Technology adoption : {bp.technology_adoption or 'not specified'}
- Social media activity: {bp.social_media_activity or 'moderate'}
- Key concerns        : {concerns}
- Likely stance       : {bp.likely_stance_on_topic or 'mixed'}

## Typical behaviours
{behaviors}

## Persona hints
{mbti_hint}
{profession_hint}
{country_hint}
{topics_hint}
{evidence_block}

## Platform note
{platform_note}

## Task
Generate exactly {count} distinct individual agents who all belong to the
"{segment.name}" segment. Make each agent unique — vary names, ages, countries,
personality traits, and writing styles.

Output format — a JSON array of exactly {count} objects:
{_PER_AGENT_SCHEMA}
"""

    # ------------------------------------------------------------------
    # LLM call and JSON parsing
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str) -> list:
        """Call the LLM and return a parsed JSON list."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.85,   # Higher temperature for persona diversity
            max_tokens=3000,
        )
        content = response.choices[0].message.content or ""
        return self._extract_json_list(content)

    @staticmethod
    def _extract_json_list(content: str) -> list:
        """Extract and parse a JSON array from the LLM response."""
        # Strip markdown code fences
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if fence_match:
            content = fence_match.group(1).strip()
        else:
            content = content.strip()

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Find the outermost [...] block
            bracket_match = re.search(r"\[[\s\S]*\]", content)
            if bracket_match:
                result = json.loads(bracket_match.group(0))
            else:
                raise ValueError(
                    f"Could not extract JSON array from LLM response. "
                    f"Preview: {content[:200]}"
                )

        if not isinstance(result, list):
            raise ValueError(
                f"Expected JSON array from LLM, got {type(result).__name__}."
            )
        return result

    # ------------------------------------------------------------------
    # Profile construction
    # ------------------------------------------------------------------

    def _parse_profiles(
        self,
        raw_list: list,
        segment: HumanSegment,
        user_id_offset: int,
    ) -> List[OasisAgentProfile]:
        """Convert raw LLM dicts into OasisAgentProfile instances."""
        profiles: List[OasisAgentProfile] = []

        for idx, raw in enumerate(raw_list):
            if not isinstance(raw, dict):
                logger.warning(
                    "ProfileBridge: skipping non-dict item at index %d", idx
                )
                continue

            user_id = user_id_offset + idx
            user_name = str(raw.get("user_name") or f"agent_{user_id:04d}")
            # Sanitise user_name: lower-case, spaces → underscores
            user_name = re.sub(r"[^a-z0-9_]", "_", user_name.lower())

            profile = OasisAgentProfile(
                user_id=user_id,
                user_name=user_name,
                name=str(raw.get("name") or user_name),
                bio=str(raw.get("bio") or ""),
                persona=str(raw.get("persona") or ""),
                karma=int(raw.get("karma") or 1000),
                # Twitter-style fields — set sensible defaults
                friend_count=int(raw.get("friend_count") or 100),
                follower_count=int(raw.get("follower_count") or 150),
                statuses_count=int(raw.get("statuses_count") or 500),
                age=self._safe_int(raw.get("age")),
                gender=raw.get("gender"),
                mbti=raw.get("mbti"),
                country=raw.get("country"),
                profession=raw.get("profession"),
                interested_topics=raw.get("interested_topics") or [],
                # Traceability back to the source segment
                source_entity_type=segment.name,
                source_entity_uuid=segment.id,
            )
            profiles.append(profile)

        return profiles

    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Return int(value) or None if conversion fails."""
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None
