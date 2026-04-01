"""
Research synthesiser.

Takes the raw scraped material (RawSources) and the domain config and calls
the LLM to produce a structured set of human segment profiles (ResearchOutput).

The source weights from the domain config are passed as a prompt hint so the
LLM knows which evidence to treat as more authoritative. They are not used for
hard filtering — all available data is included in the context.

Token budget strategy
---------------------
To keep the prompt within a reasonable input budget we:
  - Cap academic papers at MAX_PAPERS (sorted by citation count descending)
  - Cap social posts at MAX_SOCIAL_POSTS total (split proportionally by source)
  - Cap case study snippets at MAX_CASE_STUDIES
  - Truncate abstracts, post bodies, and snippets to defined char limits
  - Include only the top N comments per post
"""

from __future__ import annotations

import json
import logging
import re
from typing import List, Optional

from openai import OpenAI

from ...config import Config
from .schemas import (
    AcademicPaper,
    CaseStudySnippet,
    DomainConfig,
    HumanSegment,
    RawSources,
    ResearchMetadata,
    ResearchOutput,
    SocialPost,
    SourceCounts,
)

logger = logging.getLogger("manasim.research.synthesiser")

# ── Token budget limits ────────────────────────────────────────────────────
MAX_PAPERS = 10
MAX_REDDIT_POSTS = 12
MAX_HN_POSTS = 8
MAX_CASE_STUDIES = 5

ABSTRACT_MAX_CHARS = 250
POST_BODY_MAX_CHARS = 350
COMMENT_MAX_CHARS = 200
SNIPPET_MAX_CHARS = 400
ARTIFACT_MAX_CHARS = 3000

COMMENTS_PER_POST = 2
# ──────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a research analyst for ManaSim, a domain-adaptive social simulation engine.
Your task is to analyse a collection of research findings about a domain and synthesise
them into a set of distinct human segment profiles.

These profiles describe the real-world human stakeholders who will be modelled as
agents in a social dynamics simulation. Each segment must be:
  - Grounded in concrete evidence from the provided sources
  - Meaningfully distinct from the other segments in values, behaviour, and attitude
  - Realistic and non-stereotyped — reflect the nuance in the evidence

Output rules:
  - Output ONLY valid JSON. No markdown, no prose, no code fences.
  - population_weights across all segments must sum to exactly 1.0.
  - id fields must follow the format "seg_001", "seg_002", etc.
  - Each segment must include 3-5 evidence_snippets (verbatim quotes or close
    paraphrases drawn from the provided sources — never invented).
"""

_OUTPUT_SCHEMA = """\
{
  "research_summary": "string — 2-3 sentence synthesis of the key findings",
  "human_segments": [
    {
      "id": "seg_001",
      "name": "string",
      "description": "string — 1-2 sentences",
      "size_hint": "string — e.g. '~25% of the population'",
      "population_weight": 0.00,
      "demographics": {
        "age_range": "string",
        "gender_distribution": "string",
        "location": "string",
        "education_level": "string"
      },
      "behavioral_profile": {
        "technology_adoption": "string",
        "social_media_activity": "string — low / moderate / high",
        "key_concerns": ["string"],
        "typical_behaviors": ["string"],
        "likely_stance_on_topic": "string"
      },
      "persona_hints": {
        "likely_mbti": ["string"],
        "example_professions": ["string"],
        "likely_countries": ["string"],
        "interested_topics": ["string"]
      },
      "evidence_snippets": ["string"]
    }
  ]
}"""


class Synthesiser:
    """
    Calls the LLM to synthesise raw scraped sources into ResearchOutput.
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
                "LLM_API_KEY is not configured. Cannot run synthesis."
            )

        self._client = OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(
        self,
        raw: RawSources,
        config: DomainConfig,
        artifact_text: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> ResearchOutput:
        """
        Synthesise raw sources into a ResearchOutput containing human segments.

        Args:
            raw:           All scraped material from the research pipeline.
            config:        Domain config (supplies weights, expected segments).
            artifact_text: Optional text extracted from the uploaded document.
            artifact_name: Optional filename of the uploaded document.

        Returns:
            A validated ResearchOutput instance.
        """
        user_message = self._build_prompt(raw, config, artifact_text)

        logger.info(
            "Synthesiser: calling LLM (%s) for domain '%s' ...",
            self._model,
            config.domain_id,
        )

        raw_json = self._call_llm(user_message)
        parsed = self._parse_and_validate(raw_json, config)

        return ResearchOutput(
            domain=config.domain_id,
            research_summary=parsed["research_summary"],
            human_segments=[
                HumanSegment.model_validate(seg)
                for seg in parsed["human_segments"]
            ],
            raw_source_counts=SourceCounts(
                semantic_scholar_papers=len(raw.academic_papers),
                reddit_posts=sum(
                    1 for p in raw.social_posts if p.source == "reddit"
                ),
                hacker_news_items=sum(
                    1 for p in raw.social_posts if p.source == "hacker_news"
                ),
                case_studies=len(raw.case_study_snippets),
            ),
            metadata=ResearchMetadata(
                domain_id=config.domain_id,
                artifact_document_name=artifact_name,
            ),
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        raw: RawSources,
        config: DomainConfig,
        artifact_text: Optional[str],
    ) -> str:
        weights = config.source_weights
        n_segments = len(config.expected_segments)
        # Allow the LLM to go one above or below the hint
        n_hint = f"{max(3, n_segments - 1)}-{n_segments + 1}"

        lines: List[str] = []

        # ── Header ──────────────────────────────────────────────────────
        lines += [
            f"## Domain: {config.domain_name}",
            config.description,
            "",
            "## Source weight guidance",
            "(Higher weight = stronger evidence signal — use as emphasis guide, not a filter)",
            f"  Academic papers   : {int(weights.semantic_scholar * 100)}%",
            f"  Reddit discussions: {int(weights.reddit * 100)}%",
            f"  Hacker News       : {int(weights.hacker_news * 100)}%",
            f"  Case studies      : {int(weights.case_studies * 100)}%",
            "",
            "## Expected segment types (hints — refine based on evidence)",
        ]
        for seg in config.expected_segments:
            lines.append(f"  - {seg}")
        lines.append("")

        # ── Artifact document ────────────────────────────────────────────
        if artifact_text and artifact_text.strip():
            excerpt = artifact_text.strip()[:ARTIFACT_MAX_CHARS]
            lines += [
                "## Uploaded artefact document (primary source)",
                excerpt,
                "...(truncated)" if len(artifact_text) > ARTIFACT_MAX_CHARS else "",
                "",
            ]

        # ── Academic papers ──────────────────────────────────────────────
        papers = sorted(
            raw.academic_papers, key=lambda p: p.citation_count, reverse=True
        )[:MAX_PAPERS]

        if papers:
            lines.append("## Academic papers")
            for i, p in enumerate(papers, 1):
                lines.append(self._format_paper(i, p))
            lines.append("")

        # ── Reddit ───────────────────────────────────────────────────────
        reddit_posts = sorted(
            [p for p in raw.social_posts if p.source == "reddit"],
            key=lambda p: p.score,
            reverse=True,
        )[:MAX_REDDIT_POSTS]

        if reddit_posts:
            lines.append("## Reddit discussions")
            for i, p in enumerate(reddit_posts, 1):
                lines.append(self._format_social_post(i, p))
            lines.append("")

        # ── Hacker News ──────────────────────────────────────────────────
        hn_posts = sorted(
            [p for p in raw.social_posts if p.source == "hacker_news"],
            key=lambda p: p.score,
            reverse=True,
        )[:MAX_HN_POSTS]

        if hn_posts:
            lines.append("## Hacker News discussions")
            for i, p in enumerate(hn_posts, 1):
                lines.append(self._format_social_post(i, p))
            lines.append("")

        # ── Case studies ─────────────────────────────────────────────────
        case_studies = raw.case_study_snippets[:MAX_CASE_STUDIES]

        if case_studies:
            lines.append("## Case studies and real-world examples")
            for i, cs in enumerate(case_studies, 1):
                lines.append(self._format_case_study(i, cs))
            lines.append("")

        # ── Availability notice ──────────────────────────────────────────
        unavailable = []
        if not raw.academic_available:
            unavailable.append("academic papers")
        if not raw.reddit_available:
            unavailable.append("Reddit")
        if not raw.hacker_news_available:
            unavailable.append("Hacker News")
        if not raw.case_studies_available:
            unavailable.append("case studies")
        if unavailable:
            lines += [
                f"Note: the following sources were unavailable during this run "
                f"and are not represented above: {', '.join(unavailable)}.",
                "",
            ]

        # ── Task instruction ─────────────────────────────────────────────
        lines += [
            f"## Task",
            f"Based on the research above, produce {n_hint} distinct human segments.",
            "Each segment must include exactly 3-5 evidence_snippets drawn from the sources above.",
            "population_weights must sum to 1.0.",
            "",
            "Output the following JSON schema exactly:",
            _OUTPUT_SCHEMA,
        ]

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_paper(idx: int, p: AcademicPaper) -> str:
        authors = ", ".join(p.authors[:3]) or "Unknown"
        if len(p.authors) > 3:
            authors += " et al."
        year = f"({p.year})" if p.year else ""
        abstract = (p.abstract or "No abstract available.")[:ABSTRACT_MAX_CHARS]
        citations = f"[{p.citation_count} citations]" if p.citation_count else ""
        return (
            f"[{idx}] {p.title} — {authors} {year} {citations}\n"
            f"    {abstract}"
        )

    @staticmethod
    def _format_social_post(idx: int, p: SocialPost) -> str:
        source_label = (
            f"r/{p.subreddit}" if p.subreddit else "Hacker News"
        )
        body = (p.body or "")[:POST_BODY_MAX_CHARS]
        comments = ""
        if p.top_comments:
            selected = p.top_comments[:COMMENTS_PER_POST]
            comments = " | ".join(
                c[:COMMENT_MAX_CHARS] for c in selected if c
            )
        return (
            f"[{idx}] [{source_label}] score={p.score} — {p.title}\n"
            + (f"    {body}\n" if body else "")
            + (f"    Comments: {comments}" if comments else "")
        )

    @staticmethod
    def _format_case_study(idx: int, cs: CaseStudySnippet) -> str:
        snippet = cs.snippet[:SNIPPET_MAX_CHARS]
        return (
            f"[{idx}] {cs.title}\n"
            f"    Query: {cs.query}\n"
            f"    {snippet}"
        )

    # ------------------------------------------------------------------
    # LLM call and JSON parsing
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str) -> dict:
        """Call the LLM and return the parsed JSON response."""
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.4,
            max_tokens=4096,
        )
        content = response.choices[0].message.content or ""
        return self._extract_json(content)

    @staticmethod
    def _extract_json(content: str) -> dict:
        """
        Extract and parse JSON from LLM response.

        Handles responses that are wrapped in markdown code fences or have
        leading/trailing prose.
        """
        # Strip markdown code fences if present
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content)
        if fence_match:
            content = fence_match.group(1).strip()
        else:
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Last-ditch attempt: find the outermost {...} block
            brace_match = re.search(r"\{[\s\S]*\}", content)
            if brace_match:
                return json.loads(brace_match.group(0))
            raise ValueError(
                f"Could not extract valid JSON from LLM response. "
                f"Response preview: {content[:200]}"
            )

    # ------------------------------------------------------------------
    # Validation and post-processing
    # ------------------------------------------------------------------

    def _parse_and_validate(self, data: dict, config: DomainConfig) -> dict:
        """
        Validate the LLM output and apply defensive corrections.

        - Ensures required keys are present
        - Assigns sequential ids if missing
        - Clamps population_weights to [0, 1] and re-normalises if needed
        """
        if "research_summary" not in data:
            data["research_summary"] = (
                f"Research synthesis for domain: {config.domain_name}."
            )

        segments = data.get("human_segments", [])
        if not segments:
            raise ValueError(
                "LLM returned zero human segments. Cannot continue."
            )

        # Assign ids if missing
        for i, seg in enumerate(segments, 1):
            if not seg.get("id"):
                seg["id"] = f"seg_{i:03d}"

            # Ensure nested dicts exist
            seg.setdefault("demographics", {})
            seg.setdefault("behavioral_profile", {})
            seg.setdefault("persona_hints", {})
            seg.setdefault("evidence_snippets", [])

            # Ensure list fields are lists
            for field in ("key_concerns", "typical_behaviors"):
                bp = seg["behavioral_profile"]
                if not isinstance(bp.get(field), list):
                    bp[field] = []

            for field in ("likely_mbti", "example_professions",
                          "likely_countries", "interested_topics"):
                ph = seg["persona_hints"]
                if not isinstance(ph.get(field), list):
                    ph[field] = []

        # Re-normalise population weights
        raw_weights = [
            float(seg.get("population_weight", 0) or 0) for seg in segments
        ]
        total = sum(raw_weights)
        if total <= 0:
            # Distribute evenly
            equal = 1.0 / len(segments)
            for seg in segments:
                seg["population_weight"] = round(equal, 4)
        elif not (0.99 <= total <= 1.01):
            logger.warning(
                "Segment weights sum to %.4f — re-normalising to 1.0.", total
            )
            for seg, w in zip(segments, raw_weights):
                seg["population_weight"] = round(w / total, 4)

        data["human_segments"] = segments
        return data
