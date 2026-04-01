"""
Pydantic schemas for the ManaSim research agent pipeline.

Three layers of models:
  1. DomainConfig   — shape of a domains/*.json file
  2. RawSource*     — scraped data from each source type
  3. ResearchOutput — final structured output consumed by profile_bridge.py
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# 1. Domain config schema (domains/*.json)
# ---------------------------------------------------------------------------

class SemanticScholarConfig(BaseModel):
    queries: List[str] = Field(..., min_length=1)
    max_results_per_query: int = Field(8, ge=1, le=50)
    year_from: Optional[int] = Field(None, ge=1990)


class RedditConfig(BaseModel):
    subreddits: List[str] = Field(..., min_length=1)
    search_queries: List[str] = Field(default_factory=list)
    max_posts_per_subreddit: int = Field(20, ge=1, le=100)
    sort: str = Field("top")
    time_filter: str = Field("year")

    @field_validator("sort")
    @classmethod
    def validate_sort(cls, v: str) -> str:
        allowed = {"hot", "new", "top", "rising", "controversial"}
        if v not in allowed:
            raise ValueError(f"sort must be one of {allowed}")
        return v

    @field_validator("time_filter")
    @classmethod
    def validate_time_filter(cls, v: str) -> str:
        allowed = {"hour", "day", "week", "month", "year", "all"}
        if v not in allowed:
            raise ValueError(f"time_filter must be one of {allowed}")
        return v


class HackerNewsConfig(BaseModel):
    queries: List[str] = Field(..., min_length=1)
    max_results_per_query: int = Field(10, ge=1, le=50)


class CaseStudiesConfig(BaseModel):
    queries: List[str] = Field(..., min_length=1)
    max_results: int = Field(5, ge=1, le=20)


class SourceWeights(BaseModel):
    semantic_scholar: float = Field(0.5, ge=0.0, le=1.0)
    reddit: float = Field(0.3, ge=0.0, le=1.0)
    hacker_news: float = Field(0.1, ge=0.0, le=1.0)
    case_studies: float = Field(0.1, ge=0.0, le=1.0)

    @field_validator("case_studies")
    @classmethod
    def weights_sum_to_one(cls, v: float, info: Any) -> float:
        # Only validate on the last field (case_studies)
        data = info.data
        total = (
            data.get("semantic_scholar", 0.0)
            + data.get("reddit", 0.0)
            + data.get("hacker_news", 0.0)
            + v
        )
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"source_weights must sum to 1.0 (got {total:.3f})"
            )
        return v


class DomainConfig(BaseModel):
    domain_id: str
    domain_name: str
    description: str

    semantic_scholar: SemanticScholarConfig
    reddit: RedditConfig
    hacker_news: HackerNewsConfig
    case_studies: CaseStudiesConfig

    source_weights: SourceWeights = Field(default_factory=SourceWeights)

    expected_segments: List[str] = Field(..., min_length=1)
    default_agents_per_segment: int = Field(5, ge=1, le=50)


# ---------------------------------------------------------------------------
# 2. Raw scraped data models
# ---------------------------------------------------------------------------

class AcademicPaper(BaseModel):
    """A single result from Semantic Scholar."""
    title: str
    authors: List[str] = Field(default_factory=list)
    year: Optional[int] = None
    abstract: Optional[str] = None
    citation_count: int = 0
    url: Optional[str] = None
    fields_of_study: List[str] = Field(default_factory=list)


class SocialPost(BaseModel):
    """A single post from Reddit or Hacker News."""
    source: str  # "reddit" | "hacker_news"
    title: str
    body: Optional[str] = None
    author: Optional[str] = None
    score: int = 0
    comment_count: int = 0
    url: Optional[str] = None
    subreddit: Optional[str] = None  # Reddit only
    top_comments: List[str] = Field(default_factory=list)


class CaseStudySnippet(BaseModel):
    """A summarised web result for a case study query."""
    query: str
    title: str
    url: Optional[str] = None
    snippet: str
    llm_summary: Optional[str] = None  # Populated by synthesiser


class RawSources(BaseModel):
    """All scraped material before synthesis."""
    academic_papers: List[AcademicPaper] = Field(default_factory=list)
    social_posts: List[SocialPost] = Field(default_factory=list)
    case_study_snippets: List[CaseStudySnippet] = Field(default_factory=list)

    # Flags indicating which scrapers ran successfully
    academic_available: bool = False
    reddit_available: bool = False
    hacker_news_available: bool = False
    case_studies_available: bool = False


# ---------------------------------------------------------------------------
# 3. Human segment and research output models
# ---------------------------------------------------------------------------

class Demographics(BaseModel):
    age_range: Optional[str] = None
    gender_distribution: Optional[str] = None
    location: Optional[str] = None
    education_level: Optional[str] = None


class BehavioralProfile(BaseModel):
    technology_adoption: Optional[str] = None
    social_media_activity: Optional[str] = None
    key_concerns: List[str] = Field(default_factory=list)
    typical_behaviors: List[str] = Field(default_factory=list)
    likely_stance_on_topic: Optional[str] = None


class PersonaHints(BaseModel):
    """Soft hints passed to the LLM persona generator — not enforced."""
    likely_mbti: List[str] = Field(default_factory=list)
    example_professions: List[str] = Field(default_factory=list)
    likely_countries: List[str] = Field(default_factory=list)
    interested_topics: List[str] = Field(default_factory=list)


class HumanSegment(BaseModel):
    """A synthesised human segment profile."""
    id: str                          # e.g. "seg_001"
    name: str
    description: str
    size_hint: Optional[str] = None  # e.g. "~20% of teaching population"
    population_weight: float = Field(..., ge=0.0, le=1.0)

    demographics: Demographics = Field(default_factory=Demographics)
    behavioral_profile: BehavioralProfile = Field(default_factory=BehavioralProfile)
    persona_hints: PersonaHints = Field(default_factory=PersonaHints)

    # Short verbatim or paraphrased quotes from raw sources that support
    # this segment definition — surfaced to the persona generator for realism
    evidence_snippets: List[str] = Field(default_factory=list)


class SourceCounts(BaseModel):
    semantic_scholar_papers: int = 0
    reddit_posts: int = 0
    hacker_news_items: int = 0
    case_studies: int = 0


class ResearchMetadata(BaseModel):
    domain_id: str
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z"
    )
    artifact_document_name: Optional[str] = None


class ResearchOutput(BaseModel):
    """
    Top-level output of the research agent.

    Consumed by profile_bridge.py to generate OasisAgentProfile instances.
    Can also be serialised to JSON for caching / inspection.
    """
    domain: str
    research_summary: str

    human_segments: List[HumanSegment] = Field(..., min_length=1)

    raw_source_counts: SourceCounts = Field(default_factory=SourceCounts)
    metadata: ResearchMetadata

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()
