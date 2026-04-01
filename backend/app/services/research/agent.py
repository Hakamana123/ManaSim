"""
Research agent orchestrator.

This is the entry point for the pre-simulation research pipeline. It:

  1. Loads the domain config from domains/<domain_id>.json
  2. Runs all configured scrapers concurrently (Semantic Scholar, Reddit,
     Hacker News, case studies)
  3. Passes the scraped material to the LLM synthesiser to produce human
     segment profiles
  4. Optionally passes the segment profiles through the profile bridge to
     produce ready-to-use OasisAgentProfile instances

Typical call from a Flask route or a standalone script:

    agent = ResearchAgent()
    output = agent.run(
        domain_id="education",
        artifact_text=extracted_pdf_text,
        artifact_name="edtech_report_2025.pdf",
    )
    # output is a ResearchOutput — pass to ProfileBridge or serialise to JSON

Concurrency note
----------------
The four scrapers hit independent external APIs so they are run concurrently
via ThreadPoolExecutor. The total wall-clock time is dominated by the slowest
scraper (typically Semantic Scholar due to its 1 req/s rate limit) rather
than the sum of all scraper times.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from ..oasis_profile_generator import OasisAgentProfile
from .domain_loader import DomainLoader
from .profile_bridge import ProfileBridge
from .schemas import DomainConfig, RawSources, ResearchOutput
from .scrapers import (
    CaseStudiesScraper,
    HackerNewsScraper,
    RedditScraper,
    SemanticScholarScraper,
)
from .synthesiser import Synthesiser

logger = logging.getLogger("manasim.research.agent")


class ResearchAgent:
    """
    Orchestrates the full research pipeline for a single simulation run.

    All scraper instances are created fresh per ResearchAgent instance so
    the agent is safe to re-use across multiple `run()` calls within the
    same process lifetime.
    """

    def __init__(self, domains_dir: Optional[str] = None) -> None:
        self._loader = DomainLoader(domains_dir=domains_dir)

        # Scrapers — initialised once, reused across run() calls
        self._semantic_scholar = SemanticScholarScraper()
        self._reddit = RedditScraper()
        self._hacker_news = HackerNewsScraper()
        self._case_studies = CaseStudiesScraper()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        domain_id: str,
        artifact_text: Optional[str] = None,
        artifact_name: Optional[str] = None,
    ) -> ResearchOutput:
        """
        Execute the full research pipeline and return structured findings.

        Args:
            domain_id:     Domain identifier matching a file in domains/.
                           Case-insensitive (e.g. "education", "organisation").
            artifact_text: Optional text extracted from the user's uploaded
                           document. Passed to the synthesiser as a primary
                           source.
            artifact_name: Optional filename of the uploaded document (used
                           for metadata only).

        Returns:
            A ResearchOutput containing human segment profiles and metadata.

        Raises:
            DomainNotFoundError:   if no config exists for domain_id.
            DomainValidationError: if the config file fails schema validation.
            ValueError:            if the LLM synthesis step fails completely.
        """
        config = self._loader.load(domain_id)
        logger.info(
            "ResearchAgent: starting pipeline for domain '%s' (%s)",
            config.domain_id, config.domain_name,
        )

        raw = self._scrape(config)
        self._log_scrape_summary(raw)

        synthesiser = Synthesiser()
        output = synthesiser.synthesise(
            raw=raw,
            config=config,
            artifact_text=artifact_text,
            artifact_name=artifact_name,
        )

        logger.info(
            "ResearchAgent: synthesis complete — %d segments produced for '%s'",
            len(output.human_segments), config.domain_id,
        )
        return output

    def run_with_profiles(
        self,
        domain_id: str,
        artifact_text: Optional[str] = None,
        artifact_name: Optional[str] = None,
        agents_per_segment: Optional[int] = None,
        output_platform: str = "reddit",
        parallel_bridge: bool = False,
    ) -> tuple[ResearchOutput, List[OasisAgentProfile]]:
        """
        Execute the full pipeline including agent profile generation.

        Convenience wrapper that calls run() and then the ProfileBridge so
        callers can get both the research output and the ready-to-use profiles
        in a single call.

        Args:
            domain_id:          Domain identifier.
            artifact_text:      Optional uploaded document text.
            artifact_name:      Optional filename.
            agents_per_segment: Agents to generate per segment. Defaults to
                                the domain config's default_agents_per_segment.
            output_platform:    "reddit" or "twitter".
            parallel_bridge:    Whether to generate segment profiles in parallel.

        Returns:
            (ResearchOutput, List[OasisAgentProfile])
        """
        output = self.run(
            domain_id=domain_id,
            artifact_text=artifact_text,
            artifact_name=artifact_name,
        )

        config = self._loader.load(domain_id)
        count = agents_per_segment or config.default_agents_per_segment

        bridge = ProfileBridge()
        profiles = bridge.generate(
            research=output,
            agents_per_segment=count,
            output_platform=output_platform,
            parallel=parallel_bridge,
        )

        logger.info(
            "ResearchAgent: profile bridge complete — %d total agent profiles",
            len(profiles),
        )
        return output, profiles

    def list_domains(self) -> List[str]:
        """Return all available domain IDs from the domains/ directory."""
        return self._loader.list_domain_ids()

    def load_config(self, domain_id: str) -> DomainConfig:
        """Load and return the DomainConfig for inspection without running the pipeline."""
        return self._loader.load(domain_id)

    # ------------------------------------------------------------------
    # Scraping
    # ------------------------------------------------------------------

    def _scrape(self, config: DomainConfig) -> RawSources:
        """
        Run all scrapers concurrently and aggregate results into RawSources.

        Each scraper runs in its own thread. Failures are caught per-scraper
        so a single source outage does not block the others.
        """
        raw = RawSources()

        tasks = {
            "semantic_scholar": lambda: self._semantic_scholar.scrape(
                config.semantic_scholar
            ),
            "reddit": lambda: self._reddit.scrape(config.reddit),
            "hacker_news": lambda: self._hacker_news.scrape(config.hacker_news),
            "case_studies": lambda: self._case_studies.scrape(config.case_studies),
        }

        results: dict = {}

        with ThreadPoolExecutor(max_workers=4) as pool:
            future_to_name = {
                pool.submit(fn): name for name, fn in tasks.items()
            }
            for future in as_completed(future_to_name):
                name = future_to_name[future]
                try:
                    results[name] = future.result()
                    logger.debug(
                        "Scraper '%s' completed: %d items",
                        name, len(results[name]),
                    )
                except Exception as exc:
                    logger.error(
                        "Scraper '%s' raised an exception: %s", name, exc
                    )
                    results[name] = []

        # Populate RawSources
        raw.academic_papers = results.get("semantic_scholar", [])
        raw.academic_available = bool(raw.academic_papers)

        reddit_posts = results.get("reddit", [])
        hn_posts = results.get("hacker_news", [])
        raw.social_posts = reddit_posts + hn_posts
        raw.reddit_available = bool(reddit_posts) and self._reddit.available
        raw.hacker_news_available = bool(hn_posts)

        raw.case_study_snippets = results.get("case_studies", [])
        raw.case_studies_available = bool(raw.case_study_snippets)

        return raw

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_scrape_summary(raw: RawSources) -> None:
        reddit_count = sum(1 for p in raw.social_posts if p.source == "reddit")
        hn_count = sum(1 for p in raw.social_posts if p.source == "hacker_news")

        logger.info(
            "Scrape summary — academic: %d | reddit: %d | HN: %d | case studies: %d",
            len(raw.academic_papers),
            reddit_count,
            hn_count,
            len(raw.case_study_snippets),
        )

        if not raw.academic_available:
            logger.warning("Semantic Scholar returned no papers.")
        if not raw.reddit_available:
            logger.warning(
                "Reddit data unavailable. Check REDDIT_* environment variables."
            )
        if not raw.hacker_news_available:
            logger.warning("Hacker News returned no results.")
        if not raw.case_studies_available:
            logger.warning(
                "Case studies unavailable. "
                "Ensure duckduckgo-search is installed: pip install duckduckgo-search"
            )
