"""
Case studies scraper.

Uses duckduckgo-search (ddgs library, no API key required) to search the
open web for documented real-world examples, reports, and case studies
relevant to the simulation domain.

Each query from the domain config is run in sequence. Results are returned
as CaseStudySnippet objects containing the title, URL, and the body snippet
that DuckDuckGo extracts from the page. The llm_summary field is left None
here — the synthesiser may use the raw snippet directly.

Install: pip install duckduckgo-search
"""

from __future__ import annotations

import logging
import time
from typing import List

from ..schemas import CaseStudiesConfig, CaseStudySnippet

logger = logging.getLogger("manasim.research.case_studies")

# Polite delay between queries
_REQUEST_INTERVAL = 1.5


class CaseStudiesScraper:
    """
    Searches the web for case study results using DuckDuckGo text search.

    Degrades gracefully if the duckduckgo-search package is not installed.
    """

    def __init__(self) -> None:
        self._ddgs_cls = None
        self._available = False
        self._init_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if duckduckgo-search is installed and ready."""
        return self._available

    def scrape(self, config: CaseStudiesConfig) -> List[CaseStudySnippet]:
        """
        Run all case study queries in *config* and return deduplicated snippets.

        Args:
            config: CaseStudiesConfig from the domain config file.

        Returns:
            Deduplicated list of CaseStudySnippet objects. Returns an empty
            list (never raises) if the library is absent or all queries fail.
        """
        if not self._available:
            return []

        seen_urls: set = set()
        snippets: List[CaseStudySnippet] = []

        for query in config.queries:
            try:
                batch = self._fetch_query(query, config.max_results)
                for snippet in batch:
                    key = snippet.url or snippet.title
                    if key not in seen_urls:
                        seen_urls.add(key)
                        snippets.append(snippet)
            except Exception as exc:
                logger.warning(
                    "Case studies query '%s' failed: %s", query, exc
                )

            time.sleep(_REQUEST_INTERVAL)

        logger.info(
            "Case studies: collected %d unique results across %d queries",
            len(snippets),
            len(config.queries),
        )
        return snippets

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Lazy-import duckduckgo_search and verify it is usable."""
        try:
            from duckduckgo_search import DDGS  # noqa: PLC0415
            self._ddgs_cls = DDGS
            self._available = True
            logger.info("Case studies scraper initialised (DuckDuckGo).")
        except ImportError:
            logger.warning(
                "Case studies scraper disabled — 'duckduckgo-search' is not "
                "installed. Run: pip install duckduckgo-search"
            )

    def _fetch_query(
        self, query: str, max_results: int
    ) -> List[CaseStudySnippet]:
        """Fetch DuckDuckGo text results for a single query string."""
        results: List[CaseStudySnippet] = []

        with self._ddgs_cls() as ddgs:
            hits = list(ddgs.text(query, max_results=max_results))

        for hit in hits:
            title = hit.get("title") or "Untitled"
            url = hit.get("href")
            body = hit.get("body") or ""

            results.append(
                CaseStudySnippet(
                    query=query,
                    title=title,
                    url=url,
                    snippet=body,
                )
            )

        return results
