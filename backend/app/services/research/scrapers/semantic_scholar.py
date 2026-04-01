"""
Semantic Scholar scraper.

Uses the public Semantic Scholar Graph API (no API key required for basic use).
Fetches academic papers matching the queries defined in a DomainConfig and
returns a list of AcademicPaper objects.

Rate limits (unauthenticated):
  - 1 request/second for the search endpoint
  - Automatic retry with exponential back-off on 429 responses

API docs: https://api.semanticscholar.org/graph/v1
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import requests

from ..schemas import AcademicPaper, SemanticScholarConfig

logger = logging.getLogger("manasim.research.semantic_scholar")

_BASE_URL = "https://api.semanticscholar.org/graph/v1"
_SEARCH_ENDPOINT = f"{_BASE_URL}/paper/search"

# Fields requested from the API — keep this list small to reduce payload size
_FIELDS = "title,authors,year,abstract,citationCount,externalIds,fieldsOfStudy"

# Seconds to wait between successive query requests (respect 1 req/s limit)
_REQUEST_INTERVAL = 1.1

# Retry settings for 429 / transient errors
_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0  # seconds; actual wait = _BACKOFF_BASE ** attempt


class SemanticScholarScraper:
    """
    Fetches academic papers from the Semantic Scholar Graph API.

    Each configured query is run in sequence. Results are deduplicated by
    paper ID before being returned. Papers without an abstract are included
    but flagged via a missing abstract field so the synthesiser can
    down-weight them.
    """

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "User-Agent": "ManaSim-ResearchAgent/1.0",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape(self, config: SemanticScholarConfig) -> List[AcademicPaper]:
        """
        Run all queries in *config* and return deduplicated AcademicPaper list.

        Args:
            config: SemanticScholarConfig from the domain config file.

        Returns:
            List of AcademicPaper instances, deduplicated by Semantic Scholar
            paper ID. Returns an empty list (never raises) on total failure.
        """
        seen_ids: set = set()
        papers: List[AcademicPaper] = []

        for query in config.queries:
            try:
                batch = self._fetch_query(query, config)
                for paper in batch:
                    # Use URL as a stable dedup key
                    key = paper.url or paper.title
                    if key not in seen_ids:
                        seen_ids.add(key)
                        papers.append(paper)
            except Exception as exc:
                logger.warning(
                    "Semantic Scholar query '%s' failed: %s", query, exc
                )

            # Respect rate limit between queries
            time.sleep(_REQUEST_INTERVAL)

        logger.info(
            "Semantic Scholar: collected %d unique papers across %d queries",
            len(papers),
            len(config.queries),
        )
        return papers

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_query(
        self, query: str, config: SemanticScholarConfig
    ) -> List[AcademicPaper]:
        """Fetch one page of results for a single query string."""
        params: dict = {
            "query": query,
            "limit": config.max_results_per_query,
            "fields": _FIELDS,
        }
        if config.year_from:
            params["year"] = f"{config.year_from}-"

        response = self._get_with_retry(_SEARCH_ENDPOINT, params)
        data = response.json()

        raw_papers = data.get("data", [])
        return [self._parse_paper(p) for p in raw_papers if p]

    def _get_with_retry(self, url: str, params: dict) -> requests.Response:
        """GET with exponential back-off on 429 or transient server errors."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)

                if resp.status_code == 429:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Semantic Scholar rate-limited (429). "
                        "Waiting %.1fs before retry %d/%d.",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Semantic Scholar server error %d. "
                        "Waiting %.1fs before retry %d/%d.",
                        resp.status_code, wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                resp.raise_for_status()
                return resp

            except requests.RequestException as exc:
                last_exc = exc
                wait = _BACKOFF_BASE ** (attempt + 1)
                logger.warning(
                    "Semantic Scholar request error: %s. "
                    "Waiting %.1fs before retry %d/%d.",
                    exc, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Semantic Scholar: all {_MAX_RETRIES} retries exhausted. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _parse_paper(raw: dict) -> AcademicPaper:
        """Convert a raw API result dict into an AcademicPaper."""
        authors = [
            a.get("name", "") for a in raw.get("authors", []) if a.get("name")
        ]

        # Build a stable URL from external IDs when available
        url: Optional[str] = None
        ext_ids = raw.get("externalIds") or {}
        if doi := ext_ids.get("DOI"):
            url = f"https://doi.org/{doi}"
        elif ss_id := raw.get("paperId"):
            url = f"https://www.semanticscholar.org/paper/{ss_id}"

        return AcademicPaper(
            title=raw.get("title") or "Untitled",
            authors=authors,
            year=raw.get("year"),
            abstract=raw.get("abstract"),
            citation_count=raw.get("citationCount") or 0,
            url=url,
            fields_of_study=raw.get("fieldsOfStudy") or [],
        )
