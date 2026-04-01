"""
Hacker News scraper.

Uses the Algolia HN Search API — no API key required.

Endpoint docs: https://hn.algolia.com/api

Two requests are made per query:
  1. Search stories   — high-score items matching the query
  2. Search comments  — top-voted comment threads on the same query

This dual approach captures both the "headline signal" (what the HN crowd
found worth sharing) and the "discussion signal" (how they actually reacted),
which maps well onto the opinion-leader and early-adopter segments typical
of HN demographics.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import requests

from ..schemas import HackerNewsConfig, SocialPost

logger = logging.getLogger("manasim.research.hacker_news")

_ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
_SEARCH_URL = f"{_ALGOLIA_BASE}/search"

# Seconds between successive requests — Algolia is generous but be polite
_REQUEST_INTERVAL = 0.5

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

# Characters kept from each comment body
_MAX_COMMENT_LENGTH = 400

# Top comments to fetch per story (via a second Algolia call on the item)
_TOP_COMMENTS_PER_STORY = 5

# Maximum story body / comment text length stored on the SocialPost
_MAX_BODY_LENGTH = 800


class HackerNewsScraper:
    """
    Fetches stories and comment threads from Hacker News via Algolia.

    For each query in the domain config, the scraper:
      - Fetches the top-scoring stories
      - For each story, retrieves up to _TOP_COMMENTS_PER_STORY top comments
        using the Algolia items endpoint (sorted by points)
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

    def scrape(self, config: HackerNewsConfig) -> List[SocialPost]:
        """
        Run all queries in *config* and return deduplicated SocialPost list.

        Args:
            config: HackerNewsConfig from the domain config file.

        Returns:
            Deduplicated list of SocialPost objects. Returns an empty list
            (never raises) on total failure.
        """
        seen_ids: set = set()
        posts: List[SocialPost] = []

        for query in config.queries:
            try:
                batch = self._fetch_query(query, config.max_results_per_query)
                for post in batch:
                    key = post.url or post.title
                    if key not in seen_ids:
                        seen_ids.add(key)
                        posts.append(post)
            except Exception as exc:
                logger.warning(
                    "Hacker News query '%s' failed: %s", query, exc
                )

            time.sleep(_REQUEST_INTERVAL)

        logger.info(
            "Hacker News: collected %d unique items across %d queries",
            len(posts),
            len(config.queries),
        )
        return posts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_query(self, query: str, limit: int) -> List[SocialPost]:
        """Fetch stories and their top comments for a single query."""
        params = {
            "query": query,
            "tags": "story",          # stories only in the first pass
            "hitsPerPage": limit,
            "attributesToRetrieve": (
                "objectID,title,url,author,points,num_comments,"
                "story_text,created_at"
            ),
        }

        resp = self._get_with_retry(_SEARCH_URL, params)
        hits = resp.json().get("hits", [])

        posts: List[SocialPost] = []
        for hit in hits:
            story_id = hit.get("objectID")
            top_comments = self._fetch_top_comments(story_id) if story_id else []
            posts.append(self._parse_hit(hit, top_comments))
            time.sleep(_REQUEST_INTERVAL)

        return posts

    def _fetch_top_comments(self, story_id: str) -> List[str]:
        """
        Retrieve the top comments for a story using the Algolia items endpoint.

        Sorted by points descending so the most upvoted comments come first.
        """
        params = {
            "tags": f"comment,story_{story_id}",
            "hitsPerPage": _TOP_COMMENTS_PER_STORY,
            "attributesToRetrieve": "comment_text,points",
        }

        try:
            resp = self._get_with_retry(_SEARCH_URL, params)
            hits = resp.json().get("hits", [])
            # Sort by points descending (Algolia doesn't guarantee order here)
            hits_sorted = sorted(
                hits, key=lambda h: h.get("points") or 0, reverse=True
            )
            comments: List[str] = []
            for h in hits_sorted[:_TOP_COMMENTS_PER_STORY]:
                text = h.get("comment_text") or ""
                text = _strip_html(text)
                if text:
                    comments.append(text[:_MAX_COMMENT_LENGTH])
            return comments
        except Exception as exc:
            logger.debug(
                "Could not fetch comments for HN story %s: %s", story_id, exc
            )
            return []

    def _get_with_retry(self, url: str, params: dict) -> requests.Response:
        """GET with exponential back-off on 429 or transient server errors."""
        last_exc: Optional[Exception] = None

        for attempt in range(_MAX_RETRIES):
            try:
                resp = self._session.get(url, params=params, timeout=self._timeout)

                if resp.status_code == 429:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Algolia HN rate-limited (429). "
                        "Waiting %.1fs before retry %d/%d.",
                        wait, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    continue

                if resp.status_code >= 500:
                    wait = _BACKOFF_BASE ** (attempt + 1)
                    logger.warning(
                        "Algolia HN server error %d. "
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
                    "Hacker News request error: %s. "
                    "Waiting %.1fs before retry %d/%d.",
                    exc, wait, attempt + 1, _MAX_RETRIES,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Hacker News: all {_MAX_RETRIES} retries exhausted. "
            f"Last error: {last_exc}"
        )

    @staticmethod
    def _parse_hit(hit: dict, top_comments: List[str]) -> SocialPost:
        """Convert an Algolia hit dict into a SocialPost."""
        story_text = hit.get("story_text") or ""
        body = _strip_html(story_text)[:_MAX_BODY_LENGTH] if story_text else None

        # External URL for link posts; fall back to the HN discussion page
        object_id = hit.get("objectID", "")
        external_url = hit.get("url") or f"https://news.ycombinator.com/item?id={object_id}"

        return SocialPost(
            source="hacker_news",
            title=hit.get("title") or "Untitled",
            body=body,
            author=hit.get("author"),
            score=hit.get("points") or 0,
            comment_count=hit.get("num_comments") or 0,
            url=external_url,
            subreddit=None,
            top_comments=top_comments,
        )


def _strip_html(text: str) -> str:
    """
    Remove common HTML tags from Algolia-returned story/comment text.

    Algolia stores HN text with minimal HTML markup (<p>, <a>, <i>, <b>,
    <code>, <pre>). This function does a lightweight tag strip without
    pulling in an HTML parser dependency.
    """
    import re
    # Replace block-level tags with a space to preserve word boundaries
    text = re.sub(r"<(?:p|br|/p)[^>]*>", " ", text, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text
