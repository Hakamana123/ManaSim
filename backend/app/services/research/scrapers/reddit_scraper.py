"""
Reddit scraper.

Uses PRAW (Python Reddit API Wrapper) in read-only mode.

Required environment variables (set in .env):
    REDDIT_CLIENT_ID      — from https://www.reddit.com/prefs/apps
    REDDIT_CLIENT_SECRET  — from the same page
    REDDIT_USER_AGENT     — e.g. "ManaSim/1.0 by your_reddit_username"

If any of these are absent the scraper logs a warning and returns an empty
list, allowing the pipeline to degrade gracefully to the remaining sources.

PRAW docs: https://praw.readthedocs.io/
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from ..schemas import RedditConfig, SocialPost

logger = logging.getLogger("manasim.research.reddit")

# Number of top-level comments to capture per post
_TOP_COMMENTS_PER_POST = 5

# Maximum body length kept per post (characters) — keeps token usage bounded
_MAX_BODY_LENGTH = 1000

# Maximum comment length kept per comment
_MAX_COMMENT_LENGTH = 300


def _load_credentials() -> Optional[dict]:
    """
    Read Reddit credentials from the environment.

    Returns a dict of kwargs for praw.Reddit, or None if any required key
    is missing.
    """
    client_id = os.environ.get("REDDIT_CLIENT_ID", "").strip()
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "").strip()
    user_agent = os.environ.get("REDDIT_USER_AGENT", "").strip()

    missing = [
        name
        for name, val in [
            ("REDDIT_CLIENT_ID", client_id),
            ("REDDIT_CLIENT_SECRET", client_secret),
            ("REDDIT_USER_AGENT", user_agent),
        ]
        if not val
    ]

    if missing:
        logger.warning(
            "Reddit scraper disabled — missing environment variable(s): %s. "
            "Set these in .env to enable social data collection.",
            ", ".join(missing),
        )
        return None

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "user_agent": user_agent,
    }


class RedditScraper:
    """
    Fetches posts from Reddit subreddits and search queries.

    Two collection strategies are run per subreddit:
      1. Top posts of the configured time period (broad coverage)
      2. Keyword search within the subreddit for each search_query

    Top-level comments are collected for each post to capture the
    discussion texture that is most useful for segment synthesis.
    """

    def __init__(self) -> None:
        self._praw: Optional[object] = None  # praw.Reddit instance
        self._available = False
        self._init_client()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if Reddit credentials are present and PRAW initialised."""
        return self._available

    def scrape(self, config: RedditConfig) -> List[SocialPost]:
        """
        Collect posts from all subreddits defined in *config*.

        Args:
            config: RedditConfig from the domain config file.

        Returns:
            Deduplicated list of SocialPost objects. Returns an empty list
            (never raises) if credentials are absent or all requests fail.
        """
        if not self._available:
            return []

        seen_ids: set = set()
        posts: List[SocialPost] = []

        for subreddit_name in config.subreddits:
            try:
                batch = self._scrape_subreddit(subreddit_name, config)
                for post in batch:
                    key = post.url or post.title
                    if key not in seen_ids:
                        seen_ids.add(key)
                        posts.append(post)
            except Exception as exc:
                logger.warning(
                    "Reddit: failed to scrape r/%s: %s", subreddit_name, exc
                )

        logger.info(
            "Reddit: collected %d unique posts from %d subreddit(s)",
            len(posts),
            len(config.subreddits),
        )
        return posts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_client(self) -> None:
        """Initialise the PRAW client if credentials are available."""
        try:
            import praw  # noqa: PLC0415 — lazy import, optional dependency
        except ImportError:
            logger.warning(
                "Reddit scraper disabled — 'praw' is not installed. "
                "Run: pip install praw"
            )
            return

        creds = _load_credentials()
        if creds is None:
            return

        try:
            self._praw = praw.Reddit(**creds, read_only=True)
            # Trigger a lightweight API call to validate credentials early
            _ = self._praw.user.me()  # returns None in read-only mode; fine
            self._available = True
            logger.info("Reddit scraper initialised (read-only mode).")
        except Exception as exc:
            logger.warning(
                "Reddit scraper: credential validation failed: %s", exc
            )

    def _scrape_subreddit(
        self, subreddit_name: str, config: RedditConfig
    ) -> List[SocialPost]:
        """Collect top posts + search-query posts from one subreddit."""
        subreddit = self._praw.subreddit(subreddit_name)
        posts: List[SocialPost] = []

        # --- Strategy 1: top posts of the configured time window ---
        try:
            top_posts = subreddit.top(
                time_filter=config.time_filter,
                limit=config.max_posts_per_subreddit,
            )
            for submission in top_posts:
                posts.append(self._parse_submission(submission, subreddit_name))
        except Exception as exc:
            logger.warning(
                "Reddit r/%s top posts failed: %s", subreddit_name, exc
            )

        # --- Strategy 2: keyword search within the subreddit ---
        for query in config.search_queries:
            try:
                search_results = subreddit.search(
                    query,
                    sort=config.sort,
                    time_filter=config.time_filter,
                    limit=min(10, config.max_posts_per_subreddit),
                )
                for submission in search_results:
                    posts.append(self._parse_submission(submission, subreddit_name))
            except Exception as exc:
                logger.warning(
                    "Reddit r/%s search '%s' failed: %s",
                    subreddit_name, query, exc,
                )

        return posts

    def _parse_submission(self, submission: object, subreddit_name: str) -> SocialPost:
        """Convert a PRAW Submission object into a SocialPost."""
        # Collect top-level comments, skip MoreComments objects
        top_comments: List[str] = []
        try:
            submission.comments.replace_more(limit=0)
            for comment in submission.comments.list()[:_TOP_COMMENTS_PER_POST]:
                body = getattr(comment, "body", "") or ""
                if body and body != "[deleted]" and body != "[removed]":
                    top_comments.append(body[:_MAX_COMMENT_LENGTH])
        except Exception as exc:
            logger.debug("Could not fetch comments for post '%s': %s",
                         getattr(submission, "id", "?"), exc)

        # Selftext may be absent on link posts
        raw_body = getattr(submission, "selftext", "") or ""
        body = raw_body[:_MAX_BODY_LENGTH] if raw_body else None

        return SocialPost(
            source="reddit",
            title=getattr(submission, "title", "") or "",
            body=body,
            author=str(getattr(submission, "author", None) or ""),
            score=getattr(submission, "score", 0) or 0,
            comment_count=getattr(submission, "num_comments", 0) or 0,
            url=f"https://reddit.com{getattr(submission, 'permalink', '')}",
            subreddit=subreddit_name,
            top_comments=top_comments,
        )
