"""
Research agent scrapers package.

Exports one scraper class per data source. Each scraper is independently
optional — missing credentials or packages cause a warning and graceful
degradation rather than a hard failure.
"""

from .case_studies import CaseStudiesScraper
from .hacker_news import HackerNewsScraper
from .reddit_scraper import RedditScraper
from .semantic_scholar import SemanticScholarScraper

__all__ = [
    "CaseStudiesScraper",
    "HackerNewsScraper",
    "RedditScraper",
    "SemanticScholarScraper",
]
