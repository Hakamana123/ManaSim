"""
ManaSim research agent package.

Runs a pre-simulation research pipeline that scrapes academic, social, and
web sources for a given domain, then synthesises structured human segment
profiles consumed by the OASIS agent profile generator.

Typical usage:

    from backend.app.services.research import ResearchAgent

    agent = ResearchAgent()

    # Research only
    output = agent.run("education", artifact_text=pdf_text)

    # Research + agent profile generation in one call
    output, profiles = agent.run_with_profiles(
        "education",
        artifact_text=pdf_text,
        artifact_name="report.pdf",
        output_platform="reddit",
    )
"""

from .agent import ResearchAgent
from .domain_loader import DomainLoader, DomainNotFoundError, DomainValidationError
from .schemas import DomainConfig, HumanSegment, ResearchOutput

__all__ = [
    "ResearchAgent",
    "DomainLoader",
    "DomainNotFoundError",
    "DomainValidationError",
    "DomainConfig",
    "HumanSegment",
    "ResearchOutput",
]
