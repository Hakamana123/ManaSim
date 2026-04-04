# Pluggable rubric implementations for the validation layer.
#
# Concrete rubric modules are imported here so that RubricMeta registers them
# into RUBRIC_REGISTRY on package import.  The scorer resolves rubrics by name
# through that registry rather than importing concrete classes directly.

from .base import BaseRubric, RUBRIC_REGISTRY

# Trigger auto-registration of all built-in rubrics
from . import education       # noqa: F401  registers NAME="education"
from . import organisation    # noqa: F401  registers NAME="organisation"

__all__ = ["BaseRubric", "RUBRIC_REGISTRY"]
