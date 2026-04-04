# Pluggable rubric implementations for the validation layer.
#
# Each rubric subclasses BaseRubric and is auto-registered in RUBRIC_REGISTRY
# when its module is imported.  The scorer resolves rubrics by name through
# that registry rather than importing concrete classes directly.

from .base import BaseRubric, RUBRIC_REGISTRY

__all__ = ["BaseRubric", "RUBRIC_REGISTRY"]
