"""
Backward-compatibility facade for quality checks.
Re-exports from anemia_ai.preprocessing.image_quality.
"""

from anemia_ai.preprocessing.image_quality import assess_image_quality

__all__ = ["assess_image_quality"]
