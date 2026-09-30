"""
Test fixtures package.
"""

from tests.fixtures.synthetic_samples import (
    generate_blurry_image,
    generate_overexposed_image,
    generate_polished_nail,
    generate_synthetic_nail,
    generate_underexposed_image,
    generate_wood_desk_ood,
)

__all__ = [
    "generate_synthetic_nail",
    "generate_blurry_image",
    "generate_underexposed_image",
    "generate_overexposed_image",
    "generate_polished_nail",
    "generate_wood_desk_ood",
]
