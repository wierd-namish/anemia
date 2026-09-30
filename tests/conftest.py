"""
Pytest configuration and global fixtures.
"""

import os
import sys
from pathlib import Path
import pytest

# Add project root and src to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tests.fixtures.synthetic_samples import (
    generate_synthetic_nail,
    generate_blurry_image,
    generate_underexposed_image,
    generate_overexposed_image,
    generate_polished_nail,
    generate_wood_desk_ood,
)


@pytest.fixture
def sample_healthy_nail():
    return generate_synthetic_nail(is_anemic=False)


@pytest.fixture
def sample_anemic_nail():
    return generate_synthetic_nail(is_anemic=True)


@pytest.fixture
def sample_blurry_image():
    return generate_blurry_image()


@pytest.fixture
def sample_ood_desk():
    return generate_wood_desk_ood()
