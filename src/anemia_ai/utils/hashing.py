"""
Cryptographic hashing utilities for artifact verification.
"""

import hashlib
from pathlib import Path
from typing import Union


def compute_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA256 checksum of a file in 64KB chunks."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for hash calculation: {path}")

    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()
