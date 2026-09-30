"""
Deterministic External Model Downloader.
Downloads and validates Hugging Face JetX-GT/nail-anemia-detector artifacts with SHA256 integrity verification.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anemia_ai.config.settings import get_settings
from anemia_ai.models.jetx_nail import ensure_jetx_artifacts_downloaded


def main() -> None:
    settings = get_settings()
    print("=" * 70)
    print("ANEMIA AI — MODEL DOWNLOAD & ARTIFACT SYNCHRONIZATION")
    print("=" * 70)
    print(f"Target Cache Directory: {settings.jetx_cache_dir}")

    hashes = ensure_jetx_artifacts_downloaded(settings.jetx_cache_dir)
    print("\nVerified Downloaded Artifacts:")
    for fname, sha in hashes.items():
        print(f"  - {fname:25}: SHA256 {sha}")

    print("\n[OK] Model synchronization complete.")


if __name__ == "__main__":
    main()
