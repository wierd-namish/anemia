"""
Automated Quality Gate Runner.
Runs environment checks, secret scanning, model checkpoint verification, and complete test suites.
"""

import os
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR))


def run_secret_scan() -> bool:
    print("\n[GATE 1] Running Secret & Credential Scan...")
    patterns = [
        re.compile(r'(?i)(api[_-]?key|secret|password|bearer|credential)\s*[:=]\s*[\'"][^\'"]{8,}'),
        re.compile(r'(?i)ghp_[a-zA-Z0-9]{36}'),
        re.compile(r'(?i)AKIA[0-9A-Z]{16}'),
    ]

    findings = []
    for root, dirs, files in os.walk(BASE_DIR):
        if any(x in root for x in ["__pycache__", ".git", "Fingernails", ".pytest_cache"]):
            continue
        for f in files:
            if f.endswith((".pyc", ".joblib", ".pth", ".pt", ".png", ".jpg", ".jpeg")):
                continue
            p = os.path.join(root, f)
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                    for lno, line in enumerate(fh, 1):
                        for pat in patterns:
                            if pat.search(line):
                                findings.append((p, lno, line.strip()))
            except Exception:
                pass

    if findings:
        print(f"  [FAIL] Detected {len(findings)} potential secret patterns:")
        for f, l, s in findings[:5]:
            print(f"    {f}:{l} -> {s[:60]}")
        return False

    print("  [PASS] Zero secrets or credentials detected.")
    return True


def run_checkpoint_verification() -> bool:
    print("\n[GATE 2] Verifying Model Checkpoints & Artifacts...")
    from scripts.verify_checkpoint import verify_checkpoints
    return verify_checkpoints()


def run_tests() -> bool:
    print("\n[GATE 3] Executing Automated Unit, API, and Inference Tests...")
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(str(BASE_DIR / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=1)
    result = runner.run(suite)
    return result.wasSuccessful()


def main():
    print("=" * 70)
    print("ANEMIA AI — PRODUCTION QUALITY GATES & VERIFICATION")
    print("=" * 70)

    g1 = run_secret_scan()
    g2 = run_checkpoint_verification()
    g3 = run_tests()

    print("\n" + "=" * 70)
    print("QUALITY GATE SUMMARY:")
    print(f"  1. Secret & Privacy Scan:   {'PASS' if g1 else 'FAIL'}")
    print(f"  2. Model Checkpoint Audit:  {'PASS' if g2 else 'FAIL'}")
    print(f"  3. Automated Test Suite:    {'PASS' if g3 else 'FAIL'}")
    print("=" * 70)

    if g1 and g2 and g3:
        print("ALL QUALITY GATES PASSED! REPOSITORY IS PRODUCTION-READY.")
        sys.exit(0)
    else:
        print("ONE OR MORE QUALITY GATES FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
