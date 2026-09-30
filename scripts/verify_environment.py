"""
Environment and Hardware Verification Script.
Checks Python version, PyTorch version, CUDA runtime, GPU acceleration, and core dependencies.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from anemia_ai.version import __version__


def verify_environment() -> bool:
    print("=" * 70)
    print(f"ANEMIA AI (v{__version__}) — ENVIRONMENT & RUNTIME VERIFICATION")
    print("=" * 70)

    # 1. Python Version
    py_ver = sys.version.split()[0]
    print(f"Python Version:       {py_ver}")
    assert sys.version_info >= (3, 9), "Python 3.9+ is required."

    # 2. PyTorch & CUDA
    print(f"PyTorch Version:      {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available:       {cuda_available}")

    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU Device:           {gpu_name}")
        print(f"CUDA Version:         {cuda_ver}")
        print(f"Total VRAM:           {vram_gb:.2f} GB")
    else:
        print("Running in CPU-only mode.")

    # 3. Core Dependencies
    dependencies = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("fastapi", "FastAPI"),
        ("pydantic", "Pydantic"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("sklearn", "Scikit-Learn"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("joblib", "Joblib"),
    ]

    print("\nVerifying Core Dependencies:")
    for mod_name, label in dependencies:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "loaded")
            print(f"  [OK] {label:15}: {ver}")
        except ImportError as e:
            print(f"  [FAIL] {label:15}: NOT FOUND ({e})")
            return False

    print("\n" + "=" * 70)
    print("ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
