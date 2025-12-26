"""Evidence Suite - Quick Setup Script
Run this to install dependencies and verify the setup.
"""

import subprocess
import sys
from pathlib import Path


def main():
    print("=" * 50)
    print("  Evidence Suite: Setup")
    print("=" * 50)

    project_dir = Path(__file__).parent

    # Step 1: Create virtual environment (optional)
    print("\n[1/4] Checking Python version...")
    print(f"  Python: {sys.version}")

    # Step 2: Install dependencies
    print("\n[2/4] Installing dependencies...")
    print("  This may take a few minutes (downloading BERT model)...")

    requirements_file = project_dir / "requirements.txt"

    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], check=True
        )
        print("  Dependencies installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: Failed to install dependencies: {e}")
        return 1

    # Step 3: Verify Tesseract
    print("\n[3/4] Verifying Tesseract OCR...")
    try:
        result = subprocess.run(
            ["tesseract", "--version"], check=False, capture_output=True, text=True
        )
        version = result.stdout.split("\n")[0]
        print(f"  Tesseract: {version}")
    except FileNotFoundError:
        print("  WARNING: Tesseract not found in PATH")
        print("  OCR features may not work without Tesseract")

    # Step 4: Quick import test
    print("\n[4/4] Testing imports...")
    try:
        sys.path.insert(0, str(project_dir))
        from core.models import EvidencePacket
        from pipeline import EvidencePipeline

        print("  Core imports: OK")
    except ImportError as e:
        print(f"  ERROR: Import failed: {e}")
        return 1

    print("\n" + "=" * 50)
    print("  Setup Complete!")
    print("=" * 50)
    print("\nTo run the pipeline test:")
    print(f"  cd {project_dir}")
    print("  python run_pipeline.py")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
