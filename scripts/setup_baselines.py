#!/usr/bin/env python3
"""
Download model weights for Phishpedia and PhishIntention baselines.
Uses gdown to fetch from Google Drive.

Usage:
  py -3 scripts/setup_baselines.py
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINES_DIR = PROJECT_ROOT / "baselines_repos"

# Google Drive file IDs for each baseline
PHISHPEDIA_MODELS = {
    "rcnn_bet365.pth": "1tE2Mu5WC8uqCxei3XqAd7AWaP5JTmVWH",
    "faster_rcnn.yaml": "1Q6lqjpl4exW7q_dPbComcj0udBMDl8CW",
    "resnetv2_rgb_new.pth.tar": "1H0Q_DbdKPLFcZee8I14K62qV7TTy7xvS",
    "expand_targetlist.zip": "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I",
    "domain_map.pkl": "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1",
}

PHISHINTENTION_MODELS = {
    "layout_detector.pth": "1HWjE5Fv-c3nCDzLCBc7I3vClP1IeuP_I",
    "crp_classifier.pth.tar": "1igEMRz0vFBonxAILeYMRWTyd7A9sRirO",
    "crp_locator.pth": "1_O5SALqaJqvWoZDrdIVpsZyCnmSkzQcm",
    "ocr_pretrained.pth.tar": "15pfVWnZR-at46gqxd50cWhrXemP8oaxp",
    "ocr_siamese.pth.tar": "1BxJf5lAcNEnnC0In55flWZ89xwlYkzPk",
    "expand_targetlist.zip": "1fr5ZxBKyDiNZ_1B6rRAfZbAHBBoUjZ7I",
    "domain_map.pkl": "1qSdkSSoCYUkZMKs44Rup_1DPBxHnEKl1",
}


def ensure_gdown():
    """Ensure gdown is installed."""
    try:
        import gdown
        return True
    except ImportError:
        print("Installing gdown...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "--quiet"])
        return True


def download_file(file_id: str, output_path: str):
    """Download a file from Google Drive using gdown."""
    import gdown
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"  Downloading {os.path.basename(output_path)}...")
    try:
        gdown.download(url, output_path, quiet=False)
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"    ✓ {size_mb:.1f} MB")
            return True
        else:
            print(f"    ✗ Download failed")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False


def extract_targetlist(models_dir: Path):
    """Extract expand_targetlist.zip and flatten if needed."""
    zip_path = models_dir / "expand_targetlist.zip"
    if not zip_path.exists():
        print("  ✗ expand_targetlist.zip not found")
        return False

    target_dir = models_dir / "expand_targetlist"
    if target_dir.exists() and len(list(target_dir.iterdir())) > 0:
        print("  ✓ expand_targetlist already extracted")
        return True

    print("  Extracting expand_targetlist.zip...")
    target_dir.mkdir(exist_ok=True)
    with zipfile.ZipFile(str(zip_path), 'r') as zf:
        zf.extractall(str(target_dir))

    # Flatten if nested
    nested = target_dir / "expand_targetlist"
    if nested.exists() and nested.is_dir():
        print("  Flattening nested directory...")
        import shutil
        for item in nested.iterdir():
            dest = target_dir / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        shutil.rmtree(str(nested))

    num_files = len(list(target_dir.rglob("*")))
    print(f"    ✓ Extracted {num_files} files")
    return True


def setup_phishpedia():
    """Download Phishpedia model weights."""
    print("\n" + "=" * 60)
    print("Setting up Phishpedia")
    print("=" * 60)

    models_dir = BASELINES_DIR / "Phishpedia" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for filename, file_id in PHISHPEDIA_MODELS.items():
        output_path = str(models_dir / filename)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"  ✓ {filename} already exists")
            success += 1
            continue
        if download_file(file_id, output_path):
            success += 1

    # Extract targetlist
    extract_targetlist(models_dir)

    print(f"\nPhishpedia: {success}/{len(PHISHPEDIA_MODELS)} models downloaded")
    return success == len(PHISHPEDIA_MODELS)


def setup_phishintention():
    """Download PhishIntention model weights."""
    print("\n" + "=" * 60)
    print("Setting up PhishIntention")
    print("=" * 60)

    models_dir = BASELINES_DIR / "PhishIntention" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    success = 0
    for filename, file_id in PHISHINTENTION_MODELS.items():
        output_path = str(models_dir / filename)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            print(f"  ✓ {filename} already exists")
            success += 1
            continue
        if download_file(file_id, output_path):
            success += 1

    # Extract targetlist
    extract_targetlist(models_dir)

    print(f"\nPhishIntention: {success}/{len(PHISHINTENTION_MODELS)} models downloaded")
    return success == len(PHISHINTENTION_MODELS)


def install_detectron2():
    """Install detectron2 (CPU version)."""
    print("\n" + "=" * 60)
    print("Installing detectron2 (CPU)")
    print("=" * 60)

    try:
        import detectron2
        print(f"  ✓ detectron2 already installed: {detectron2.__version__}")
        return True
    except ImportError:
        pass

    # Try the pre-built CPU-only wheels first
    print("  Installing detectron2 from source (CPU-only)...")
    try:
        # Method 1: try pre-built wheel index
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--extra-index-url", "https://miropsota.github.io/torch_packages_builder",
             "detectron2"],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0:
            print("  ✓ detectron2 installed via pre-built wheel")
            return True
    except Exception:
        pass

    # Method 2: from GitHub
    try:
        print("  Trying GitHub source install...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--no-build-isolation",
             "git+https://github.com/facebookresearch/detectron2.git"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print("  ✓ detectron2 installed from GitHub")
            return True
        else:
            print(f"  ✗ GitHub install failed: {result.stderr[:200]}")
    except Exception as e:
        print(f"  ✗ GitHub install error: {e}")

    return False


def install_dependencies():
    """Install all required pip packages."""
    print("\n" + "=" * 60)
    print("Installing dependencies")
    print("=" * 60)

    packages = [
        "torch", "torchvision", "opencv-python", "Pillow",
        "pyyaml", "gdown", "tqdm", "numpy", "scipy",
        "scikit-learn",
    ]
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_").split("[")[0])
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  Installing {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                capture_output=True
            )


def verify_setup():
    """Verify that everything is properly set up."""
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    # Check Phishpedia models
    pp_models = BASELINES_DIR / "Phishpedia" / "models"
    pp_ok = True
    for name in PHISHPEDIA_MODELS:
        if name.endswith(".zip"):
            folder = pp_models / name.replace(".zip", "")
            exists = folder.exists() and len(list(folder.iterdir())) > 0
        else:
            exists = (pp_models / name).exists()
        status = "✓" if exists else "✗"
        if not exists:
            pp_ok = False
        print(f"  Phishpedia/{name}: {status}")

    # Check PhishIntention models
    pi_models = BASELINES_DIR / "PhishIntention" / "models"
    pi_ok = True
    for name in PHISHINTENTION_MODELS:
        if name.endswith(".zip"):
            folder = pi_models / name.replace(".zip", "")
            exists = folder.exists() and len(list(folder.iterdir())) > 0
        else:
            exists = (pi_models / name).exists()
        status = "✓" if exists else "✗"
        if not exists:
            pi_ok = False
        print(f"  PhishIntention/{name}: {status}")

    # Check detectron2
    try:
        import detectron2
        print(f"  detectron2: ✓ ({detectron2.__version__})")
    except ImportError:
        print("  detectron2: ✗")
        pi_ok = False
        pp_ok = False

    print(f"\nPhishpedia ready: {'YES' if pp_ok else 'NO'}")
    print(f"PhishIntention ready: {'YES' if pi_ok else 'NO'}")

    return pp_ok, pi_ok


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-models", action="store_true",
                        help="Skip model weight downloads")
    parser.add_argument("--skip-detectron2", action="store_true",
                        help="Skip detectron2 installation")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only verify existing setup")
    args = parser.parse_args()

    if args.verify_only:
        verify_setup()
        sys.exit(0)

    ensure_gdown()

    if not args.skip_models:
        install_dependencies()
        setup_phishpedia()
        setup_phishintention()

    if not args.skip_detectron2:
        install_detectron2()

    verify_setup()
