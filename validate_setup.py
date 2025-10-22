"""Validate project setup and check for common issues."""

import sys
from pathlib import Path


def check_structure():
    """Check if project structure is correct."""
    print("Checking project structure...")

    required_dirs = [
        "src",
        "configs",
        "data",
        "outputs",
        "outputs/csv",
        "outputs/plots",
        "outputs/snippets"
    ]

    required_files = [
        "src/__init__.py",
        "src/run.py",
        "src/data.py",
        "src/features.py",
        "src/model.py",
        "src/intervene.py",
        "src/metrics.py",
        "src/snippets.py",
        "src/plots.py",
        "src/report.py",
        "src/utils.py",
        "configs/default.yaml",
        "requirements.txt",
        "README.md"
    ]

    missing = []

    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            missing.append(f"Directory: {dir_path}")

    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(f"File: {file_path}")

    if missing:
        print("❌ Missing required files/directories:")
        for item in missing:
            print(f"  - {item}")
        return False
    else:
        print("✅ Project structure is correct")
        return True


def check_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")

    required_packages = [
        "torch",
        "transformer_lens",
        "sae_lens",
        "transformers",
        "numpy",
        "pandas",
        "matplotlib",
        "tqdm",
        "yaml"
    ]

    missing = []

    for package in required_packages:
        try:
            if package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print("❌ Missing required packages:")
        for package in missing:
            print(f"  - {package}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed")
        return True


def check_features_csv():
    """Check if features CSV exists and is valid."""
    print("\nChecking features CSV...")

    csv_path = Path("data/neuronpedia_features.csv")

    if not csv_path.exists():
        print("⚠️  Features CSV not found at data/neuronpedia_features.csv")
        print("   Using template for testing...")

        template_path = Path("data/neuronpedia_features_template.csv")
        if template_path.exists():
            print("   Template found, you can use it for testing")
            return True
        else:
            print("❌ Template also not found!")
            return False

    # Try to load and validate
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)

        required_cols = {"layer", "feature_id", "label", "label_confidence"}
        if not required_cols.issubset(df.columns):
            print(f"❌ CSV missing required columns. Need: {required_cols}")
            return False

        num_features = len(df)
        num_layers = df["layer"].nunique()

        print(f"✅ Features CSV is valid")
        print(f"   - {num_features} features across {num_layers} layers")

        return True

    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return False


def check_config():
    """Check if default config is valid."""
    print("\nChecking configuration...")

    config_path = Path("configs/default.yaml")

    if not config_path.exists():
        print("❌ Default config not found")
        return False

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

        required_keys = [
            "model_name",
            "sae_release",
            "hook",
            "layers",
            "alpha",
            "corpus_name"
        ]

        missing_keys = [k for k in required_keys if k not in config]

        if missing_keys:
            print(f"❌ Config missing required keys: {missing_keys}")
            return False

        print("✅ Configuration is valid")
        return True

    except Exception as e:
        print(f"❌ Error reading config: {e}")
        return False


def check_gpu():
    """Check GPU availability."""
    print("\nChecking GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✅ GPU available: {device_name}")
            print(f"   - VRAM: {total_mem:.1f} GB")

            if total_mem < 16:
                print("   ⚠️  Less than 16GB VRAM, you may need to reduce batch size")
        else:
            print("⚠️  No GPU available, will use CPU (slower)")

        return True

    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("SAE Intervention Pipeline - Setup Validation")
    print("=" * 60)

    checks = [
        ("Project Structure", check_structure),
        ("Dependencies", check_dependencies),
        ("Features CSV", check_features_csv),
        ("Configuration", check_config),
        ("GPU", check_gpu)
    ]

    results = []

    for name, check_fn in checks:
        try:
            result = check_fn()
            results.append(result)
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"\nPassed: {passed}/{total} checks")

    if passed == total:
        print("\n✅ All checks passed! You're ready to run the experiment.")
        print("\nNext steps:")
        print("  1. (Optional) Edit configs/default.yaml")
        print("  2. Run: python -m src.run")
        print("  3. Check results in outputs/report.md")
    else:
        print("\n❌ Some checks failed. Please fix the issues above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
