#!/usr/bin/env python3
"""
Quick verification script to test the environment setup
"""

import sys


def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")

    packages = {
        "numpy": "NumPy",
        "scipy": "SciPy",
        "tensorflow": "TensorFlow",
        "numba": "Numba",
        "tqdm": "tqdm",
    }

    failed = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name} - {e}")
            failed.append(name)

    return len(failed) == 0, failed


def test_files():
    """Test that required files and directories exist"""
    import os

    print("\nTesting file structure...")

    required_files = ["utils.py", "main.ipynb", "glove.6B.300d.txt"]

    required_dirs = [
        "KGs/dbp_fr_en",
        "KGs/dbp_ja_en",
        "KGs/dbp_zh_en",
        "KGs/srprs_de_en",
        "KGs/srprs_fr_en",
        "translated_ent_name",
    ]

    failed_files = []
    for f in required_files:
        if os.path.exists(f):
            print(f"  ✓ {f}")
        else:
            print(f"  ✗ {f} (missing)")
            failed_files.append(f)

    for d in required_dirs:
        if os.path.isdir(d):
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ (missing)")
            failed_files.append(d)

    return len(failed_files) == 0, failed_files


def test_data_integrity():
    """Test that dataset files are readable"""
    import os
    import json

    print("\nTesting dataset integrity...")

    datasets = ["dbp_fr_en", "dbp_ja_en", "dbp_zh_en", "srprs_de_en", "srprs_fr_en"]
    failed = []

    for dataset in datasets:
        try:
            # Test entity names
            with open(f"translated_ent_name/{dataset}.json", "r") as f:
                ent_names = json.load(f)

            # Test KG files
            kg_files = [
                "ent_ids_1",
                "ent_ids_2",
                "ref_ent_ids",
                "triples_1",
                "triples_2",
            ]
            for kg_file in kg_files:
                path = f"KGs/{dataset}/{kg_file}"
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing {path}")

            print(f"  ✓ {dataset} ({len(ent_names)} entities)")
        except Exception as e:
            print(f"  ✗ {dataset} - {e}")
            failed.append(dataset)

    return len(failed) == 0, failed


def main():
    print("=" * 60)
    print("SEU Entity Alignment - Environment Verification")
    print("=" * 60)

    all_passed = True

    # Test imports
    imports_ok, failed_imports = test_imports()
    if not imports_ok:
        print(f"\n⚠ Warning: Failed to import: {', '.join(failed_imports)}")
        print("  Run: pip install -r requirements.txt")
        all_passed = False

    # Test files
    files_ok, failed_files = test_files()
    if not files_ok:
        print(f"\n⚠ Warning: Missing files: {', '.join(failed_files)}")
        if "glove.6B.300d.txt" in failed_files:
            print("  Run: ./setup_environment.sh to download GloVe embeddings")
        all_passed = False

    # Test data integrity
    data_ok, failed_data = test_data_integrity()
    if not data_ok:
        print(f"\n⚠ Warning: Issues with datasets: {', '.join(failed_data)}")
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Environment is ready.")
        print("\nYou can now run experiments with:")
        print("  ./run_all_experiments.sh")
    else:
        print("✗ Some tests failed. Please fix the issues above.")
        print("\nTo set up the environment, run:")
        print("  ./setup_environment.sh")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
