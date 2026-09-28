#!/usr/bin/env python
"""Build the leakage-safe T0 feature store: python scripts/build_features.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import features

if __name__ == "__main__":
    features.run()