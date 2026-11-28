"""Test bootstrap: make the training modules importable from the repo root."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAINING = ROOT / "training"
for path in (str(ROOT), str(TRAINING)):
    if path not in sys.path:
        sys.path.insert(0, path)
