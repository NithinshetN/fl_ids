"""Run regression and synthetic integration tests with the active Python."""
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest", "-q", *sys.argv[1:]], cwd=root))
