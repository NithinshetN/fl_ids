"""Convenience entry point for independent local-only models."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import main

if __name__ == "__main__":
    sys.argv[1:1] = ["--algorithm", "local"]
    main()
