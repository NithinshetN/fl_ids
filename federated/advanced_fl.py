"""Run FedProx or SCAFFOLD using the common experiment engine."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_experiments import main

if __name__ == "__main__":
    main()
