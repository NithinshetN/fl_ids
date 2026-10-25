from pathlib import Path
import argparse
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocessing.preprocess import input_files


def main():
    p = argparse.ArgumentParser(description="Inspect CSV labels and columns before preprocessing.")
    p.add_argument("--input", required=True, type=Path)
    a = p.parse_args()
    files, ignored = input_files(a.input)
    if ignored:
        print("Ignored class-count sidecars:", ignored)
    for path in files:
        counts = pd.Series(dtype="int64")
        rows = 0
        for frame in pd.read_csv(path, chunksize=100000, encoding_errors="replace"):
            frame.columns = frame.columns.str.strip()
            if "Label" not in frame:
                raise ValueError(f"Missing Label column: {path}")
            rows += len(frame)
            counts = counts.add(frame.Label.value_counts(dropna=False), fill_value=0)
        print(f"{path}: {rows:,} rows")
        print(counts.astype(int).to_string())


if __name__ == "__main__":
    main()
