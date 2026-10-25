"""Summarize exactly one experiment suite, never mix stale/global result files."""
from pathlib import Path
import argparse
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json

SCORES = ["accuracy", "balanced_accuracy", "majority_baseline_test_accuracy", "weighted_precision", "weighted_recall", "weighted_f1",
          "macro_precision", "macro_recall", "macro_f1", "attack_false_negative_rate",
          "benign_false_positive_rate", "training_seconds"]


def compare(root):
    root = Path(root)
    rows, missing_classes = [], set()
    for path in sorted(root.glob("*/metrics.json")):
        c, m = read_json(path.parent / "config.json"), read_json(path)
        missing_classes.update(m.get("missing_raw_classes", []))
        rows.append({"run": path.parent.name, **c, **{k: m.get(k) for k in SCORES},
                     "conditional_epsilon_bound": m.get("conditional_epsilon_bound"),
                     "full_objective_data_coverage": m["full_objective_data_coverage"]})
    if not rows:
        failures = [read_json(p) for p in sorted(root.glob("*/failure.json"))]
        if not failures:
            raise FileNotFoundError("No completed experiments in the requested directory")
        text = "# Experiment results\n\nNo configurations completed successfully.\n\n"
        text += "\n".join(f"- {f['run']}: {f['error']}" for f in failures)
        (root / "RESULTS.md").write_text(text + "\n", encoding="utf-8")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df.to_csv(root / "comparison.csv", index=False)
    keys = ["algorithm", "mode", "noise", "rounds", "local_epochs", "lr", "mu", "alpha", "clients", "clip_norm", "delta"]
    summary = df.groupby(keys, dropna=False)[SCORES].agg(["mean", "std", "count"])
    summary.to_csv(root / "summary.csv")
    lines = ["# Experiment results", "", "These are measurements from this run directory only.", "",
             "| Method | Distribution | Noise | Seeds | Accuracy mean | Macro F1 mean | Macro F1 SD |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for (alg, mode, noise), g in df.groupby(["algorithm", "mode", "noise"]):
        sd = f"{g.macro_f1.std():.4f}" if len(g) > 1 else "n/a"
        lines.append(f"| {alg} | {mode} | {noise:g} | {len(g)} | {g.accuracy.mean():.4f} | {g.macro_f1.mean():.4f} | {sd} |")
    lines += ["", "Use macro F1 and per-class reports to assess minority attacks. Weighted scores can hide missed rare attacks.",
              "Local scores are means of independent local models, not ensemble predictions.",
              "SCAFFOLD uses uniform client weights; compare it with fedavg_uniform as well as sample-weighted FedAvg.",
              "Privacy noise=0 is a clipping-only control; fedavg_uniform is its unclipped control.",
              "Conditional epsilon is per run only, and is not an end-to-end guarantee for this benchmark."]
    if not df.full_objective_data_coverage.all():
        lines += ["", "**REPORT COVERAGE LIMITATION:** missing class groups: " + ", ".join(sorted(missing_classes)) +
                  ". Passing the selected label scope does not establish the report's missing-class objectives."]
    baseline = df.majority_baseline_test_accuracy.dropna()
    if not baseline.empty:
        lines += ["", f"Majority-only baseline test accuracy: {baseline.iloc[0]:.4%}. "
                  "Read macro F1, balanced accuracy, attack FNR and per-class recall alongside accuracy."]
    failures = [read_json(p) for p in sorted(root.glob("*/failure.json"))]
    if failures:
        lines += ["", "## Failed configurations", "", "These configurations diverged and are absent from the numeric averages:"]
        lines += [f"- {f['run']}: {f['error']}" for f in failures]
    (root / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Comparison saved to {root / 'comparison.csv'}")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path)
    compare(p.parse_args().results)
