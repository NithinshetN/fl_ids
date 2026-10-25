"""Create comparison, validation-convergence, privacy and confusion-matrix figures."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json


def plot(root):
    root = Path(root)
    df = pd.read_csv(root / "comparison.csv")
    out = root / "plots"; out.mkdir(exist_ok=True)
    group = df.groupby(["algorithm", "mode", "noise"]).macro_f1.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(12, max(4, .35 * len(group))))
    ax.barh(range(len(group)), group["mean"], xerr=group["std"].fillna(0))
    ax.set_yticks(range(len(group)), [f"{a}/{m}/noise={n:g}" for a, m, n in group.index])
    ax.set(xlabel="Test macro F1 (mean +/- seed SD)", xlim=(0, 1), title="IDS comparison")
    fig.tight_layout(); fig.savefig(out / "comparison.png", dpi=160); plt.close(fig)
    history_rows = []
    for folder in sorted(root.iterdir()):
        if not (folder / "history.json").exists():
            continue
        hist = read_json(folder / "history.json")
        if hist:
            config = read_json(folder / "config.json")
            for row in hist:
                history_rows.append({**row, "algorithm": config["algorithm"],
                                     "mode": config["mode"], "noise": config["noise"]})
        metrics = read_json(folder / "metrics.json")
        if "confusion_matrix" in metrics:
            cm = np.array(metrics["confusion_matrix"])
            normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
            f, a = plt.subplots(figsize=(8, 7))
            im = a.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            names = metrics["class_names"]
            a.set_xticks(range(len(names)), names, rotation=65, ha="right")
            a.set_yticks(range(len(names)), names)
            a.set(xlabel="Predicted", ylabel="True", title=folder.name)
            f.colorbar(im, ax=a, label="Row-normalized fraction")
            f.tight_layout(); f.savefig(out / f"{folder.name}_confusion.png", dpi=130); plt.close(f)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    history_df = pd.DataFrame(history_rows)
    for ax, panel in zip(axes.ravel(), ["iid", "noniid", "organization", "privacy"]):
        if not history_df.empty:
            selected = history_df[history_df.algorithm == "privacy"] if panel == "privacy" else history_df[
                (history_df["mode"] == panel) & (history_df.algorithm != "privacy")]
            for (alg, noise), g in selected.groupby(["algorithm", "noise"]):
                stats = g.groupby("round").validation_macro_f1.agg(["mean", "std"])
                label = f"noise={noise:g}" if panel == "privacy" else alg
                line, = ax.plot(stats.index, stats["mean"], marker=".", label=label)
                sd = stats["std"].fillna(0)
                ax.fill_between(stats.index, stats["mean"] - sd, stats["mean"] + sd,
                                color=line.get_color(), alpha=.1)
        ax.set(xlabel="Round", ylabel="Validation macro F1", title=panel)
        ax.legend(fontsize=8)
    fig.suptitle("Validation convergence: seed mean and standard deviation")
    fig.tight_layout(); fig.savefig(out / "convergence.png", dpi=160); plt.close(fig)
    private = df[df.algorithm == "privacy"]
    if not private.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        g = private.groupby("noise").agg(f1=("macro_f1", "mean"), acc=("accuracy", "mean"), eps=("conditional_epsilon_bound", "mean"))
        axes[0].plot(g.index, g.f1, "o-", label="Macro F1")
        axes[0].plot(g.index, g.acc, "s-", label="Accuracy")
        axes[0].set(xlabel="Noise multiplier (0 = clipping only)", ylabel="Test score")
        axes[0].legend()
        finite = g[g.eps.notna()]
        axes[1].plot(finite.eps, finite.f1, "o-")
        axes[1].set(xlabel="Conditional epsilon bound per run", ylabel="Test macro F1")
        fig.suptitle("Privacy mechanism study; not end-to-end DP")
        fig.tight_layout(); fig.savefig(out / "privacy_tradeoff.png", dpi=160); plt.close(fig)
    print(f"Plots saved to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, type=Path)
    plot(p.parse_args().results)
