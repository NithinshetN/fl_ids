"""Run an individual experiment or the four-question comparison suite."""
from pathlib import Path
import argparse
from dataclasses import replace
from common import ROOT, save_json
from experiments import Config, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/main")
    p.add_argument("--output", type=Path, default=ROOT / "results/main")
    p.add_argument("--suite", action="store_true")
    p.add_argument("--algorithm", choices=["centralized", "local", "fedavg", "fedavg_uniform", "fedprox", "scaffold", "privacy"], default="fedavg")
    p.add_argument("--mode", choices=["iid", "noniid", "organization"], default="noniid")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=.3)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--mu", type=float, default=.01)
    p.add_argument("--noise", type=float, default=1.0)
    p.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, .5, 1.0, 2.0])
    p.add_argument("--clip-norm", type=float, default=.1)
    p.add_argument("--delta", type=float, default=1e-5)
    a = p.parse_args()
    if a.output.exists():
        p.error("Output exists; use a new --output to preserve earlier experiments.")
    if a.suite and a.clients != 5:
        p.error("The full suite includes five organization profiles and needs --clients 5.")
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.noise_levels)) != len(a.noise_levels):
        p.error("Duplicate seeds or noise levels would overwrite experiment names.")
    failures = []
    for seed in a.seeds:
        base = Config(algorithm=a.algorithm, mode=a.mode, seed=seed, clients=a.clients,
                      alpha=a.alpha, rounds=a.rounds, local_epochs=a.local_epochs,
                      batch_size=a.batch_size, lr=a.lr, mu=a.mu, noise=a.noise,
                      clip_norm=a.clip_norm, delta=a.delta)
        configs = [base]
        if a.suite:
            configs = [replace(base, algorithm="centralized", mode="iid", noise=0)]
            for mode in ("iid", "noniid", "organization"):
                for alg in ("local", "fedavg", "fedavg_uniform", "fedprox", "scaffold"):
                    configs.append(replace(base, algorithm=alg, mode=mode, noise=0))
            configs += [replace(base, algorithm="privacy", mode="noniid", noise=n) for n in a.noise_levels]
        for cfg in configs:
            name = f"{cfg.algorithm}_{cfg.mode}_seed{seed}"
            if cfg.algorithm == "privacy":
                name += f"_noise{cfg.noise:g}"
            try:
                run(a.data, a.output / name, cfg)
            except FloatingPointError as exc:
                # Divergence is an experimental outcome, never silently omitted.
                failure = {"run": name, "error": str(exc)}
                save_json(a.output / name / "failure.json", failure)
                failures.append(failure)
                print(f"FAILED {name}: {exc}", flush=True)
    from evaluation.compare_results import compare
    compare(a.output)
    if failures:
        save_json(a.output / "failures.json", failures)
        raise SystemExit(f"{len(failures)} configuration(s) diverged; see failures.json and RESULTS.md.")


if __name__ == "__main__":
    main()
