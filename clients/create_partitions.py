"""Reproducible IID, Dirichlet and organization-profile partitions."""
from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, read_json, save_json

PROFILES = ["dos", "brute_force", "web", "botnet", "mixed"]


def partition(y, names, mode="iid", clients=5, alpha=.3, seed=42):
    if clients < 2 or len(y) < clients or alpha <= 0:
        raise ValueError("Need >=2 clients, >=1 row/client and alpha >0.")
    if mode == "organization" and clients != 5:
        raise ValueError("Organization mode represents exactly five organizations.")
    rng = np.random.default_rng(seed)
    split_dos_profiles = "Web_Attack" not in names and "DoS" in names and "DDoS" in names
    profiles = ["dos", "ddos", "brute_force", "botnet", "mixed"] if split_dos_profiles else PROFILES
    groups = [[] for _ in range(clients)]
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        if mode == "iid":
            # Rotate remainder recipients to avoid concentrating rare classes in client 0.
            order = rng.permutation(clients)
            for cid, part in zip(order, np.array_split(idx, clients)):
                groups[cid].extend(part)
            continue
        if mode == "noniid":
            p = rng.dirichlet(np.full(clients, alpha))
        elif mode == "organization":
            name = names[int(c)]
            owner = {"DoS": 0, "DDoS": 0, "DoS_DDoS": 0, "Brute_Force": 1,
                     "Web_Attack": 2, "Bot": 3}.get(name, 4)
            if split_dos_profiles:
                owner = {"DoS": 0, "DDoS": 1, "Brute_Force": 2, "Bot": 3}.get(name, 4)
            p = np.full(clients, .05)
            p[owner] = .8
            if name == "Benign":
                p = np.full(clients, .2)
        else:
            raise ValueError(f"Unknown partition mode: {mode}")
        counts = rng.multinomial(len(idx), p / p.sum())
        for cid, part in enumerate(np.split(idx, np.cumsum(counts)[:-1])):
            groups[cid].extend(part)
    # Repair empty clients deterministically, preserving the exact disjoint cover.
    repairs = 0
    for cid in range(clients):
        if not groups[cid]:
            donor = max(range(clients), key=lambda j: len(groups[j]))
            groups[cid].append(groups[donor].pop())
            repairs += 1
    indices = [rng.permutation(np.asarray(g, dtype=np.int64)) for g in groups]
    merged = np.concatenate(indices)
    if len(merged) != len(y) or len(np.unique(merged)) != len(y):
        raise AssertionError("Client partitions must cover each training row exactly once.")
    global_p = np.bincount(y, minlength=len(names)) / len(y)
    summary = {"mode": mode, "seed": seed, "alpha": alpha, "empty_client_repairs": repairs,
               "organization_profile_scheme": profiles if mode == "organization" else None,
               "clients": {}, "mean_total_variation_from_global": 0.0}
    distances = []
    for cid, idx in enumerate(indices):
        counts = np.bincount(y[idx], minlength=len(names))
        tv = float(np.abs(counts / len(idx) - global_p).sum() / 2)
        distances.append(tv)
        summary["clients"][str(cid)] = {
            "profile": profiles[cid] if mode == "organization" else mode,
            "rows": len(idx), "class_counts": dict(zip(names, counts.tolist())),
            "total_variation_from_global": tv}
    summary["mean_total_variation_from_global"] = float(np.mean(distances))
    summary["missing_organization_attacks"] = [n for n in ["Web_Attack", "Bot", "Brute_Force"] if n not in names]
    return indices, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data/processed/main")
    p.add_argument("--mode", choices=["iid", "noniid", "organization"], default="noniid")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--alpha", type=float, default=.3)
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()
    d = np.load(a.data / "train.npz")
    meta = read_json(a.data / "metadata.json")
    indices, summary = partition(d["y"], meta["class_names"], a.mode, a.clients, a.alpha, a.seed)
    out = a.data / "clients" / f"{a.mode}_seed{a.seed}_alpha{a.alpha}_n{a.clients}"
    if out.exists():
        raise FileExistsError(f"Choose a new partition seed/configuration: {out}")
    out.mkdir(parents=True)
    for cid, idx in enumerate(indices):
        np.savez_compressed(out / f"client_{cid}.npz", X=d["X"][idx], y=d["y"][idx], indices=idx)
    save_json(out / "summary.json", summary)
    print(out)


if __name__ == "__main__":
    main()
