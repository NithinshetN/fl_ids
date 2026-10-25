"""Common experiment engine; all methods share data, initialization and SGD settings."""
from pathlib import Path
from dataclasses import dataclass, asdict
import time
import importlib.metadata
import numpy as np
import torch
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from common import save_json, read_json, sha256, seed_everything
from clients.create_partitions import partition
from models.ids_model import build_model, get_weights, set_weights, train_local, predict
from federated.aggregation import average, scaffold_control
from federated.privacy import private_average, accounting
from evaluation.metrics import evaluate_predictions


@dataclass
class Config:
    algorithm: str = "fedavg"
    mode: str = "noniid"
    seed: int = 42
    clients: int = 5
    alpha: float = .3
    rounds: int = 10
    local_epochs: int = 1
    batch_size: int = 256
    lr: float = .01
    mu: float = .01
    noise: float = 0.0
    clip_norm: float = 0.1
    delta: float = 1e-5


def validate(config):
    if not all(np.isfinite(v) for v in [config.lr, config.mu, config.noise, config.clip_norm, config.delta, config.alpha]):
        raise ValueError("Numeric configuration values must be finite")
    if config.algorithm not in {"centralized", "local", "fedavg", "fedavg_uniform", "fedprox", "scaffold", "privacy"}:
        raise ValueError("Unknown algorithm")
    if min(config.rounds, config.local_epochs, config.batch_size) < 1:
        raise ValueError("rounds, local_epochs and batch_size must be positive")
    if config.lr <= 0 or config.mu < 0 or config.noise < 0 or config.clip_norm <= 0:
        raise ValueError("Invalid optimizer/privacy arguments")
    if not 0 < config.delta < 1:
        raise ValueError("delta must be between 0 and 1")


def load_data(path):
    path = Path(path)
    meta = read_json(path / "metadata.json")
    sets = {}
    for name in ("train", "validation", "test"):
        if sha256(path / f"{name}.npz") != meta["split_sha256"][name]:
            raise ValueError(f"{name} data changed after preprocessing; prepare a new dataset.")
        with np.load(path / f"{name}.npz") as d:
            sets[name] = {key: d[key].copy() for key in d.files}
    return meta, sets


def flower_average(weight_sets, sizes, round_number):
    """Use Flower's real FedAvg strategy; sequential simulation needs no Ray/network."""
    results = [(None, FitRes(status=Status(code=Code.OK, message=""),
                parameters=ndarrays_to_parameters(ws), num_examples=int(n), metrics={}))
               for ws, n in zip(weight_sets, sizes)]
    parameters, _ = FedAvg(inplace=False, fit_metrics_aggregation_fn=lambda _: {}).aggregate_fit(round_number, results, [])
    if parameters is None:
        raise RuntimeError("Flower did not aggregate client updates")
    return parameters_to_ndarrays(parameters)


def evaluate(model, split, names):
    predictions, loss = predict(model, split["X"], split["y"])
    result = evaluate_predictions(split["y"], predictions, names)
    result["loss"] = loss
    return result


def run(data_path, output, config):
    validate(config)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to mix/overwrite experiments: {output}")
    meta, sets = load_data(data_path)
    names = meta["class_names"]
    train = sets["train"]
    indices, summary = partition(train["y"], names, config.mode, config.clients, config.alpha, config.seed)
    seed_everything(config.seed)
    model = build_model(meta["n_features"], len(names))
    initial = get_weights(model)
    output.mkdir(parents=True)
    save_json(output / "config.json", asdict(config))
    save_json(output / "partition_summary.json", summary)
    np.savez_compressed(output / "partition_indices.npz", **{f"client_{i}": v for i, v in enumerate(indices)})
    save_json(output / "provenance.json", {
        "dataset_metadata": meta, "initial_weight_sha256": __import__("hashlib").sha256(
            b"".join(w.tobytes() for w in initial)).hexdigest(),
        "versions": {p: importlib.metadata.version(p) for p in ["numpy", "pandas", "torch", "flwr", "scikit-learn"]},
        "budget": "Each training row is visited rounds * local_epochs times; optimizer-step counts differ by partition.",
        "selection": "Fixed final round; validation curves are diagnostic. Test evaluated only after training.",
        "federation": "Sequential client simulation; Flower FedAvg strategy; no network isolation or privacy boundary."})
    start = time.perf_counter()
    total_steps = 0
    history = []
    if config.algorithm == "local":
        local_results = []
        for cid, idx in enumerate(indices):
            client_model = build_model(meta["n_features"], len(names)); set_weights(client_model, initial)
            _, steps, _ = train_local(client_model, train["X"][idx], train["y"][idx],
                config.rounds * config.local_epochs, config.batch_size, config.lr, config.seed + cid)
            metrics = evaluate(client_model, sets["test"], names)
            local_results.append({"client": cid, "training_rows": len(idx), "metrics": metrics})
            total_steps += steps
        save_json(output / "local_metrics.json", local_results)
        # Mean of independent local-model scores on the identical common test set, not an ensemble.
        result = {key: float(np.mean([d["metrics"][key] for d in local_results]))
                  for key, value in local_results[0]["metrics"].items()
                  if isinstance(value, (int, float)) and value is not None}
        result["interpretation"] = "Unweighted mean over local-only client models on the common test set."
        result["macro_f1_std_across_clients"] = float(np.std([d["metrics"]["macro_f1"] for d in local_results]))
    else:
        gw = initial
        cg = [np.zeros_like(w) for w in gw]
        ci = [[np.zeros_like(w) for w in gw] for _ in indices]
        for rnd in range(1, config.rounds + 1):
            if config.algorithm == "centralized":
                gw, steps, _ = train_local(model, train["X"], train["y"], config.local_epochs,
                    config.batch_size, config.lr, config.seed + rnd * 1000)
                total_steps += steps
            else:
                locals_, sizes, new_controls = [], [], []
                for cid, idx in enumerate(indices):
                    client_model = build_model(meta["n_features"], len(names)); set_weights(client_model, gw)
                    lw, steps, _ = train_local(client_model, train["X"][idx], train["y"][idx],
                        config.local_epochs, config.batch_size, config.lr, config.seed + rnd * 1000 + cid,
                        global_weights=gw, mu=config.mu if config.algorithm == "fedprox" else 0,
                        control_global=cg if config.algorithm == "scaffold" else None,
                        control_client=ci[cid] if config.algorithm == "scaffold" else None)
                    locals_.append(lw); sizes.append(len(idx)); total_steps += steps
                    if config.algorithm == "scaffold":
                        new_controls.append(scaffold_control(ci[cid], cg, gw, lw, steps, config.lr))
                if config.algorithm == "privacy":
                    updates = [[w - g for w, g in zip(ws, gw)] for ws in locals_]
                    change = private_average(updates, config.clip_norm, config.noise)
                    gw = [g + d for g, d in zip(gw, change)]
                elif config.algorithm == "scaffold":
                    # Original SCAFFOLD objective is a uniform average over clients.
                    gw = average(locals_, [1] * len(sizes))
                    ci = new_controls
                    cg = average(ci, [1] * len(ci))  # full participation, initially zero controls
                else:
                    gw = flower_average(locals_, [1] * len(sizes) if config.algorithm == "fedavg_uniform" else sizes, rnd)
                set_weights(model, gw)
            val = evaluate(model, sets["validation"], names)
            row = {"round": rnd, "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
                   "validation_macro_f1": val["macro_f1"], "validation_weighted_f1": val["weighted_f1"]}
            history.append(row)
            print(f"{config.algorithm}/{config.mode} seed={config.seed} round={rnd}/{config.rounds} val_macro_f1={val['macro_f1']:.4f}", flush=True)
        result = evaluate(model, sets["test"], names)
        torch.save({"state_dict": model.state_dict(), "n_features": meta["n_features"],
                    "class_names": names}, output / "model.pt")
        if config.algorithm == "privacy":
            private = accounting(config.rounds, config.noise, config.delta)
            save_json(output / "privacy_accounting.json", private)
            result["conditional_epsilon_bound"] = private["conditional_epsilon_bound"]
    result.update({"training_seconds": time.perf_counter() - start, "optimizer_steps": total_steps,
                   "majority_baseline_test_accuracy": meta.get("majority_baseline_test_accuracy"),
                   "missing_raw_classes": meta["missing_raw_classes"],
                   "label_scope_satisfied": meta.get("label_scope_satisfied", meta["full_objective_data_coverage"]),
                   "full_objective_data_coverage": meta["full_objective_data_coverage"]})
    save_json(output / "history.json", history)
    save_json(output / "metrics.json", result)
    return result
