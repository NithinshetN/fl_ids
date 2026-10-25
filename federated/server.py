"""Flower FedAvg network demonstration, bound to localhost by default."""
from pathlib import Path
import argparse
import sys
import flwr as fl
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import save_json, seed_everything, sha256
from experiments import load_data, evaluate
from models.ids_model import build_model, get_weights, set_weights


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--address", default="127.0.0.1:8080")
    a = p.parse_args()
    if min(a.rounds, a.local_epochs, a.batch_size) < 1 or a.clients < 2 or a.lr <= 0:
        p.error("Invalid training parameters")
    if a.output.exists():
        p.error("Choose a new output directory")
    a.output.mkdir(parents=True)
    meta, data = load_data(a.data)
    seed_everything(a.seed)
    model = build_model(meta["n_features"], len(meta["class_names"]))
    history = []
    latest = {}

    def evaluate_round(rnd, parameters, config):
        set_weights(model, parameters)
        metrics = evaluate(model, data["validation"], meta["class_names"])
        history.append({"round": rnd, "validation_loss": metrics["loss"],
                        "validation_macro_f1": metrics["macro_f1"]})
        latest["round"] = rnd
        save_json(a.output / "history.json", history)
        return metrics["loss"], {"macro_f1": metrics["macro_f1"]}

    class StrictFedAvg(fl.server.strategy.FedAvg):
        def aggregate_fit(self, server_round, results, failures):
            ids = [r.metrics.get("cid") for _, r in results]
            if failures or sorted(ids) != list(range(a.clients)):
                raise RuntimeError("A client failed, is missing, or has a duplicate/wrong ID")
            return super().aggregate_fit(server_round, results, failures)

    strategy = StrictFedAvg(fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=a.clients, min_available_clients=a.clients, min_evaluate_clients=0,
        accept_failures=False, inplace=False,
        initial_parameters=fl.common.ndarrays_to_parameters(get_weights(model)),
        evaluate_fn=evaluate_round,
        on_fit_config_fn=lambda rnd: {"local_epochs": a.local_epochs, "batch_size": a.batch_size,
            "lr": a.lr, "seed": a.seed, "round": rnd,
            "metadata_sha256": sha256(a.data / "metadata.json")})
    fl.server.start_server(server_address=a.address,
        config=fl.server.ServerConfig(num_rounds=a.rounds, round_timeout=180), strategy=strategy)
    if latest.get("round") != a.rounds:
        raise RuntimeError("Training did not complete all rounds")
    save_json(a.output / "metrics.json", evaluate(model, data["test"], meta["class_names"]))
    torch.save(model.state_dict(), a.output / "model.pt")


if __name__ == "__main__":
    main()
