"""Flower loopback client: load only this client's prepared partition."""
from pathlib import Path
import argparse
import sys
import numpy as np
import flwr as fl
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import read_json, seed_everything, sha256
from models.ids_model import build_model, get_weights, set_weights, train_local, predict


class IDSClient(fl.client.NumPyClient):
    def __init__(self, cid, partition_dir, metadata):
        self.cid = cid
        self.meta = read_json(metadata)
        self.fingerprint = sha256(metadata)
        d = np.load(Path(partition_dir) / f"client_{cid}.npz")
        self.X, self.y = d["X"], d["y"]
        self.model = build_model(self.meta["n_features"], len(self.meta["class_names"]))

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        if config["metadata_sha256"] != self.fingerprint:
            raise ValueError("Client and server dataset metadata differ")
        set_weights(self.model, parameters)
        weights, steps, loss = train_local(self.model, self.X, self.y,
            int(config["local_epochs"]), int(config["batch_size"]), float(config["lr"]),
            int(config["seed"]) + int(config["round"]) * 1000 + self.cid)
        return weights, len(self.y), {"cid": self.cid, "steps": steps, "loss": loss}

    def evaluate(self, parameters, config):
        # Disabled by the server. No global test rows are loaded by clients.
        set_weights(self.model, parameters)
        pred, loss = predict(self.model, self.X, self.y)
        return loss, len(self.y), {"accuracy": float(np.mean(pred == self.y))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cid", type=int, required=True)
    p.add_argument("--partitions", type=Path, required=True)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--server", default="127.0.0.1:8080")
    a = p.parse_args()
    seed_everything(42)
    client = IDSClient(a.cid, a.partitions, a.metadata)
    fl.client.start_client(server_address=a.server, client=client.to_client(), insecure=True)


if __name__ == "__main__":
    main()
