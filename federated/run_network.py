"""Launch one local Flower server and all clients, with logs and bounded cleanup."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
import os
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, read_json, save_json
from clients.create_partitions import partition


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--mode", choices=["iid", "noniid", "organization"], default="noniid")
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=.3)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--timeout", type=int, default=600)
    a = p.parse_args()
    a.data, a.output = a.data.resolve(), a.output.resolve()
    if a.output.exists():
        p.error("Choose a new --output directory")
    if min(a.rounds, a.local_epochs, a.timeout) < 1 or not 1 <= a.port <= 65535:
        p.error("Invalid rounds/epochs/timeout/port")
    d = np.load(a.data / "train.npz")
    meta = read_json(a.data / "metadata.json")
    indices, summary = partition(d["y"], meta["class_names"], a.mode, a.clients, a.alpha, a.seed)
    parts = a.output / "partitions"; parts.mkdir(parents=True)
    save_json(parts / "summary.json", summary)
    for cid, idx in enumerate(indices):
        np.savez_compressed(parts / f"client_{cid}.npz", X=d["X"][idx], y=d["y"][idx])
    logs, processes = [], []
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    address = f"127.0.0.1:{a.port}"

    def launch(name, script, args):
        log = open(a.output / f"{name}.log", "w", encoding="utf-8"); logs.append(log)
        proc = subprocess.Popen([sys.executable, str(ROOT / script), *map(str, args)],
                                stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, creationflags=flags)
        processes.append(proc)
        return proc

    try:
        server = launch("server", "federated/server.py", ["--data", a.data, "--output", a.output / "server",
            "--clients", a.clients, "--rounds", a.rounds, "--local-epochs", a.local_epochs,
            "--seed", a.seed, "--address", address])
        # Wait for the server to bind before launching clients (especially on Windows).
        ready_deadline = time.monotonic() + min(60, a.timeout)
        while "gRPC server running" not in (a.output / "server.log").read_text(encoding="utf-8", errors="replace"):
            if server.poll() is not None or time.monotonic() > ready_deadline:
                raise RuntimeError(f"Server failed to become ready; read {a.output / 'server.log'}")
            time.sleep(.2)
        for cid in range(a.clients):
            launch(f"client_{cid}", "federated/client.py", ["--cid", cid, "--partitions", parts,
                "--metadata", a.data / "metadata.json", "--server", address])
        deadline = time.monotonic() + a.timeout
        while any(proc.poll() is None for proc in processes):
            if any(proc.poll() not in (None, 0) for proc in processes):
                raise RuntimeError(f"A Flower process failed. Read logs in {a.output}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"Flower timed out; read logs in {a.output}")
            time.sleep(.2)
        if server.returncode != 0 or not (a.output / "server/metrics.json").exists():
            raise RuntimeError("Flower did not produce final test metrics")
        print(f"Flower network run completed. Results: {a.output / 'server/metrics.json'}")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        for log in logs:
            log.close()


if __name__ == "__main__":
    main()
