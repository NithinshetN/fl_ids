"""Trusted-server Gaussian mechanism on the equal-client mean of clipped updates.

Accounting is conditional on fixed/public preprocessing, partition membership and
evaluation data. The offline benchmark as a whole is NOT end-to-end DP.
"""
import math
import numpy as np
from federated.aggregation import average


def clip_update(update, clip_norm):
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    norm = math.sqrt(sum(float(np.square(w.astype(np.float64)).sum()) for w in update))
    return [w * min(1.0, clip_norm / max(norm, 1e-30)) for w in update]


def private_average(updates, clip_norm, noise_multiplier, rng=None):
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be nonnegative")
    clipped = [clip_update(u, clip_norm) for u in updates]
    mean = average(clipped, [1] * len(clipped))
    # Replace-one-client adjacency, fixed N: sensitivity of mean <= 2C/N.
    sensitivity = 2 * clip_norm / len(updates)
    # Default generator is entropy-seeded independently of published training seeds.
    rng = np.random.default_rng() if rng is None else rng
    return [w + rng.normal(0, noise_multiplier * sensitivity, w.shape).astype(w.dtype) for w in mean]


def accounting(rounds, noise_multiplier, delta=1e-5):
    if rounds < 1 or noise_multiplier < 0 or not 0 < delta < 1:
        raise ValueError("Invalid privacy accounting arguments")
    rho = rounds / (2 * noise_multiplier ** 2) if noise_multiplier > 0 else None
    epsilon = rho + 2 * math.sqrt(rho * math.log(1 / delta)) if rho is not None else None
    return {"noise_multiplier": noise_multiplier, "delta": delta, "rho": rho,
            "conditional_epsilon_bound": epsilon, "rounds_composed": rounds,
            "adjacency": "Replace all records of one client; fixed roster and uniform client weights.",
            "mechanism": "Gaussian noise std = noise_multiplier * 2 * clip_norm / n_clients",
            "trust": "Trusted aggregator observes unclipped client updates; no secure aggregation.",
            "accounting_scope": "All released global rounds within this one run; no subsampling amplification.",
            "conditions": "Fixed/public preprocessing, feature schema, partitions, initialization and evaluation set.",
            "end_to_end_dp_claim": False,
            "limitation": "Pooled preprocessing, data-dependent partitioning, metadata and nonprivate baselines are not DP. "
                          "This is a benchmark mechanism study; epsilon is conditional, not a guarantee for all outputs. "
                          "Releasing multiple runs needs composition across runs. NumPy noise is research software, not a hardened DP sampler."}
