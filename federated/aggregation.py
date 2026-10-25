"""Aggregation and SCAFFOLD option-II control update, independently testable."""
import numpy as np


def average(weight_sets, sizes):
    if not weight_sets or len(weight_sets) != len(sizes) or any(n <= 0 for n in sizes):
        raise ValueError("Aggregation requires positive sizes for all clients.")
    if any(len(ws) != len(weight_sets[0]) for ws in weight_sets):
        raise ValueError("Inconsistent parameter counts")
    fractions = np.asarray(sizes, dtype=np.float64) / sum(sizes)
    result = []
    for tensors in zip(*weight_sets):
        if any(w.shape != tensors[0].shape for w in tensors):
            raise ValueError("Inconsistent tensor shapes")
        result.append(sum(w.astype(np.float64) * f for w, f in zip(tensors, fractions)).astype(tensors[0].dtype))
    return result


def scaffold_control(old_client, global_control, global_weights, local_weights, steps, lr):
    if steps <= 0 or lr <= 0:
        raise ValueError("SCAFFOLD requires positive local steps and learning rate.")
    # c_i_new = c_i - c + (x_global - y_local)/(K * eta_local)
    return [ci - c + (x - y) / (steps * lr)
            for ci, c, x, y in zip(old_client, global_control, global_weights, local_weights)]
