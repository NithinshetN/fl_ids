"""Shared MLP and SGD local solver for every comparison."""
import numpy as np
import torch
from torch import nn
from common import seed_everything


def build_model(input_dim, num_classes):
    # No BatchNorm running buffers: every exchanged tensor is a trainable parameter.
    return nn.Sequential(nn.Linear(input_dim, 128), nn.LeakyReLU(.2),
                         nn.Linear(128, 64), nn.LeakyReLU(.2),
                         nn.Linear(64, 64), nn.LeakyReLU(.2), nn.Linear(64, num_classes))


def get_weights(model):
    return [p.detach().cpu().numpy().copy() for p in model.parameters()]


def set_weights(model, weights):
    params = list(model.parameters())
    if len(params) != len(weights):
        raise ValueError("Parameter count mismatch")
    with torch.no_grad():
        for p, w in zip(params, weights):
            if tuple(p.shape) != tuple(w.shape) or not np.isfinite(w).all():
                raise ValueError("Invalid parameter shape or non-finite values")
            p.copy_(torch.as_tensor(w, dtype=p.dtype))


def train_local(model, X, y, epochs, batch_size, lr, seed, global_weights=None,
                mu=0.0, control_global=None, control_client=None):
    if len(y) == 0 or epochs < 1 or batch_size < 1 or lr <= 0 or mu < 0:
        raise ValueError("Invalid local-training configuration")
    seed_everything(seed)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    xt, yt = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    rng = torch.Generator().manual_seed(seed)
    anchor = [torch.as_tensor(w) for w in global_weights] if global_weights is not None else None
    correction = None
    if control_global is not None:
        correction = [torch.as_tensor(cg - ci) for cg, ci in zip(control_global, control_client)]
    steps, total_loss, seen = 0, 0.0, 0
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=rng)
        for idx in order.split(batch_size):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(xt[idx]), yt[idx])
            if mu:
                if anchor is None:
                    raise ValueError("FedProx requires global parameters")
                loss = loss + mu / 2 * sum((p - w).square().sum() for p, w in zip(model.parameters(), anchor))
            if not torch.isfinite(loss):
                raise FloatingPointError("Training diverged; reduce learning rate or inspect features.")
            loss.backward()
            if correction is not None:
                for p, c in zip(model.parameters(), correction):
                    p.grad.add_(c)
            opt.step()
            total_loss += float(loss.detach()) * len(idx)
            seen += len(idx); steps += 1
    weights = get_weights(model)
    if not all(np.isfinite(w).all() for w in weights):
        raise FloatingPointError("Non-finite model weights")
    return weights, steps, total_loss / seen


def predict(model, X, y, batch_size=4096):
    model.eval()
    preds, losses = [], []
    with torch.no_grad():
        for start in range(0, len(y), batch_size):
            xb = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32)
            yb = torch.as_tensor(y[start:start + batch_size], dtype=torch.long)
            logits = model(xb)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Non-finite evaluation logits; the configuration diverged.")
            preds.append(logits.argmax(1).numpy())
            losses.append(float(nn.functional.cross_entropy(logits, yb, reduction="sum")))
    if not np.isfinite(losses).all():
        raise FloatingPointError("Non-finite evaluation loss; the configuration diverged.")
    return np.concatenate(preds), sum(losses) / len(y)
