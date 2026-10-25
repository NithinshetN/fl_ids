import numpy as np
import pandas as pd
import pytest
import torch
from joblib import load
from common import seed_everything
from preprocessing.preprocess import map_label, split_indices, stratified_subset, prepare
from clients.create_partitions import partition
from evaluation.metrics import evaluate_predictions
from federated.aggregation import average, scaffold_control
from federated.privacy import clip_update, private_average, accounting
from models.ids_model import build_model, get_weights, set_weights, train_local
from experiments import flower_average, Config, run


@pytest.mark.parametrize("label,expected", [
    ("Web Attack - Brute Force", "Web_Attack"), ("Web Attack \ufffd XSS", "Web_Attack"),
    ("Web Attack – Sql Injection", "Web_Attack"), ("FTP-Patator", "Brute_Force"),
    ("DoS Hulk", "DoS"), ("DDoS", "DDoS"), ("DoS_DDoS", "DoS_DDoS"), ("unknown", None)])
def test_labels(label, expected):
    assert map_label(label) == expected


@pytest.mark.parametrize("mode", ["iid", "noniid", "organization"])
def test_partitions_cover_train_once(mode):
    names = ["Benign", "DoS", "Brute_Force", "Web_Attack", "Bot"]
    y = np.repeat(np.arange(5), 100)
    parts, _ = partition(y, names, mode, alpha=.01)
    repeated, _ = partition(y, names, mode, alpha=.01)
    assert all(len(v) > 0 for v in parts)
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(len(y)))
    assert all(np.array_equal(a, b) for a, b in zip(parts, repeated))
    if mode == "organization":
        assert np.sum(y[parts[0]] == 1) > 50
        assert np.sum(y[parts[2]] == 3) > 50


def test_empty_client_repair():
    parts, _ = partition(np.zeros(5, dtype=int), ["Benign"], "noniid", alpha=.001)
    assert sorted(map(len, parts)) == [1] * 5


def test_rare_class_split_and_quota():
    y = np.repeat(np.arange(4), [100, 1, 2, 3])
    chosen = stratified_subset(y, 30, np.random.default_rng(7))
    assert len(chosen) == len(set(chosen)) == 30
    train, val, test = split_indices(y[chosen], 42)
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
    assert np.array_equal(np.sort(np.concatenate([train, val, test])), np.arange(30))
    assert 1 in y[chosen][train]


def test_detection_false_negative_is_attack_to_benign_only():
    m = evaluate_predictions([0, 1, 2, 2], [1, 2, 0, 2], ["Benign", "DoS", "Bot"])
    assert m["attack_false_negative_rate"] == pytest.approx(1 / 3)
    assert m["attack_classification_error_rate"] == pytest.approx(2 / 3)
    assert m["benign_false_positive_rate"] == 1
    m = evaluate_predictions([0], [0], ["Benign", "DoS", "Bot"])
    assert len(m["confusion_matrix"]) == 3 and m["attack_false_negative_rate"] is None


def test_weighted_aggregation_matches_flower():
    a, b = [np.array([1., 3.], dtype=np.float32)], [np.array([5., 7.], dtype=np.float32)]
    expected = np.array([4., 6.])
    np.testing.assert_allclose(average([a, b], [1, 3])[0], expected)
    np.testing.assert_allclose(flower_average([a, b], [1, 3], 1)[0], expected)


def test_scaffold_control_sign_and_corrected_sgd():
    # With a constant gradient g=3, c=2, ci=1, a single step gives corrected g=4.
    out = scaffold_control([np.array([1.])], [np.array([2.])],
                           [np.array([10.])], [np.array([9.6])], 1, .1)
    np.testing.assert_allclose(out[0], [3.])
    seed_everything(42)
    m = build_model(2, 2)
    initial = get_weights(m)
    zero = [np.zeros_like(w) for w in initial]
    x = np.array([[1., 2.]], dtype=np.float32); y = np.array([1])
    plain, _, _ = train_local(m, x, y, 1, 1, .1, 42)
    set_weights(m, initial)
    corrected, _, _ = train_local(m, x, y, 1, 1, .1, 42,
                                 control_global=[w + 1 for w in zero], control_client=zero)
    for a, b in zip(plain, corrected):
        np.testing.assert_allclose(b, a - .1, atol=1e-6)


def test_fedprox_zero_equals_fedavg_and_positive_restrains_drift():
    seed_everything(3)
    m = build_model(2, 2); initial = get_weights(m)
    x = np.array([[1., 2.], [2., 3.]], dtype=np.float32); y = np.array([0, 1])
    plain, _, _ = train_local(m, x, y, 4, 2, .01, 3)
    set_weights(m, initial)
    zero, _, _ = train_local(m, x, y, 4, 2, .01, 3, global_weights=initial, mu=0)
    for a, b in zip(plain, zero):
        np.testing.assert_array_equal(a, b)
    set_weights(m, initial)
    prox, _, _ = train_local(m, x, y, 4, 2, .01, 3, global_weights=initial, mu=10)
    norm = lambda ws: sum(float(((a - b) ** 2).sum()) for a, b in zip(ws, initial))
    assert norm(prox) < norm(plain)


def test_clipping_noise_scale_and_accounting():
    clipped = clip_update([np.array([3., 4.]), np.array([12.])], 2)
    assert np.sqrt(sum((w ** 2).sum() for w in clipped)) == pytest.approx(2)
    out = private_average([[np.array([3., 4.])], [np.array([0., 0.])]], 2, 0)[0]
    np.testing.assert_allclose(out, [.6, .8])
    class Spy:
        def normal(self, loc, scale, shape):
            assert scale == pytest.approx(3 * 2 * 2 / 2)
            return np.zeros(shape)
    private_average([[np.zeros(2)], [np.zeros(2)]], 2, 3, Spy())
    assert accounting(4, 2)["rho"] == .5
    assert accounting(4, 2)["conditional_epsilon_bound"] < accounting(4, 1)["conditional_epsilon_bound"]
    assert accounting(4, 0)["conditional_epsilon_bound"] is None


@pytest.fixture
def prepared(tmp_path):
    rows = []
    raw_labels = ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "FTP-Patator",
                  "Web Attack - XSS", "Bot", "Infiltration", "Heartbleed"]
    for c, name in enumerate(raw_labels):
        for i in range(12):
            rows.append({" f1": float(c * 100 + i), " f2": float(i * i + c), "constant": 1, " Label": name})
    rows += [rows[0].copy(), {" f1": np.inf, " f2": 0, "constant": 1, " Label": "BENIGN"}]
    # Contradictory labels for identical features must both be removed.
    rows += [{" f1": -999., " f2": -1., "constant": 1, " Label": "BENIGN"},
             {" f1": -999., " f2": -1., "constant": 1, " Label": "Bot"}]
    csv = tmp_path / "raw.csv"; pd.DataFrame(rows).to_csv(csv, index=False)
    out = tmp_path / "prepared"
    meta = prepare(csv, out, 0)
    return out, meta, csv


def test_preprocessing_no_leakage(prepared):
    out, meta, csv = prepared
    assert meta["full_objective_data_coverage"]
    assert meta["exact_duplicates_removed"] == 1
    assert meta["invalid_rows_removed"] == 1
    assert meta["conflicting_rows_removed"] == 2
    assert meta["train_constant_features_removed"] == ["constant"]
    train, val, test = [np.load(out / f"{s}.npz") for s in ["train", "validation", "test"]]
    assert not set(train["row_ids"]) & set(test["row_ids"])
    assert not set(val["row_ids"]) & set(test["row_ids"])
    np.testing.assert_allclose(train["X"].mean(0), [0, 0], atol=1e-6)
    original = pd.read_csv(csv).iloc[:108, :2].to_numpy()
    scaler = load(out / "preprocessor.joblib")["scaler"]
    np.testing.assert_allclose(scaler.mean_, original[train["row_ids"]].mean(0))


@pytest.mark.parametrize("algorithm", ["centralized", "local", "fedavg", "fedavg_uniform", "fedprox", "scaffold", "privacy"])
def test_every_algorithm_runs(prepared, tmp_path, algorithm):
    data, _, _ = prepared
    metrics = run(data, tmp_path / algorithm, Config(algorithm=algorithm, rounds=2, batch_size=8, noise=.05))
    assert 0 <= metrics["accuracy"] <= 1
    assert np.isfinite(metrics["loss"])
    assert metrics["full_objective_data_coverage"]
    with pytest.raises(FileExistsError):
        run(data, tmp_path / algorithm, Config())


def test_missing_classes_requires_explicit_override(tmp_path):
    frame = pd.DataFrame({"x": range(20), "Label": ["Benign"] * 10 + ["DoS_DDoS"] * 10})
    path = tmp_path / "legacy.csv"; frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Full objective coverage unavailable"):
        prepare(path, tmp_path / "strict")
    meta = prepare(path, tmp_path / "limited", allow_incomplete=True)
    assert not meta["full_objective_data_coverage"] and meta["legacy_merged_labels"]


def test_all_failed_configs_are_reported(tmp_path):
    from common import save_json
    from evaluation.compare_results import compare
    save_json(tmp_path / "failed/failure.json", {"run": "failed", "error": "diverged"})
    assert compare(tmp_path).empty
    assert "No configurations completed" in (tmp_path / "RESULTS.md").read_text()


def test_nonfinite_config_rejected():
    from experiments import validate
    with pytest.raises(ValueError, match="finite"):
        validate(Config(noise=float("nan")))


def test_nonprivate_repeatability(prepared, tmp_path):
    data, _, _ = prepared
    cfg = Config(algorithm="fedavg", rounds=2, batch_size=8)
    a = run(data, tmp_path / "a", cfg)
    b = run(data, tmp_path / "b", cfg)
    assert a["confusion_matrix"] == b["confusion_matrix"]
    assert a["loss"] == b["loss"]
