import inspect
import numpy as np
import pandas as pd
import pytest
from preprocessing.preprocess import prepare, map_label, input_files, PROJECT_CLASSES
from clients.create_partitions import partition
from models.ids_model import build_model, get_weights, set_weights
from evaluation.metrics import evaluate_predictions
from experiments import run, Config


def eight_class_csv(path, per_class=12):
    frame = pd.DataFrame({"flow_feature": np.arange(len(PROJECT_CLASSES) * per_class),
                          "Label": np.repeat(PROJECT_CLASSES, per_class)})
    frame.loc[frame.Label == "Brute_Force", "Label"] = "Brute Force"
    frame.to_csv(path, index=False)
    return frame


def test_new_scope_separates_dos_and_ddos(tmp_path):
    csv = tmp_path / "traffic.csv"
    eight_class_csv(csv)
    meta = prepare(csv, tmp_path / "data")
    assert meta["class_names"] == PROJECT_CLASSES
    assert meta["class_to_id"]["DoS"] != meta["class_to_id"]["DDoS"]
    assert meta["label_scope_satisfied"]
    assert not meta["full_objective_data_coverage"]
    assert meta["missing_raw_classes"] == ["Web_Attack"]
    assert map_label("Brute Force") == "Brute_Force"
    with pytest.raises(ValueError, match="Web_Attack"):
        prepare(csv, tmp_path / "strict_report", label_scope="report9")


def test_default_preserves_more_than_100k_rows(tmp_path):
    csv = tmp_path / "traffic.csv"
    frame = eight_class_csv(csv, per_class=12501)
    meta = prepare(csv, tmp_path / "data")
    assert inspect.signature(prepare).parameters["sample_size"].default == 0
    assert meta["sampled_rows"] == len(frame) == 100008
    assert sum(v["rows"] for v in meta["splits"].values()) == 100008


def test_directory_ignores_only_known_sidecar(tmp_path):
    eight_class_csv(tmp_path / "traffic.csv")
    pd.DataFrame({"Class": ["Benign"], "Records": [12], "Percentage": [12.5]}).to_csv(
        tmp_path / "sample_class_counts.csv", index=False)
    files, ignored = input_files(tmp_path)
    assert [p.name for p in files] == ["traffic.csv"]
    assert ignored == ["sample_class_counts.csv"]
    meta = prepare(tmp_path, tmp_path / "prepared")
    assert meta["ignored_auxiliary_csv_files"] == ignored
    assert meta["input_rows"] == 96
    pd.DataFrame({"wrong": [1]}).to_csv(tmp_path / "unexpected.csv", index=False)
    with pytest.raises(ValueError, match="missing Label"):
        prepare(tmp_path, tmp_path / "invalid")


def test_organization_has_real_ddos_profile():
    y = np.repeat(np.arange(8), 200)
    parts, summary = partition(y, PROJECT_CLASSES, mode="organization")
    assert summary["organization_profile_scheme"] == ["dos", "ddos", "brute_force", "botnet", "mixed"]
    assert summary["clients"]["1"]["class_counts"]["DDoS"] > 130
    assert summary["clients"]["2"]["class_counts"]["Brute_Force"] > 130
    assert summary["missing_organization_attacks"] == ["Web_Attack"]
    assert len(np.unique(np.concatenate(parts))) == len(y)


def test_balanced_accuracy_exposes_majority_classifier():
    m = evaluate_predictions([0, 0, 0, 1], [0, 0, 0, 0], ["Benign", "DoS", "DDoS"])
    assert m["accuracy"] == .75
    assert m["balanced_accuracy"] == .5
    assert m["macro_recall"] == pytest.approx(1 / 3)
    assert m["attack_false_negative_rate"] == 1


def test_training_and_artifacts_have_eight_classes(tmp_path):
    csv = tmp_path / "traffic.csv"; eight_class_csv(csv)
    meta = prepare(csv, tmp_path / "prepared")
    m = run(tmp_path / "prepared", tmp_path / "run", Config(rounds=1, batch_size=8))
    assert len(m["confusion_matrix"]) == 8
    assert len(m["class_names"]) == 8
    assert m["label_scope_satisfied"] and not m["full_objective_data_coverage"]
    assert m["majority_baseline_test_accuracy"] == meta["majority_baseline_test_accuracy"]
    old = build_model(1, 7); new = build_model(1, 8)
    with pytest.raises(ValueError, match="shape"):
        set_weights(new, get_weights(old))
