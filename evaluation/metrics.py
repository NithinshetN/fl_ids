"""Multiclass metrics and binary attack/benign detection rates."""
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report


def evaluate_predictions(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    b = class_names.index("Benign")
    attacks = [i for i in labels if i != b]
    attack_total = int(cm[attacks, :].sum())
    benign_total = int(cm[b, :].sum())
    attack_to_benign = int(cm[attacks, b].sum())
    result = {"accuracy": float(accuracy_score(y_true, y_pred)),
              "balanced_accuracy": float(np.mean(np.diag(cm)[cm.sum(axis=1) > 0] / cm.sum(axis=1)[cm.sum(axis=1) > 0])),
              "attack_false_negative_rate": attack_to_benign / attack_total if attack_total else None,
              "benign_false_positive_rate": (benign_total - int(cm[b, b])) / benign_total if benign_total else None,
              "attack_detection_recall": 1 - attack_to_benign / attack_total if attack_total else None,
              "attack_classification_error_rate":
                  (attack_total - sum(int(cm[i, i]) for i in attacks)) / attack_total if attack_total else None,
              "confusion_matrix": cm.tolist(), "class_names": class_names,
              "classification_report": classification_report(y_true, y_pred, labels=labels,
                  target_names=class_names, zero_division=0, output_dict=True)}
    for average in ("weighted", "macro"):
        p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                   average=average, zero_division=0)
        result.update({f"{average}_precision": float(p), f"{average}_recall": float(r), f"{average}_f1": float(f)})
    result["macro_definition"] = "All configured classes, including zero-support classes (zero_division=0)."
    return result
