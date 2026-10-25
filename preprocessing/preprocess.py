"""Clean, deduplicate, sample, split, then fit preprocessing on training only."""
from pathlib import Path
import argparse
import sys
import re
import warnings
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, save_json, sha256

RAW_CLASSES = ["Benign", "DoS", "DDoS", "PortScan", "Brute_Force",
               "Web_Attack", "Bot", "Infiltration", "Heartbleed"]
PROJECT_CLASSES = [c for c in RAW_CLASSES if c != "Web_Attack"]
LEGACY_CLASSES = ["Benign", "Bot", "Brute_Force", "DoS_DDoS", "Heartbleed",
                  "Infiltration", "PortScan", "Web_Attack"]
IDENTIFIERS = {"flow id", "source ip", "destination ip", "src ip", "dst ip",
               "timestamp", "source port", "fwd header length.1"}


def map_label(value):
    text = str(value).strip().replace("\xa0", " ")
    key = re.sub(r"[^a-z0-9]", "", text.lower())
    if key.startswith("webattack") or key == "web":
        return "Web_Attack"
    aliases = {"benign": "Benign", "bot": "Bot", "botnet": "Bot",
               "ftp patator": "Brute_Force", "ftppatator": "Brute_Force",
               "sshpatator": "Brute_Force", "bruteforce": "Brute_Force",
               "portscan": "PortScan", "infiltration": "Infiltration",
               "heartbleed": "Heartbleed", "ddos": "DDoS",
               "dosddos": "DoS_DDoS", "dos": "DoS"}
    if key.startswith("dos") and key != "dosddos":
        return "DoS"
    return aliases.get(key)


def stratified_subset(y, limit, rng):
    """Proportional quotas with >=3 per observed class where possible (no oversampling)."""
    if limit <= 0 or limit >= len(y):
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    minimum = np.minimum(counts, 3)
    if limit < minimum.sum():
        raise ValueError("Sample size too small to preserve classes; increase --sample-size.")
    desired = counts * (limit / len(y))
    quotas = np.minimum(counts, np.maximum(minimum, np.floor(desired).astype(int)))
    while quotas.sum() < limit:
        eligible = np.where(quotas < counts)[0]
        j = eligible[np.argmax((desired - quotas)[eligible])]
        quotas[j] += 1
    while quotas.sum() > limit:
        eligible = np.where(quotas > minimum)[0]
        j = eligible[np.argmax((quotas - desired)[eligible])]
        quotas[j] -= 1
    selected = np.concatenate([rng.choice(np.flatnonzero(y == c), q, replace=False)
                               for c, q in zip(classes, quotas)])
    rng.shuffle(selected)
    return selected


def split_indices(y, seed):
    """70/10/20 per class. Singletons stay in train; two rows give train/test."""
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n = len(idx)
        nt = max(1, int(round(n * .2))) if n >= 2 else 0
        nv = max(1, int(round(n * .1))) if n >= 3 else 0
        nt = min(nt, n - nv - 1)
        test.extend(idx[:nt]); val.extend(idx[nt:nt + nv]); train.extend(idx[nt + nv:])
    result = [np.asarray(v, dtype=np.int64) for v in (train, val, test)]
    for v in result:
        rng.shuffle(v)
    if any(len(v) == 0 for v in result):
        raise ValueError("Insufficient data for nonempty train/validation/test splits.")
    return result


def input_files(input_path):
    """Exclude the known class-count sidecar, never interpret it as traffic."""
    input_path = Path(input_path)
    candidates = sorted(input_path.glob("*.csv")) if input_path.is_dir() else [input_path]
    files, ignored = [], []
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(path)
        columns = [str(c).strip() for c in pd.read_csv(path, nrows=0, encoding="utf-8-sig").columns]
        if path.name.endswith("_class_counts.csv") and columns == ["Class", "Records", "Percentage"]:
            ignored.append(path.name)
        else:
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No traffic CSV input found at {input_path}")
    return files, ignored


def prepare(input_path, output, sample_size=0, seed=42, allow_incomplete=False, label_scope="project8"):
    input_path, output = Path(input_path), Path(output)
    files, ignored_files = input_files(input_path)
    if sample_size < 0 or label_scope not in {"project8", "report9"}:
        raise ValueError("sample_size must be >=0 and label_scope project8 or report9")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}. Choose a new --output directory.")
    frames, sources, feature_names = [], [], None
    unknown, invalid_rows, input_rows, input_label_counts = {}, 0, 0, {}
    for path in files:
        sources.append({"file": path.name, "sha256": sha256(path)})
        for frame in pd.read_csv(path, chunksize=100000, encoding="utf-8-sig", encoding_errors="replace"):
            frame.columns = [str(c).strip().replace("\ufeff", "") for c in frame.columns]
            if frame.columns.duplicated().any() or "Label" not in frame:
                raise ValueError(f"Ambiguous columns or missing Label column in {path}")
            columns = [c for c in frame.columns if c != "Label" and c.lower() not in IDENTIFIERS
                       and not c.lower().startswith("unnamed:")]
            if feature_names is None:
                feature_names = columns
            if set(columns) != set(feature_names):
                raise ValueError(f"Feature schema differs in {path}")
            input_rows += len(frame)
            for label, count in frame.Label.astype(str).value_counts().items():
                input_label_counts[label] = input_label_counts.get(label, 0) + int(count)
            labels = frame.Label.map(map_label)
            for name, count in frame.loc[labels.isna(), "Label"].astype(str).value_counts().items():
                unknown[name] = unknown.get(name, 0) + int(count)
            numeric = frame[feature_names].apply(pd.to_numeric, errors="coerce")
            finite = np.isfinite(numeric.to_numpy()).all(axis=1)
            invalid_rows += int((~finite & labels.notna()).sum())
            keep = finite & labels.notna()
            cleaned = numeric.loc[keep].copy()
            cleaned["Label"] = labels.loc[keep]
            frames.append(cleaned)
    if unknown:
        raise ValueError(f"Unknown labels (none silently discarded): {unknown}")
    data = pd.concat(frames, ignore_index=True)
    if data.empty:
        raise ValueError("No valid rows remain.")
    # Remove every conflicting feature vector, rather than arbitrarily choosing a label.
    exact_duplicates = int(data.duplicated().sum())
    data = data.drop_duplicates()
    conflicting = data.duplicated(subset=feature_names, keep=False)
    conflicting_rows = int(conflicting.sum())
    data = data.loc[~conflicting].reset_index(drop=True)
    legacy = "DoS_DDoS" in set(data.Label)
    if legacy and ({"DoS", "DDoS"} & set(data.Label)):
        raise ValueError("Do not mix raw labels with the pre-merged DoS_DDoS dataset.")
    expected = LEGACY_CLASSES if legacy else RAW_CLASSES
    names = [c for c in expected if c in set(data.Label)]
    if "Benign" not in names or len(names) < 2:
        raise ValueError("An IDS comparison requires Benign and at least one attack class.")
    missing = sorted(set(RAW_CLASSES) - set(names))
    complete = not missing and not legacy
    required = PROJECT_CLASSES if label_scope == "project8" else RAW_CLASSES
    missing_scope = sorted(set(required) - set(names))
    scope_satisfied = not missing_scope and not legacy
    if not scope_satisfied and not allow_incomplete:
        raise ValueError(f"Full objective coverage unavailable for {label_scope}. Missing: {missing_scope}; merged labels: {legacy}. "
                         "Provide the missing labels, or --allow-incomplete for explicitly limited experiments.")
    if not complete:
        warnings.warn(f"LIMITED DATASET: missing {missing}; legacy merged labels={legacy}")
    y = data.Label.map({c: i for i, c in enumerate(names)}).to_numpy(np.int64)
    chosen = stratified_subset(y, sample_size, np.random.default_rng(seed))
    X, y = data[feature_names].to_numpy(np.float64)[chosen], y[chosen]
    train, val, test = split_indices(y, seed)
    variable = np.ptp(X[train], axis=0) > 0
    dropped = [c for c, keep in zip(feature_names, variable) if not keep]
    features = [c for c, keep in zip(feature_names, variable) if keep]
    if not features:
        raise ValueError("No variable training features remain.")
    X = X[:, variable]
    scaler = StandardScaler().fit(X[train])
    output.mkdir(parents=True)
    splits = {}
    for name, idx in zip(("train", "validation", "test"), (train, val, test)):
        transformed = scaler.transform(X[idx])
        if not np.isfinite(transformed).all() or np.abs(transformed).max() > np.finfo(np.float32).max:
            raise ValueError("Scaling generated non-finite/overflow values.")
        np.savez_compressed(output / f"{name}.npz", X=transformed.astype(np.float32),
                            y=y[idx], row_ids=chosen[idx])
        splits[name] = {"rows": len(idx), "class_counts": {
            c: int((y[idx] == i).sum()) for i, c in enumerate(names)}}
    dump({"scaler": scaler, "feature_names": features}, output / "preprocessor.joblib")
    coverage_notes = [f"{c} has {int((y == i).sum())} rows; interpret rare-class metrics cautiously."
                      for i, c in enumerate(names) if (y == i).sum() < 50]
    majority_id = int(np.bincount(y[train], minlength=len(names)).argmax())
    metadata = {"schema_version": 3, "source_files": sources, "input_rows": input_rows,
                "ignored_auxiliary_csv_files": ignored_files, "input_label_counts": input_label_counts,
                "label_scope": label_scope, "required_class_names": required,
                "label_scope_satisfied": scope_satisfied, "missing_scope_classes": missing_scope,
                "class_to_id": {name: i for i, name in enumerate(names)},
                "training_majority_class": names[majority_id],
                "majority_baseline_test_accuracy": float(np.mean(y[test] == majority_id)),
                "invalid_rows_removed": invalid_rows, "exact_duplicates_removed": exact_duplicates,
                "conflicting_rows_removed": conflicting_rows, "requested_sample_size": sample_size,
                "sampled_rows": len(y), "seed": seed, "class_names": names,
                "feature_names": features, "n_features": len(features),
                "train_constant_features_removed": dropped, "splits": splits,
                "full_objective_data_coverage": complete, "missing_raw_classes": missing,
                "legacy_merged_labels": legacy, "notes": coverage_notes,
                "sampling": "Proportional after cleaning/deduplication; minimum 3 per class where available.",
                "privacy_scope": "Offline benchmark preprocessing uses pooled training statistics; not end-to-end DP."}
    for name in splits:
        metadata.setdefault("split_sha256", {})[name] = sha256(output / f"{name}.npz")
    save_json(output / "metadata.json", metadata)
    print(f"Prepared {len(y):,} rows, {len(features)} features, {len(names)} classes at {output}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/CICIDS2017_stratified_300k.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/processed/main")
    parser.add_argument("--sample-size", type=int, default=0, help="0 (default) uses all cleaned rows; positive values explicitly subsample")
    parser.add_argument("--label-scope", choices=["project8", "report9"], default="project8",
                        help="project8 requires eight groups; report9 additionally requires Web_Attack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-incomplete", action="store_true")
    a = parser.parse_args()
    if a.sample_size < 0:
        parser.error("--sample-size must be >=0")
    prepare(a.input, a.output, a.sample_size, a.seed, a.allow_incomplete, a.label_scope)


if __name__ == "__main__":
    main()
