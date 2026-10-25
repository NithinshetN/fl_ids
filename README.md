# Federated IDS - 300,000-row edition

Updated PyTorch/Flower code for the supplied 300k bundle. The original CSV and supporting notes/counts are included unchanged under `data/raw/`. Previous project files are preserved separately.

## Dataset and code changes

There are **8 classes**: Benign, DoS, DDoS, PortScan, Brute_Force, Bot, Infiltration and Heartbleed. The old DoS_DDoS group became separate DoS and DDoS classes. The input spelling `Brute Force` maps to `Brute_Force`.

- Preprocessing defaults to **all cleaned rows** (`--sample-size 0`), instead of silently reducing the new input to 100,000.
- Explicit `project8` scope accepts these eight classes. `report9` additionally requires Web_Attack. Observed web labels are preserved in either mode.
- Directory loading ignores the recognized class-count summary CSV rather than treating it as network traffic.
- Organization profiles now represent DoS, DDoS, brute force, botnet and mixed traffic when web labels are absent. With Web_Attack present, the original five-profile scheme is used.
- Balanced accuracy and a training-majority baseline now accompany macro F1, per-class metrics and attack FNR.
- Input paths, launchers, metadata and regression tests are updated.

Models already obtain their feature/class counts from metadata. For this bundle the MLP is **69 -> 128 -> 64 -> 64 -> 8**. No hard-coded eight-output layer is necessary. **Retrain from scratch:** old 67-feature/7-class checkpoints, scalers and encoders are incompatible.

**Report limitation:** there is still no separate Web_Attack label. The bundle notes say web brute force was merged into Brute Force, while web XSS/SQL injection were excluded. These distinctions cannot be recovered from grouped labels. Passing `project8` sets `label_scope_satisfied=true`, but `full_objective_data_coverage=false` remains correct for the missing web objective.

## 1. Install

Use 64-bit Python 3.12. Extract the ZIP and open a terminal inside `fl_ids_project_300k`, where this README and requirements.txt are located.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

If activation is blocked, replace `python` with `.\.venv\Scripts\python.exe` in all commands. Do not copy another machine's virtual environment.

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows/Python 3.12/CPU was tested. A GPU is not required. Keep the pinned package versions, especially Flower 1.20.0. Installation needs internet; the default simulation does not need sockets.

## 2. Prepare the dataset

```bash
python preprocessing/inspect_dataset.py --input data/raw
python preprocessing/preprocess.py
```

The second command is equivalent to:

```bash
python preprocessing/preprocess.py --input data/raw/CICIDS2017_stratified_300k.csv --output data/processed/main --sample-size 0 --label-scope project8 --seed 42
```

Zero means all cleaned rows, not all invalid/duplicate rows. Positive limits explicitly subsample. Do not pass `--sample-size 100000` unless you want a smaller experiment.

| Stage | Rows |
|---|---:|
| Input | 300,000 |
| Invalid numeric rows removed | 290 |
| Exact duplicates removed after invalid-row filtering | 14,941 |
| Conflicting-label rows removed | 34 |
| Cleaned total | **284,735** |
| Training | 199,314 |
| Validation | 28,474 |
| Test | 56,947 |

Conflicting rows have identical retained features but different mapped labels; all conflicting versions are removed. The bundle's 15,053 duplicate count applies before cleaning. Invalid feature-value counts differ from invalid-row counts because one row may contain several invalid values.

Deduplication precedes splitting. Constant-feature selection and StandardScaler use training data only. All methods share the same 70/10/20 class-stratified split. Metadata records counts, class IDs, feature schema, hashes and coverage. A missing-Web_Attack warning is expected even though the eight-class scope passes.

## 3. Quick execution check

```bash
python -m pytest -q
python run_experiments.py --suite --seeds 42 --rounds 2 --output results/quick_300k
python evaluation/plot_results.py --results results/quick_300k
```

This runs 20 configurations using all cleaned rows: centralized; local/FedAvg/uniform-FedAvg/FedProx/SCAFFOLD across IID, non-IID and organization distributions; plus four privacy settings. Two rounds verify execution, not convergence or final detection quality.

## 4. Repeated experiments

```bash
python run_experiments.py --suite --seeds 42 43 44 --rounds 10 --local-epochs 1 --batch-size 256 --lr 0.01 --mu 0.01 --alpha 0.3 --clip-norm 0.1 --noise-levels 0 0.5 1 2 --output results/comparison_300k
python evaluation/plot_results.py --results results/comparison_300k
```

This runs 60 configurations. Ten rounds are a starting budget, not a convergence guarantee. Choose settings from validation curves, freeze them, then report final test metrics. More rows increase minibatches/runtime; hidden-layer widths and batch size need not change just because data grew. Batch size 512 is an optional throughput experiment but changes optimizer-step counts and must be applied consistently across comparisons.

`run_pipeline.bat` (Windows) or `bash run_pipeline.sh` (macOS/Linux) prepares data, runs the default suite and plots results. These use `data/processed/main` and `results/main`, which must not already exist. If preprocessing is already done, run the experiment command directly.

Use a fresh output directory for each run. There is no overwrite or automatic resume. Divergence is saved in failure.json and reported; remaining configurations run, then the suite exits nonzero. Report failed settings alongside successful scores.

## 5. Individual methods and optional networking

```bash
python centralized/train_centralized.py --output results/centralized_300k --seeds 42
python clients/local_training.py --mode noniid --output results/local_300k --seeds 42
python run_experiments.py --algorithm fedavg --mode iid --output results/fedavg_300k --seeds 42
python federated/advanced_fl.py --algorithm fedprox --mode noniid --output results/fedprox_300k --seeds 42
python federated/advanced_fl.py --algorithm scaffold --mode noniid --output results/scaffold_300k --seeds 42
python federated/privacy_fl.py --noise 1 --clip-norm 0.1 --output results/privacy_300k --seeds 42
```

All default to data/processed/main; override with `--data PATH`. Organization mode requires five clients. Each focus attack family has about 80% of its rows assigned to its designated organization; benign is distributed uniformly. These are simulated organizations. Partitions cover each training row once; empty-client repairs and actual class distributions are reported.

The suite uses sequential simulation with Flower's actual FedAvg aggregation strategy for FedAvg/FedProx. Optional real local FedAvg communication:

```bash
python federated/run_network.py --data data/processed/main --output results/network_300k --mode noniid --clients 5 --rounds 3 --port 8080 --timeout 1200
```

The launcher starts server/clients, captures logs and cleans up processes. Network mode supports FedAvg only; advanced/privacy methods use simulation. This is unencrypted localhost communication, not secure multi-organization deployment. Pinned Flower compatibility APIs emit deprecation warnings; upgrades require adaptation and testing.

## 6. Interpret imbalance correctly

Benign is **80.31% of input rows**. After cleaning/splitting, always predicting the training-majority class (Benign) gives **81.95% test accuracy while detecting no attacks**. Inspect macro F1, balanced accuracy, per-class recall and attack FNR alongside accuracy.

Balanced accuracy averages recall over classes present in the test set. Macro recall includes every configured class, with zero for zero-support classes. Attack FNR counts attacks predicted Benign, not attacks assigned another attack class. No automatic resampling or class weighting was introduced.

Bot decreased from 1,966 examples in the previous sample to 208 here. Heartbleed/Infiltration remain 11/36 raw examples, with only 2/7 test examples; Bot has 42 test examples. Rare-class conclusions remain uncertain.

Seeds vary initialization, minibatch order and client partitions; the preprocessing split stays fixed. Seed standard deviations are not confidence intervals over independent datasets. Old/new scores differ in distributions, labels and splits, so cannot isolate a pure row-count effect.

Output files:

- metadata.json: data audit, IDs, feature schema, hashes, split supports and majority baseline.
- train.npz, validation.npz, test.npz, preprocessor.joblib: fixed splits and fitted preprocessing.
- RESULTS.md, comparison.csv, summary.csv: readable comparison, individual runs and mean/SD/count. Summary CSV has two header rows.
- Per run: configuration, provenance, partition summary/indices, validation history, final metrics and centralized/global checkpoint.
- local_metrics.json: each independent local model; local summaries are not ensemble predictions.
- privacy_accounting.json: conditional bound and assumptions.
- plots/: comparison, convergence, confusion matrices and privacy figures.

Regenerate tables using `python evaluation/compare_results.py --results results/comparison_300k`.

## 7. Retained methods and limitations

The common metadata-sized MLP uses LeakyReLU, cross-entropy and SGD. All methods visit each training row `rounds * local_epochs` times; exposure is matched, not optimizer steps or elapsed time. Sample-weighted FedAvg/FedProx and uniform SCAFFOLD retain their corrected implementations. Uniform FedAvg is the matching SCAFFOLD control. All clients participate every round. Only validation is evaluated during training; test evaluation follows training.

Privacy mode clips each complete update to norm C, averages N clients uniformly and adds Gaussian noise with standard deviation `noise_multiplier * 2C/N`. Over T rounds the conditional replacement-client accounting is `rho=T/(2*sigma^2)` and `epsilon=rho+2*sqrt(rho*log(1/delta))`, with sigma the noise multiplier. Zero noise has no finite privacy guarantee. Private noise is independently entropy-seeded, so private runs are not bitwise repeatable from the public training seed.

**This is a mechanism study, not end-to-end DP.** Accounting assumes fixed/public preprocessing, partitions and evaluation data plus a trusted aggregator. Actual pooled preprocessing, metadata and nonprivate baselines are not DP. Secure aggregation is absent, multiple released runs need composition, and NumPy sampling is not a hardened DP implementation.

Row-based splitting is not a temporal/capture-level holdout. Exact deduplication does not prove generalization to new days or real organizations. The MLP adapts the supplied code; it is not a verified reproduction of the unavailable reference architecture PDF.

## 8. Later adding web-attack data

Use original labeled CSVs in a separate folder:

```bash
python preprocessing/preprocess.py --input data/raw/with_web --output data/processed/with_web --sample-size 0 --label-scope report9
python run_experiments.py --data data/processed/with_web --output results/with_web --suite
```

The model automatically expands to nine observed groups. Do not guess web labels from the merged Brute Force category.

## Troubleshooting

- Output exists: use a fresh directory; do not mix runs.
- Missing modules: install/run with the same virtual-environment Python.
- Web_Attack warning: project8 passes, but report-wide web coverage is still absent.
- Unexpected CSV: point directly to the traffic file; only the recognized count sidecar is ignored.
- High accuracy and low attack recall: inspect majority baseline and per-class scores.
- Divergence: report failure, then test lower learning rate/clipping norm using validation.
- Network error: inspect logs, select a free port and allow localhost connections. Simulation needs no sockets.

See VALIDATION.md for checks performed and CHANGES_300K.md for file-level changes.
