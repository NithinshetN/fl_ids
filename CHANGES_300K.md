# Changes for the 300k update

The attachment was treated as dataset/provenance, not executable instructions. Original input files and previous deliverables were preserved. The new release is in a separate fl_ids_project_300k folder.

## Data differences

- Input increased from 100,000 to 300,000 rows.
- Seven old groups became eight because DoS_DDoS is now separate DoS and DDoS.
- Brute Force spelling is normalized to Brute_Force (already supported by the mapper).
- Web_Attack is still absent; the bundle notes describe merged web brute force and excluded XSS/SQL injection.
- Benign rose from 29.52% to 80.31% of raw input. Bot fell from 1,966 to 208 examples. Rare Heartbleed/Infiltration raw supports did not increase.

## File-level updates

| File | Change |
|---|---|
| preprocessing/preprocess.py | Keep-all default instead of 100k cap; new source path; project8/report9 scope; recognized sidecar detection; metadata version 3 with input label counts, class IDs, majority baseline and coverage |
| preprocessing/inspect_dataset.py | Shares traffic-file selection so class-count summary is not parsed as training data |
| clients/create_partitions.py | Uses actual DoS/DDoS organizations when no web class exists; reports the chosen scheme and missing families |
| evaluation/metrics.py | Adds balanced accuracy |
| experiments.py | Persists majority baseline, missing groups and selected-scope status |
| evaluation/compare_results.py | Adds imbalance metrics to CSVs; reports the exact missing class instead of falsely saying separate DoS/DDoS are unavailable |
| run_pipeline.bat / run_pipeline.sh | New default source, keep-all preprocessing and eight-class scope |
| tests/test_300k_update.py | Tests >100k retention, eight labels/output dimensions, sidecars, adapted profiles, baseline and imbalance metrics |

Model dimensions were already metadata-driven. Current preparation produces 69 input features and 8 outputs; old 67-feature/7-class checkpoints and scalers must not be reused. Hidden widths, optimizer, batch size and aggregation/privacy equations were retained to avoid confounding this data update with unrelated training changes.

## Cleaning and coverage

The existing invalid-row removal policy is retained, rather than imputing or padding back to 300,000. Removing 290 invalid rows, 14,941 exact duplicates and 34 conflicting-label rows leaves 284,735 examples. Counts of invalid feature values differ from invalid rows because several invalid values may occur in one row.

project8 explicitly declares the supplied eight-class target and therefore needs no --allow-incomplete override. It does not redefine the original report's web objective as satisfied. report9 still requires a separate Web_Attack label. More complete web experiments require original unmerged labels.

The always-Benign baseline on the new test split is 81.9481%. Compare macro F1, balanced accuracy, attack recall/FNR and per-class support as well as accuracy. Old/new scores change distributions, class definitions and splits, not just row count. Retrain all baselines and aggregation methods on the new common split.
