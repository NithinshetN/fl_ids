# Validation of the 300k edition

Date: 25 October 2026. Windows, Python 3.12.14, CPU PyTorch 2.5.1+cpu, Flower 1.20.0, NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2.

## Checks completed

- **36 automated tests passed.** Existing regression tests cover preprocessing, disjoint partitions, FedAvg/FedProx/SCAFFOLD, privacy calculations and all algorithms. New tests cover eight-class mapping, separate DoS/DDoS, retaining more than 100k rows by default, sidecar handling, report9 rejection when web data is absent, organization-profile adaptation, balanced accuracy, actual eight-class training and rejection of seven-class weights in an eight-class model.
- Preprocessing ran on the entire supplied 300k CSV. Result: 284,735 cleaned rows, 69 features and 8 classes. Split: 199,314 train, 28,474 validation, 56,947 test.
- **20 experiments completed** on that full cleaned dataset with seed 42, two rounds, one local epoch, batch size 256, SGD lr 0.01, mu 0.01, alpha 0.3, clipping norm 0.1, and privacy noise levels 0/0.5/1/2. This exercises every method and all three distributions.

Tests emitted expected missing-Web_Attack warnings plus dependency deprecation warnings. They did not fail. Network process launchers were retained from the previously tested release; this update's real-data validation used sequential Flower aggregation rather than rerunning socket tests.

## Results are execution evidence, not final accuracy claims

The two-round centralized run had approximately 89.43% accuracy and macro F1 0.2922. Several federated runs predicted only Benign: accuracy 81.95%, macro F1 0.1126, balanced accuracy 0.125 and attack FNR 1.0. These values show undertraining and the imbalance trap; they must not be presented as successful detection merely because accuracy exceeds 80%.

This release did not tune on test scores, impose an accuracy target, or automatically rebalance the test data. Use validation-based training/hyperparameter selection and repeated seeds for final analysis. Do not infer that one algorithm is generally best from two rounds and one seed.

The eight-class scope passes. Full report coverage does not: Web_Attack is absent. Rare-class test support is only 2 Heartbleed and 7 Infiltration records. The privacy mechanism's bounds remain conditional, not end-to-end DP.

## Included evidence

validation_results/ includes dataset metadata, the 20-run tables, configurations, provenance, client summaries, histories, per-class metrics and conditional privacy reports. Plots can be regenerated from these files. Prepared arrays, checkpoints and client-index archives from validation are omitted to reduce ZIP size; ordinary runs generate them. Original raw CSV bytes and bundle provenance notes are included under data/raw/.
