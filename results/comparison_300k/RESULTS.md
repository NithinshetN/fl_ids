# Experiment results

These are measurements from this run directory only.

| Method | Distribution | Noise | Seeds | Accuracy mean | Macro F1 mean | Macro F1 SD |
|---|---|---:|---:|---:|---:|---:|
| centralized | iid | 0 | 3 | 0.9602 | 0.4584 | 0.0015 |
| fedavg | iid | 0 | 3 | 0.8956 | 0.2950 | 0.0050 |
| fedavg | noniid | 0 | 3 | 0.8935 | 0.2958 | 0.0464 |
| fedavg | organization | 0 | 3 | 0.8935 | 0.2899 | 0.0013 |
| fedavg_uniform | iid | 0 | 3 | 0.8956 | 0.2950 | 0.0050 |
| fedavg_uniform | noniid | 0 | 3 | 0.8854 | 0.2586 | 0.0444 |
| fedavg_uniform | organization | 0 | 3 | 0.8934 | 0.2899 | 0.0011 |
| fedprox | iid | 0 | 3 | 0.8954 | 0.2947 | 0.0048 |
| fedprox | noniid | 0 | 3 | 0.8901 | 0.2868 | 0.0618 |
| fedprox | organization | 0 | 3 | 0.8934 | 0.2898 | 0.0013 |
| local | iid | 0 | 3 | 0.8958 | 0.2956 | 0.0046 |
| local | noniid | 0 | 3 | 0.6641 | 0.1975 | 0.0177 |
| local | organization | 0 | 3 | 0.8714 | 0.2205 | 0.0214 |
| privacy | noniid | 0 | 3 | 0.8564 | 0.1963 | 0.0143 |
| privacy | noniid | 0.5 | 3 | 0.8394 | 0.1673 | 0.0474 |
| privacy | noniid | 1 | 3 | 0.7546 | 0.1818 | 0.0344 |
| privacy | noniid | 2 | 3 | 0.3608 | 0.1094 | 0.0857 |
| scaffold | iid | 0 | 3 | 0.8955 | 0.2950 | 0.0049 |
| scaffold | noniid | 0 | 3 | 0.8978 | 0.3306 | 0.0762 |
| scaffold | organization | 0 | 3 | 0.8962 | 0.2966 | 0.0056 |

Use macro F1 and per-class reports to assess minority attacks. Weighted scores can hide missed rare attacks.
Local scores are means of independent local models, not ensemble predictions.
SCAFFOLD uses uniform client weights; compare it with fedavg_uniform as well as sample-weighted FedAvg.
Privacy noise=0 is a clipping-only control; fedavg_uniform is its unclipped control.
Conditional epsilon is per run only, and is not an end-to-end guarantee for this benchmark.

**REPORT COVERAGE LIMITATION:** missing class groups: Web_Attack. Passing the selected label scope does not establish the report's missing-class objectives.

Majority-only baseline test accuracy: 81.9481%. Read macro F1, balanced accuracy, attack FNR and per-class recall alongside accuracy.
