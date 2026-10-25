# Experiment results

These are measurements from this run directory only.

| Method | Distribution | Noise | Seeds | Accuracy mean | Macro F1 mean | Macro F1 SD |
|---|---|---:|---:|---:|---:|---:|
| centralized | iid | 0 | 1 | 0.8943 | 0.2922 | n/a |
| fedavg | iid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedavg | noniid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedavg | organization | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedavg_uniform | iid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedavg_uniform | noniid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedavg_uniform | organization | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedprox | iid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedprox | noniid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| fedprox | organization | 0 | 1 | 0.8195 | 0.1126 | n/a |
| local | iid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| local | noniid | 0 | 1 | 0.6468 | 0.1138 | n/a |
| local | organization | 0 | 1 | 0.8325 | 0.1425 | n/a |
| privacy | noniid | 0 | 1 | 0.8193 | 0.1951 | n/a |
| privacy | noniid | 0.5 | 1 | 0.8327 | 0.1768 | n/a |
| privacy | noniid | 1 | 1 | 0.3319 | 0.0765 | n/a |
| privacy | noniid | 2 | 1 | 0.3520 | 0.0821 | n/a |
| scaffold | iid | 0 | 1 | 0.8195 | 0.1126 | n/a |
| scaffold | noniid | 0 | 1 | 0.8624 | 0.2064 | n/a |
| scaffold | organization | 0 | 1 | 0.8195 | 0.1126 | n/a |

Use macro F1 and per-class reports to assess minority attacks. Weighted scores can hide missed rare attacks.
Local scores are means of independent local models, not ensemble predictions.
SCAFFOLD uses uniform client weights; compare it with fedavg_uniform as well as sample-weighted FedAvg.
Privacy noise=0 is a clipping-only control; fedavg_uniform is its unclipped control.
Conditional epsilon is per run only, and is not an end-to-end guarantee for this benchmark.

**REPORT COVERAGE LIMITATION:** missing class groups: Web_Attack. Passing the selected label scope does not establish the report's missing-class objectives.

Majority-only baseline test accuracy: 81.9481%. Read macro F1, balanced accuracy, attack FNR and per-class recall alongside accuracy.
