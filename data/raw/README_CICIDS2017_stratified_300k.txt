CICIDS2017 stratified 300,000-record sample
==============================================

Main file: CICIDS2017_stratified_300k.csv
Rows: 300,000
Columns: 79 (78 traffic features + Label)
Sampling random seed: 42
Labels: 8 agreed project groups

Class counts:
       Class  Records  Percentage
      Benign   240925     80.3083
         DoS    26779      8.9263
        DDoS    13570      4.5233
    PortScan    16845      5.6150
 Brute Force     1626      0.5420
         Bot      208      0.0693
Infiltration       36      0.0120
  Heartbleed       11      0.0037

Label mapping:
- BENIGN -> Benign
- DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest -> DoS
- DDoS -> DDoS
- PortScan -> PortScan
- FTP-Patator, SSH-Patator, Web Attack - Brute Force -> Brute Force
- Bot -> Bot
- Infiltration -> Infiltration
- Heartbleed -> Heartbleed

Excluded original labels because they do not fit the eight agreed groups:
- Web Attack - XSS: 652 records in the source archive
- Web Attack - Sql Injection: 21 records in the source archive
Total excluded for label-scope consistency: 673 records.

Sampling method:
A reproducible, random, without-replacement stratified sample was selected from
all eight CSV files in the uploaded archive. Class allocation is proportional
among the six non-rare classes; all available Infiltration (36) and Heartbleed
(11) rows were retained because their source counts are extremely small. This
means rare classes are slightly overrepresented relative to strict proportional
sampling, but no rare examples were discarded.

Data quality notes:
- Column names were trimmed and the Label column was standardized to the agreed
  eight class names.
- 15,053 exact duplicate rows were found within this 300,000-row
  sample. They were retained to preserve the requested exact sample size and
  the original sampled distribution. Before model evaluation, avoid allowing
  identical records to appear in both training and test sets; ideally deduplicate
  or use a duplicate-aware/grouped split and document the choice.
- 141 missing values were found in the 'Flow Bytes/s' feature.
- 439 infinite numeric feature values were found.
  Replace +/- infinity with missing values and fit any imputation/scaling only
  on the training data. Do not fit preprocessing on the held-out test set.
- No non-numeric feature values were detected in the 78 traffic feature columns
  when coercing to numeric (apart from missing values).
- This file is a stratified sample, not a fully preprocessed model-ready tensor.
  Keep Label as the target and do not use it as an input feature.

Recommended next step:
Create one fixed stratified train/test split, audit duplicates to prevent leakage,
fit preprocessing on training data only, and reuse the same held-out test set for
centralized and federated comparisons.
