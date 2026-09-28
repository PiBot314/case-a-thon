# Model metrics (time-based grouped CV, DEVELOPMENT cohort)

Primary: **xgboost**, threshold 0.32, calibrated=True

| model   |       n |   prevalence |   auc_pr |   roc_auc |   brier |   ece |   recall@top10% |   precision@top10% |   recall@top50 |   lift_at_frac |
|:--------|--------:|-------------:|---------:|----------:|--------:|------:|----------------:|-------------------:|---------------:|---------------:|
| logreg  | 687.667 |        0.465 |    0.61  |     0.678 |   0.231 | 0.071 |           0.142 |               0.66 |          0.108 |          1.417 |
| xgboost | 687.667 |        0.465 |    0.615 |     0.681 |   0.231 | 0.073 |           0.146 |               0.68 |          0.106 |          1.46  |

## Pooled OOF (primary)
|                  |        0 |
|:-----------------|---------:|
| n                | 2063     |
| prevalence       |    0.465 |
| auc_pr           |    0.617 |
| roc_auc          |    0.688 |
| brier            |    0.221 |
| ece              |    0     |
| recall@top10%    |    0.149 |
| precision@top10% |    0.694 |
| recall@top50     |    0.035 |
| lift_at_frac     |    1.492 |