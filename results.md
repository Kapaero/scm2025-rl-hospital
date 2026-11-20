# Re-implementation of the experiment (RL + clustering + GOAP)

Calibrated parameters: {"lam_test": 1.362, "lam_train": 0.8, "crit_share": 0.282, "boxes": 6, "hw_scale": 0.8, "len_scale": 3.157, "quantile": 0.991, "qbin": 2, "noise": 0.0}

## Case study (single run, seed 1; as in the paper)
| Method | Critical, min (mean ± SD, n) | Non-critical, min (mean ± SD, n) | Out of contour |
|---|---|---|---|
| GOAP | 13.6 ± 0.6 (70) | 22.2 ± 3.3 (203) | 0.00 |
| Pure RL | 48.9 ± 23.0 (70) | 53.2 ± 21.6 (203) | 0.00 |
| RL + clustering + GOAP | 17.7 ± 7.2 (70) | 21.6 ± 6.5 (203) | 0.48 |

Targets read from the paper figures (mean, min): GOAP 14 / 21; pure RL 53 / 49; hybrid 19 / 23 (critical / non-critical).

## Check on 30 other seeds (101-130), same load as the case study
| Method | Critical, min (mean over seeds ± SD between seeds) | Non-critical, min |
|---|---|---|
| GOAP | 13.7 ± 0.2 | 22.7 ± 1.2 |
| Pure RL | 51.5 ± 10.9 | 52.4 ± 10.8 |
| RL + clustering + GOAP | 18.8 ± 0.9 | 21.6 ± 1.1 |
Share of decisions out of contour (hybrid): 0.48

## Control: load as in the training sample
| Method | Critical, min (mean over seeds ± SD between seeds) | Non-critical, min |
|---|---|---|
| GOAP | 13.3 ± 0.1 | 15.0 ± 0.9 |
| Pure RL | 14.7 ± 0.9 | 14.6 ± 0.9 |
| RL + clustering + GOAP | 14.6 ± 0.8 | 14.6 ± 0.8 |
Share of decisions out of contour (hybrid): 0.00

Share of test-regime states out of contour: 0.26; in the training regime it is 0.05 by construction of the threshold (quantile 0.991).