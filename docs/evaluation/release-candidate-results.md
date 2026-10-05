# Recorded energy_public_v1 release-candidate results

These are the unchanged evaluation metrics from the verified `energy_public_v1` release candidate, evaluated before the source-only first release. No model was retrained and no metric was recalculated during release preparation. The evaluated checkpoint, fitted preprocessing and generated reports are retained locally but excluded from the first public commit.

## Held-out test: 38,807 windows

| Method | MAE (kWh) | RMSE (kWh) |
| --- | ---: | ---: |
| Public Informer | 5.87175854569893 | 8.607393349984621 |
| Persistence | 8.126241477053108 | 11.888868459709448 |
| Training station mean | 5.801776442521896 | 8.707518861819395 |

Station mean has lower MAE; the neural model has lower RMSE. These are public-profile candidate results, not paper-reported results. The README rounds the same values to three decimals. Test evaluation follows validation-only model selection (epoch 13 of 18 completed epochs, seed 42).

## Validation: 38,906 windows

| Method | MAE (kWh) | RMSE (kWh) |
| --- | ---: | ---: |
| Public Informer | 5.1624725356533645 | 7.465317362864173 |
| Persistence | 7.101272245926078 | 10.434371130061662 |
| Training station mean | 5.084359812625394 | 7.497471814165697 |

## Supplementary matched-cohort comparison: 18,336 shared test windows

The retained generated comparison evaluated frozen public and prior internal models on identical held-out targets/histories, with different feature sets and training cohorts. It is not a controlled feature ablation. Its generated JSON is excluded from the first commit. Internal artifacts are not distributed or used as an inference fallback; this comparison cannot be independently regenerated from the source-only tree.

| Method | MAE (kWh) | RMSE (kWh) |
| --- | ---: | ---: |
| Prior internal model (historical comparison only) | 6.091349604644076 | 8.77932910027814 |
| Public Informer | 5.8710463587160095 | 8.58744307094017 |
| Persistence | 8.115845386125654 | 11.840949841068026 |
| Canonical training station mean | 5.821075502340582 | 8.691375824312008 |

## Provenance and limitations

The primary and validation values above were transcribed exactly from the retained candidate's `models/energy_public_v1/training_report.json`; supplementary values came from `docs/evaluation/matched-cohort-comparison.json`. Both JSON files are excluded by the explicit ten-file policy. This documentation retains aggregate scores, not learned parameters or recorded histories.

Candidate dataset SHA-256: `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c`.

Candidate training-script SHA-256: `e373b3dfeb971ff4cd5266467293eee17d1a24c31f358eb24accc8d686812c8c`.

Candidate manifest SHA-256: `a5b837ca50c8aabec294fed9ec54bf18de9e0a11ac1236b64c35fda5826a289b`.

One seed, known stations, overlapping rolling one-session windows and naive recorded wall-clock completion times limit these results. They do not establish differential privacy, unseen-station performance, fixed-origin forecasting or fidelity to the IEEE architecture. See [methodology](../methodology.md). Regeneration follows [README Training](../../README.md#training); a regenerated bundle reports its own scores rather than inheriting the candidate's recorded metrics.
