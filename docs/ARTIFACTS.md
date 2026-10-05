# Source-only first release: artifact policy

The initial public Git commit excludes **all ten files** in [artifact-decision-files.txt](artifact-decision-files.txt). Trained weights, fitted preprocessing and these associated data-derived files are not redistributed. Each path has an exact root-anchored `.gitignore` entry. Unrelated JSON, metadata, checkpoints and model files are not globally ignored.

| Excluded path | Role |
| --- | --- |
| `models/energy_public_v1/informer_checkpoint.pth` | Trained neural weights and embedded public schema |
| `models/energy_public_v1/minmax_scaler.joblib` | Fitted feature scaling parameters |
| `models/energy_public_v1/target_scaler.joblib` | Fitted Energy scaling parameters |
| `models/energy_public_v1/label_encoders.joblib` | Fitted station/port/plug vocabularies |
| `models/energy_public_v1/feature_metadata.json` | Generated feature contract and dataset checksum |
| `models/energy_public_v1/training_report.json` | Generated training scores, settings and provenance |
| `models/energy_public_v1/prediction_catalog.json` | Generated empty-history catalog and dataset checksum |
| `models/energy_public_v1/manifest.json` | SHA-256 integrity mapping for seven bundle members |
| `frontend/src/data/inputCatalog.json` | Exported fitted categorical choices |
| `docs/evaluation/matched-cohort-comparison.json` | Generated aggregate comparison report |

Local review copies are retained unchanged and ignored. Never force-add them to the first commit. The dataset and IEEE publication are not redistributed.

Follow [README Training](../README.md#training) to regenerate all eight `energy_public_v1` bundle files from the separately acquired CSV into a new empty local directory. Set `PUBLIC_MODEL_PATH` to that checkpoint. Every manifest member is required: API startup fails with a regeneration explanation for a missing, corrupt or incompatible bundle. It never selects internal/private artifacts or another profile as a fallback.

The frontend builds from a source-defined schema with empty category choices, then gets supported categories from the validated public API contract. It does not require or regenerate the excluded exported frontend catalog. The private-dependent matched-cohort report is optional historical evidence, not a prerequisite for training or inference. Its recorded aggregate metrics and the main release-candidate evaluation remain in [results documentation](evaluation/release-candidate-results.md); none were modified or recomputed.
