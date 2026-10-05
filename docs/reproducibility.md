# Reproducibility and recorded evidence

The initial public release is source-only. All ten paths in [artifact-decision-files.txt](artifact-decision-files.txt), including the eight-file `energy_public_v1` bundle, are excluded from the first commit. Local review copies remain byte-identical to the verified release candidate. Its metrics are preserved unchanged in [results documentation](evaluation/release-candidate-results.md) and [methodology](methodology.md).

The public trainer remains unchanged in `backend/train_energy_public.py` so its recorded SHA-256 remains verifiable. `training/train.py` delegates to it from the repository root and requires a new empty output directory named `energy_public_v1`. Model computation, preprocessing, splits, selection and training behavior are unchanged. No retraining or metric recalculation was performed during source-only release preparation.

To recreate the bundle:

1. Follow [data acquisition](../data/README.md) and verify the CSV checksum.
2. Install the pinned dependencies using the root [README](../README.md#installation). The verified environment is Windows / Python 3.13.1 / CPU; recorded versions also appear in `docs/verification/runtime-checks.json`.
3. From the root, run `.venv/Scripts/python.exe training/train.py --csv 'data/ChargePoint Data CY20Q4.csv' --output local/retrain/energy_public_v1 --epochs 20 --patience 5 --seed 42`.
4. Set `PUBLIC_MODEL_PATH` to the absolute path of `local/retrain/energy_public_v1/informer_checkpoint.pth`, then start FastAPI as documented.

Training writes weights, three fitted preprocessing files, feature metadata, the training report, an empty-history catalog and a manifest hashing all seven members. No old bundle may be overwritten. The regenerated report records its own history, exact schema, package versions, seed, data checksum and source hashes. Different data, seeds or environments constitute a new run; recorded candidate metrics are not guaranteed or silently replaced.

The frontend requires no fitted catalog file: source supplies only the schema and empty choices, and the API supplies validated categories after bundle loading. API startup fails if artifacts are missing, corrupt or incompatible; it never substitutes an internal/private bundle.

Source tests need neither fitted artifacts nor the CSV. Run the README's `pytest -m "not artifacts"`, frontend tests and build. Optional `pytest -m artifacts` checks require a complete local bundle selected through `PUBLIC_MODEL_PATH`; missing artifacts are skipped. Structural test inputs are explicitly constructed, not real sessions.

The existing runtime/browser/training-data and three-case numerical-equivalence JSON reports are historical verification of the release candidate with artifacts present. Numerical-equivalence cases are integration evidence, not held-out metrics. [Source-only checks](verification/source-only-release-checks.json) separately record this preparation. The supplementary matched-cohort comparison depends on private artifacts and cannot be recreated from the source-only tree; it is not required for the public workflow and is not a controlled ablation.

Linux/macOS/GPU and future upstream changes have not been validated. Requirements are exact observed pins, not a cross-platform hashed wheel lock.
