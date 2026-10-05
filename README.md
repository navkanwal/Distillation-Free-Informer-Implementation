# EV Charging Energy Forecasting with Informer

A research-oriented implementation for predicting the **Energy (kWh) of the next completed EV charging session** at a known charging station using its previous 24 completed sessions.

The project combines a PyTorch sequence model with a FastAPI inference service and a React/Vite interface. The public version is intentionally source-only: raw data, fitted preprocessing objects, trained weights, and private/internal artifacts are not committed.

## Associated research

This repository is related to my IEEE research on EV charging load forecasting:

**[IEEE Xplore — Document 11438439](https://ieeexplore.ieee.org/document/11438439)**

The public implementation is a related engineering implementation and should not be treated as an exact code release or numerical reproduction of the IEEE paper.

## What the project does

For one charging station, the model receives the previous **24 completed charging sessions** and predicts the Energy delivered in the **next completed session**.

The public input schema uses operational information such as:

- historical Energy (kWh)
- connection duration
- charging duration
- port type
- plug type
- completion hour
- completion weekday
- charging station

Persistent user, device, and session identifiers are deliberately excluded from the public interface.

```mermaid
flowchart LR
    D[City of Palo Alto EV charging data] --> P[Public-feature preprocessing]
    P --> H[Previous 24 completed sessions]
    H --> M[Informer model]
    M --> E[Next-session Energy prediction]
    E --> A[FastAPI]
    A --> U[React research interface]
```

## Results

The following values are the recorded held-out results for the `energy_public_v1` release candidate.

| Method | MAE (kWh) | RMSE (kWh) |
| --- | ---: | ---: |
| Public Informer | 5.872 | **8.607** |
| Persistence | 8.126 | 11.889 |
| Training-station mean | **5.802** | 8.708 |

The Informer performs substantially better than persistence. The station-mean baseline has slightly better MAE, while the Informer has the best RMSE.

These numbers describe the public release candidate, **not the results reported in the associated IEEE paper**.

## Model

The public network uses:

- 10 input features
- embedding dimension of 64
- sinusoidal positional encoding
- 4 attention heads
- dropout of 0.1
- encoder attention
- decoder cross-attention
- residual connections and normalization
- scalar Energy prediction from the final sequence position

The current public implementation uses dense attention. The historical class name `ProbSparseAttention` is retained in parts of the codebase, but that name should not be interpreted as a claim that the released model is using the original Informer ProbSparse mechanism.

## Dataset

The project uses the City of Palo Alto **Electric Vehicle Charging Station Usage (July 2011 – Dec 2020)** dataset.

The source dataset contains **259,415 rows**.

The CSV is not redistributed in this repository. See [`data/README.md`](data/README.md) for the expected filename, source, placement, and verification information.

Training examples are formed station-by-station:

1. order completed sessions chronologically;
2. take 24 consecutive valid sessions from the same station;
3. use those sessions as the input sequence;
4. predict the Energy of the following completed session.

The nominal split is chronological 70/15/15. The recorded release-candidate window counts are:

- Train: 179,769
- Validation: 38,906
- Test: 38,807

## Repository structure

```text
backend/       FastAPI service, model/preprocessing code and API tests
frontend/      React/Vite research interface
training/      Training entry point
models/        Local model bundle location; generated artifacts are ignored
data/          Dataset download/setup instructions
docs/          Methodology, evaluation, verification and reproducibility notes
```

## Run locally

The project was validated on Windows with:

- Python 3.13.1
- Node.js 22.13.1
- npm 10.9.2
- CPU execution

The commands below assume you are running them from the repository root.

### 1. Create a Python environment

**Command Prompt:**

```bat
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

If you also want to run the backend test suite, install the test requirements instead:

```bat
python -m pip install -r backend\requirements-test.txt
```

### 2. Install the frontend dependencies

```bat
cd frontend
npm ci
cd ..
```

### 3. Prepare the model artifacts

The trained model and fitted preprocessing objects are intentionally not committed to the public repository.

You therefore have two options:

- use a locally generated valid `energy_public_v1` bundle; or
- retrain the public model from the dataset.

To retrain, first place the dataset as described in [`data/README.md`](data/README.md), then run:

```bat
.\.venv\Scripts\python.exe training\train.py --csv "data\ChargePoint Data CY20Q4.csv" --output local\retrain\energy_public_v1 --epochs 20 --patience 5 --seed 42
```

Training creates the model checkpoint, scalers, encoders, metadata, prediction catalog, training report, and manifest required by the API.

The output directory must be new or empty.

### 4. Start FastAPI

If your generated bundle is in `local/retrain/energy_public_v1`, set `PUBLIC_MODEL_PATH` before starting the API.

**PowerShell:**

```powershell
$env:PUBLIC_MODEL_PATH = (Resolve-Path 'local/retrain/energy_public_v1/informer_checkpoint.pth').Path
cd backend
../.venv/Scripts/python.exe -m uvicorn app.public_main:app --reload --host 127.0.0.1 --port 8000
```

If you are using **Command Prompt**, set the variable to the absolute path of the checkpoint:

```bat
set "PUBLIC_MODEL_PATH=C:\path\to\this-repository\local\retrain\energy_public_v1\informer_checkpoint.pth"
cd backend
..\.venv\Scripts\python.exe -m uvicorn app.public_main:app --reload --host 127.0.0.1 --port 8000
```

The API will be available at:

```text
http://127.0.0.1:8000
```

If a complete bundle is already stored under:

```text
models/energy_public_v1/
```

you can omit `PUBLIC_MODEL_PATH`.

The backend intentionally refuses to start if the model bundle is missing, incomplete, corrupted, or incompatible. It does not silently fall back to an older/private model.

### 5. Start the frontend

Open a second terminal from the repository root:

```bat
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Then open:

```text
http://127.0.0.1:5173
```

The frontend sends the previous 24 completed sessions to the FastAPI service and displays the predicted next-session Energy in kWh.

## API

The prediction endpoint expects 24 complete public-schema observations:

```json
{
  "features": [
    "... 24 completed-session objects, oldest first ..."
  ]
}
```

The interface obtains the valid categorical choices from the API contract and calculates the cyclic completion-hour and weekday features.

Important validation rules include:

- Energy and durations must be non-negative;
- charging duration cannot exceed connection duration;
- all 24 rows must belong to the selected station;
- invalid or unknown categorical values are rejected;
- additional undeclared fields are rejected.

A successful prediction returns the next-session Energy estimate in **kWh**.

## Testing

### Backend

From the repository root:

```bat
cd backend
..\.venv\Scripts\python.exe -m pytest tests -m "not artifacts" -q
```

These tests do not require the trained model bundle.

After generating a valid model bundle and setting `PUBLIC_MODEL_PATH`, artifact integration tests can also be run:

```bat
..\.venv\Scripts\python.exe -m pytest tests -m artifacts -q
```

### Frontend

```bat
cd frontend
npm test
npm run build
```

## Privacy and public-release scope

This repository excludes persistent identifiers such as:

- User ID
- MAC address
- driver postal code
- event/session ID
- EVSE ID
- equipment serial number

It also excludes raw data, real recorded histories, historical internal models, fitted preprocessing objects, and trained weights from the public commit.

This is a **data-minimization choice**, not a claim of differential privacy. Operational charging histories may still be sensitive and should only be used when appropriately authorized.

For the detailed field review, see [`docs/feature-audit.md`](docs/feature-audit.md).

## Reproducibility

Generated model bundles include a manifest containing hashes for the required artifacts. Training metadata records the schema, seed, dataset checksum, package versions, and evaluation information needed to inspect a run.

More detail is available in:

- [`docs/methodology.md`](docs/methodology.md)
- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/evaluation/release-candidate-results.md`](docs/evaluation/release-candidate-results.md)

## Limitations

A few limitations are worth keeping explicit:

- the recorded model uses a single training seed;
- evaluation covers known stations rather than unseen-station generalization;
- adjacent 24-session windows overlap;
- the station-mean baseline has slightly better MAE;
- recorded completion timestamps are naive wall-clock values;
- the API does not independently verify the chronology of submitted observations;
- removing identifiers does not guarantee differential privacy;
- the public implementation is related to, but not claimed to be numerically identical to, the associated IEEE paper.
