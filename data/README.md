# Dataset acquisition and local use

- **Dataset:** Electric Vehicle Charging Station Usage (July 2011 - Dec 2020).
- **Source:** City of Palo Alto Open Data, infrastructure/utility charging records.
- **Official acquisition location:** https://data.paloalto.gov/dataviews/257812/electric-vehicle-charging-station-usage-july-2011-dec-2020/ . This URL was verified for the public release. The source page links the full file through its Information / Data collected from entry; the tabular export may have a row limit.
- **Expected local filename:** `ChargePoint Data CY20Q4.csv`.
- **Expected training-copy SHA-256:** `e65f5f5d3861cdd6bf3da2de97aabe590d7708e001db1532202eec52081ba82c`.
- **Local placement:** `data/ChargePoint Data CY20Q4.csv` relative to the repository root. CSV files are ignored; never force-add them.

The dataset is **not redistributed** by this repository. Review source terms and access authorization before acquisition/use. The original source contains sensitive fields; the public trainer explicitly reads only station name, historical Energy, duration, port/plug class, and completion time. No raw source records belong in documentation, test fixtures, frontend assets or Git.

With dependencies installed, run from the repository root:

```powershell
Get-FileHash -Algorithm SHA256 'data/ChargePoint Data CY20Q4.csv'
.venv/Scripts/python.exe training/train.py --csv 'data/ChargePoint Data CY20Q4.csv' --output local/retrain/energy_public_v1 --epochs 20 --patience 5 --seed 42
```

The training entrypoint performs preprocessing and evaluation. Choose a new empty output directory named `energy_public_v1`; it refuses overwrites. The initial public commit includes no trained/fitted bundle. This command recreates all eight required `energy_public_v1` files; set `PUBLIC_MODEL_PATH` to the absolute path of its `informer_checkpoint.pth` before starting FastAPI. Setup/source tests do not train a model. `local/` is ignored. Results from a different checksum, package environment or seed must be reported as a new run, not as reproductions of the saved scores.
