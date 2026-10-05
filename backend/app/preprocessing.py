from pathlib import Path
from typing import Any
import json
import math
import hashlib

import joblib
import numpy as np
import pandas as pd
import torch


ENERGY_TARGET = "Energy (kWh)"
TARGET_UNAVAILABLE_MESSAGE = (
    "Energy prediction is unavailable: the installed model was trained to predict "
    "Model Number, not Energy (kWh). A verified Energy model and matching "
    "preprocessing artifacts are required."
)


class SequenceValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        self.errors = errors
        super().__init__("Correct the highlighted historical input values.")


class PreprocessingArtifacts:
    def __init__(self, models_dir: Path):
        with (models_dir / "feature_metadata.json").open("r", encoding="utf-8") as source:
            self.metadata = json.load(source)
        self.verified_energy_bundle = False
        self.target_scaler = None
        if self.metadata.get("schema_version") == 2:
            manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
            required = {"informer_checkpoint.pth", "feature_metadata.json", "minmax_scaler.joblib", "target_scaler.joblib", "label_encoders.joblib", "training_report.json", "prediction_catalog.json"}
            if set(manifest) != required:
                raise ValueError("The model bundle manifest is incomplete.")
            for name in required:
                if hashlib.sha256((models_dir / name).read_bytes()).hexdigest() != manifest[name]:
                    raise ValueError("The model bundle failed its integrity check.")
            self.target_scaler = joblib.load(models_dir / "target_scaler.joblib")
            if list(self.target_scaler.feature_names_in_) != [ENERGY_TARGET] or self.target_scaler.n_features_in_ != 1:
                raise ValueError("The target scaler does not match Energy.")
            report = json.loads((models_dir / "training_report.json").read_text(encoding="utf-8"))
            if (report.get("target_column") != ENERGY_TARGET or report.get("unit") != "kWh"
                    or report.get("completed_epochs", 0) < 1
                    or report.get("data_sha256") != self.metadata.get("training_data_sha256")
                    or self.metadata.get("target_unit") != "kWh"
                    or self.metadata.get("prediction_horizon") != 1):
                raise ValueError("The Energy training contract is invalid.")
        self.scaler = joblib.load(models_dir / "minmax_scaler.joblib")
        self.label_encoders = joblib.load(models_dir / "label_encoders.joblib")
        self.feature_columns = self.metadata["feature_columns"]
        self.categorical_columns = [column for column in self.feature_columns if column in self.label_encoders]
        self.numeric_columns = [column for column in self.feature_columns if column not in self.label_encoders]
        self.input_window = int(self.metadata["input_window"])
        self.input_dim = int(self.metadata["input_dim"])
        self.target_column = self.metadata["target_column"]
        if (self.input_window, self.input_dim) != (24, 28):
            raise ValueError("The installed model input dimensions are not supported.")
        if len(self.feature_columns) != self.input_dim or len(set(self.feature_columns)) != self.input_dim:
            raise ValueError("The installed feature ordering is invalid.")
        if list(self.scaler.feature_names_in_) != self.feature_columns:
            raise ValueError("The installed scaler feature ordering does not match the model.")
        self.category_options = {
            column: [str(value) for value in encoder.classes_ if isinstance(value, (str, np.str_))]
            for column, encoder in self.label_encoders.items()
        }
        self.category_sets = {column: set(values) for column, values in self.category_options.items()}

    @property
    def energy_prediction_available(self) -> bool:
        return self.verified_energy_bundle and self.target_column == ENERGY_TARGET and ENERGY_TARGET in self.numeric_columns

    @staticmethod
    def issue(row: int | None, feature: str | None, code: str, message: str) -> dict[str, Any]:
        return {"row": row, "feature": feature, "code": code, "message": message}

    def validate_rows(self, rows: list[dict[str, Any]]) -> None:
        errors = []
        if len(rows) != self.input_window:
            errors.append(self.issue(None, None, "sequence_length", "Exactly 24 historical observations are required."))
        expected = set(self.feature_columns)
        station = rows[0].get("Station Name") if rows else None
        for index, row in enumerate(rows):
            missing = expected - set(row)
            for column in self.feature_columns:
                value = row.get(column)
                if column in missing or value is None or (isinstance(value, str) and value == ""):
                    errors.append(self.issue(index, column, "missing", f"{column} is required."))
                    continue
                if column in self.category_sets:
                    if not isinstance(value, str) or value not in self.category_sets[column]:
                        errors.append(self.issue(index, column, "category", f"Unsupported value for {column}."))
                else:
                    try:
                        finite_number = not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)
                    except (OverflowError, ValueError):
                        finite_number = False
                    if not finite_number:
                        errors.append(self.issue(index, column, "number", f"{column} must be a finite number."))
            if set(row) - expected:
                errors.append(self.issue(index, None, "structure", "This observation contains unsupported fields."))
            row_station = row.get("Station Name")
            if isinstance(station, str) and station in self.category_sets["Station Name"] and isinstance(row_station, str) and row_station in self.category_sets["Station Name"] and row_station != station:
                errors.append(self.issue(index, "Station Name", "station", "All 24 observations must belong to the selected station."))
        if errors:
            raise SequenceValidationError(errors)

    def to_model_input(self, rows: list[dict[str, Any]]) -> torch.Tensor:
        self.validate_rows(rows)
        frame = pd.DataFrame(rows, columns=self.feature_columns)
        for column in self.categorical_columns:
            frame[column] = self.label_encoders[column].transform(frame[column])
        with np.errstate(over="ignore", invalid="ignore"):
            values = np.asarray(self.scaler.transform(frame[self.feature_columns]), dtype=np.float32)
        if values.shape != (self.input_window, self.input_dim):
            raise ValueError("The preprocessed sequence has an invalid shape.")
        if not np.isfinite(values).all():
            errors = [
                self.issue(int(row), self.feature_columns[int(column)], "number", "This number is outside the supported processing range.")
                for row, column in np.argwhere(~np.isfinite(values))
            ]
            raise SequenceValidationError(errors)
        return torch.from_numpy(values).unsqueeze(0)

    def inverse_energy(self, scaled_value: float) -> float:
        """Use only the dedicated scaler from a verified Energy bundle."""
        if not self.energy_prediction_available:
            raise ValueError("The installed target cannot be converted to Energy (kWh).")
        scale = float(self.target_scaler.scale_[0])
        offset = float(self.target_scaler.min_[0])
        if not math.isfinite(scaled_value) or not math.isfinite(scale) or scale == 0:
            raise ValueError("The prediction cannot be inverse-transformed.")
        energy = (scaled_value - offset) / scale
        if not math.isfinite(energy):
            raise ValueError("The inverse-transformed prediction must be finite.")
        return energy

    def contract(self) -> dict[str, Any]:
        return {
            "inputWindow": self.input_window,
            "inputDim": self.input_dim,
            "featureColumns": self.feature_columns,
            "numericColumns": self.numeric_columns,
            "categoricalColumns": self.categorical_columns,
            "categoryOptions": self.category_options,
            "stationOptions": self.category_options["Station Name"],
            "fixedValues": {column: values[0] for column, values in self.category_options.items() if len(values) == 1},
            "targetColumn": self.target_column,
            "predictionHorizon": int(self.metadata["prediction_horizon"]),
            "horizonLabel": "Next observation",
            "energyPredictionAvailable": self.energy_prediction_available,
            "availabilityMessage": None if self.energy_prediction_available else TARGET_UNAVAILABLE_MESSAGE,
        }
