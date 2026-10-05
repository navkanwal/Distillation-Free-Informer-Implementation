from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Exactly 24 complete observations using original raw feature values."""

    model_config = ConfigDict(extra="forbid", strict=True)
    features: Annotated[list[dict[str, Any]], Field(min_length=24, max_length=24)]


class PredictionResponse(BaseModel):
    predicted_energy_kwh: float
    target_column: Literal["Energy (kWh)"] = "Energy (kWh)"
    unit: Literal["kWh"] = "kWh"
    prediction_horizon: Literal[1] = 1
    forecast_horizon: Literal["Next observation"] = "Next observation"
