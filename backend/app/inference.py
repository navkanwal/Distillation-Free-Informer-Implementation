from pathlib import Path

import torch
from torch import nn

from .preprocessing import PreprocessingArtifacts


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested before a checkpoint is loaded."""


class TargetMismatchError(RuntimeError):
    """The loaded checkpoint is not a verified Energy forecasting model."""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :].to(x.device)


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = (d_model // n_heads) ** -0.5
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)
        batch_size, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        head_dim = self.d_model // self.n_heads
        queries = queries.view(batch_size, query_len, self.n_heads, head_dim).transpose(1, 2)
        keys = keys.view(batch_size, key_len, self.n_heads, head_dim).transpose(1, 2)
        values = values.view(batch_size, key_len, self.n_heads, head_dim).transpose(1, 2)
        attention = torch.softmax(torch.matmul(queries, keys.transpose(-2, -1)) * self.scale, dim=-1)
        output = torch.matmul(attention, values)
        return output.transpose(1, 2).contiguous().view(batch_size, query_len, self.d_model)


class InformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention = self.attention(x, x, x)
        return self.norm(x + self.dropout(self.linear(attention)))


class InformerDecoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention = ProbSparseAttention(d_model, n_heads)
        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attention = self.attention(x, memory, memory)
        return self.norm(x + self.dropout(self.linear(attention)))


class Informer(nn.Module):
    def __init__(self, input_dim: int, d_model: int, n_heads: int, dropout: float, seq_len: int):
        super().__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.encoder = InformerEncoder(d_model, n_heads, dropout)
        self.decoder = InformerDecoder(d_model, n_heads, dropout)
        self.projection = nn.Linear(d_model, 1)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_enc = self.positional_encoding(self.input_embedding(x_enc))
        memory = self.encoder(x_enc)
        output = self.decoder(x_enc, memory)
        return self.projection(output[:, -1, :])


class InferenceService:
    def __init__(self, model_path: Path, device_name: str = "auto"):
        self.model_path = model_path
        self.device = self._select_device(device_name)
        self.model: Informer | None = None
        self.preprocessing: PreprocessingArtifacts | None = None

    @staticmethod
    def _select_device(device_name: str) -> torch.device:
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_name)

    def load(self) -> None:
        self.model = None
        self.preprocessing = None
        self.preprocessing = PreprocessingArtifacts(self.model_path.parent)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"model checkpoint not found: {self.model_path}")
        preprocessing = self.preprocessing
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        verified_energy = preprocessing.metadata.get("schema_version") == 2
        if verified_energy:
            if self.model_path.name != "informer_checkpoint.pth" or checkpoint.get("contract") != preprocessing.metadata:
                raise ValueError("Checkpoint and preprocessing contracts do not match.")
            config = preprocessing.metadata["model_config"]
            if config.get("input_dim") != preprocessing.input_dim or config.get("seq_len") != preprocessing.input_window:
                raise ValueError("The model architecture does not match its input contract.")
        else:
            config = dict(input_dim=preprocessing.input_dim, d_model=64, n_heads=4, dropout=0.1, seq_len=preprocessing.input_window)
        model = Informer(**config).to(self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        preprocessing.verified_energy_bundle = verified_energy
        self.model = model
        self.preprocessing = preprocessing

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def predict(self, rows: list[dict[str, object]]) -> float:
        if self.preprocessing is None:
            raise ModelUnavailableError("model is not loaded")
        model_input = self.preprocessing.to_model_input(rows).to(self.device)
        if not self.preprocessing.energy_prediction_available:
            raise TargetMismatchError("The installed checkpoint does not predict Energy (kWh).")
        if self.model is None:
            raise ModelUnavailableError("model is not loaded")
        with torch.no_grad():
            output = self.model(model_input)
        if tuple(output.shape) != (1, 1) or not torch.isfinite(output).all().item():
            raise RuntimeError("The model returned an invalid prediction.")
        return self.preprocessing.inverse_energy(float(output.item()))
