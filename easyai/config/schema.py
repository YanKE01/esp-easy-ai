"""JSON configuration schema for esp-easy-ai."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Backend(str, Enum):
    ESP_DL = "esp-dl"
    TFLITE_MICRO = "tflite-micro"


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class DatasetFormat(str, Enum):
    WINDOWED_PER_CLASS_CSV = "windowed_per_class_csv"
    WINDOWED_REGRESSION_CSV = "windowed_regression_csv"
    CONTINUOUS_PER_CLASS_CSV = "continuous_per_class_csv"


class FeatureLayout(str, Enum):
    INTERLEAVED = "interleaved"
    GROUPED = "grouped"


# ---------------------------------------------------------------------------
# Sub-models: Task
# ---------------------------------------------------------------------------


class TaskConfig(BaseModel):
    input_dim: Literal[1] = 1
    type: TaskType


# ---------------------------------------------------------------------------
# Sub-models: Transform item
# ---------------------------------------------------------------------------

SUPPORTED_TRANSFORM_TYPES = {
    "Normalize",
    "Rescale",
    "Identity",
}


class TransformItem(BaseModel):
    """A single transform operation. `type` is the unified DSL name,
    remaining fields are passed through as kwargs."""

    type: str
    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def check_type(cls, v: str) -> str:
        if v not in SUPPORTED_TRANSFORM_TYPES:
            raise ValueError(
                f"Unknown transform type '{v}'. Supported: {sorted(SUPPORTED_TRANSFORM_TYPES)}"
            )
        return v


class TransformsConfig(BaseModel):
    train: List[TransformItem] = Field(default_factory=list)
    val: List[TransformItem] = Field(default_factory=list)

    @field_validator("train", "val", mode="before")
    @classmethod
    def normalize_none_to_empty_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str) and v.lower() == "none":
            return []
        return v


# ---------------------------------------------------------------------------
# Sub-models: Calibration
# ---------------------------------------------------------------------------


class CalibrationConfig(BaseModel):
    source: Literal["reuse_val", "reuse_train"] = "reuse_val"
    num_samples: int = Field(gt=0, default=256)


# ---------------------------------------------------------------------------
# Sub-models: Target normalization (regression)
# ---------------------------------------------------------------------------


class TargetNormalize(BaseModel):
    """For single-output: mean/std are floats.
    For multi-output: mean/std are lists of same length as output_dim."""

    mean: Union[float, List[float]]
    std: Union[float, List[float]]


# ---------------------------------------------------------------------------
# Sub-models: Dataset (by format)
# ---------------------------------------------------------------------------


class DatasetBase(BaseModel):
    """Common fields across all dataset formats."""

    root: str
    batch_size: int = Field(gt=0, default=64)
    num_workers: int = Field(ge=0, default=4)
    input_shape: List[int]
    transforms: Optional[TransformsConfig] = None
    calibration: Optional[CalibrationConfig] = None

    @field_validator("input_shape")
    @classmethod
    def check_input_shape(cls, v: List[int]) -> List[int]:
        if len(v) != 2:
            raise ValueError(f"input_shape must be [C, L] (1D), got length {len(v)}")
        if any(d <= 0 for d in v):
            raise ValueError("All dimensions in input_shape must be > 0")
        return v


class WindowedPerClassDataset(DatasetBase):
    format: Literal["windowed_per_class_csv"]
    train_dir: str = "train"
    val_dir: str = "test"
    delimiter: Optional[str] = None
    num_features: int = Field(gt=0)
    feature_layout: FeatureLayout = FeatureLayout.INTERLEAVED
    has_header: bool = False
    num_classes: Optional[int] = Field(default=None, gt=0)
    class_names: Optional[List[str]] = None

    @model_validator(mode="after")
    def check_shape_vs_features(self) -> "WindowedPerClassDataset":
        if self.input_shape[0] != self.num_features:
            raise ValueError(
                f"input_shape[0]={self.input_shape[0]} must equal num_features={self.num_features}"
            )
        return self


class WindowedRegressionDataset(DatasetBase):
    format: Literal["windowed_regression_csv"]
    train_dir: str = "train"
    val_dir: str = "test"
    delimiter: Optional[str] = None
    num_features: int = Field(gt=0)
    feature_layout: FeatureLayout = FeatureLayout.INTERLEAVED
    has_header: bool = False
    label_start_col: int
    label_end_col: int
    output_dim: Optional[int] = Field(default=None, gt=0)
    target_normalize: Optional[TargetNormalize] = None

    @model_validator(mode="after")
    def check_shape_vs_features(self) -> "WindowedRegressionDataset":
        if self.input_shape[0] != self.num_features:
            raise ValueError(
                f"input_shape[0]={self.input_shape[0]} must equal num_features={self.num_features}"
            )
        return self

    @model_validator(mode="after")
    def check_label_cols(self) -> "WindowedRegressionDataset":
        s, e = self.label_start_col, self.label_end_col
        # Negative indices are checked at data-load time against actual col count.
        # Here we only validate that if both are non-negative, s <= e.
        if s >= 0 and e >= 0 and s > e:
            raise ValueError(f"label_start_col ({s}) must be <= label_end_col ({e})")
        # If output_dim is given, check consistency when both cols are non-negative.
        if self.output_dim is not None and s >= 0 and e >= 0:
            expected = e - s + 1
            if self.output_dim != expected:
                raise ValueError(
                    f"output_dim={self.output_dim} but label range gives {expected} columns"
                )
        return self


class ContinuousPerClassDataset(DatasetBase):
    format: Literal["continuous_per_class_csv"]
    train_dir: str = "train"
    val_dir: str = "test"
    delimiter: Optional[str] = None
    num_features: int = Field(gt=0)
    has_header: bool = False
    time_column: Optional[Union[int, str]] = None
    feature_columns: Optional[List[str]] = None
    window_size: int = Field(gt=0)
    window_stride: int = Field(gt=0)
    num_classes: Optional[int] = Field(default=None, gt=0)
    class_names: Optional[List[str]] = None

    @model_validator(mode="after")
    def check_shape_vs_window(self) -> "ContinuousPerClassDataset":
        if self.input_shape[0] != self.num_features:
            raise ValueError(
                f"input_shape[0]={self.input_shape[0]} must equal num_features={self.num_features}"
            )
        if self.input_shape[1] != self.window_size:
            raise ValueError(
                f"input_shape[1]={self.input_shape[1]} must equal window_size={self.window_size}"
            )
        return self


DatasetConfig = Union[
    WindowedPerClassDataset,
    WindowedRegressionDataset,
    ContinuousPerClassDataset,
]


# ---------------------------------------------------------------------------
# Sub-models: Model layer
# ---------------------------------------------------------------------------

SUPPORTED_LAYER_TYPES = {
    "Conv1d",
    "Linear",
    "BatchNorm1d",
    "MaxPool1d",
    "AvgPool1d",
    "GlobalAvgPool1d",
    "Flatten",
    "Dropout",
    "ReLU",
    "ReLU6",
    "LeakyReLU",
    "Sigmoid",
    "Tanh",
    "Softmax",
}


class LayerConfig(BaseModel):
    """A single layer in the model. `type` must be one of the unified layer
    names; remaining fields are constructor kwargs."""

    type: str
    model_config = {"extra": "allow"}

    @field_validator("type")
    @classmethod
    def check_type(cls, v: str) -> str:
        if v not in SUPPORTED_LAYER_TYPES:
            raise ValueError(
                f"Unknown layer type '{v}'. Supported: {sorted(SUPPORTED_LAYER_TYPES)}"
            )
        return v


class ModelConfig(BaseModel):
    layers: List[LayerConfig] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Sub-models: Training
# ---------------------------------------------------------------------------


class OptimizerConfig(BaseModel):
    type: str  # Unified name: Adam, SGD
    model_config = {"extra": "allow"}


class LrSchedulerConfig(BaseModel):
    type: str  # Unified name: CosineAnnealingLR, StepLR
    model_config = {"extra": "allow"}


class EarlyStoppingConfig(BaseModel):
    patience: int = Field(gt=0)
    monitor: str = "val_accuracy"


class CheckpointConfig(BaseModel):
    save_best: bool = True
    monitor: str = "val_accuracy"


class TrainingConfig(BaseModel):
    epochs: int = Field(gt=0)
    optimizer: OptimizerConfig
    lr_scheduler: Optional[LrSchedulerConfig] = None
    loss: str  # Unified name: cross_entropy, mse, l1, huber
    metrics: List[str] = Field(default_factory=lambda: ["accuracy"])
    early_stopping: Optional[EarlyStoppingConfig] = None
    checkpoint: Optional[CheckpointConfig] = None
    device: Literal["auto", "cuda", "cpu"] = "auto"


# ---------------------------------------------------------------------------
# Sub-models: Quantization
# ---------------------------------------------------------------------------


class QuantizationConfig(BaseModel):
    enable: bool = True
    method: Literal["ptq"] = "ptq"
    dtype: Literal["int8"] = "int8"
    target: str = Field(default="esp32s3", min_length=1)
    checkpoint: Literal["best", "last"] = "best"


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


class EasyAIConfig(BaseModel):
    backend: Backend
    task: TaskConfig
    project_name: str = Field(min_length=1)
    output_dir: str = Field(min_length=1)
    seed: Optional[int] = None

    dataset: DatasetConfig = Field(discriminator="format")

    model: ModelConfig
    training: TrainingConfig
    quantization: Optional[QuantizationConfig] = None

    # ---- Cross-field validators ----

    @model_validator(mode="after")
    def check_task_vs_dataset_format(self) -> "EasyAIConfig":
        """Ensure dataset format matches the task type."""
        fmt = self.dataset.format
        task_type = self.task.type

        valid_combos = {
            TaskType.CLASSIFICATION: {
                DatasetFormat.WINDOWED_PER_CLASS_CSV,
                DatasetFormat.CONTINUOUS_PER_CLASS_CSV,
            },
            TaskType.REGRESSION: {DatasetFormat.WINDOWED_REGRESSION_CSV},
        }
        allowed = valid_combos[task_type]
        if fmt not in allowed:
            raise ValueError(
                f"Dataset format '{fmt}' is not valid for task type '{task_type}'. "
                f"Expected one of: {[f.value for f in allowed]}"
            )
        return self

    @model_validator(mode="after")
    def check_quantization_target_for_backend(self) -> "EasyAIConfig":
        if self.quantization is None:
            return self
        if self.backend == Backend.ESP_DL:
            if self.quantization.target not in {"esp32s3", "esp32p4"}:
                raise ValueError(
                    "quantization.target must be 'esp32s3' or 'esp32p4' when backend='esp-dl'"
                )
        return self


# ---------------------------------------------------------------------------
# Internal schema-level validation helper
# ---------------------------------------------------------------------------


def _validate_schema(raw: dict) -> EasyAIConfig:
    """Validate a parsed config object against the schema only."""
    return EasyAIConfig.model_validate(raw)


def load_config(path: Union[str, Path]) -> EasyAIConfig:
    """Backward-compatible alias for full file validation."""
    from easyai.config.validation import validate_config_file

    return validate_config_file(path)
