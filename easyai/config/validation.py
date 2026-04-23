"""Unified project-level validation entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from easyai.config.dataset_check import validate_dataset_files
from easyai.config.simple_mode import complete_small_white_user_config
from easyai.config.schema import EasyAIConfig, _validate_schema
from easyai.model import validate_model_definition


def validate_config(raw: dict) -> EasyAIConfig:
    """Validate a parsed config object against the schema only."""
    return _validate_schema(complete_small_white_user_config(raw))


def validate_config_file(path: Union[str, Path]) -> EasyAIConfig:
    """Validate a config file end-to-end.

    This is the user-facing entrypoint used by the CLI:
    1. Read and parse JSON
    2. Validate config schema and cross-field rules
    3. Validate referenced dataset files
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON at line {e.lineno}, column {e.colno}: {e.msg}"
        ) from e

    cfg = validate_config(raw)

    errors = validate_dataset_files(cfg, str(path))
    if errors:
        raise ValueError(
            "Dataset validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    validate_model_definition(cfg)

    return cfg


def load_config(path: Union[str, Path]) -> EasyAIConfig:
    """Backward-compatible alias for full file validation."""
    return validate_config_file(path)
