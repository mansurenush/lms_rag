from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    
    return Path(__file__).resolve().parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """
    Загружает YAML-конфиг.
    По умолчанию берёт `configs/config.yaml` в корне репозитория.
    """
    root = _repo_root()
    if config_path is None:
        config_path = root / "configs" / "config.yaml"

    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    
    for section in cfg.values():
        if isinstance(section, dict):
            for k, v in list(section.items()):
                if isinstance(v, str):
                    section[k] = os.path.expandvars(os.path.expanduser(v))

    return cfg

