from dataclasses import fields
from importlib import resources
from pathlib import Path
from typing import Mapping, Union

import yaml

from pcc.dataset.dataset import PCCDataset
from pcc.params import PCCConfig


def _config_from_mapping(data: Mapping) -> PCCConfig:
    if "pcc" in data and isinstance(data["pcc"], Mapping):
        data = data["pcc"]

    allowed = {field.name for field in fields(PCCConfig)}
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"Unknown PCC config field(s): {', '.join(unknown)}")

    return PCCConfig(**dict(data))


def load_pcc_config(path: Union[str, Path]) -> PCCConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _config_from_mapping(data)


def load_default_config() -> PCCConfig:
    if hasattr(resources, "files"):
        default_yaml = (
            resources.files("pcc.configs")
            .joinpath("default.yaml")
            .read_text(encoding="utf-8")
        )
    else:
        default_yaml = resources.read_text("pcc.configs", "default.yaml")
    return _config_from_mapping(yaml.safe_load(default_yaml) or {})


def dataset_from_yaml(path: Union[str, Path], size: int, seed: int = 0) -> PCCDataset:
    return PCCDataset(size=size, cfg=load_pcc_config(path), seed=seed)
