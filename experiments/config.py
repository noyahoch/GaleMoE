"""
Config loader.

Loads base.yaml first, then deep-merges the condition-specific yaml on top.
CLI key=value overrides can be applied afterwards.

Usage:
    cfg = load_config("configs/gated_svd.yaml")
    cfg = load_config("configs/gated_svd.yaml", overrides={"training.lr": 2e-4})
    print(cfg.training.lr)
"""

import os
from typing import Any

import yaml


class Namespace:
    """Recursive dot-access wrapper around a nested dict."""

    def __init__(self, d: dict):
        for k, v in d.items():
            setattr(self, k, Namespace(v) if isinstance(v, dict) else v)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, Namespace) else v
        return out

    def __repr__(self):
        return f"Namespace({self.__dict__})"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(condition_path: str, overrides: dict[str, Any] | None = None) -> Namespace:
    """
    Load base.yaml and merge condition-specific yaml on top.

    condition_path: path to a condition yaml (e.g. "configs/gated_svd.yaml")
    overrides: flat dict with dot-notation keys, e.g. {"training.lr": 2e-4}
    """
    base_path = os.path.join(os.path.dirname(condition_path), "base.yaml")

    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    with open(condition_path) as f:
        condition_cfg = yaml.safe_load(f)

    cfg = _deep_merge(cfg, condition_cfg)

    # apply CLI overrides: "training.lr" -> cfg["training"]["lr"]
    if overrides:
        for key, val in overrides.items():
            parts = key.split(".")
            node = cfg
            for part in parts[:-1]:
                node = node[part]
            node[parts[-1]] = val

    return Namespace(cfg)
