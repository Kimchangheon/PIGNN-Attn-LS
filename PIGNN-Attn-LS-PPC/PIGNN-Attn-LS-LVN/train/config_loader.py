from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Mapping, Optional

import yaml


def load_yaml_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping/dict, got: {type(data).__name__}")
    return data


def get(d: Mapping[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, Mapping) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_update(base: Dict[str, Any], other: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively update a dict; returns base."""
    for k, v in other.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            deep_update(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def as_path_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return value
    raise TypeError("parquet_paths must be a string or list of strings")


def env_expand_paths(paths: list[str]) -> list[str]:
    return [os.path.expandvars(os.path.expanduser(p)) for p in paths]
