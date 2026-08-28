"""Safe YAML configuration loading with deterministic deep merges."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_PRE_STAGE3_CONFIG_ARCHIVE = Path("history") / "pre_stage3_2026-06-21" / "configs"


def resolve_config_path(path: str | Path) -> Path:
    """Resolve active configs first, then the fixed read-only pre-Stage-3 archive."""

    requested = Path(path)
    if requested.exists():
        return requested
    for parent in requested.parents:
        if parent.name != "configs":
            continue
        relative = requested.relative_to(parent)
        archived = parent.parent / _PRE_STAGE3_CONFIG_ARCHIVE / relative
        if archived.exists():
            return archived
        break
    return requested


def _deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _expand_environment(value: Any) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if name not in os.environ:
                raise KeyError(f"Environment variable {name!r} is not defined.")
            return os.environ[name]

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def _parse_override(override: str) -> tuple[list[str], Any]:
    if "=" not in override:
        raise ValueError(f"Override must use dotted.path=value syntax: {override!r}")
    dotted_path, raw_value = override.split("=", 1)
    keys = [part for part in dotted_path.split(".") if part]
    if not keys:
        raise ValueError(f"Override path is empty: {override!r}")
    return keys, yaml.safe_load(raw_value)


def _set_nested(config: dict[str, Any], keys: list[str], value: Any) -> None:
    cursor = config
    for key in keys[:-1]:
        existing = cursor.get(key)
        if existing is None:
            cursor[key] = {}
        elif not isinstance(existing, dict):
            raise TypeError(f"Cannot set child key under non-mapping value at {key!r}.")
        cursor = cursor[key]
    cursor[keys[-1]] = value


def load_config(
    paths: str | Path | Iterable[str | Path],
    overrides: Iterable[str] = (),
) -> dict[str, Any]:
    """Load one or more YAML files, deep-merge them, and apply dotted overrides."""

    if isinstance(paths, (str, Path)):
        paths = [paths]

    config: dict[str, Any] = {}
    for raw_path in paths:
        path = resolve_config_path(raw_path)
        with path.open("r", encoding="utf-8") as stream:
            loaded = yaml.safe_load(stream) or {}
        if not isinstance(loaded, Mapping):
            raise TypeError(f"Top-level configuration must be a mapping: {path}")
        config = _deep_merge(config, loaded)

    for override in overrides:
        keys, value = _parse_override(override)
        _set_nested(config, keys, value)

    return _expand_environment(config)


def get_config_value(config: Mapping[str, Any], dotted_path: str) -> Any:
    """Return a required value selected by a dotted path."""

    value: Any = config
    for key in dotted_path.split("."):
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"Missing configuration value: {dotted_path}")
        value = value[key]
    return value
