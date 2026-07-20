from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import MISSING, fields
from typing import Any, cast

type Scalar = int | float | str


def table(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a TOML table")
    return cast(Mapping[str, Any], value)


def strict_fields(values: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {', '.join(unknown)}")


def exact_fields(
    values: Mapping[str, Any],
    expected: set[str],
    name: str,
    *,
    incomplete_message: str | None = None,
) -> None:
    strict_fields(values, expected, name)
    missing = sorted(expected - set(values))
    if missing:
        message = incomplete_message or f"{name} fields are incomplete: {', '.join(missing)}"
        raise ValueError(message)


def non_empty_string(values: Mapping[str, Any], key: str, table_name: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{table_name}.{key} must be a non-empty string")
    return value


def is_lowercase_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def sha256_string(values: Mapping[str, Any], key: str, table_name: str) -> str:
    value = non_empty_string(values, key, table_name)
    if not is_lowercase_sha256(value):
        raise ValueError(f"{table_name}.{key} must be a lowercase SHA-256 digest")
    return value


def scalar_mapping(
    values: Mapping[str, Any],
    key: str,
    table_name: str,
) -> dict[str, Scalar]:
    mapping = table(values.get(key), f"{table_name}.{key}")
    if not mapping or any(
        not isinstance(name, str) or not name or isinstance(value, bool) or not isinstance(value, int | float | str) for name, value in mapping.items()
    ):
        raise ValueError(f"{table_name}.{key} contains an invalid value")
    return {name: cast(Scalar, value) for name, value in mapping.items()}


def mapping_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_defaults[T](cls: type[T], value: object, table_name: str) -> T:
    values = table(value, table_name)
    definitions = {definition.name: definition for definition in fields(cast(Any, cls))}
    strict_fields(values, set(definitions), table_name)
    parsed: dict[str, Any] = {}
    for key, item in values.items():
        default = definitions[key].default
        if default is MISSING:
            raise TypeError(f"{cls.__name__}.{key} must define a default")
        if not _matches_default_type(item, default):
            raise ValueError(f"{table_name}.{key} has the wrong type")
        parsed[key] = item
    return cls(**parsed)


def _matches_default_type(value: object, default: object) -> bool:
    if isinstance(default, bool):
        return isinstance(value, bool)
    if isinstance(default, int):
        return isinstance(value, int) and not isinstance(value, bool)
    if isinstance(default, float):
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, type(default))


def positive(value: float, name: str, *, allow_zero: bool = False) -> None:
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")


def positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def non_negative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def float_list(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    if any(not isinstance(item, int | float) or isinstance(item, bool) for item in value):
        raise ValueError(f"{name} must contain only numbers")
    values = tuple(float(cast(int | float, item)) for item in value)
    if any(item < 0 for item in values) or len(values) != len(set(values)):
        raise ValueError(f"{name} must contain unique non-negative numbers")
    return values
