from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast, final


@final
class RunDirectory:
    def __init__(self, root: Path, *, resume: bool = False) -> None:
        if resume:
            if not root.is_dir():
                raise FileNotFoundError(f"cannot resume missing run directory: {root}")
        else:
            if root.exists():
                raise FileExistsError(f"run directory already exists: {root}")
            root.mkdir(parents=True)
        self.root = root

    def path(self, relative: str | Path) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, value: object) -> Path:
        path = self.path(relative)
        write_json_atomic(path, value)
        return path

    def write_text(self, relative: str | Path, value: str) -> Path:
        path = self.path(relative)
        write_text_atomic(path, value)
        return path


def load_json_object(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object in {path}")
    return cast(Mapping[str, Any], value)


def json_compatible_object(value: Mapping[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(value)))


def write_json_atomic(path: Path, value: object) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    Path(temporary).replace(path)
