from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from functools import lru_cache
from importlib.metadata import distribution
from pathlib import Path
from typing import Final, Protocol

_HASH_CHUNK_BYTES: Final = 1024 * 1024
type FileStatIdentity = tuple[str, int, int, int, int]
type DistributionFileIdentity = tuple[str, FileStatIdentity]


class _Digest(Protocol):
    def update(self, data: bytes, /) -> None: ...


def file_stat_identity(path: Path) -> FileStatIdentity:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    return (
        str(resolved),
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_ctime_ns,
        stat.st_ino,
    )


def directory_files(root: Path) -> tuple[Path, ...]:
    files = []
    for directory, _, filenames in os.walk(root):
        for filename in filenames:
            candidate = Path(directory, filename)
            if candidate.is_file():
                files.append(candidate)
    return tuple(sorted(files))


def path_stat_identity(path: Path) -> tuple[FileStatIdentity, ...]:
    resolved = path.resolve(strict=True)
    if resolved.is_file():
        return (file_stat_identity(resolved),)
    return tuple(file_stat_identity(candidate) for candidate in directory_files(resolved))


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(f"artifact path does not exist: {path}")
    root = path.resolve(strict=True)
    return _sha256_directory(root, directory_files(root))


def _update_file_digest(digest: _Digest, path: Path) -> None:
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)


def _sha256_directory(root: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for child in files:
        digest.update(child.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        _update_file_digest(digest, child)
        digest.update(b"\0")
    return digest.hexdigest()


@lru_cache(maxsize=4096)
def _sha256_path_cached(
    resolved: str,
    identity: tuple[FileStatIdentity, ...],
) -> str:
    path = Path(resolved)
    if path.is_file():
        return sha256_file(path)
    return _sha256_directory(
        path,
        (Path(file_identity[0]) for file_identity in identity),
    )


def cached_sha256_path(path: Path) -> str:
    resolved = path.resolve(strict=True)
    return _sha256_path_cached(str(resolved), path_stat_identity(resolved))


@lru_cache(maxsize=32)
def _installed_distribution_identity(
    name: str,
    version: str,
    identity: tuple[DistributionFileIdentity, ...],
) -> dict[str, str]:
    digest = hashlib.sha256()
    for relative, file_identity in identity:
        path = Path(file_identity[0])
        encoded = relative.replace("\\", "/").encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(file_identity[1].to_bytes(8, "big"))
        _update_file_digest(digest, path)
    return {
        "name": name,
        "version": version,
        "content_sha256": digest.hexdigest(),
    }


def installed_distribution_identity(name: str) -> dict[str, str]:
    installed = distribution(name)
    files = installed.files
    if files is None:
        raise FileNotFoundError(f"installed distribution has no file inventory: {name}")
    identity = []
    for relative in sorted(files, key=str):
        path = Path(str(installed.locate_file(relative)))
        if not path.is_file():
            raise FileNotFoundError(f"installed distribution file is missing: {relative}")
        identity.append((str(relative), file_stat_identity(path)))
    return dict(
        _installed_distribution_identity(
            name,
            installed.version,
            tuple(identity),
        ),
    )
