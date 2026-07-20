from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def source_state(project_root: Path) -> tuple[str, bool, str]:
    commit = subprocess.run(
        ("git", "-C", str(project_root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    difference = subprocess.run(
        ("git", "-C", str(project_root), "diff", "--binary", "HEAD"),
        check=True,
        capture_output=True,
    ).stdout
    untracked_output = subprocess.run(
        ("git", "-C", str(project_root), "ls-files", "--others", "--exclude-standard", "-z"),
        check=True,
        capture_output=True,
    ).stdout
    digest = hashlib.sha256(b"continuous-tokenizer-dirty-state\0")
    digest.update(len(difference).to_bytes(8, "big"))
    digest.update(difference)
    untracked = tuple(path for path in untracked_output.decode().split("\0") if path)
    for relative in sorted(untracked):
        encoded_path = relative.encode()
        contents = (project_root / relative).read_bytes()
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return commit, bool(difference or untracked), digest.hexdigest()


def find_project_root(path: Path) -> Path:
    start = path.resolve()
    directory = start if start.is_dir() else start.parent
    for candidate in (directory, *directory.parents):
        if (candidate / ".git").exists() and (candidate / "uv.lock").is_file():
            return candidate
    raise FileNotFoundError(f"could not find project root above {path}")
