from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path

_PACKAGE = "continuous_tokenizer"
_PACKAGE_PREFIX = f"{_PACKAGE}."

# These are the target one-way dependency boundaries.
_FORBIDDEN_IMPORTS = {
    "backbone": frozenset({"codec", "input", "output"}),
    "codec": frozenset({"input", "output"}),
    "input": frozenset({"output"}),
    "output": frozenset({"input"}),
}
_FOUNDATION_ALLOWED_IMPORTS = {
    "contracts": frozenset({"contracts"}),
    "artifacts": frozenset({"artifacts", "contracts"}),
    "runtime": frozenset({"artifacts", "contracts", "runtime"}),
    "data": frozenset({"contracts", "data"}),
    "training": frozenset({"contracts", "runtime", "training"}),
}
_REPORTING_ALLOWED_IMPORTS = frozenset({"artifacts", "contracts", "reporting"})


def _package_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = (node.module,)
        else:
            continue
        for name in names:
            if name.startswith(_PACKAGE_PREFIX):
                imports.add(name.removeprefix(_PACKAGE_PREFIX))
    return imports


class DependencyBoundaryTests(unittest.TestCase):
    def test_foundation_packages_only_import_inward(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / _PACKAGE
        violations: set[tuple[str, str]] = set()
        for package, allowed in _FOUNDATION_ALLOWED_IMPORTS.items():
            for path in (source_root / package).rglob("*.py"):
                importer = ".".join(path.relative_to(source_root).with_suffix("").parts)
                for imported in _package_imports(path):
                    if imported.partition(".")[0] not in allowed:
                        violations.add((importer, imported))

        self.assertEqual(violations, set())

    def test_superseded_modules_are_removed(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / _PACKAGE
        removed = (
            "backbone/attention.py",
            "backbone/deployment.py",
            "backbone/runtime.py",
            "codec/cache.py",
            "codec/checkpoint.py",
            "codec/shared.py",
            "experiment/artifacts.py",
            "experiment/corpus.py",
            "experiment/optimization.py",
            "experiment/progress.py",
            "experiment/spec.py",
            "experiment/toml_parse.py",
            "input/benchmark.py",
            "input/distillation.py",
            "input/prefill_benchmark.py",
            "input/reconstruction_training.py",
            "input/tokenizer_benchmark.py",
            "input/training.py",
            "input/training_cache.py",
            "input/training_selection.py",
            "input/vocabulary_training.py",
            "output/trajectory.py",
            "input/runner.py",
            "output/runner.py",
        )
        self.assertEqual(
            [relative for relative in removed if (source_root / relative).exists()],
            [],
        )
        self.assertFalse((source_root / "experiment").exists())

    def test_restructured_domain_packages_have_no_reexports(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / _PACKAGE
        initializers = (
            source_root / "campaigns/__init__.py",
            source_root / "commands/__init__.py",
            source_root / "input/benchmark/__init__.py",
            source_root / "input/training/__init__.py",
            source_root / "reporting/__init__.py",
            source_root / "search/__init__.py",
        )

        self.assertTrue(all(path.is_file() for path in initializers))
        self.assertEqual(
            {str(path.relative_to(source_root)): _package_imports(path) for path in initializers},
            {
                "campaigns/__init__.py": set(),
                "commands/__init__.py": set(),
                "input/benchmark/__init__.py": set(),
                "input/training/__init__.py": set(),
                "reporting/__init__.py": set(),
                "search/__init__.py": set(),
            },
        )

    def test_contracts_are_stdlib_only(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / _PACKAGE
        violations: set[tuple[str, str]] = set()
        for path in (source_root / "contracts").rglob("*.py"):
            importer = ".".join(path.relative_to(source_root).with_suffix("").parts)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = (alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module is not None:
                    names = (node.module,)
                else:
                    continue
                for name in names:
                    root = name.partition(".")[0]
                    if root not in sys.stdlib_module_names and not name.startswith(f"{_PACKAGE}.contracts"):
                        violations.add((importer, name))
        self.assertEqual(violations, set())

    def test_reporting_has_no_runtime_orchestration_dependencies(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / _PACKAGE
        violations: set[tuple[str, str]] = set()
        for path in (source_root / "reporting").rglob("*.py"):
            importer = ".".join(path.relative_to(source_root).with_suffix("").parts)
            for imported in _package_imports(path):
                if imported.partition(".")[0] not in _REPORTING_ALLOWED_IMPORTS:
                    violations.add((importer, imported))
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            if any(
                (isinstance(node, ast.Import) and any(alias.name == "torch" or alias.name.startswith("torch.") for alias in node.names))
                or (isinstance(node, ast.ImportFrom) and node.module is not None and (node.module == "torch" or node.module.startswith("torch.")))
                for node in ast.walk(tree)
            ):
                violations.add((importer, "torch"))
        self.assertEqual(violations, set())

    def test_no_new_imports_cross_target_package_boundaries(self) -> None:
        source_root = Path(__file__).parents[1] / "src" / _PACKAGE
        violations: set[tuple[str, str]] = set()
        for path in source_root.rglob("*.py"):
            importer = ".".join(path.relative_to(source_root).with_suffix("").parts)
            importer_package = importer.partition(".")[0]
            forbidden = _FORBIDDEN_IMPORTS.get(importer_package, ())
            for imported in _package_imports(path):
                if imported.partition(".")[0] in forbidden:
                    violations.add((importer, imported))

        self.assertEqual(violations, set())


if __name__ == "__main__":
    unittest.main()
