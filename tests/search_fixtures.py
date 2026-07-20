from __future__ import annotations

import unittest
from importlib import import_module, util
from types import ModuleType


def _optional_search_module(name: str) -> ModuleType:
    if util.find_spec("optuna") is None or util.find_spec("tomli_w") is None:
        raise unittest.SkipTest("install the search dependency group")
    return import_module(name)
