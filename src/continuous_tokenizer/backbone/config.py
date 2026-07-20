from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, AutoModelForImageTextToText


def _config_value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def text_config(config: Any) -> Any:
    return _config_value(config, "text_config", config)


def tie_word_embeddings(config: Any) -> bool:
    return bool(_config_value(config, "tie_word_embeddings", False))


def input_table_is_removable(config: Any) -> bool:
    return bool(_config_value(config, "removable_input_table", not tie_word_embeddings(config)))


def model_loader(config: Any) -> Any:
    return AutoModelForImageTextToText if _config_value(config, "text_config") is not None else AutoModelForCausalLM


def build_model_from_config(config: Any) -> Any:
    return model_loader(config).from_config(config)
