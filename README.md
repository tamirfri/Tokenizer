# Continuous Byte Tokenizer

This repository tests whether a small continuous byte codec can replace a large input
embedding table and produce denser, exactly reversible byte spans.

The experiment intentionally separates tokenizer claims from language-model claims. It
measures embedding-table reconstruction, parameter memory, and bytes per global position. It
does not claim preserved model quality without a later logits evaluation.

## Setup

```bash
uv sync
uv run tokenizer --help
```

## Commands

```bash
uv run tokenizer inspect Qwen/Qwen3-0.6B
uv run tokenizer train Qwen/Qwen3-0.6B --output-dir checkpoints/qwen3
uv run tokenizer segment Qwen/Qwen3-0.6B checkpoints/qwen3/medium.pt "hello"
uv run tokenizer benchmark Qwen/Qwen3-0.6B checkpoints/qwen3/medium.pt
uv run tokenizer run-all openai/gpt-oss-20b --output-dir results/gpt-oss
```

`train` downloads only the tokenizer, configuration, checkpoint index, and the Safetensors
shard containing the input embedding matrix. GPT-OSS still requires several gigabytes of free
disk space for that shard.

## Verification

```bash
uv run ruff format --check .
uv run ruff check .
uv run ty check src tests
uv run pytest
```
