from __future__ import annotations

import hashlib
import json
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, cast, final

import torch

from continuous_tokenizer.backbone.assets import ModelAssets
from continuous_tokenizer.codec.batches import build_byte_batch
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.contracts.input import InputGateSpec
from continuous_tokenizer.contracts.input_study import (
    INPUT_SELECTION_CANDIDATES,
    INPUT_SELECTION_RULE,
)
from continuous_tokenizer.contracts.parsing import is_lowercase_sha256, mapping_fingerprint
from continuous_tokenizer.contracts.prospective_subset import (
    PROSPECTIVE_INPUT_SUBSET_ALGORITHM,
)

VOCABULARY_SUBSET_ALGORITHM: Final = PROSPECTIVE_INPUT_SUBSET_ALGORITHM
BINARY_SPAN_ALGORITHM: Final = "python_random_getrandbits"


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def spans_sha256(spans: Sequence[bytes]) -> str:
    digest = hashlib.sha256()
    for span in spans:
        digest.update(len(span).to_bytes(4, "big"))
        digest.update(span)
    return digest.hexdigest()


@final
@dataclass(frozen=True, slots=True)
class VocabularySubset:
    requested_rows: int
    token_ids: tuple[int, ...]
    sha256: str
    algorithm: str = VOCABULARY_SUBSET_ALGORITHM

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "token_ids": list(self.token_ids),
        }


@final
@dataclass(frozen=True, slots=True)
class RegisteredVocabularySubsetRequest:
    requested_rows: int
    subset_seed: int
    subset_sha256: str
    algorithm: str
    work_units: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.requested_rows, int) or isinstance(self.requested_rows, bool) or self.requested_rows < 1:
            raise ValueError("prospective vocabulary subset rows must be positive")
        if not isinstance(self.subset_seed, int) or isinstance(self.subset_seed, bool) or not -(1 << 63) <= self.subset_seed < 1 << 63:
            raise ValueError("prospective vocabulary subset seed must be a signed 64-bit integer")
        if not is_lowercase_sha256(self.subset_sha256):
            raise ValueError("prospective vocabulary subset hash must be a lowercase SHA-256 digest")
        if self.algorithm != VOCABULARY_SUBSET_ALGORITHM:
            raise ValueError("prospective vocabulary subset algorithm is unsupported")
        if (
            not self.work_units
            or tuple(sorted(self.work_units)) != self.work_units
            or len({name for name, _ in self.work_units}) != len(self.work_units)
            or any(
                not isinstance(name, str) or not name or not isinstance(value, int) or isinstance(value, bool) or value < 0 for name, value in self.work_units
            )
        ):
            raise ValueError("prospective work units must be sorted unique non-negative integers")

    def execution_fingerprint(self, experiment_fingerprint: str) -> str:
        return mapping_fingerprint(
            {
                "experiment_fingerprint": experiment_fingerprint,
                "prospective_input_subset": {
                    "requested_rows": self.requested_rows,
                    "subset_seed": self.subset_seed,
                    "subset_sha256": self.subset_sha256,
                    "algorithm": self.algorithm,
                    "work_units": dict(self.work_units),
                },
            },
        )


@final
@dataclass(frozen=True, slots=True)
class CandidateLengthRequest:
    assets: ModelAssets
    validation_data: bytes
    candidate_lengths: Sequence[int]
    binary_samples_per_length: int
    seed: int
    batch_size: int
    vocabulary_token_ids: Sequence[int] | None = None


def registered_vocabulary_subset(
    assets: ModelAssets,
    requested_rows: int,
    seed: int,
) -> VocabularySubset:
    vocabulary = assets.vocabulary
    if requested_rows == 0:
        token_ids = vocabulary.compatibility_ids
        algorithm = "complete_compatibility_vocabulary"
    else:
        eligible = tuple(token_id for token_id in vocabulary.compatibility_ids if len(vocabulary.bytes_for(token_id)) > 1)
        if requested_rows > len(eligible):
            raise ValueError("registered vocabulary subset exceeds available non-atomic rows")
        ranked_by_length: dict[int, list[tuple[bytes, int]]] = {}
        seed_bytes = seed.to_bytes(8, "big", signed=True)
        for token_id in eligible:
            payload = vocabulary.bytes_for(token_id)
            rank = hashlib.sha256(
                seed_bytes + token_id.to_bytes(8, "big") + payload,
            ).digest()
            ranked_by_length.setdefault(len(payload), []).append((rank, token_id))
        by_length = {length: deque(sorted(rows)) for length, rows in ranked_by_length.items()}
        lengths = sorted(
            by_length,
            key=lambda length: hashlib.sha256(
                seed_bytes + length.to_bytes(8, "big"),
            ).digest(),
        )
        selected = []
        while len(selected) < requested_rows:
            for length in lengths:
                rows = by_length[length]
                if rows:
                    _, token_id = rows.popleft()
                    selected.append(token_id)
                    if len(selected) == requested_rows:
                        break
        token_ids = tuple(sorted(selected))
        algorithm = VOCABULARY_SUBSET_ALGORITHM
    content = [
        {
            "token_id": token_id,
            "bytes": vocabulary.bytes_for(token_id).hex(),
        }
        for token_id in token_ids
    ]
    return VocabularySubset(
        requested_rows=requested_rows,
        token_ids=token_ids,
        sha256=_sha256_json(content),
        algorithm=algorithm,
    )


def deterministic_binary_spans(
    lengths: Sequence[int],
    samples_per_length: int,
    seed: int,
) -> dict[int, tuple[bytes, ...]]:
    if samples_per_length < 1:
        raise ValueError("binary samples per length must be positive")
    result: dict[int, tuple[bytes, ...]] = {}
    for length in lengths:
        randomizer = random.Random((seed << 8) ^ length)
        result[length] = tuple(bytes(randomizer.getrandbits(8) for _ in range(length)) for _ in range(samples_per_length))
    return result


def exact_corpus_spans(
    data: bytes,
    lengths: Sequence[int],
    samples_per_length: int,
) -> dict[int, tuple[bytes, ...]]:
    result: dict[int, tuple[bytes, ...]] = {}
    for length in lengths:
        available = len(data) // length
        selected = min(samples_per_length, available)
        result[length] = tuple(data[index * length : (index + 1) * length] for index in range(selected))
    return result


def exact_vocabulary_spans(
    assets: ModelAssets,
    lengths: Sequence[int],
    token_ids: Sequence[int] | None = None,
) -> dict[int, tuple[bytes, ...]]:
    selected = assets.vocabulary.compatibility_ids if token_ids is None else token_ids
    by_length = {length: [] for length in lengths}
    for token_id in selected:
        payload = assets.vocabulary.bytes_for(token_id)
        if len(payload) in by_length:
            by_length[len(payload)].append(payload)
    return {length: tuple(values) for length, values in by_length.items()}


@torch.inference_mode()
def exact_length_metrics(
    codec: InputByteCodec,
    spans_by_length: Mapping[int, Sequence[bytes]],
    *,
    batch_size: int,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("candidate metric batch size must be positive")
    result: dict[str, Any] = {}
    codec.eval()
    for length, source in spans_by_length.items():
        spans = tuple(source)
        if any(len(span) != length for span in spans):
            raise ValueError("candidate spans do not match their registered exact length")
        reconstructed = 0
        for start in range(0, len(spans), batch_size):
            selected = list(spans[start : start + batch_size])
            if not selected:
                continue
            logical_rows = len(selected)
            selected.extend([selected[-1]] * (batch_size - logical_rows))
            batch = build_byte_batch(
                selected,
                max_span=codec.max_span,
                device=codec.device,
            )
            _, matches = codec.encode_and_reconstruction_matches(
                batch.byte_values,
                batch.valid_mask,
            )
            reconstructed += int(matches[:logical_rows].sum().cpu().item())
        candidates = len(spans)
        source_bytes = candidates * length
        fallback_positions = reconstructed + (candidates - reconstructed) * length
        result[str(length)] = {
            "candidate_length": length,
            "candidates": candidates,
            "source_bytes": source_bytes,
            "reconstructed_candidates": reconstructed,
            "reconstruction_fraction": reconstructed / max(candidates, 1),
            "accepted_spans": reconstructed,
            "accepted_bytes": reconstructed * length,
            "accepted_bytes_fraction": reconstructed * length / max(source_bytes, 1),
            "positions_with_atomic_fallback": fallback_positions,
            "bytes_per_position_with_atomic_fallback": source_bytes / max(fallback_positions, 1),
            "spans_sha256": spans_sha256(spans),
        }
    return result


def candidate_length_report(
    codec: InputByteCodec,
    request: CandidateLengthRequest,
) -> dict[str, Any]:
    binary = deterministic_binary_spans(
        request.candidate_lengths,
        request.binary_samples_per_length,
        request.seed,
    )
    validation = exact_corpus_spans(
        request.validation_data,
        request.candidate_lengths,
        request.binary_samples_per_length,
    )
    vocabulary = exact_vocabulary_spans(
        request.assets,
        request.candidate_lengths,
        request.vocabulary_token_ids,
    )
    return {
        "candidate_lengths": list(request.candidate_lengths),
        "source_dtype": str(codec.dtype),
        "vocabulary": {
            "metrics": exact_length_metrics(codec, vocabulary, batch_size=request.batch_size),
            "content_sha256": spans_sha256(
                tuple(span for length in request.candidate_lengths for span in vocabulary[length]),
            ),
        },
        "wikitext_validation": {
            "metrics": exact_length_metrics(codec, validation, batch_size=request.batch_size),
            "content_sha256": hashlib.sha256(request.validation_data).hexdigest(),
            "bytes": len(request.validation_data),
        },
        "arbitrary_binary": {
            "metrics": exact_length_metrics(codec, binary, batch_size=request.batch_size),
            "content_sha256": spans_sha256(
                tuple(span for length in request.candidate_lengths for span in binary[length]),
            ),
            "algorithm": BINARY_SPAN_ALGORITHM,
            "seed": request.seed,
        },
    }


def input_behavior_gates(
    validation: Mapping[str, Any],
    gates: InputGateSpec,
) -> dict[str, bool]:
    segmented = cast(
        Mapping[str, float],
        cast(Mapping[str, Any], validation["teacher_forced"])["segmented"],
    )
    generation = cast(Mapping[str, Any], validation["generation"])
    nll_delta = float(segmented["student_nll"]) - float(segmented["teacher_nll"])
    similarity = generation.get("segmented_mean_byte_similarity")
    return {
        "maximum_segmented_mean_kl": (float(segmented["mean_kl"]) <= gates.maximum_segmented_mean_kl),
        "maximum_segmented_nll_delta": (nll_delta <= gates.maximum_segmented_nll_delta),
        "minimum_segmented_top1_agreement": (float(segmented["top1_agreement"]) >= gates.minimum_segmented_top1_agreement),
        "minimum_segmented_generation_byte_similarity": (
            isinstance(similarity, int | float) and not isinstance(similarity, bool) and float(similarity) >= gates.minimum_segmented_generation_byte_similarity
        ),
    }


def _density_eligible(
    tokenizer: Mapping[str, Any],
    gates: InputGateSpec,
) -> bool:
    acceptance = cast(Mapping[str, Any], tokenizer["acceptance"])
    density = cast(Mapping[str, Any], tokenizer["density"])
    return (
        acceptance["density"] is True
        and density["round_trip"] is True
        and float(density["native_tokens_per_continuous_token"]) >= gates.minimum_native_tokens_per_continuous_token
    )


def _selection_score(
    candidate: Mapping[str, Any],
) -> tuple[float, float, float, float, float, float, float, int]:
    validation = cast(Mapping[str, Any], candidate["validation"])
    segmented = cast(
        Mapping[str, float],
        cast(Mapping[str, Any], validation["teacher_forced"])["segmented"],
    )
    generation = cast(Mapping[str, float], validation["generation"])
    tokenizer = cast(Mapping[str, Any], candidate["tokenizer"])
    density = cast(Mapping[str, float], tokenizer["density"])
    nll_delta = float(segmented["student_nll"]) - float(segmented["teacher_nll"])
    name = str(candidate["name"])
    return (
        -float(density["native_tokens_per_continuous_token"]),
        float(segmented["mean_kl"]),
        nll_delta,
        -float(segmented["top1_agreement"]),
        -float(generation["segmented_mean_byte_similarity"]),
        float(segmented["mean_js"]),
        float(segmented["student_nll"]),
        INPUT_SELECTION_CANDIDATES.index(name),
    )


def select_input_candidate(
    candidates: Sequence[Mapping[str, Any]],
    gates: InputGateSpec | None = None,
) -> dict[str, Any]:
    gates = InputGateSpec() if gates is None else gates
    by_name = {str(candidate.get("name")): candidate for candidate in candidates}
    if set(by_name) != set(INPUT_SELECTION_CANDIDATES) or len(by_name) != len(candidates):
        raise ValueError("input selection requires each registered candidate exactly once")
    eligibility = {
        str(candidate["name"]): {
            "exact_held_out_density": _density_eligible(
                cast(Mapping[str, Any], candidate["tokenizer"]),
                gates,
            ),
            "behavioral_similarity_gates": input_behavior_gates(
                cast(Mapping[str, Any], candidate["validation"]),
                gates,
            ),
        }
        for candidate in candidates
    }
    feasible = [
        candidate
        for candidate in candidates
        if eligibility[str(candidate["name"])]["exact_held_out_density"] is True
        and all(
            cast(
                Mapping[str, bool],
                eligibility[str(candidate["name"])]["behavioral_similarity_gates"],
            ).values()
        )
    ]
    eligible_names = frozenset(str(candidate["name"]) for candidate in feasible)
    scores = {str(candidate["name"]): _selection_score(candidate) for candidate in candidates}
    pool = feasible or list(candidates)
    ranked = sorted(pool, key=lambda candidate: scores[str(candidate["name"])])
    selected = ranked[0]
    reported = sorted(
        candidates,
        key=lambda candidate: (
            str(candidate["name"]) not in eligible_names,
            scores[str(candidate["name"])],
        ),
    )
    return {
        "selection_rule": INPUT_SELECTION_RULE,
        "selection_metrics_split": "validation",
        "untouched_by_training": True,
        "gate_policy": (
            "Require exact held-out density and every registered behavioral-similarity gate. "
            "Report embedding alignment and compactness independently. If none pass, rank all "
            "candidates and mark selection_feasible false."
        ),
        "score_order": [
            "maximum_held_out_native_tokens_per_continuous_token",
            "minimum_segmented_mean_kl",
            "minimum_segmented_nll_delta",
            "maximum_segmented_top1_agreement",
            "maximum_segmented_generation_byte_similarity",
            "minimum_segmented_mean_js",
            "minimum_segmented_student_nll",
            "registered_candidate_order",
        ],
        "selected_candidate": selected["name"],
        "selection_feasible": bool(feasible),
        "ranked_candidates": [
            {
                "name": candidate["name"],
                "score": list(scores[str(candidate["name"])]),
                "eligible": str(candidate["name"]) in eligible_names,
                **eligibility[str(candidate["name"])],
                "embedding_alignment_passed": cast(
                    Mapping[str, Any],
                    candidate["tokenizer"],
                )["acceptance"]["embedding_fit"],
                "compactness_passed": cast(
                    Mapping[str, Any],
                    candidate["tokenizer"],
                )["acceptance"]["compactness"],
            }
            for candidate in reported
        ],
    }
