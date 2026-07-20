from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any, Final, Literal, Self, cast, final

from continuous_tokenizer.contracts.parsing import (
    is_lowercase_sha256,
    mapping_fingerprint,
)

STATE_BUDGET_VERSION: Final = 1
STATE_BUDGET_MAXIMUM_RATIO: Final = 1.0
STATE_BUDGET_CONCLUSION: Final = "fits_within_preregistered_joint_tensor_state_budget"
STATE_BUDGET_SCOPE: Final = "cross_directional_prerequisite"
REFERENCE_DEDUPLICATION_POLICY: Final = "count_tied_vocabulary_once_else_input_and_output_separately"
CONTROL_DEDUPLICATION_POLICY: Final = "count_shared_control_ids_and_copied_input_control_rows_once"
STATE_BUDGET_RUN_IDENTITY_FIELDS: Final = (
    "model_id",
    "model_revision",
    "dataset_id",
    "dataset_revision",
    "embedding_tensor",
    "source_dtype",
    "seed",
    "source_commit",
    "source_dirty",
    "source_state_sha256",
    "dependency_lock_sha256",
    "installed_package",
    "claim_vocabulary_sha256",
    "source_assets",
)
STATE_BUDGET_PROJECT_IDENTITY_FIELDS: Final = (
    "source_commit",
    "source_dirty",
    "source_state_sha256",
    "dependency_lock_sha256",
    "installed_package",
    "claim_vocabulary_sha256",
)

type StateBudgetVerdict = Literal["supported", "unsupported"]
_DTYPE_BYTES: Final = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int16": 2,
    "torch.float32": 4,
    "torch.int32": 4,
    "torch.float64": 8,
    "torch.int64": 8,
}


def _strict_mapping(
    value: object,
    cls: type[Any],
    label: str,
) -> Mapping[str, Any]:
    expected = {field.name for field in fields(cls)}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are not canonical")
    return cast(Mapping[str, Any], value)


def _integer(value: object, label: str, *, positive: bool = False) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{label} must be a {qualifier} integer")
    return value


def _sha256(value: object, label: str) -> str:
    if not is_lowercase_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return cast(str, value)


def _string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


@final
@dataclass(frozen=True, slots=True)
class StateBudgetConfig:
    version: int = STATE_BUDGET_VERSION
    maximum_ratio: float = STATE_BUDGET_MAXIMUM_RATIO
    conclusion: str = STATE_BUDGET_CONCLUSION

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget config")
        if values["version"] != STATE_BUDGET_VERSION:
            raise ValueError("state-budget config version is unsupported")
        maximum = values["maximum_ratio"]
        if not isinstance(maximum, int | float) or isinstance(maximum, bool) or float(maximum) != STATE_BUDGET_MAXIMUM_RATIO:
            raise ValueError("state-budget maximum ratio is fixed at 1.0")
        if values["conclusion"] != STATE_BUDGET_CONCLUSION:
            raise ValueError("state-budget conclusion is not supported")
        return cls()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class StateBudgetNonClaims:
    combined_runtime_tested: bool = False
    continuous_feedback_tested: bool = False
    physical_omission_tested: bool = False
    resident_memory_reduction_tested: bool = False
    peak_memory_reduction_tested: bool = False

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget non-claim flags")
        if any(item is not False for item in values.values()):
            raise ValueError("state-budget non-claim flags must all be false")
        return cls()

    def to_dict(self) -> dict[str, bool]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class StateBudgetTensor:
    name: str
    shape: tuple[int, ...]
    dtype: str
    bytes: int
    sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget tensor")
        raw_shape = values["shape"]
        if not isinstance(raw_shape, Sequence) or isinstance(
            raw_shape,
            str | bytes,
        ):
            raise ValueError("state-budget tensor shape must be an array")
        shape = tuple(_integer(dimension, "state-budget tensor dimension") for dimension in raw_shape)
        dtype = _string(values["dtype"], "state-budget tensor dtype")
        byte_count = _integer(
            values["bytes"],
            "state-budget tensor bytes",
        )
        if dtype not in _DTYPE_BYTES:
            raise ValueError("state-budget tensor dtype is unsupported")
        expected_bytes = math.prod(shape) * _DTYPE_BYTES[dtype]
        if byte_count != expected_bytes:
            raise ValueError(
                "state-budget tensor bytes do not match shape and dtype",
            )
        return cls(
            name=_string(values["name"], "state-budget tensor name"),
            shape=shape,
            dtype=dtype,
            bytes=byte_count,
            sha256=_sha256(values["sha256"], "state-budget tensor hash"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "shape": list(self.shape),
        }


@final
@dataclass(frozen=True, slots=True)
class StateBudgetIdentity:
    source_commit: str
    source_dirty: bool
    source_state_sha256: str
    dependency_lock_sha256: str
    installed_package_sha256: str
    claim_vocabulary_sha256: str
    model_config_sha256: str
    input_embedding_sha256: str
    tokenizer_vocabulary_sha256: str
    input_contract_sha256: str
    output_contract_sha256: str

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget identity")
        if not isinstance(values["source_dirty"], bool):
            raise ValueError("state-budget source dirty flag must be boolean")
        return cls(
            source_commit=_string(
                values["source_commit"],
                "state-budget source commit",
            ),
            source_dirty=values["source_dirty"],
            **{
                name: _sha256(
                    values[name],
                    f"state-budget {name.replace('_', ' ')}",
                )
                for name in (
                    "source_state_sha256",
                    "dependency_lock_sha256",
                    "installed_package_sha256",
                    "claim_vocabulary_sha256",
                    "model_config_sha256",
                    "input_embedding_sha256",
                    "tokenizer_vocabulary_sha256",
                    "input_contract_sha256",
                    "output_contract_sha256",
                )
            },
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class StateBudgetArithmetic:
    input_codec_bytes: int
    output_codec_bytes: int
    atomic_byte_rows_bytes: int
    shared_control_id_bytes: int
    shared_control_row_bytes: int
    candidate_tensor_state_bytes: int
    reference_input_table_bytes: int
    reference_output_head_bytes: int
    reference_tensor_state_bytes: int

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget arithmetic")
        return cls(
            **{
                name: _integer(
                    values[name],
                    f"state-budget {name.replace('_', ' ')}",
                )
                for name in values
            },
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@final
@dataclass(frozen=True, slots=True)
class StateBudgetSeedResult:
    model_id: str
    model_revision: str
    seed: int
    tie_word_embeddings: bool
    identity: StateBudgetIdentity
    input_checkpoint_sha256: str
    output_checkpoint_sha256: str
    input_inventory: tuple[StateBudgetTensor, ...]
    output_inventory: tuple[StateBudgetTensor, ...]
    reference_inventory: tuple[StateBudgetTensor, ...]
    input_inventory_sha256: str
    output_inventory_sha256: str
    reference_inventory_sha256: str
    reference_deduplication_policy: str
    control_deduplication_policy: str
    arithmetic: StateBudgetArithmetic
    ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget seed result")
        tied = values["tie_word_embeddings"]
        ratio = values["ratio"]
        if not isinstance(tied, bool):
            raise ValueError("state-budget tied-table flag must be boolean")
        if not isinstance(ratio, int | float) or isinstance(ratio, bool) or float(ratio) < 0:
            raise ValueError("state-budget ratio must be non-negative")
        inventories = {
            name: _tensor_inventory(
                values[name],
                f"state-budget {name.replace('_', ' ')}",
            )
            for name in (
                "input_inventory",
                "output_inventory",
                "reference_inventory",
            )
        }
        result = cls(
            model_id=_string(values["model_id"], "state-budget model ID"),
            model_revision=_string(
                values["model_revision"],
                "state-budget model revision",
            ),
            seed=_integer(values["seed"], "state-budget seed"),
            tie_word_embeddings=tied,
            identity=StateBudgetIdentity.from_mapping(values["identity"]),
            input_checkpoint_sha256=_sha256(
                values["input_checkpoint_sha256"],
                "state-budget input checkpoint hash",
            ),
            output_checkpoint_sha256=_sha256(
                values["output_checkpoint_sha256"],
                "state-budget output checkpoint hash",
            ),
            input_inventory=inventories["input_inventory"],
            output_inventory=inventories["output_inventory"],
            reference_inventory=inventories["reference_inventory"],
            input_inventory_sha256=_sha256(
                values["input_inventory_sha256"],
                "state-budget input inventory hash",
            ),
            output_inventory_sha256=_sha256(
                values["output_inventory_sha256"],
                "state-budget output inventory hash",
            ),
            reference_inventory_sha256=_sha256(
                values["reference_inventory_sha256"],
                "state-budget reference inventory hash",
            ),
            reference_deduplication_policy=_fixed_policy(
                values["reference_deduplication_policy"],
                REFERENCE_DEDUPLICATION_POLICY,
                "reference",
            ),
            control_deduplication_policy=_fixed_policy(
                values["control_deduplication_policy"],
                CONTROL_DEDUPLICATION_POLICY,
                "control",
            ),
            arithmetic=StateBudgetArithmetic.from_mapping(
                values["arithmetic"],
            ),
            ratio=float(ratio),
        )
        _validate_seed_result(result)
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            **asdict(self),
            "identity": self.identity.to_dict(),
            "input_inventory": [tensor.to_dict() for tensor in self.input_inventory],
            "output_inventory": [tensor.to_dict() for tensor in self.output_inventory],
            "reference_inventory": [tensor.to_dict() for tensor in self.reference_inventory],
            "arithmetic": self.arithmetic.to_dict(),
        }


def _tensor_inventory(
    value: object,
    label: str,
) -> tuple[StateBudgetTensor, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError(f"{label} must be an array")
    rows = tuple(StateBudgetTensor.from_mapping(row) for row in value)
    names = [row.name for row in rows]
    if not rows or len(names) != len(set(names)):
        raise ValueError(f"{label} must be non-empty and contain unique rows")
    return rows


def _fixed_policy(value: object, expected: str, label: str) -> str:
    if value != expected:
        raise ValueError(f"state-budget {label} deduplication policy is invalid")
    return expected


def _named_tensor(
    rows: Sequence[StateBudgetTensor],
    name: str,
) -> StateBudgetTensor:
    matches = [row for row in rows if row.name == name]
    if len(matches) != 1:
        raise ValueError(
            f"state-budget inventory requires exactly one {name!r} row",
        )
    return matches[0]


def _codec_bytes(rows: Sequence[StateBudgetTensor]) -> int:
    codec_rows = [row for row in rows if row.name.startswith("codec.")]
    if not codec_rows:
        raise ValueError("state-budget inventory has no codec tensors")
    return sum(row.bytes for row in codec_rows)


def inventory_sha256(rows: Sequence[StateBudgetTensor]) -> str:
    return mapping_fingerprint(
        {"tensors": [row.to_dict() for row in rows]},
    )


def reference_inventory(
    model: tuple[str, str],
    source_assets: Mapping[str, object],
    metadata: Mapping[str, object],
    input_bytes: object,
    output_bytes: object,
) -> tuple[StateBudgetTensor, ...]:
    shape = metadata.get("source_shape")
    dtype = metadata.get("source_dtype")
    tied = metadata.get("tie_word_embeddings")
    model_config = source_assets.get("model_config")
    input_embedding = source_assets.get("input_embedding_tensor")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, str | bytes)
        or not isinstance(dtype, str)
        or not dtype
        or not isinstance(tied, bool)
        or not isinstance(model_config, Mapping)
        or not isinstance(input_embedding, Mapping)
    ):
        raise ValueError("state-budget reference tensor evidence is not canonical")
    model_config_sha256 = _sha256(
        model_config.get("sha256"),
        "state-budget model config hash",
    )
    input_embedding_sha256 = _sha256(
        input_embedding.get("sha256"),
        "state-budget input embedding hash",
    )
    tensor_shape = list(shape)
    input_row = StateBudgetTensor.from_mapping(
        {
            "name": "native.tied_vocabulary" if tied else "native.input_embedding",
            "shape": tensor_shape,
            "dtype": dtype,
            "bytes": input_bytes,
            "sha256": input_embedding_sha256,
        }
    )
    if tied:
        if output_bytes != input_bytes:
            raise ValueError("state-budget tied reference tensor bytes do not match")
        return (input_row,)
    output_descriptor = {
        "model_id": model[0],
        "model_revision": model[1],
        "model_config_sha256": model_config_sha256,
        "name": "native.output_head",
        "shape": tensor_shape,
        "dtype": dtype,
        "bytes": output_bytes,
    }
    return (
        input_row,
        StateBudgetTensor.from_mapping(
            {
                "name": output_descriptor["name"],
                "shape": tensor_shape,
                "dtype": dtype,
                "bytes": output_bytes,
                "sha256": mapping_fingerprint(output_descriptor),
            }
        ),
    )


def derive_state_budget_arithmetic(
    input_inventory: Sequence[StateBudgetTensor],
    output_inventory: Sequence[StateBudgetTensor],
    reference_inventory: Sequence[StateBudgetTensor],
    *,
    tied: bool,
) -> StateBudgetArithmetic:
    input_ids = _named_tensor(input_inventory, "controls.ids")
    output_ids = _named_tensor(output_inventory, "controls.ids")
    if input_ids != output_ids:
        raise ValueError(
            "state-budget shared control-ID inventory rows do not match",
        )
    control_rows = _named_tensor(
        input_inventory,
        "controls.embeddings",
    )
    if any(row.name == "controls.embeddings" for row in output_inventory):
        raise ValueError(
            "state-budget output inventory must not duplicate control rows",
        )
    atomic_rows = _named_tensor(
        input_inventory,
        "codec.byte_embeddings",
    )
    expected_reference_names = {"native.tied_vocabulary"} if tied else {"native.input_embedding", "native.output_head"}
    if {row.name for row in reference_inventory} != expected_reference_names:
        raise ValueError(
            "state-budget reference inventory does not match tied layout",
        )
    input_codec_bytes = _codec_bytes(input_inventory)
    output_codec_bytes = _codec_bytes(output_inventory)
    input_reference = _named_tensor(
        reference_inventory,
        "native.tied_vocabulary" if tied else "native.input_embedding",
    )
    output_reference = input_reference if tied else _named_tensor(reference_inventory, "native.output_head")
    reference_bytes = sum(row.bytes for row in reference_inventory)
    if reference_bytes <= 0:
        raise ValueError("state-budget reference inventory must be positive")
    return StateBudgetArithmetic(
        input_codec_bytes=input_codec_bytes,
        output_codec_bytes=output_codec_bytes,
        atomic_byte_rows_bytes=atomic_rows.bytes,
        shared_control_id_bytes=input_ids.bytes,
        shared_control_row_bytes=control_rows.bytes,
        candidate_tensor_state_bytes=(input_codec_bytes + output_codec_bytes + input_ids.bytes + control_rows.bytes),
        reference_input_table_bytes=input_reference.bytes,
        reference_output_head_bytes=output_reference.bytes,
        reference_tensor_state_bytes=reference_bytes,
    )


def _validate_seed_result(result: StateBudgetSeedResult) -> None:
    expected_hashes = (
        inventory_sha256(result.input_inventory),
        inventory_sha256(result.output_inventory),
        inventory_sha256(result.reference_inventory),
    )
    if expected_hashes != (
        result.input_inventory_sha256,
        result.output_inventory_sha256,
        result.reference_inventory_sha256,
    ):
        raise ValueError("state-budget inventory hash mismatch")
    expected_arithmetic = derive_state_budget_arithmetic(
        result.input_inventory,
        result.output_inventory,
        result.reference_inventory,
        tied=result.tie_word_embeddings,
    )
    if result.arithmetic != expected_arithmetic:
        raise ValueError("state-budget arithmetic does not match inventories")
    reference_bytes = expected_arithmetic.reference_tensor_state_bytes
    if not math.isclose(
        result.ratio,
        expected_arithmetic.candidate_tensor_state_bytes / reference_bytes,
    ):
        raise ValueError("state-budget ratio does not match arithmetic")


@final
@dataclass(frozen=True, slots=True)
class StateBudgetResult:
    version: int
    evidence_scope: str
    operational_status: str
    config: StateBudgetConfig
    conclusion: str
    verdict: StateBudgetVerdict
    non_claims: StateBudgetNonClaims
    per_seed: tuple[StateBudgetSeedResult, ...]
    worst_case_ratio: float

    @classmethod
    def from_mapping(cls, value: object) -> Self:
        values = _strict_mapping(value, cls, "state-budget result")
        config = StateBudgetConfig.from_mapping(values["config"])
        if values["version"] != STATE_BUDGET_VERSION:
            raise ValueError("state-budget result version is unsupported")
        if values["evidence_scope"] != STATE_BUDGET_SCOPE:
            raise ValueError("state-budget evidence scope is invalid")
        if values["operational_status"] != "completed":
            raise ValueError("state-budget operational status is invalid")
        if values["conclusion"] != STATE_BUDGET_CONCLUSION:
            raise ValueError("state-budget conclusion is not supported")
        verdict = values["verdict"]
        if verdict not in {"supported", "unsupported"}:
            raise ValueError("state-budget verdict is invalid")
        worst = values["worst_case_ratio"]
        if not isinstance(worst, int | float) or isinstance(worst, bool) or float(worst) < 0:
            raise ValueError("state-budget worst-case ratio must be non-negative")
        rows = _seed_results(values["per_seed"])
        result = cls(
            version=STATE_BUDGET_VERSION,
            evidence_scope=STATE_BUDGET_SCOPE,
            operational_status="completed",
            config=config,
            conclusion=STATE_BUDGET_CONCLUSION,
            verdict=cast(StateBudgetVerdict, verdict),
            non_claims=StateBudgetNonClaims.from_mapping(
                values["non_claims"],
            ),
            per_seed=rows,
            worst_case_ratio=float(worst),
        )
        expected_worst = max(row.ratio for row in rows)
        expected_verdict = "supported" if expected_worst <= config.maximum_ratio else "unsupported"
        if not math.isclose(result.worst_case_ratio, expected_worst):
            raise ValueError(
                "state-budget worst-case ratio differs from per-seed results",
            )
        if result.verdict != expected_verdict:
            raise ValueError(
                "state-budget verdict differs from preregistered ratio gate",
            )
        return result

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "evidence_scope": self.evidence_scope,
            "operational_status": self.operational_status,
            "config": self.config.to_dict(),
            "conclusion": self.conclusion,
            "verdict": self.verdict,
            "non_claims": self.non_claims.to_dict(),
            "per_seed": [row.to_dict() for row in self.per_seed],
            "worst_case_ratio": self.worst_case_ratio,
        }


def _seed_results(value: object) -> tuple[StateBudgetSeedResult, ...]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ValueError("state-budget per-seed results must be an array")
    rows = tuple(StateBudgetSeedResult.from_mapping(row) for row in value)
    identities = [(row.model_id, row.model_revision, row.seed) for row in rows]
    if len(rows) != 6 or len(identities) != len(set(identities)):
        raise ValueError(
            "state-budget requires six unique model-seed results",
        )
    return rows
