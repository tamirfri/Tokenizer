from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from continuous_tokenizer.contracts.claim_derivation import (
    MINIMUM_LATENCY_REPETITIONS,
    MINIMUM_LATENCY_WARMUPS,
)
from continuous_tokenizer.contracts.claims import (
    ClaimVerdict,
    claim_records,
    directional_claims,
)
from continuous_tokenizer.contracts.experiment import TokenizerMode
from continuous_tokenizer.contracts.profiles import DIAGNOSTIC_PROFILE_NAME
from continuous_tokenizer.reporting.discovery import (
    acceptance_rows,
    artifact_profile,
    training_progress_rows,
)
from continuous_tokenizer.reporting.replication import encoding_cache_speedup
from continuous_tokenizer.reporting.shared import (
    canonical_claim_lines,
    current_design_lines,
    display_name,
    input_headline_line,
    optional_mapping,
    optional_metric,
    status_lines,
)


def _gate_verdict(value: bool | None) -> str:
    match value:
        case None:
            return "N/A"
        case True:
            return "PASS"
        case False:
            return "FAIL"


def artifact_claim_records(
    result: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    stored = result.get("claims")
    if isinstance(stored, list) and all(isinstance(record, Mapping) for record in stored):
        return cast(list[Mapping[str, Any]], stored)
    mode = cast(TokenizerMode, result["mode"])
    reason = (
        "single-run artifacts expose registered measurements and gates; project-level sealed replication evidence is required for a directional claim verdict"
    )
    verdicts: dict[str, ClaimVerdict] = {definition.claim_id: "incomplete" for definition in directional_claims(mode)}
    return cast(
        list[Mapping[str, Any]],
        claim_records(mode, verdicts, reason=reason),
    )


def _optimizer_policy(metadata: Mapping[str, str | int]) -> str:
    adjustment = metadata.get("muon_adjust_lr_fn", "unknown")
    if "hidden_matrix_parameters" in metadata:
        return (
            f"hidden matrices={metadata['hidden_matrix_parameters']}; "
            "output and non-matrices="
            f"{metadata.get('output_and_non_matrix_parameters', 'unknown')}; "
            f"Muon LR adjustment={adjustment}"
        )
    return (
        f"2D matrices={metadata.get('matrix_parameters', 'unknown')}; "
        f"non-matrices={metadata.get('non_matrix_parameters', 'unknown')}; "
        f"Muon LR adjustment={adjustment}"
    )


def _tokenizer_runtime_lines(tokenizer: Mapping[str, Any]) -> list[str]:
    runs = tokenizer.get("segmentation_runs", ())
    if not runs:
        return []
    compilation = tokenizer.get("compilation", {})
    speedup = encoding_cache_speedup(runs)
    warmup = f"`{compilation.get('warmup_seconds', 0):.6f}` seconds" if compilation.get("enabled") else "`not applicable`"
    speedup_label = "`not measured`" if speedup is None else f"`{speedup:.4f}x`"
    lines = [
        "",
        "## Tokenizer Runtime",
        "",
        f"- Compiler warm-up: {warmup}",
        f"- Warm encoding-cache median speedup: {speedup_label}",
        "",
        "| Mode | Median seconds | P95 seconds | Runs | Hit rate | Cache tensor bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {run['mode']} | {run['seconds']:.6f} | "
        f"{run.get('p95_seconds', run['seconds']):.6f} | "
        f"{run.get('repetitions', 1)} | {run.get('cache_hit_rate', 0):.2%} | "
        f"{run.get('cache_tensor_bytes', 0):,} |"
        for run in runs
    )
    lines.extend(
        [
            "",
            (
                "Compiler warm-up is excluded from cache timings. Cache modes are interleaved across "
                "repetitions; the cache may affect latency only and must preserve identical spans."
            ),
        ]
    )
    return lines


def _vocabulary_lines(summary: Mapping[str, Any] | None) -> list[str]:
    if summary is None:
        return []
    fields = (
        ("Vocabulary rows", "vocabulary_size"),
        ("Ordinary tokens", "ordinary_tokens"),
        ("Compatibility tokens", "compatibility_tokens"),
        ("Duplicate aliases", "duplicate_aliases"),
        ("Ambiguous byte payloads", "ambiguous_byte_sequences"),
        ("Structural controls", "control_tokens"),
        ("Unavailable rows", "unavailable_rows"),
        ("Out-of-table controls", "out_of_table_controls"),
        ("Atomic bytes", "atomic_bytes"),
    )
    return [
        "",
        "## Native Tokenizer Vocabulary",
        "",
        *(f"- {label}: `{summary[name]}`" for label, name in fields),
    ]


def _verification_lines(verification: Mapping[str, Any] | None) -> list[str]:
    if not verification or not verification.get("provided"):
        return ["", "## Lifecycle Verification", "", "- Preflight verification: `not provided`"]
    lines = [
        "",
        "## Lifecycle Verification",
        "",
        "- Preflight verification: `provided`",
        f"- All checks passed: `{verification.get('all_passed')}`",
        f"- Source commit: `{verification.get('source_commit', 'n/a')}`",
        f"- Source state SHA-256: `{verification.get('source_state_sha256', 'n/a')}`",
        f"- Dependency lock SHA-256: `{verification.get('dependency_lock_sha256', 'n/a')}`",
        "",
        "| Check | Passed | Return code | Seconds | Log SHA-256 |",
        "|---|:---:|---:|---:|---|",
    ]
    checks = verification.get("checks", {})
    if isinstance(checks, Mapping):
        for name, check in checks.items():
            if not isinstance(check, Mapping):
                continue
            lines.append(
                f"| {name} | {check.get('passed')} | {check.get('return_code')} | {float(check.get('seconds', 0)):.3f} | `{check.get('log_sha256', 'n/a')}` |"
            )
    return lines


def _scalar_findings_lines(title: str, value: Mapping[str, Any] | None) -> list[str]:
    if not value:
        return []
    rows: list[tuple[str, Any]] = []

    def collect(prefix: str, item: Any) -> None:
        if isinstance(item, Mapping):
            for name, nested in item.items():
                collect(f"{prefix}.{name}" if prefix else str(name), nested)
        elif isinstance(item, Sequence) and not isinstance(item, str | bytes):
            for index, nested in enumerate(item):
                collect(f"{prefix}[{index}]", nested)
        elif item is None or isinstance(item, str | int | float | bool):
            rows.append((prefix, item))

    collect("", value)
    return [
        "",
        f"## {title}",
        "",
        "| Finding | Stored value |",
        "|---|---|",
        *(f"| `{name}` | `{item}` |" for name, item in rows),
    ]


def _training_progress_lines(progress: Sequence[Mapping[str, Any]]) -> list[str]:
    rows = training_progress_rows(progress)
    if not rows:
        return []
    lines = [
        "",
        "## Training Convergence",
        "",
        "| Phase | Epoch | Loss | Normalized RMSE | Cosine P01 | Cosine P50 | Source-dtype equal | Reconstruction | Density | Selected |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        selected = row["selected"]
        if selected is True:
            selected_label = "yes"
        elif selected is False:
            selected_label = "no"
        else:
            selected_label = "n/a"
        lines.append(
            f"| {row['phase']} | {row['epoch']} | "
            f"{optional_metric(row['training_loss'])} | "
            f"{optional_metric(row['normalized_rmse'])} | "
            f"{optional_metric(row['cosine_p01'])} | "
            f"{optional_metric(row['cosine_p50'])} | "
            f"{optional_metric(row['source_dtype_equal'], percentage=True)} | "
            f"{optional_metric(row['reconstruction'], percentage=True)} | "
            f"{optional_metric(row['native_tokens_per_continuous_token'])} | "
            f"{selected_label} |"
        )
    lines.extend(
        [
            "",
            "Optimization uses FP32 parameters. Convergence rows, checkpoint selection, and headline evidence are measured in the source embedding dtype.",
        ]
    )
    return lines


def _output_artifact_report(result: Mapping[str, Any]) -> str:
    experiment = result["experiment"]
    output = result["output"]
    gates = result["gates"]
    profile = experiment["training"]["profile"]
    stop_control = output.get("stop_control", {})
    gate_spec = experiment["gates"]
    gate_definitions = (
        ("direct_feedback", "direct_feedback_equality", ">=", "minimum_direct_feedback_equality"),
        (
            "native_tokens_per_attempted_macro_step",
            "native_tokens_per_attempted_macro_step",
            ">=",
            "minimum_native_tokens_per_attempted_macro_step",
        ),
        ("rollout_event_agreement", "rollout_event_agreement", ">=", "minimum_rollout_event_agreement"),
        (
            "candidate_reference_state_ratio",
            "candidate_reference_state_ratio",
            "<=",
            "maximum_candidate_reference_state_ratio",
        ),
    )
    defined_gates = {definition[0] for definition in gate_definitions}
    lines = [
        f"# {experiment['name']}",
        "",
        "## Headline Verdict",
        "",
        f"- Stored run scientific verdict: **{str(result['scientific_verdict']).upper()}**",
        (f"- Primary output density operand: native tokens per attempted macro-step `{output.get('native_tokens_per_attempted_macro_step')}`"),
        (
            "- Exactness prerequisites: direct feedback "
            f"`{output.get('direct_feedback_equality')}`; invalid events "
            f"`{output.get('invalid_events')}`; valid non-empty termination "
            f"`{output.get('valid_non_empty_termination')}`; rollout agreement "
            f"`{output.get('rollout_event_agreement')}`"
        ),
        "",
        *current_design_lines(),
        "",
        "## Run Status",
        "",
        *status_lines(result),
        f"- Tokenizer profile: {profile}",
        "- Native vocabulary head used during inference: no",
        "- Feedback policy: deterministic longest native-byte match",
        f"- Stop-control policy: `{stop_control.get('policy', 'not recorded')}`",
        f"- Stop-control token IDs: `{stop_control.get('token_ids', [])}`",
        f"- Observed oracle stop controls: `{stop_control.get('oracle_events', 'not recorded')}`",
        f"- Observed predicted stop controls: `{stop_control.get('predicted_events', 'not recorded')}`",
        "- Physical output-head omission: `measured by independent deployment evidence`",
        "",
        "## Canonical Directional Claims",
        "",
        *canonical_claim_lines(artifact_claim_records(result)),
        "",
        "## Registered Gates",
        "",
        "| Gate | Measured | Requirement | Stored verdict |",
        "|---|---:|---:|:---:|",
    ]
    for gate_name, metric_name, operator, threshold_name in gate_definitions:
        measured = output.get(metric_name)
        passed = gates.get(gate_name)
        lines.append(
            f"| {display_name(gate_name)} | "
            f"`{'N/A' if measured is None else measured}` | "
            f"`{operator} {gate_spec[threshold_name]}` | "
            f"{_gate_verdict(passed if isinstance(passed, bool) else None)} |"
        )
    for gate_name, passed in gates.items():
        if gate_name in defined_gates:
            continue
        measured = output.get(gate_name)
        lines.append(f"| {display_name(gate_name)} | `{measured}` | `== True` | {_gate_verdict(passed if isinstance(passed, bool) else None)} |")
    unsupported = [display_name(name) for name, passed in gates.items() if passed is False]
    not_applicable = [display_name(name) for name, metric_name, _, _ in gate_definitions if output.get(metric_name) is None]
    lines.extend(_vocabulary_lines(optional_mapping(result.get("vocabulary"))))
    lines.extend(_verification_lines(optional_mapping(result.get("verification"))))
    lines.extend(_scalar_findings_lines("Output Raw Findings", output))
    lines.extend(
        [
            "",
            "## Unsupported And Not Applicable Findings",
            "",
            "- Unsupported: " + (", ".join(unsupported) if unsupported else "none"),
            "- Not applicable: " + (", ".join(not_applicable) if not_applicable else "none"),
            "",
            f"Scientific verdict: **{str(result['scientific_verdict']).upper()}**",
            "",
            "Run completion and scientific support are separate: a completed run may report unsupported output-only hypotheses.",
        ]
    )
    return "\n".join(lines) + "\n"


def _input_density_lines(tokenizer: Mapping[str, Any]) -> list[str]:
    lines = [
        "",
        "## Input Position Findings",
        "",
        "| Alignment | Native tokens | Continuous tokens | Bytes/continuous token | Native tokens/continuous token | Round-trip |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for density_name in ("density", "native_aligned_segmentation"):
        finding = tokenizer.get(density_name)
        if isinstance(finding, Mapping):
            lines.append(
                f"| {finding['alignment']} | {finding['native_tokens']} | "
                f"{finding['continuous_tokens']} | {finding['bytes_per_continuous_token']:.6f} | "
                f"{finding['native_tokens_per_continuous_token']:.6f} | "
                f"{finding['round_trip']} |"
            )
    fixtures = tokenizer.get("raw_byte_fixtures")
    if not isinstance(fixtures, Mapping):
        return lines
    lines.extend(
        [
            "",
            "## Raw Binary Fixture Findings",
            "",
            "| Fixture | Bytes | Spans | Bytes/span | Atomic spans | Round-trip | Span lengths |",
            "|---|---:|---:|---:|---:|:---:|---|",
        ]
    )
    for name, fixture in fixtures.items():
        if isinstance(fixture, Mapping):
            lines.append(
                f"| {name} | {fixture['bytes']} | {fixture['spans']} | "
                f"{fixture['bytes_per_span']:.6f} | {fixture['atomic_spans']} | "
                f"{fixture['round_trip']} | `{fixture['span_lengths']}` |"
            )
    return lines


def _input_efficiency_lines(llm: Any) -> list[str]:
    if llm is None:
        return []
    native = llm["performance"]["native"]
    segmented = llm["performance"]["segmented"]
    cache_reduced = segmented["materialized_cache_bytes"] < native["materialized_cache_bytes"]
    flops_reduced = segmented["total_analytical_flops"] < native["total_analytical_flops"]
    lines = [
        f"- Registered-prompt-set model-cache reduction: **{'OBSERVED' if cache_reduced else 'NOT OBSERVED'}**",
        f"- Registered-prompt-set analytical FLOP reduction: **{'OBSERVED' if flops_reduced else 'NOT OBSERVED'}**",
    ]
    options = llm.get("options", {})
    measurement = llm["performance"].get("measurement", {})
    prompt_count = measurement.get("prompt_count", 0)
    expected_pairs = prompt_count * measurement.get("repetitions", 0)
    raw_pairs = llm["performance"].get("raw_pairs")
    timing_ready = (
        options.get("performance_prompts") == prompt_count
        and options.get("warmups") == measurement.get("warmups")
        and options.get("repetitions") == measurement.get("repetitions")
        and measurement.get("warmups", 0) >= MINIMUM_LATENCY_WARMUPS
        and measurement.get("repetitions", 0) >= MINIMUM_LATENCY_REPETITIONS
        and measurement.get("expected_raw_pairs") == expected_pairs
        and measurement.get("recorded_raw_pairs") == expected_pairs
        and isinstance(raw_pairs, list)
        and len(raw_pairs) == expected_pairs
        and measurement.get("prompt_order") == "cyclic_rotation_by_repetition"
        and measurement.get("path_order") == "cyclic_rotation_by_pair_execution_order"
    )
    timing_improved = segmented["time_to_first_logit_median_seconds"] < native["time_to_first_logit_median_seconds"]
    timing_label = "OBSERVED" if timing_improved else "NOT OBSERVED"
    if not timing_ready:
        timing_label = "INSUFFICIENT MEASUREMENT"
    lines.extend(
        (
            f"- End-to-end time-to-first-logit improvement: **{timing_label}**",
            "- Frozen-model behavioral similarity: **REGISTERED GATES REPORTED IN THE HEADLINE**",
        )
    )
    return lines


def _input_llm_evidence_lines(llm: Any, *, synthetic: bool) -> list[str]:
    if llm is None:
        return (
            [
                "",
                "Synthetic evidence validates the implementation and report pipeline only. It does not support a real-model research claim.",
            ]
            if synthetic
            else []
        )
    native = llm["performance"]["native"]
    segmented = llm["performance"]["segmented"]
    positions = llm["positions"]
    native_cache_bytes = native["materialized_cache_bytes"]
    cache_reduction = None if not native_cache_bytes else 1 - segmented["materialized_cache_bytes"] / native_cache_bytes
    flop_reduction = 1 - segmented["total_analytical_flops"] / native["total_analytical_flops"]
    ttfl_speedup = native["time_to_first_logit_median_seconds"] / segmented["time_to_first_logit_median_seconds"]
    measurement = llm["performance"]["measurement"]
    lines = [
        "- Teacher-forced evaluation policy: `batch size 8`; scalar-versus-batched calibration must pass for the exact execution identity",
        f"- Compatibility KL: `{llm['teacher_forced']['compatibility']['mean_kl']:.6f}`",
        f"- Segmented KL: `{llm['teacher_forced']['segmented']['mean_kl']:.6f}`",
        (
            "- Native positions/segmented position: "
            f"`{positions['native_positions_per_segmented_position']:.4f}x` "
            f"(`{positions['segmented']:.0f}` segmented vs `{positions['native']:.0f}` native positions)"
        ),
        (f"- Registered performance prompt set: `{measurement['prompt_count']}` prompts; SHA-256 `{measurement['prompt_set_sha256']}`"),
        "- Registered-prompt-set model-cache reduction: " + ("`n/a`" if cache_reduction is None else f"`{cache_reduction:.2%}`"),
        f"- Registered-prompt-set analytical prefill FLOP reduction: `{flop_reduction:.2%}`",
        f"- Measured end-to-end time-to-first-logit speedup: `{ttfl_speedup:.4f}x`",
        f"- Segmented greedy generation exact agreement: `{llm['generation']['segmented_exact_fraction']:.2%}`",
        "",
        (
            "LLM behavior is evaluated against the prospectively registered comparative-similarity "
            "tolerances; passing is not a non-inferiority result. The measured "
            "time-to-first-logit value includes tokenizer preparation and uses raw paired "
            "prompt-by-repetition rows with rotated prompt and path order. Analytical FLOPs report "
            "registered-prompt-set totals for backbone, codec encode, codec validation/decode, and total compute. "
            "Compatibility and segmented timing use a warm encoding cache; tokenizer-only reports "
            "retain disabled, cold, and warm timings."
        ),
        "Single-run observations are not project-level support; that requires the fixed three-seed replication artifact.",
    ]
    diagnostics = llm.get("diagnostics")
    if isinstance(diagnostics, Mapping):
        native_continuation = diagnostics.get("native_continuation")
        if isinstance(native_continuation, Mapping):
            diagnostic_metrics = native_continuation["teacher_forced"]
            lines.insert(
                9,
                (
                    "- Native-continuation compressed-prompt diagnostic: "
                    f"`{diagnostic_metrics['mean_kl']:.6f}` KL; mechanism-only and "
                    "excluded from acceptance, claims, and performance"
                ),
            )
    lines.extend(_scalar_findings_lines("Frozen Backbone Raw Findings", llm))
    return lines


def _input_unsupported_lines(
    acceptance: Mapping[str, Any],
) -> list[str]:
    unsupported = [
        name
        for name, passed in (
            ("Known-token compatibility", acceptance["embedding_fit"]),
            (
                "Exact held-out position compression",
                acceptance["density"],
            ),
            ("Candidate-state compactness", acceptance.get("compactness")),
        )
        if passed is False
    ]
    return [
        "",
        "## Unsupported And Not Applicable Findings",
        "",
        "- Unsupported: " + (", ".join(unsupported) if unsupported else "none"),
        "- Not applicable: Output tokenization and output density",
        "",
        "Output tokenization and output density are outside this input-only artifact.",
    ]


def _input_headline_lines(
    result: Mapping[str, Any],
    tokenizer: Mapping[str, Any],
    llm: Mapping[str, Any] | None,
) -> list[str]:
    density = cast(Mapping[str, Any], tokenizer["density"])
    fit = cast(Mapping[str, Any], tokenizer["embedding_fit"])
    fixtures = cast(
        Mapping[str, Any],
        tokenizer.get("raw_byte_fixtures", {}),
    )
    strata = cast(Mapping[str, Any], tokenizer.get("density_strata", {}))
    held_out = cast(Mapping[str, Any], strata.get("wikitext", {}))
    segmented = (
        {}
        if llm is None
        else cast(
            Mapping[str, Any],
            cast(Mapping[str, Any], llm.get("teacher_forced", {})).get(
                "segmented",
                {},
            ),
        )
    )
    generation = {} if llm is None else cast(Mapping[str, Any], llm.get("generation", {}))
    gates = cast(Mapping[str, Any], result.get("gates", {}))
    nll_delta = None if "student_nll" not in segmented or "teacher_nll" not in segmented else float(segmented["student_nll"]) - float(segmented["teacher_nll"])
    return [
        "## Headline Verdict",
        "",
        input_headline_line(),
        (f"- Stored run verdict for usable exact held-out input position compression: **{str(result['scientific_verdict']).upper()}**"),
        f"- Exact held-out byte round-trip operand: `{density.get('round_trip')}`",
        (f"- Stored raw-byte fixture round-trip operands: `{[value.get('round_trip') for value in fixtures.values() if isinstance(value, Mapping)]}`"),
        (
            "- Stored held-out window round-trip operands: "
            f"`{[value.get('empirical_round_trip') for value in held_out.get('windows', ()) if isinstance(value, Mapping)]}`"
        ),
        (f"- Held-out native/continuous position ratio: `{density.get('native_tokens_per_continuous_token')}`"),
        (
            "- Registered behavioral-similarity operands: "
            f"KL `{segmented.get('mean_kl')}`; "
            f"NLL delta `{nll_delta}`; "
            f"top-1 `{segmented.get('top1_agreement')}`; "
            "generation-byte similarity "
            f"`{generation.get('segmented_mean_byte_similarity')}`"
        ),
        f"- Stored behavioral gate verdicts: `{dict(gates)}`",
        (f"- Independent embedding-alignment operand: normalized RMSE `{fit.get('normalized_rmse')}`"),
        "",
        (
            "Exactness and the position ratio are joint operands of position "
            "compression. Embedding alignment is reported independently and does "
            "not determine the headline."
        ),
        "",
    ]


def artifact_report(result: Mapping[str, Any]) -> str:
    if result["mode"] == "output_only":
        return _output_artifact_report(result)
    tokenizer = result["tokenizer"]
    llm = result.get("llm")
    optimizer = result.get("training", {}).get("optimizer", {})
    acceptance = tokenizer["acceptance"]
    reconstruction = tokenizer["embedding_fit"]["reconstruction_fraction"]
    exact = tokenizer["embedding_fit"]["exact_fraction"]
    native_tokens_per_continuous_token = tokenizer["density"]["native_tokens_per_continuous_token"]
    synthetic = tokenizer["model"]["id"].startswith("continuous-tokenizer/synthetic-")
    profile = artifact_profile(result)
    diagnostic = profile == DIAGNOSTIC_PROFILE_NAME and not synthetic
    lines = [
        "# Input-Only Continuous Byte Tokenizer Artifact",
        "",
        *_input_headline_lines(
            result,
            tokenizer,
            cast(Mapping[str, Any], llm) if isinstance(llm, Mapping) else None,
        ),
        *current_design_lines(),
        "",
        "## Run Status",
        "",
        *status_lines(result),
        f"- Experiment: `{result['experiment']['name']}`",
        f"- Model: `{tokenizer['model']['id']}`",
        f"- Revision: `{tokenizer['model']['revision']}`",
        f"- Checkpoint: `{tokenizer['checkpoint']['sha256']}`",
        f"- Tokenizer encoder attention: `{tokenizer['codec']['query_heads']}Q/{tokenizer['codec']['key_value_heads']}KV GQA`",
        f"- Tokenizer optimizer: `{_optimizer_policy(optimizer)}`",
        f"- Tokenizer profile: `{profile or 'unknown'}`",
        (f"- Stored tokenizer component-gate bundle: **{'PASSED' if acceptance['overall'] else 'NOT MET'}**"),
        "",
        "## Canonical Directional Claims",
        "",
        *canonical_claim_lines(artifact_claim_records(result)),
    ]
    if diagnostic:
        lines.append("- This Small-profile run is diagnostic and cannot establish a directional project claim.")
    gates = acceptance_rows(result)
    if gates:
        lines.extend(
            [
                "",
                "## Registered Gates",
                "",
                "| Gate | Measured | Requirement | Verdict |",
                "|---|---:|---:|:---:|",
                *(f"| {row['gate']} | `{row['measured']}` | `{row['operator']} {row['threshold']}` | {_gate_verdict(row['passed'])} |" for row in gates),
            ]
        )
    lines.extend(_vocabulary_lines(optional_mapping(result.get("vocabulary"))))
    lines.extend(_verification_lines(optional_mapping(result.get("verification"))))
    fit = tokenizer["embedding_fit"]
    lines.extend(
        [
            "",
            "## Retrieval Findings",
            "",
            f"- Retrieval queries: `{fit['retrieval_queries']}`",
            f"- Retrieval candidates: `{fit['retrieval_candidates']}`",
            f"- Retrieval top-1 fraction: `{fit['retrieval_top1_fraction']:.6f}`",
            f"- Retrieval top-5 fraction: `{fit['retrieval_top5_fraction']:.6f}`",
        ]
    )
    lines.extend(_input_density_lines(tokenizer))
    lines.extend(
        _scalar_findings_lines(
            "Input Selection Findings",
            optional_mapping(result.get("input_selection")),
        )
    )
    lines.extend(
        _scalar_findings_lines(
            "Distillation Findings",
            optional_mapping(result.get("distillation")),
        )
    )
    lines.extend(
        _scalar_findings_lines(
            "Ablation Findings",
            optional_mapping(result.get("ablations")),
        )
    )
    lines.extend(_input_efficiency_lines(llm))
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            f"- Vocabulary reconstruction: `{reconstruction:.2%}`",
            f"- Source-dtype equal rows: `{exact:.2%}`",
            f"- Native tokens/continuous token: `{native_tokens_per_continuous_token:.4f}x`",
            f"- Candidate/reference state ratio: `{tokenizer['compactness']['candidate_reference_state_ratio']:.4f}x`",
            f"- Reference input-table state: `{tokenizer['compactness']['reference_state_bytes']:,}` bytes",
            f"- Candidate state: `{tokenizer['compactness']['candidate_state_bytes']:,}` bytes",
            f"- Warm encoding-cache tensor payload: `{tokenizer['compactness']['warm_cache_tensor_bytes']:,}` bytes",
            "- Compactness is a state-size comparison; resident reference state remains reported separately.",
            "- Cache memory scope: tensor payload only; whole-process RSS is reported separately",
        ]
    )
    lines.extend(_tokenizer_runtime_lines(tokenizer))
    lines.extend(_training_progress_lines(result.get("training_progress", ())))
    lines.extend(_input_llm_evidence_lines(llm, synthetic=synthetic))
    lines.extend(_input_unsupported_lines(acceptance))
    return "\n".join(lines) + "\n"
