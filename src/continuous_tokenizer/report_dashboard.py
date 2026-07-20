from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from continuous_tokenizer.artifacts.evidence import EVIDENCE_MANIFEST_FILENAME
from continuous_tokenizer.artifacts.hashing import (
    FileStatIdentity,
    cached_sha256_path,
    file_stat_identity,
)
from continuous_tokenizer.contracts.claims import (
    CURRENT_DESIGN_NOTICE,
    DEFAULT_PERFORMANCE_LIMITATION,
    INPUT_HEADLINE,
)
from continuous_tokenizer.reporting.artifact_markdown import (
    artifact_claim_records,
)
from continuous_tokenizer.reporting.discovery import (
    ArtifactRun,
    PerformanceAblationArtifact,
    ProjectArtifact,
    ReplicationArtifact,
    ReportArtifact,
    SearchArtifact,
    sealed_artifact_paths,
)
from continuous_tokenizer.reporting.shared import (
    claim_role_groups,
    claim_verdicts,
    display_name,
    input_headline_operands,
    optional_mapping,
)

_INPUT_PERFORMANCE_METRICS = frozenset(
    {
        "prompt_cache_ratio",
        "prefill_flops_ratio",
        "time_to_first_logit_ratio",
        "tokenizer_latency_ratio",
    },
)


@lru_cache(maxsize=256)
def _immutable_text(path: str, identity: FileStatIdentity) -> str:
    del identity
    return Path(path).read_text(encoding="utf-8")


def _sequence(value: object) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str | bytes) else ()


def _mappings(value: object) -> tuple[Mapping[str, Any], ...]:
    return tuple(mapping for item in _sequence(value) if (mapping := optional_mapping(item)) is not None)


def _metric(value: object, format_spec: str = "") -> str:
    if value is None:
        return "N/A"
    return format(value, format_spec) if format_spec else str(value)


def render_status(st: Any, artifact: ReportArtifact, evidence_scope: str) -> None:
    columns = st.columns(4)
    columns[0].metric("Operational status", artifact.operational_status.upper())
    columns[1].metric("Scientific verdict", artifact.scientific_verdict.upper())
    columns[2].metric("Evidence scope", evidence_scope.upper())
    columns[3].metric("Mode", display_name(artifact.mode).upper())
    st.caption("Operational completion and scientific support are independent stored findings. The dashboard does not recompute either verdict.")


def render_current_design_boundary(st: Any) -> None:
    st.warning(CURRENT_DESIGN_NOTICE)
    st.info(DEFAULT_PERFORMANCE_LIMITATION)


def render_performance_ablation(
    st: Any,
    artifact: PerformanceAblationArtifact,
    ablation: Mapping[str, Any],
) -> None:
    st.subheader("Operational performance ablation")
    st.warning(
        "This artifact is operational and secondary evidence only. It cannot promote a final "
        "deployment claim, the primary density-and-behavior headline, or the future joint "
        "state-budget prerequisite.",
    )
    st.info(DEFAULT_PERFORMANCE_LIMITATION)
    st.write("Optimizations", ablation.get("optimization_ids", ()))
    st.write("Exact semantic digest", ablation.get("semantic_sha256"))
    rows = []
    conditions = optional_mapping(ablation.get("conditions")) or {}
    for name, value in conditions.items():
        condition = optional_mapping(value) or {}
        summary = optional_mapping(condition.get("summary")) or {}
        rows.append(
            {
                "condition": name,
                "warmups": condition.get("warmups"),
                "paired repetitions": condition.get("repetitions"),
                "baseline median seconds": summary.get("baseline_median_seconds"),
                "optimized median seconds": summary.get("optimized_median_seconds"),
                "optimized/baseline ratio": summary.get("median_ratio"),
                "95% low": summary.get("confidence_95_low"),
                "95% high": summary.get("confidence_95_high"),
            },
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    render_status(st, artifact, str(ablation.get("evidence_scope", "operational_secondary")))


def render_verification(
    st: Any,
    result: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
) -> None:
    verification = None if result is None else optional_mapping(result.get("verification"))
    if verification is None and manifest is not None:
        verification = optional_mapping(manifest.get("verification"))
    st.subheader("Verification and preflight")
    if not verification or not verification.get("provided"):
        st.info("Preflight verification was not provided.")
        return
    columns = st.columns(2)
    columns[0].metric(
        "Verification checks",
        "PASSED" if verification.get("all_passed") is True else "NOT PASSED",
    )
    columns[1].metric("Source commit", verification.get("source_commit", "N/A"))
    st.caption(f"Source state: {verification.get('source_state_sha256', 'N/A')} · Dependency lock: {verification.get('dependency_lock_sha256', 'N/A')}")
    checks = optional_mapping(verification.get("checks"))
    if checks:
        st.dataframe(
            [
                {
                    "check": name,
                    "passed": check.get("passed"),
                    "return_code": check.get("return_code"),
                    "seconds": check.get("seconds"),
                    "log_sha256": check.get("log_sha256"),
                }
                for name, value in checks.items()
                if (check := optional_mapping(value)) is not None
            ],
            hide_index=True,
            width="stretch",
        )


def render_failure(st: Any, failure: Mapping[str, Any] | None, *, title: str = "Failure") -> None:
    if not failure:
        return
    st.subheader(title)
    st.error(f"{failure.get('type', 'Error')}: {failure.get('message', failure)}")


def render_vocabulary(st: Any, result: Mapping[str, Any]) -> None:
    vocabulary = optional_mapping(result.get("vocabulary"))
    if not vocabulary:
        return
    st.subheader("Vocabulary coverage")
    fields = (
        ("Vocabulary rows", "vocabulary_size"),
        ("Compatibility rows", "compatibility_tokens"),
        ("Duplicate aliases", "duplicate_aliases"),
        ("Ambiguous payloads", "ambiguous_byte_sequences"),
        ("Unavailable rows", "unavailable_rows"),
        ("Structural controls", "control_tokens"),
    )
    st.dataframe(
        [{"finding": label, "stored value": vocabulary.get(name)} for label, name in fields],
        hide_index=True,
        width="stretch",
    )


def _stored_gate_rows(gates: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name, value in gates.items():
        gate = optional_mapping(value)
        rows.append(
            {
                "gate": display_name(name),
                "measured": None if gate is None else gate.get("measured"),
                "requirement": (None if gate is None else f"{gate.get('operator', '')} {gate.get('threshold', '')}".strip()),
                "stored verdict": value if gate is None else gate.get("passed"),
            }
        )
    return rows


def render_gate_findings(st: Any, gates: Mapping[str, Any] | None) -> None:
    if not gates:
        return
    rows = _stored_gate_rows(gates)
    st.subheader("Stored gate findings")
    st.dataframe(rows, hide_index=True, width="stretch")
    unsupported = [row["gate"] for row in rows if row["stored verdict"] is False]
    not_applicable = [row["gate"] for row in rows if row["stored verdict"] is None]
    st.write("Unsupported gates", unsupported or "None")
    st.write("Not-applicable gates", not_applicable or "None")


def render_input_findings(st: Any, result: Mapping[str, Any]) -> None:
    tokenizer = optional_mapping(result.get("tokenizer"))
    if not tokenizer:
        st.info("No completed input-tokenizer findings were stored.")
        return
    fit = optional_mapping(tokenizer.get("embedding_fit")) or {}
    st.subheader("Retrieval findings")
    st.dataframe(
        [
            {"finding": "Queries", "stored value": fit.get("retrieval_queries")},
            {"finding": "Candidates", "stored value": fit.get("retrieval_candidates")},
            {"finding": "Top-1 fraction", "stored value": fit.get("retrieval_top1_fraction")},
            {"finding": "Top-5 fraction", "stored value": fit.get("retrieval_top5_fraction")},
        ],
        hide_index=True,
        width="stretch",
    )
    density_rows = []
    for name in ("density", "native_aligned_segmentation"):
        finding = optional_mapping(tokenizer.get(name))
        if finding:
            density_rows.append({"finding": name, **finding})
    if density_rows:
        st.subheader("Input density findings")
        st.dataframe(density_rows, hide_index=True, width="stretch")
    fixtures = optional_mapping(tokenizer.get("raw_byte_fixtures"))
    if fixtures:
        st.subheader("Raw byte fixtures")
        st.dataframe(
            [{"fixture": name, **finding} for name, value in fixtures.items() if (finding := optional_mapping(value)) is not None],
            hide_index=True,
            width="stretch",
        )
    render_gate_findings(st, optional_mapping(tokenizer.get("gates")))


def render_input_performance(
    st: Any,
    llm: Mapping[str, Any] | None,
) -> None:
    if not llm:
        return
    calibration = optional_mapping(llm.get("calibration"))
    st.subheader("Batch-size-8 evaluation calibration")
    st.caption(
        "Teacher-forced evidence uses the single current batch-size-8 path and is eligible only "
        "after scalar-versus-batched numerical calibration passes for the exact execution identity."
    )
    if calibration:
        st.json(calibration)
    performance = optional_mapping(llm.get("performance"))
    if not performance:
        return
    measurement = optional_mapping(performance.get("measurement"))
    if not measurement:
        return
    st.subheader("Registered input performance measurement")
    st.caption(
        f"{measurement.get('prompt_count', 'N/A')} content-hashed prompts · "
        f"{measurement.get('repetitions', 'N/A')} paired repetitions · "
        f"prompt set {measurement.get('prompt_set_sha256', 'N/A')}"
    )
    rows = []
    for mode in ("native", "compatibility", "segmented"):
        finding = optional_mapping(performance.get(mode))
        if finding:
            rows.append(
                {
                    "mode": mode,
                    "registered prompt-set positions": finding.get("positions"),
                    "registered prompt-set cache bytes": finding.get("materialized_cache_bytes"),
                    "registered prompt-set analytical FLOPs": finding.get("total_analytical_flops"),
                    "paired TTFL median seconds": finding.get("time_to_first_logit_median_seconds"),
                    "paired observations": finding.get("timing_observations"),
                }
            )
    st.dataframe(rows, hide_index=True, width="stretch")


def render_output_findings(st: Any, result: Mapping[str, Any]) -> None:
    output = optional_mapping(result.get("output"))
    if not output:
        st.info("No completed output-tokenizer findings were stored.")
        return
    stop_control = optional_mapping(output.get("stop_control")) or {}
    deployment = optional_mapping(output.get("deployment")) or {}
    columns = st.columns(4)
    columns[0].metric(
        "Direct feedback",
        _metric(output.get("direct_feedback_equality"), ".2%"),
    )
    columns[1].metric(
        "Rollout event agreement",
        _metric(output.get("rollout_event_agreement"), ".2%"),
    )
    columns[2].metric(
        "Native positions ratio",
        _metric(output.get("native_tokens_per_attempted_macro_step"), ".3f"),
    )
    columns[3].metric("Output verdict", str(result["scientific_verdict"]).upper())
    st.subheader("Stop control")
    st.write("Policy", stop_control.get("policy", "Not recorded"))
    st.write("Token IDs", stop_control.get("token_ids", ()))
    st.write("Oracle events", stop_control.get("oracle_events", "Not recorded"))
    st.write("Predicted events", stop_control.get("predicted_events", "Not recorded"))
    if control_evidence := optional_mapping(output.get("control_evidence")):
        st.subheader("Control evidence")
        st.dataframe(
            [{"finding": display_name(name), "stored value": str(value)} for name, value in control_evidence.items()],
            hide_index=True,
            width="stretch",
        )
    st.subheader("Deployment")
    st.dataframe(
        [{"finding": display_name(name), "stored value": str(value)} for name, value in deployment.items()],
        hide_index=True,
        width="stretch",
    )
    omission = deployment.get("physical_output_head_omission")
    if isinstance(omission, str) and omission.startswith("not_applicable"):
        st.info(f"Physical output-head omission: {omission.replace('_', ' ')}")
    render_gate_findings(st, optional_mapping(result.get("gates")))


def render_run_overview(
    st: Any,
    artifact: ArtifactRun,
    result: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    failure: Mapping[str, Any] | None,
) -> None:
    st.subheader("Headline verdict")
    st.metric("Stored run scientific verdict", artifact.scientific_verdict.upper())
    st.caption("This is the stored verdict. The dashboard does not recompute acceptance.")
    render_current_design_boundary(st)
    render_status(st, artifact, artifact.evidence_scope)
    render_failure(st, failure)
    if result:
        st.subheader("Canonical directional claims")
        _render_claim_groups(
            st,
            artifact_claim_records(result),
            evidence_manifest_sha256=cached_sha256_path(
                artifact.directory / "manifest-final.json",
            ),
        )
        render_vocabulary(st, result)
        if artifact.mode == "output_only":
            render_output_findings(st, result)
        else:
            tokenizer = optional_mapping(result.get("tokenizer"))
            acceptance = None if tokenizer is None else optional_mapping(tokenizer.get("acceptance"))
            if acceptance:
                st.subheader("Stored input verdicts")
                st.dataframe(
                    [{"finding": display_name(name), "stored verdict": value} for name, value in acceptance.items()],
                    hide_index=True,
                    width="stretch",
                )
    render_verification(st, result, manifest)


def render_run_evidence(st: Any, artifact: ArtifactRun, result: Mapping[str, Any] | None) -> None:
    if not result:
        st.info("The run failed before a result artifact was stored.")
        return
    if artifact.mode == "output_only":
        render_output_findings(st, result)
    else:
        render_input_findings(st, result)
        render_input_performance(st, optional_mapping(result.get("llm")))
    for name in ("distillation", "ablations", "llm"):
        finding = optional_mapping(result.get(name))
        if finding:
            with st.expander(f"{display_name(name)} raw findings"):
                st.json(finding)


def _aggregate_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "metric": display_name(name),
            "count": metric.get("count"),
            "mean": metric.get("mean"),
            "minimum": metric.get("minimum"),
            "maximum": metric.get("maximum"),
            "95% CI low": metric.get("confidence_95_low"),
            "95% CI high": metric.get("confidence_95_high"),
            "raw per-seed values": metric.get("raw_values"),
        }
        for name, value in metrics.items()
        if (metric := optional_mapping(value)) is not None
    ]


def _claim_rows(
    values: object,
    *,
    evidence_manifest_sha256: str,
) -> list[dict[str, Any]]:
    return [
        {
            "claim ID": claim.get("claim_id"),
            "label": claim.get("label"),
            "role": claim.get("role"),
            "category": claim.get("category"),
            "basis": claim.get("basis"),
            "applicability": claim.get("applicability"),
            "gate or policy": claim.get("gate_or_policy"),
            "producer": claim.get("producer_symbol"),
            "evidence pointers": claim.get("evidence_pointers"),
            "evidence manifest SHA-256": evidence_manifest_sha256,
            "denominator / sample context": claim.get("denominator_context"),
            "verdict": claim.get("verdict"),
            "reason": claim.get("reason"),
        }
        for claim in _mappings(values)
    ]


def _render_claim_groups(
    st: Any,
    values: object,
    *,
    evidence_manifest_sha256: str,
) -> None:
    claims = _mappings(values)
    for role, records in claim_role_groups(claims):
        st.markdown(f"**{display_name(role)} claims**")
        st.dataframe(
            _claim_rows(
                records,
                evidence_manifest_sha256=evidence_manifest_sha256,
            ),
            hide_index=True,
            width="stretch",
        )


def _statement_rows(values: object, kind: str) -> list[dict[str, Any]]:
    return [
        {
            "paper label": trace.get("paper_label"),
            "statement / contract ID": trace.get("statement_id"),
            "implementation symbols": trace.get("implementation_symbols"),
            "validating test IDs": trace.get("validating_test_ids"),
            "canonical artifact / JSON pointers": trace.get("canonical_artifact_pointers"),
            "model / seed denominator": trace.get("model_seed_denominator"),
            "verdict": trace.get("verdict"),
            "reason": trace.get("reason"),
        }
        for trace in _mappings(values)
        if trace.get("kind") == kind
    ]


def _project_claim_trace_rows(
    values: object,
    *,
    cross_model: bool,
) -> list[dict[str, Any]]:
    return [
        {
            "paper label": trace.get("paper_label"),
            "claim ID": trace.get("claim_id"),
            "evidence class": trace.get("evidence_class"),
            "producer": trace.get("producer_symbol"),
            "canonical artifact / JSON pointers": trace.get("canonical_artifact_pointers"),
            "parent model pointers and seeds": trace.get("parent_model_evidence"),
            "model / seed denominator": trace.get("model_seed_denominator"),
            "verdict": trace.get("verdict"),
            "reason": trace.get("reason"),
        }
        for trace in _mappings(values)
        if (str(trace.get("claim_id", "")).endswith("cross_model_confirmation") == cross_model)
    ]


def _verdict_rows(verdicts: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"category": display_name(name), "stored verdict": verdict} for name, verdict in verdicts.items()]


def render_replication_overview(
    st: Any,
    artifact: ReplicationArtifact,
    replication: Mapping[str, Any],
) -> None:
    st.subheader("Headline verdict")
    st.metric(
        "Stored replication scientific verdict",
        artifact.scientific_verdict.upper(),
    )
    render_current_design_boundary(st)
    if artifact.mode == "input_only":
        st.caption(INPUT_HEADLINE)
        verdicts = claim_verdicts(_mappings(replication.get("claims")))
        for label, claim_id in (
            (
                "Exact held-out position compression",
                "input.held_out_position_compression",
            ),
            (
                "Registered behavioral similarity",
                "input.registered_behavioral_similarity_tolerances",
            ),
            (
                "Independent full-vocabulary alignment",
                "input.full_vocabulary_embedding_compatibility",
            ),
        ):
            st.write(label, verdicts.get(claim_id, "not recorded"))
    render_status(st, artifact, str(replication["evidence_scope"]))
    claims = _sequence(replication.get("claims"))
    if claims:
        st.subheader("Claims")
        _render_claim_groups(
            st,
            claims,
            evidence_manifest_sha256=cached_sha256_path(
                artifact.directory / EVIDENCE_MANIFEST_FILENAME,
            ),
        )
    category_verdicts = optional_mapping(replication.get("category_verdicts"))
    if category_verdicts:
        st.subheader("Independent category verdicts")
        st.dataframe(
            _verdict_rows(category_verdicts),
            hide_index=True,
            width="stretch",
        )
    render_verification(st, replication, None)


def render_replication_evidence(st: Any, replication: Mapping[str, Any]) -> None:
    runs = []
    for run in _mappings(replication.get("runs")):
        row = {
            "seed": run.get("seed"),
            "operational status": run.get("operational_status"),
            "scientific verdict": run.get("scientific_verdict"),
        }
        metrics = optional_mapping(run.get("metrics")) or {}
        if replication["mode"] == "output_only":
            row["direct feedback"] = metrics.get("direct_feedback_equality")
            row["native tokens per attempt"] = metrics.get(
                "native_tokens_per_attempted_macro_step",
            )
        else:
            row.update(
                {
                    "exact held-out bytes": metrics.get("density_exact"),
                    "native tokens/continuous token": metrics.get("native_tokens_per_continuous_token"),
                    "candidate/reference state ratio": metrics.get(
                        "candidate_reference_state_ratio",
                    ),
                    "normalized RMSE": metrics.get("normalized_rmse"),
                    "segmented mean KL": metrics.get("segmented_mean_kl"),
                    "segmented NLL delta": metrics.get("segmented_nll_delta"),
                    "segmented top-1 agreement": metrics.get("segmented_top1_agreement"),
                    "generation-byte similarity": metrics.get("segmented_generation_byte_similarity"),
                }
            )
        runs.append(row)
    runs.extend(
        {
            "seed": failed.get("seed"),
            "operational status": failed.get("operational_status"),
            "scientific verdict": failed.get("scientific_verdict"),
            "failure": failed.get("failure"),
        }
        for failed in _mappings(replication.get("failed_runs"))
    )
    st.subheader("Per-seed runs")
    st.dataframe(runs, hide_index=True, width="stretch")
    metrics = optional_mapping(replication.get("metrics")) or {}
    primary_metrics = {name: value for name, value in metrics.items() if name not in _INPUT_PERFORMANCE_METRICS}
    performance_metrics = {name: value for name, value in metrics.items() if name in _INPUT_PERFORMANCE_METRICS}
    primary_rows = _aggregate_rows(primary_metrics)
    if primary_rows:
        st.subheader("Primary and independent metrics with 95% intervals")
        st.dataframe(primary_rows, hide_index=True, width="stretch")
    performance_rows = _aggregate_rows(performance_metrics)
    if performance_rows:
        st.subheader("Secondary final-only performance")
        st.caption(
            "Ratios are descriptive unless the corresponding canonical claim is supported. "
            "Research throughput is operational only. Runtime speed does not support the future state budget.",
        )
        st.dataframe(performance_rows, hide_index=True, width="stretch")


def render_project(
    st: Any,
    artifact: ProjectArtifact,
    project: Mapping[str, Any],
) -> None:
    st.subheader("Headline verdict")
    st.metric(
        "Stored project scientific verdict",
        artifact.scientific_verdict.upper(),
    )
    render_current_design_boundary(st)
    if artifact.mode == "input_only":
        st.caption(INPUT_HEADLINE)
        verdicts = claim_verdicts(_mappings(project.get("claims")))
        columns = st.columns(3)
        for column, (label, claim_id) in zip(
            columns,
            (
                (
                    "Exact position compression",
                    "input.held_out_position_compression",
                ),
                (
                    "Behavioral similarity",
                    "input.registered_behavioral_similarity_tolerances",
                ),
                (
                    "Independent alignment",
                    "input.full_vocabulary_embedding_compatibility",
                ),
            ),
            strict=True,
        ):
            column.metric(
                label,
                verdicts.get(claim_id, "not recorded").upper(),
            )
        headline_rows = []
        for model in _mappings(project.get("models")):
            identity = optional_mapping(model.get("model")) or {}
            replication = optional_mapping(model.get("replication")) or {}
            headline_rows.extend(
                {
                    "model": identity.get("id"),
                    **input_headline_operands(run),
                }
                for run in _mappings(replication.get("runs"))
            )
        st.dataframe(
            headline_rows,
            hide_index=True,
            width="stretch",
        )
        st.caption("Stored operands only; alignment remains independent and the dashboard does not recompute acceptance.")
    render_status(st, artifact, str(project["evidence_scope"]))
    statement_traces = project.get("statement_traces")
    claim_traces = project.get("claim_traces")
    st.subheader("Protocol proofs")
    st.dataframe(
        _statement_rows(statement_traces, "protocol"),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Software validation")
    st.dataframe(
        _statement_rows(statement_traces, "software"),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Per-model empirical support")
    st.caption("Favorable search, prospective selection, and diagnostic values are non-final and cannot populate final empirical verdicts.")
    st.dataframe(
        _project_claim_trace_rows(claim_traces, cross_model=False),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Cross-model confirmation")
    st.dataframe(
        _project_claim_trace_rows(claim_traces, cross_model=True),
        hide_index=True,
        width="stretch",
    )
    st.subheader("Project claims")
    _render_claim_groups(
        st,
        project.get("claims"),
        evidence_manifest_sha256=cached_sha256_path(
            artifact.directory / EVIDENCE_MANIFEST_FILENAME,
        ),
    )
    category_verdicts = optional_mapping(project.get("category_verdicts")) or {}
    verdicts = {
        **category_verdicts,
        "cross model": project.get("cross_model_verdict", "not recorded"),
    }
    st.subheader("Independent category verdicts")
    st.dataframe(
        _verdict_rows(verdicts),
        hide_index=True,
        width="stretch",
    )
    models = _mappings(project.get("models"))
    if models:
        st.subheader("Equal primary model replications")
        st.dataframe(
            [
                {
                    "model": (optional_mapping(entry.get("model")) or {}).get("id"),
                    "revision": (optional_mapping(entry.get("model")) or {}).get("revision"),
                    "quality verdict": entry.get("quality_verdict"),
                    "replication complete": (optional_mapping(entry.get("replication")) or {}).get(
                        "replication_complete",
                    ),
                }
                for entry in models
            ],
            hide_index=True,
            width="stretch",
        )
    render_verification(st, project, None)
    for entry in models:
        replication = optional_mapping(entry.get("replication"))
        if replication is not None:
            render_replication_evidence(st, replication)


def render_state_budget(
    st: Any,
    budget: Mapping[str, Any],
) -> None:
    st.subheader("Future prerequisite verdict")
    st.warning(
        "Joint ordinary-vocabulary tensor-state arithmetic only. This is not memory evidence and does not demonstrate physical removal or a combined runtime."
    )
    config = optional_mapping(budget.get("config")) or {}
    columns = st.columns(3)
    columns[0].metric(
        "Stored verdict",
        str(budget["verdict"]).upper(),
    )
    columns[1].metric(
        "Worst-case ratio",
        _metric(budget.get("worst_case_ratio"), ".6f"),
    )
    columns[2].metric(
        "Registered maximum",
        _metric(config.get("maximum_ratio"), ".6f"),
    )
    st.subheader("Mandatory non-claims")
    non_claims = optional_mapping(budget.get("non_claims")) or {}
    st.dataframe(
        [
            {
                "non-claim flag": name,
                "stored value": value,
            }
            for name, value in non_claims.items()
        ],
        hide_index=True,
        width="stretch",
    )
    st.subheader("Per-model, per-seed tensor arithmetic")
    rows = []
    for row in _mappings(budget.get("per_seed")):
        arithmetic = optional_mapping(row.get("arithmetic")) or {}
        rows.append(
            {
                "model": row.get("model_id"),
                "seed": row.get("seed"),
                "tied vocabulary": row.get("tie_word_embeddings"),
                "input codec bytes": arithmetic.get("input_codec_bytes"),
                "output codec bytes": arithmetic.get("output_codec_bytes"),
                "atomic byte-row bytes": arithmetic.get("atomic_byte_rows_bytes"),
                "shared control-ID bytes": arithmetic.get("shared_control_id_bytes"),
                "shared control-row bytes": arithmetic.get("shared_control_row_bytes"),
                "candidate total bytes": arithmetic.get("candidate_tensor_state_bytes"),
                "reference input bytes": arithmetic.get("reference_input_table_bytes"),
                "reference output bytes": arithmetic.get("reference_output_head_bytes"),
                "deduplicated reference bytes": arithmetic.get("reference_tensor_state_bytes"),
                "ratio": row.get("ratio"),
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    st.caption(
        "The artifact stores tied-reference and shared-control deduplication "
        "policies. Streamlit displays the semantically verified arithmetic "
        "without deriving a verdict."
    )


def render_search_overview(
    st: Any,
    artifact: SearchArtifact,
    search: Mapping[str, Any],
) -> None:
    render_status(st, artifact, str(search["evidence_scope"]))
    st.warning("Search evidence is pilot-only and is never final model evidence.")
    selected_number = search.get("selected_trial")
    selected = next(
        (trial for value in _sequence(search.get("trials")) if (trial := optional_mapping(value)) is not None and trial.get("number") == selected_number),
        None,
    )
    metrics = {} if selected is None else (optional_mapping(selected.get("metrics")) or {})
    output_search = artifact.mode == "output_only"
    metric_name = "Selected direct feedback" if output_search else "Selected NRMSE"
    metric_key = "direct_feedback_equality" if output_search else "normalized_rmse"
    metric_value = metrics.get(metric_key)
    gate_name = "selected_output_passed" if output_search else "selected_alignment_passed"
    gate_value = search.get(gate_name)
    if gate_value is None:
        gate_status = "PENDING"
    elif gate_value:
        gate_status = "PASS"
    else:
        gate_status = "NOT MET"
    columns = st.columns(4)
    columns[0].metric(
        "Finished trials",
        f"{search.get('finished_trials', 0)} / {search.get('requested_trials', 0)}",
    )
    columns[1].metric("Selected trial", "N/A" if selected_number is None else selected_number)
    columns[2].metric(metric_name, _metric(metric_value, ".6f"))
    columns[3].metric("Output gates" if output_search else "Alignment gate", gate_status)
    failed_trials = search.get("failed_trials", 0)
    if failed_trials:
        st.caption(f"Failed or pruned attempts retained: {failed_trials}.")
    render_failure(st, optional_mapping(search.get("failure")), title="Search failure")
    render_verification(st, search, None)


def render_search_trials(st: Any, search: Mapping[str, Any]) -> None:
    st.subheader("All search attempts")
    st.dataframe(
        [
            {
                "trial": trial.get("number"),
                "state": trial.get("state"),
                **(optional_mapping(trial.get("parameters")) or {}),
                **(optional_mapping(trial.get("metrics")) or {}),
                "stored gates": trial.get("gates"),
                "failure or prune reason": trial.get("failure"),
            }
            for value in _sequence(search.get("trials"))
            if (trial := optional_mapping(value)) is not None
        ],
        hide_index=True,
        width="stretch",
    )
    st.subheader("Stored selection")
    st.json(
        {
            name: search.get(name)
            for name in (
                "selected_trial",
                "selected_parameters",
                "selected_metrics",
                "selected_gates",
                "selected_alignment_passed",
                "selected_compactness_passed",
                "selected_output_passed",
            )
            if name in search
        }
    )


def render_report(st: Any, directory: Path) -> None:
    sealed = sealed_artifact_paths(directory)
    for name in (
        "joint-state-budget-report.md",
        "project-report.md",
        "replication-report.md",
        "search-report.md",
        "study-report.md",
        "artifact-report.md",
    ):
        path = directory / name
        if path.is_file() and path.resolve() in sealed:
            st.markdown(
                _immutable_text(
                    str(path.resolve()),
                    file_stat_identity(path),
                ),
            )
            return
    st.info("No stored Markdown report was found.")
