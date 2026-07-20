from __future__ import annotations

import os
from importlib import import_module
from pathlib import Path
from typing import Any, Final

from safetensors import safe_open

from continuous_tokenizer.artifacts.hashing import (
    FileStatIdentity,
    cached_sha256_path,
    file_stat_identity,
)
from continuous_tokenizer.artifacts.store import load_json_object
from continuous_tokenizer.codec.checkpoints import cache_namespace, load_checkpoint
from continuous_tokenizer.codec.input import InputByteCodec
from continuous_tokenizer.input.segmentation import reconstruct, segment_bytes
from continuous_tokenizer.report_dashboard import (
    render_current_design_boundary,
    render_performance_ablation,
    render_project,
    render_replication_evidence,
    render_replication_overview,
    render_report,
    render_run_evidence,
    render_run_overview,
    render_search_overview,
    render_search_trials,
    render_state_budget,
    render_status,
)
from continuous_tokenizer.reporting.discovery import (
    ArtifactRun,
    DeploymentArtifact,
    PerformanceAblationArtifact,
    ProjectArtifact,
    ProspectiveArtifact,
    ReplicationArtifact,
    ReportArtifact,
    SearchArtifact,
    StateBudgetArtifact,
    StudyArtifact,
    artifact_index,
    discover_report_artifacts,
)

DEFAULT_ARTIFACT_ROOT: Final = os.environ.get(
    "CONTINUOUS_TOKENIZER_ARTIFACT_ROOT",
    "results",
)


def _load_streamlit() -> Any:
    try:
        return import_module("streamlit")
    except ModuleNotFoundError as error:
        raise RuntimeError("install the optional UI dependencies with `uv sync --group ui`") from error


st = _load_streamlit()
components = import_module("streamlit.components.v1")


# Streamlit excludes underscore-prefixed parameters from cache keys.
# https://docs.streamlit.io/develop/concepts/architecture/caching#excluding-input-parameters
@st.cache_data
def _load_json_file(
    path: Path,
    identity: FileStatIdentity,
) -> dict[str, Any]:
    del identity
    return dict(load_json_object(path))


@st.cache_data
def _load_text(path: Path, identity: FileStatIdentity) -> str:
    del identity
    return path.read_text(encoding="utf-8")


@st.cache_resource(scope="session")
def _load_tokenizer_checkpoint(
    path: Path,
    identity: FileStatIdentity,
) -> tuple[InputByteCodec, dict[str, Any], str]:
    del identity
    loaded = load_checkpoint(path)
    namespace = cache_namespace(
        str(loaded.metadata["model_revision"]),
        cached_sha256_path(path),
    )
    return loaded.codec, loaded.metadata, namespace


@st.cache_data
def _load_attention_layer(
    path: Path,
    layer: int,
    identity: FileStatIdentity,
) -> list[list[list[float]]]:
    del identity
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(f"layer_{layer:03d}")
    return tensor[0].tolist()


def load_json_file(path: Path) -> dict[str, Any]:
    return _load_json_file(path, file_stat_identity(path))


def load_text(path: Path) -> str:
    return _load_text(path, file_stat_identity(path))


def load_tokenizer_checkpoint(
    path: Path,
) -> tuple[InputByteCodec, dict[str, Any], str]:
    return _load_tokenizer_checkpoint(path, file_stat_identity(path))


def load_attention_layer(path: Path, layer: int) -> list[list[list[float]]]:
    return _load_attention_layer(path, layer, file_stat_identity(path))


def _optional_artifact(directory: Path, name: str) -> dict[str, Any] | None:
    path = directory / name
    return load_json_file(path) if path.is_file() else None


def _result_checkpoint(directory: Path, result: dict[str, Any] | None) -> Path | None:
    tokenizer = None if result is None else result.get("tokenizer")
    checkpoint = None if tokenizer is None else tokenizer.get("checkpoint", {}).get("path")
    if checkpoint:
        path = Path(checkpoint).expanduser()
        if path.is_file():
            return path
        relative = directory / path
        if relative.is_file():
            return relative
    candidates = sorted((directory / "checkpoints").glob("*.pt"))
    return candidates[0] if candidates else None


def selected_bytes() -> bytes | None:
    mode = st.segmented_control("Input", ("Text", "Hex", "File"), default="Text")
    if mode == "Text":
        return st.text_area("UTF-8 text", "hello world").encode()
    if mode == "Hex":
        try:
            return bytes.fromhex(st.text_area("Hexadecimal bytes", "00 ff 41 42"))
        except ValueError as error:
            st.error(str(error))
            return None
    upload = st.file_uploader("Raw byte file")
    return None if upload is None else upload.getvalue()


def render_explore(checkpoint: Path) -> None:
    st.caption(f"Input-tokenizer checkpoint: {checkpoint}")
    data = selected_bytes()
    if not data or not st.button("Segment bytes", type="primary"):
        return
    codec, metadata, namespace = load_tokenizer_checkpoint(checkpoint)
    segmentation = segment_bytes(
        codec,
        data,
        cache=codec.encoding_cache,
        namespace=namespace,
    )
    st.write(metadata)
    st.metric("Bytes per span", f"{len(data) / max(len(segmentation.spans), 1):.3f}")
    st.metric("Exact round-trip", reconstruct(segmentation.spans) == data)
    st.dataframe(
        [
            {
                "index": index,
                "length": len(span.data),
                "hex": span.data.hex(" "),
                "atomic": span.atomic,
            }
            for index, span in enumerate(segmentation.spans)
        ],
        hide_index=True,
        width="stretch",
    )
    st.write("Segmentation statistics", segmentation.stats)


def render_attention(directory: Path) -> None:
    artifact_dir = directory / "attention"
    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.is_file():
        st.info("No attention diagnostic was stored for this input checkpoint.")
        return
    metadata = load_json_file(metadata_path)
    st.warning("Attention is diagnostic only. Its eager backend and materialized weights are not performance-comparable to benchmark runs.")
    mode = st.segmented_control("Input path", ("native", "segmented"), default="native")
    mode_metadata = metadata["modes"][mode]
    columns = st.columns(3)
    columns[0].metric("Positions", mode_metadata["positions"])
    columns[1].metric("Layers", mode_metadata["layers"])
    columns[2].metric("Heads", mode_metadata["heads"])
    html_path = artifact_dir / mode_metadata["html_path"]
    if html_path.is_file() and st.toggle("BertViz model view", value=True):
        components.html(load_text(html_path), height=760, scrolling=True)
    layer, head = st.columns(2)
    selected_layer = layer.selectbox("Heatmap layer", range(mode_metadata["layers"]))
    selected_head = head.selectbox("Heatmap head", range(mode_metadata["heads"]))
    matrix = load_attention_layer(
        artifact_dir / mode_metadata["tensor_path"],
        selected_layer,
    )[selected_head]
    labels = [f"{index}: {label}" for index, label in enumerate(mode_metadata["labels"])]
    values = [
        {"query": query, "key": key, "attention": value} for query, row in zip(labels, matrix, strict=True) for key, value in zip(labels, row, strict=True)
    ]
    st.vega_lite_chart(
        spec={
            "data": {"values": values},
            "mark": "rect",
            "encoding": {
                "x": {"field": "key", "type": "ordinal", "sort": labels},
                "y": {"field": "query", "type": "ordinal", "sort": labels},
                "color": {
                    "field": "attention",
                    "type": "quantitative",
                    "scale": {"scheme": "viridis"},
                },
                "tooltip": ["query", "key", "attention"],
            },
        },
        height=max(320, min(900, len(labels) * 18)),
        width="stretch",
    )


def _render_run(artifact: ArtifactRun) -> None:
    directory = artifact.directory
    result = _optional_artifact(directory, "result.json")
    manifest = _optional_artifact(directory, "manifest-final.json")
    failure = _optional_artifact(directory, "failure.json")
    checkpoint = _result_checkpoint(directory, result) if artifact.mode == "input_only" and result is not None else None
    labels = ["Overview", "Evidence", "Report"]
    if checkpoint is not None:
        labels.extend(("Explore", "Attention"))
    tabs = st.tabs(labels)
    with tabs[0]:
        render_run_overview(st, artifact, result, manifest, failure)
    with tabs[1]:
        render_run_evidence(st, artifact, result)
    with tabs[2]:
        render_report(st, directory)
    if checkpoint is not None:
        with tabs[3]:
            render_explore(checkpoint)
        with tabs[4]:
            render_attention(directory)


def _render_replication(artifact: ReplicationArtifact) -> None:
    replication = load_json_file(artifact.directory / "replication.json")
    overview, evidence, report = st.tabs(("Overview", "Runs and 95% CIs", "Report"))
    with overview:
        render_replication_overview(st, artifact, replication)
    with evidence:
        render_replication_evidence(st, replication)
    with report:
        render_report(st, artifact.directory)


def _render_project(artifact: ProjectArtifact) -> None:
    project = load_json_file(artifact.directory / "project.json")
    overview, report = st.tabs(("Project hypotheses", "Report"))
    with overview:
        render_project(st, artifact, project)
    with report:
        render_report(st, artifact.directory)


def _render_performance_ablation(
    artifact: PerformanceAblationArtifact,
) -> None:
    ablation = load_json_file(
        artifact.directory / "performance-ablation.json",
    )
    evidence, report = st.tabs(("Operational ablation", "Report"))
    with evidence:
        render_performance_ablation(st, artifact, ablation)
    with report:
        render_report(st, artifact.directory)


def _render_search(artifact: SearchArtifact) -> None:
    search = load_json_file(artifact.directory / "search.json")
    overview, trials, report = st.tabs(("Overview", "Trials and failures", "Report"))
    with overview:
        render_search_overview(st, artifact, search)
    with trials:
        render_search_trials(st, search)
    with report:
        render_report(st, artifact.directory)


def _render_prospective(artifact: ProspectiveArtifact) -> None:
    value = load_json_file(artifact.directory / "prospective.json")
    evidence, report = st.tabs(("Non-final prospective evidence", "Report"))
    with evidence:
        st.warning(
            "Prospective smoke, screen, and selection evidence is non-final. It never enters final claim aggregation.",
        )
        render_current_design_boundary(st)
        render_status(st, artifact, artifact.tier)
        st.json(value)
    with report:
        render_report(st, artifact.directory)


def _render_standalone(
    artifact: StudyArtifact | DeploymentArtifact,
    filename: str,
) -> None:
    value = load_json_file(artifact.directory / filename)
    overview, report = st.tabs(("Evidence", "Report"))
    with overview:
        render_current_design_boundary(st)
        render_status(st, artifact, artifact.kind)
        st.json(value)
    with report:
        render_report(st, artifact.directory)


def _render_state_budget(artifact: StateBudgetArtifact) -> None:
    budget = load_json_file(
        artifact.directory / "joint-state-budget.json",
    )
    evidence, report = st.tabs(("Future prerequisite", "Report"))
    with evidence:
        render_state_budget(st, budget)
    with report:
        render_report(st, artifact.directory)


def render_artifact(artifact: ReportArtifact) -> None:
    match artifact.kind:
        case "run":
            _render_run(artifact)
        case "replication":
            _render_replication(artifact)
        case "project":
            _render_project(artifact)
        case "performance_ablation":
            _render_performance_ablation(artifact)
        case "search":
            _render_search(artifact)
        case "prospective":
            _render_prospective(artifact)
        case "study":
            _render_standalone(artifact, "result.json")
        case "deployment":
            _render_standalone(artifact, "deployment.json")
        case "state_budget":
            _render_state_budget(artifact)


def main() -> None:
    st.set_page_config(
        page_title="Continuous Byte Tokenizer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Continuous Byte Tokenizer")
    st.caption("Read-only current-design input-only and output-only research evidence. Unsupported old artifacts are excluded during semantic discovery.")
    root = Path(st.sidebar.text_input("Artifact root", DEFAULT_ARTIFACT_ROOT)).expanduser()
    artifacts = discover_report_artifacts(root)
    if not artifacts:
        st.info("No semantically verified sealed evidence artifacts found.")
        return
    selected = st.sidebar.selectbox(
        "Artifact",
        artifacts,
        index=artifact_index(artifacts, st.query_params.get("artifact")),
        format_func=lambda item: item.label,
    )
    st.sidebar.caption(f"{selected.model}\n\n{selected.directory}")
    render_artifact(selected)


if __name__ == "__main__":
    main()
