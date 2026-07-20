# Continuous Byte Tokenizers for Frozen Language Models

## Abstract

This repository tests two independent replacements for a frozen language model's vocabulary
interfaces. The input tokenizer maps variable-length byte spans to latent vectors consumed through
`inputs_embeds`. The output tokenizer maps hidden states directly to non-empty byte spans or
structural controls without invoking the native vocabulary head. Both systems preserve a frozen
backbone.

The input protocol accepts a span only after exact local reconstruction. An exact atomic-byte path
remains available for every payload. These properties guarantee representability, not compression
or model behavior. Those are empirical questions. No completed fixed-seed Qwen or Gemma artifact
currently supports the real-model hypotheses.

## Research question and contributions

Can a local continuous tokenizer reduce global sequence positions around a frozen language model
while preserving exact bytes and registered model behavior?

The paper makes four contributions:

1. A reconstruction-gated input tokenizer with exhaustive candidate evaluation and lossless
   atomic-byte fallback.
2. An independent output tokenizer that bypasses the native vocabulary head and measures exact
   native feedback.
3. Executable evidence contracts that separate protocol proofs, software validation, prospective
   studies, final model evidence, and operational measurements.
4. A future joint tensor-state hypothesis that is explicitly separate from runtime speed and
   measured memory.

Terminology is fixed. A **byte span** is discrete. A **continuous token** is one accepted ordinary
byte span occupying one model position. Its model-width representation is a **latent vector**. A
**native token** is a discrete vocabulary ID. A **tokenizer** is one complete directional system.
A **codec** is only its reversible neural byte-span component.

## Method

### Input-only tokenizer

At each byte offset, the tokenizer evaluates every candidate length independently. It accepts only
candidates whose deterministic decoder reproduces every byte and terminates at the expected private
`CODEC_EOS`. It selects the longest valid span. A failed length does not prune a longer length.
When no multi-byte candidate is valid, one exact atomic byte advances the input.

Structural controls bypass the codec and use copied frozen input rows. Ordinary accepted spans
become latent vectors. The frozen backbone receives those vectors through `inputs_embeds`.

Training is sequential. Vocabulary fitting first selects one complete encoder checkpoint. The
selected encoder is frozen while its decoder learns exact reconstruction from actual latents.
Current default input campaigns stop after vocabulary fitting and one reconstruction stage.
Frozen-backbone distillation remains an explicit diagnostic/study operation only; it is not part of
the default final design. The backbone, native head, source rows, copied byte rows, and control rows
remain frozen.

### Output-only tokenizer

The output codec predicts one non-empty byte span or structural control from a frozen hidden state.
Training targets come from deterministic greedy native-head trajectories. Emitted bytes are fed
back through deterministic native-token segmentation. Evaluation compares feedback bytes and native
token IDs directly. Output-only mode retains native input feedback and is not a combined tokenizer.

## Formal guarantees

Three statements follow from construction:

1. Every accepted multi-byte span reconstructs its original bytes and the expected `CODEC_EOS`.
2. Every finite byte sequence remains representable because each atomic byte has an exact fallback.
3. Greedy selection is locally maximal because every registered candidate length is evaluated before
   choosing the longest valid span.

These statements do not prove embedding equality, density, behavioral similarity, latency, compute
reduction, state removal, or memory reduction. Encoding caches may change latency only. They must
not change validity, segmentation, or acceptance.

## Empirical hypotheses

The primary input headline is defined once:

> **usable input compression = exact held-out position compression + registered behavioral similarity**

Exact held-out position compression requires exact held-out round-trip and at least the registered
native-to-continuous position ratio. Behavioral similarity requires the registered KL, NLL-delta,
top-1, and generation-byte tolerances. It is comparative similarity, not non-inferiority.
Performance cannot substitute for either operand.

Full-vocabulary embedding compatibility is an independent secondary claim. Approximate embedding fit
is not lossless table compression.

The output hypothesis asks whether direct feedback remains exact, every event is valid and non-empty,
rollout fidelity passes, and one macro-step represents more than one native position on held-out
text. Native-head bypass, codec size, and physical omission remain separate findings.

Performance evidence has three classes:

- Research throughput is operational evidence only.
- Tokenizer latency, direct end-to-end time to first logit, prompt-cache bytes, and compute are
  secondary deployment claims eligible only from complete final artifacts.
- Joint tensor-state budget arithmetic is a future prerequisite. Runtime or cache speedups do not
  support it.

Performance support requires the complete corrected condition matrix, raw observations, registered
warmups and repetitions, exact semantic digests, correct denominators, uncontaminated execution, and
all registered gates. Missing or contaminated evidence is incomplete. A measured regression is
unsupported. Optimization ablations remain operational and secondary; they cannot promote a final
claim by themselves.

## Experiment design

[`Qwen/Qwen3.5-0.8B`](https://huggingface.co/Qwen/Qwen3.5-0.8B) and
[`google/gemma-3-270m-it`](https://huggingface.co/google/gemma-3-270m-it) are equal primary models.
Each cross-model result requires independent Large-profile final replications at seeds `17`, `23`,
and `41` for both models. The pinned WikiText revision supplies held-out text.

Small-profile work, searches, mechanism smoke tests, feasibility screens, and candidate selections
are non-final. They never enter final claim aggregation. Final artifacts bind source state,
dependencies, model and data revisions, checkpoint identity, workload identity, raw observations,
verification inventory, and artifact hashes.

Operational completion and scientific verdict are distinct. A completed run may support,
leave incomplete, or refute a claim. Failed execution is not scientific evidence.

### Current reduced-work design

There is one current design, with no compatibility or migration layer:

- Small: local width `128`, one encoder layer, one decoder layer, four query heads, two key/value
  heads, feed-forward width `256`, and a four-times output projection.
- Large: local width `256`, two encoder layers, one decoder layer, four query heads, two key/value
  heads, feed-forward width `512`, and an eight-times output projection.
- Dynamic input segmentation evaluates candidates up to `32` bytes. Output training and oracle
  evaluation use the exact span ladder `1, 2, 4, 8`, with maximum output span `8`.
- Final input defaults use vocabulary plus reconstruction only: `10` vocabulary epochs, patience
  `2`, evaluation every `2` epochs, one reconstruction epoch over `2,048` samples, at most `512`
  corpus rows, and `32`-row cache chunks.
- Final input evaluation uses batch size `8`, `16` teacher-forced samples, `2` generation samples,
  `128` retrieval queries, and at most `2,048` held-out bytes. Batched evaluation must pass the
  registered scalar-versus-batched numerical calibration before its evidence is accepted.
- Final output defaults use `2` epochs, at most `512` corpus rows, `16` evaluation samples, `16`
  macro-steps, and `512` emitted bytes. The registered fidelity set contains `2` prompts.
- Feasibility uses `256` vocabulary rows, one vocabulary epoch, one reconstruction epoch over `64`
  samples, `256` validation bytes, `2` behavior samples, and no distillation or generation.
  Selection uses `512` rows, two vocabulary epochs, patience `1`, one alignment candidate, and two
  efficiency candidates.
- Searches use at most `2` trials and `512` vocabulary rows. Alignment feasibility uses seed `17`
  and subsets `128`, `256`, and `512`. Compression feasibility uses seed `17`, candidate lengths
  `2`, `8`, and `32`, two binary samples per length, and at most `512` validation bytes.

These reductions lower research power by reducing stages and denominators. Old artifacts are
unsupported, and current runs are not comparable to prior configurations. No new real-model result
has been produced by this redesign.

Default final runs use `1` warmup and `2` repetitions to reduce execution time. The scientific
performance floors remain `5` warmups and `20` repetitions. Therefore tokenizer-latency and direct
time-to-first-logit claims from default runs are incomplete unless an explicit full-performance run
is produced. Faster or smaller execution is not deployment-speed evidence and never substitutes for
either operand of usable input compression.

## Current results

The deterministic synthetic campaigns provide software evidence for exact protocol execution,
artifact production, and dashboard consumption. They provide no evidence for the real-model
hypotheses.

No completed three-seed Qwen or Gemma final replication is present. All real-model claims therefore
remain incomplete rather than failed. The prospective smoke, screen, and selection artifacts are
explicitly non-final.

No new real-model results exist for the current reduced-work design. Old artifacts fail current
semantic verification and are neither migrated nor displayed as current evidence.

The joint tensor-state artifact, when present, reports sealed arithmetic and five mandatory
non-claims: no combined runtime, no continuous feedback, no physical omission, no resident-memory
reduction, and no peak-memory reduction.

<!-- BEGIN GENERATED EVIDENCE LEDGER -->
## Generated evidence ledger

> [!IMPORTANT]
> This section is generated by `tokenizer readme`. Do not edit it by hand.
> Only semantically verified, sealed project artifacts can supply empirical verdicts.
> Missing final evidence is `INCOMPLETE`; it is never interpreted as zero or unsupported.

### Protocol proofs

`PROVED` is reserved for construction consequences. `VALIDATED` requires source-bound verification and both registered synthetic campaigns. Empirical claims remain separate.

- `protocol.accepted_span_exactness` — **Proposition 1 — accepted-span reconstruction is exact** — `protocol` — Status: **PROVED**
  - Statement: Every accepted multi-byte discrete byte span decodes to its exact payload and terminates at the expected private CODEC_EOS position.
  - Evidence: not required (construction proof); implementation `continuous_tokenizer.codec.input.InputByteCodec.reconstruction_matches`, `continuous_tokenizer.input.segmentation.validate_spans`; tests `test_codec_core.test_private_eos_terminates_payload_and_rejects_invalid_frames`, `test_input_evidence.test_independent_decoder_bytes_and_exact_eos_are_empirical_evidence`
  - Reason: proved by the protocol construction stated in the evidence requirement
- `protocol.atomic_fallback` — **Proposition 2 — arbitrary finite byte strings remain representable** — `protocol` — Status: **PROVED**
  - Statement: Every byte has an exact atomic latent vector fallback, so any finite byte sequence remains representable as continuous tokens.
  - Evidence: not required (construction proof); implementation `continuous_tokenizer.input.segmentation.greedy_segment`, `continuous_tokenizer.input.segmentation.reconstruct`; tests `test_input_segmentation.test_atomic_fallback_round_trips_all_byte_values`, `test_input_alignment.test_arbitrary_bytes_round_trip_through_atomic_fallback`
  - Reason: proved by the protocol construction stated in the evidence requirement
- `protocol.exhaustive_local_maximality` — **Proposition 3 — exhaustive greedy selection is locally maximal** — `protocol` — Status: **PROVED**
  - Statement: At each offset, every permitted candidate length is evaluated independently and the longest valid discrete byte span is selected.
  - Evidence: not required (construction proof); implementation `continuous_tokenizer.input.segmentation.segment_bytes`, `continuous_tokenizer.input.segmentation.greedy_segment`; tests `test_input_segmentation.test_longer_valid_span_survives_shorter_invalid_span`, `test_input_segmentation.test_longest_of_all_valid_candidates_is_selected`
  - Reason: proved by the protocol construction stated in the evidence requirement
### Software validation

- `software.cache_semantics` — **Software statement 4 — cache state cannot alter semantics** — `software` — Status: **NOT VALIDATED**
  - Statement: Encoding-cache state may change latency but cannot change accepted spans, continuous-token segmentation, or latent vectors.
  - Evidence: not supplied; implementation `continuous_tokenizer.codec.encoding_cache.EncodingCache`, `continuous_tokenizer.input.segmentation.segment_bytes`; tests `test_input_segmentation.test_cache_does_not_change_segmentation`
  - Reason: no source-bound verification and synthetic campaign inputs were supplied
- `software.backbone_immutability` — **Software statement 5 — backbone immutability is auditable** — `software` — Status: **NOT VALIDATED**
  - Statement: Input distillation and output-codec training freeze the language-model backbone and reject any parameter-fingerprint change.
  - Evidence: not supplied; implementation `continuous_tokenizer.input.training.distillation.FrozenBackboneDistiller.run`, `continuous_tokenizer.output.training.OutputCodecTrainer.run`, `continuous_tokenizer.runtime.tensors.parameter_fingerprint`; tests `test_input_distillation.test_distillation_trains_only_codec_parameters`, `test_output_training.OutputModeTests.test_output_training_preserves_frozen_backbone`
  - Reason: no source-bound verification and synthetic campaign inputs were supplied
- `software.input_path` — **Evidence ladder — deterministic input-only software path** — `software` — Status: **NOT VALIDATED**
  - Statement: The input-only path executes reconstruction-gated segmentation, continuous-token backbone input, and artifact publication in the deterministic offline campaign.
  - Evidence: not supplied; implementation `continuous_tokenizer.campaigns.input.InputExperimentRunner.run`; tests `test_input_campaign.test_synthetic_spec_runs_complete_offline_artifact`
  - Reason: no source-bound verification and synthetic campaign inputs were supplied
- `software.output_path` — **Evidence ladder — deterministic output-only software path** — `software` — Status: **NOT VALIDATED**
  - Statement: The output-only path emits non-empty byte spans or structural controls, performs deterministic native-token feedback, bypasses the native head, and publishes artifacts.
  - Evidence: not supplied; implementation `continuous_tokenizer.campaigns.output.OutputExperimentRunner.run`; tests `test_output_campaign.OutputModeTests.test_synthetic_output_campaign_proves_end_to_end_path`
  - Reason: no source-bound verification and synthetic campaign inputs were supplied
### Empirical claim ledger

### Input-only claims

**No final semantically verified sealed project evidence was supplied.**

#### Primary claims

- `input.held_out_position_compression` — **Held Out Position Compression** — `primary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: minimum_native_tokens_per_continuous_token >= 1.10
  - Evidence: `not available (no sealed project supplied)`; raw records `tokenizer-metrics.json#/density/native_tokens_per_continuous_token`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: at most 2,048 held-out WikiText bytes in every completed Large-profile final seed. no final semantically verified sealed project artifact was supplied
#### Prerequisite claims

- `input.registered_behavioral_similarity_tolerances` — **Registered Behavioral Similarity Tolerances** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: registered segmented KL, NLL delta, top-1, and generation-byte similarity tolerances all pass; this is comparative similarity, not non-inferiority
  - Evidence: `not available (no sealed project supplied)`; raw records `llm-metrics.json#/teacher_forced/segmented`, `llm-metrics.json#/generation`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: 16 teacher-forced samples and 2 generation samples in every completed Large-profile full-model seed, after batch-size-8 numerical calibration. no final semantically verified sealed project artifact was supplied
#### Secondary claims

- `input.fixed_subset_alignment_feasibility` — **Fixed Subset Alignment Feasibility** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: every staged Large-profile subset passes the registered normalized-RMSE and cosine gates under the sealed continuation rule
  - Evidence: `not available (no sealed project supplied)`; raw records `result.json#/stages`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: 128-, 256-, and 512-row deterministic vocabulary subsets at seed 17 in one sealed Large-profile study per primary model. no final semantically verified sealed project artifact was supplied
- `input.full_vocabulary_embedding_compatibility` — **Full Vocabulary Embedding Compatibility** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: registered normalized-RMSE and cosine gates all pass
  - Evidence: `not available (no sealed project supplied)`; raw records `tokenizer-metrics.json#/embedding_fit`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: all reachable canonical ordinary-token rows in every completed Large-profile final seed. no final semantically verified sealed project artifact was supplied
- `input.tokenizer_latency_improvement` — **Tokenizer Latency Improvement** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: registered warm-cache tokenizer latency is lower than disabled-cache latency after exact semantic-digest equality, raw-observation, denominator, and contamination checks pass
  - Evidence: `not available (no sealed project supplied)`; raw records `tokenizer-metrics.json#/segmentation_runs`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: disabled, cold, and warm encoding-cache conditions with at least 5 warmups and 20 raw repetitions in every completed final seed; the 1/2 default is incomplete. no final semantically verified sealed project artifact was supplied
- `input.prompt_cache_reduction` — **Prompt Cache Reduction** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: segmented/native materialized prompt-cache byte ratio < 1 after semantic, denominator, raw-observation, and contamination checks pass
  - Evidence: `not available (no sealed project supplied)`; raw records `llm-metrics.json#/performance`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: 2 paired native and segmented default prompts across completed seeds, with all semantic and denominator checks. no final semantically verified sealed project artifact was supplied
- `input.end_to_end_latency_improvement` — **End To End Latency Improvement** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: segmented/native direct time-to-first-logit ratio < 1 after the complete corrected timing matrix, semantic, raw-observation, denominator, and contamination checks pass
  - Evidence: `not available (no sealed project supplied)`; raw records `llm-metrics.json#/performance`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: paired native and segmented prompts with at least 5 warmups and 20 raw repetitions; the 1/2 default is incomplete. no final semantically verified sealed project artifact was supplied
- `input.prefill_compute_reduction` — **Prefill Compute Reduction** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: segmented/native total analytical-FLOP ratio < 1 after complete corrected conditions, semantic equality, denominator, and contamination checks pass
  - Evidence: `not available (no sealed project supplied)`; raw records `llm-metrics.json#/performance`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: paired native and segmented prompts under the declared analytical FLOP model. no final semantically verified sealed project artifact was supplied
- `input.codec_reference_compactness` — **Codec Reference Compactness** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: maximum_candidate_reference_state_ratio <= 0.50
  - Evidence: `not available (no sealed project supplied)`; raw records `tokenizer-metrics.json#/compactness/candidate_reference_state_ratio`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: one candidate/reference state inventory per completed seed. no final semantically verified sealed project artifact was supplied
- `input.physical_input_table_omission` — **Physical Input Table Omission** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: clean load proves the source input table is absent
  - Evidence: `not available (no sealed project supplied)`; raw records `deployment.json#/physical_reference_tensor_absent`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: three clean-process deployment repetitions when architecturally applicable. no final semantically verified sealed project artifact was supplied
- `input.cross_model_confirmation` — **Cross Model Confirmation** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: independent Large-profile final replications at seeds 17, 23, and 41 support exact held-out position compression and registered behavioral similarity for both models
  - Evidence: `not available (no sealed project supplied)`; raw records `project.json#/models`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: two independent primary-model replications, each containing final seeds 17, 23, and 41. no final semantically verified sealed project artifact was supplied
#### Applicability claims

- `input.input_table_removability` — **Input Table Removability** — `applicability` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: tied tables required for native output remain inapplicable
  - Evidence: `not available (no sealed project supplied)`; raw records `deployment.json#/applicability`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: one model-architecture applicability determination. no final semantically verified sealed project artifact was supplied
### Output-only claims

**No final semantically verified sealed project evidence was supplied.**

#### Primary claims

- `output.semi_autoregressive_density` — **Semi Autoregressive Density** — `primary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: minimum_native_tokens_per_attempted_macro_step >= 1.10 only after direct-feedback exactness, no-invalid-events, valid termination, and rollout fidelity are all supported
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/output_density`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: every attempted macro-step in the 16-sample current evaluation, including invalid and truncated attempts, with spans capped at 8 bytes. no final semantically verified sealed project artifact was supplied
#### Prerequisite claims

- `output.direct_feedback_exactness` — **Direct Feedback Exactness** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: byte and native-token direct-feedback equality are both 1.0
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/fidelity`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: every direct macro-event from 16 evaluation samples per completed seed, with spans capped at 8 bytes. no final semantically verified sealed project artifact was supplied
- `output.valid_non_empty_termination` — **Valid Non Empty Termination** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: valid non-empty CODEC_EOS termination is 1.0
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/valid_non_empty_termination`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: every attempted byte-span event from 16 evaluation samples per completed seed, with spans capped at 8 bytes. no final semantically verified sealed project artifact was supplied
- `output.no_invalid_events` — **No Invalid Events** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: maximum_invalid_events == 0
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/invalid_events`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: all direct and rollout events from the 16-sample, 2-prompt current evaluation. no final semantically verified sealed project artifact was supplied
- `output.control_exactness` — **Control Exactness** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: registered control coverage, precision, and recall gates pass
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/control_evidence`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: all oracle and predicted structural-control events in the 16-sample, 2-prompt current evaluation. no final semantically verified sealed project artifact was supplied
- `output.stop_exactness` — **Stop Exactness** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: stop precision and recall are both 1.0
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/stop_control`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: all oracle and predicted stop-control events in the 16-sample, 2-prompt current evaluation. no final semantically verified sealed project artifact was supplied
- `output.rollout_fidelity` — **Rollout Fidelity** — `prerequisite` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: minimum_rollout_event_agreement >= 0.50
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/rollout_event_agreement`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: the 2 registered rollout prompts, each bounded to 16 macro-steps and 512 output bytes, across completed seeds. no final semantically verified sealed project artifact was supplied
#### Secondary claims

- `output.codec_reference_compactness` — **Codec Reference Compactness** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: maximum_candidate_reference_state_ratio <= 0.50
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/candidate_reference_state_ratio`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: one candidate/reference state inventory per completed seed. no final semantically verified sealed project artifact was supplied
- `output.native_head_bypass` — **Native Head Bypass** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: native vocabulary head is never invoked
  - Evidence: `not available (no sealed project supplied)`; raw records `output-metrics.json#/candidate`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: every output-tokenizer generation call across completed seeds. no final semantically verified sealed project artifact was supplied
- `output.physical_output_head_omission` — **Physical Output Head Omission** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: clean load proves the native output head is absent
  - Evidence: `not available (no sealed project supplied)`; raw records `deployment.json#/physical_reference_tensor_absent`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: three clean-process deployment repetitions when architecturally applicable. no final semantically verified sealed project artifact was supplied
- `output.cross_model_confirmation` — **Cross Model Confirmation** — `secondary` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: independent Large-profile final replications at seeds 17, 23, and 41 support the registered output quality claims for both models
  - Evidence: `not available (no sealed project supplied)`; raw records `project.json#/models`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: two independent primary-model replications, each containing final seeds 17, 23, and 41. no final semantically verified sealed project artifact was supplied
#### Applicability claims

- `output.output_head_removability` — **Output Head Removability** — `applicability` — Verdict: **INCOMPLETE**
  - Decisive metric or policy: tied tables required for native feedback remain inapplicable
  - Evidence: `not available (no sealed project supplied)`; raw records `deployment.json#/applicability`; Evidence manifest SHA-256 `not available (no sealed project supplied)`
  - Scope: one model-architecture and native-feedback applicability determination. no final semantically verified sealed project artifact was supplied
<!-- END GENERATED EVIDENCE LEDGER -->
## Limitations

1. Greedy longest-valid segmentation is locally maximal, not globally optimal.
2. Exact reconstruction does not imply embedding equality or behavioral equivalence.
3. Distinct native rows with identical byte payloads cannot all be reproduced by one deterministic
   bytes-to-latent mapping.
4. The local codec adds parameters, latency, compilation, and transfer work.
5. Output-only feedback still uses native token IDs. A tied table therefore remains required.
6. Qwen and Gemma below one billion parameters do not establish larger-model behavior.
7. MPS measurements do not generalize to CUDA or serving hardware.
8. A passing tensor-state ratio is not a measured memory saving or proof of physical removal.
9. Performance ablations can isolate implementation effects but cannot replace final model
   replications.
10. Reduced stages and sample counts materially lower research power, and current results cannot be
    compared directly with prior configurations.

## Related work

The nearest distinction is between training byte-native models and adapting an already pretrained,
frozen backbone. The following sources motivate byte-level and learned-boundary modeling:

- Xue et al., [*ByT5: Towards a Token-Free Future with Pre-trained Byte-to-Byte
  Models*](https://aclanthology.org/2022.tacl-1.17/), TACL 2022.
- Tay et al., [*Charformer: Fast Character Transformers via Gradient-based Subword
  Tokenization*](https://arxiv.org/abs/2106.12672), ICLR 2022.
- Yu et al., [*MEGABYTE: Predicting Million-byte Sequences with Multiscale
  Transformers*](https://openreview.net/forum?id=JTmO2V9Xpz), NeurIPS 2023.
- Slagle, [*SpaceByte: Towards Deleting Tokenization from Large Language
  Modeling*](https://doi.org/10.48550/arxiv.2404.14408), NeurIPS 2024.
- Sennrich, Haddow, and Birch, [*Neural Machine Translation of Rare Words with Subword
  Units*](https://aclanthology.org/P16-1162/), ACL 2016.
- Kudo and Richardson, [*SentencePiece: A Simple and Language Independent Subword Tokenizer and
  Detokenizer for Neural Text Processing*](https://aclanthology.org/D18-2012/), EMNLP 2018.

## Reproducibility

Python 3.14 and `uv` are required. The compact runbook is:

```bash
uv sync --no-editable --reinstall-package continuous-byte-tokenizer
uv run --no-editable --group search --group ui tokenizer verify \
  --output-dir results/verification-fast
uv run --no-editable --group search --group ui tokenizer verify \
  --output-dir results/verification-complete --complete
uv run --no-editable tokenizer run experiments/synthetic/input-smoke.toml \
  --output-dir results/synthetic-input \
  --verification results/verification-complete/verification.json
uv run --no-editable tokenizer run experiments/synthetic/output-smoke.toml \
  --output-dir results/synthetic-output \
  --verification results/verification-complete/verification.json
uv run --no-editable tokenizer run \
  experiments/prospective/input/qwen35-0.8b/feasibility-screen.toml \
  --output-dir results/qwen35-input-feasibility
uv run --no-editable ruff format --check .
uv run --no-editable ruff check .
uv run --no-editable --group search ty check src tests
uv run --no-editable python -m unittest discover -s tests -v
RUN_SLOW_TESTS=1 uv run --no-editable python -m unittest discover -s tests -v
RUN_STREAMLIT_TESTS=1 uv run --no-editable --group ui \
  python -m unittest discover -s tests -p 'test_dashboard_*.py' -v
```

The default `tokenizer verify` command records format, lint, type, and fast-test checks. Final
campaign evidence requires the explicit complete inventory. The `--slow`, `--streamlit`, and
`--model-tokenizers` flags remain selectable subsets for diagnostics; `--complete` is the canonical
final-evidence command.

The read-only dashboard runs with:

```bash
uv run --no-editable --group ui streamlit run src/continuous_tokenizer/app.py
```

Detailed architecture, policy, campaign, and recovery instructions live in [`AGENTS.md`](AGENTS.md).
