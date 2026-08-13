# Prompt for Claude Code — Gemma anchors track (run on a RunPod pod)

Companion to `runpod_claude_code_prompt.md` (the SPP recipe-sweep track). Same repo, same pipeline, same conventions — read that prompt's **Setup**, **How the pipeline works**, **Supervisor**, and **Working rules** sections first and follow them identically (clone `timf34/SynthenticPersonaPretrainingSprint` to `/workspace/sprint`, `.env` with OpenRouter creds at `/workspace/sprint/assistant-axis/.env`, judge always `--judge_model openai/gpt-4.1-mini`, outputs under `/workspace/exp/`, retry supervisor + STATE.md, HF backup to `timf34/spp-assistant-axis-results`).

Pod: 1× H100/A100 80GB, ≥200GB volume. Can run in parallel with the SPP track on a different pod, or on the same pod after the SPP queue drains.

## Goal

Compute the Assistant Axis in ordinary heavily-post-trained small models as **anchors** for the SPP results: the SPP 3B models got their persona from pretraining injection; Gemma got its from conventional large-scale post-training. Same pipeline + personas + metrics → their rows join `/workspace/exp/COMPARISON.md` (marked as anchors — different architectures, so per-model metrics only, never raw directions).

## Target models (priority queue, poles first)

1. `google/gemma-3-4b-it` — closest scale anchor to the SPP 3B models
2. `google/gemma-4-E4B-it` — ~8B MatFormer; likely the model from the public "assistant axis in small Gemma" finding
3. `google/gemma-3-12b-it` — scale trend within Gemma 3
4. `google/gemma-4-12B-it` — scale trend within Gemma 4
5. `google/gemma-3-1b-it` — how small does the axis persist?

## Gemma-specific checks (do these before committing GPU-hours)

- **Multimodal wrappers:** `gemma-3-4b-it`/`12b-it` and Gemma 4 models are multimodal (`*ForConditionalGeneration`). Verify BOTH vLLM generation and the transformers-based activation extraction (`assistant_axis/internals/model.py` ProbingModel) correctly target the **language-model backbone's** decoder layers — hooks on the wrong submodule fail silently. Also `get_config` in `assistant_axis/models.py` reads `config.num_hidden_layers`, which for multimodal configs may live under `text_config` — verify the inferred layer count matches the real decoder depth, patch minimally if not (record in UPSTREAM.md).
- **Gemma 4 is a new architecture (E-series MatFormer, A4B MoE):** the vendored `uv.lock` may pin a vLLM/transformers too old to load it. Try loading first. If unsupported, upgrade vllm+transformers in the project (document versions in RESULTS.md); if it still fails after 2–3 genuine attempts, drop the Gemma 4 entries to the back of the queue and continue with Gemma 3 — do not burn the night on dependency archaeology.
- Chat template: Gemma templates handle system prompts their own way (folded into the first user turn). The pipeline's `format_conversation` detection handles this — just confirm which path it takes and record it.
- HF gating: Gemma repos require accepting the license; `HF_TOKEN` in `.env` must belong to an account that has. Fail fast with a clear message if downloads 403.

## Protocol (mirrors the SPP track, lighter gate)

- **Same 60 personas as the SPP track — this is mandatory for comparability.** Fetch `roles_60.json` from the HF backup dataset (the SPP track uploads it). If it doesn't exist yet, derive the stratified 60 per the SPP prompt's rules, save it, and upload it so both tracks converge on one list.
- **Phase 0:** smoke test per model (2 roles, `--question_count 10`, full 5-step chain) before it enters the queue.
- **Phase 1 (light):** these models are strong role-players, so no full gate — but still run the 12 SPP probe roles × 200 responses on `gemma-3-4b-it` only, and add its row to `/workspace/exp/roleplay_comparison.md`. The SPP-vs-Gemma role-play-rate gap at matched scale is a key number.
- **Phase 2:** full pipeline per model at **`--question_count 120`, `--min_count 25`** (identical to the SPP track), 60 roles + default, steps 2+3 in parallel, per-model analysis, backup after each model. Larger models are slower — expect ~2–5h/model; a queue model failing repeatedly is logged and skipped, never blocking.
- **Phase 3:** per-model RESULTS.md (PC1↔axis cosine at the middle layer, variance explained, default at the extreme, top/bottom-10 roles) and regenerate `COMPARISON.md` with an **Anchors** section beneath the SPP rows after each completion.

The comparison the morning reader wants: at ~similar scale, do the persona-pretrained SPP models show a stronger/weaker/comparable Assistant Axis and role-play flexibility than conventionally post-trained Gemma?
