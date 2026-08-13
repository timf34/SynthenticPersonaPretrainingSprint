# Prompt for Claude Code (run on the RunPod pod)

Copy everything below the line into a Claude Code session running on the RunPod pod.

Pod assumptions (set these up before starting Claude Code):
- GPU pod with 1–2× A100 80GB or H100 80GB (a 3B model fits easily on one GPU; a second GPU roughly halves generation time via the pipeline's multi-worker mode), ≥200GB volume mounted at `/workspace`.
- A `.env` file copied (scp, not git — it holds live keys) into `/workspace/sprint/assistant-axis/.env`, containing the OpenRouter credentials for the LLM judge (the judge uses the openai SDK pointed at OpenRouter; `load_dotenv()` in the pipeline picks this file up automatically) plus HF creds:
  ```
  OPENAI_API_KEY=sk-or-...        # OpenRouter key (the SDK reads this name)
  OPENAI_BASE_URL=https://openrouter.ai/api/v1
  OPENROUTER_API_KEY=sk-or-...    # same key, labeled
  HF_TOKEN=hf_...
  HF_ORG=timf34
  ```
- Docker image with CUDA + Python 3.10+ (any standard RunPod PyTorch image works; the repo installs its own deps via `uv`).

---

## Goal

Test whether the "Assistant Axis" (Lu et al. 2026, arXiv:2601.10387, safety-research/assistant-axis) exists in a small 3B model trained with **synthetic persona pretraining (SPP)** — where assistant-persona reflections were injected into a fraction of pretraining documents. The original paper found the axis in 27B–70B post-trained models (Gemma 2 27B, Qwen3 32B, Llama 3.3 70B). These SPP models are only 3B and had far lighter post-training (300k single-turn SFT examples), so two questions, in order:

1. **Phase 1 (gate):** Can these models role-play diverse personas at all? Without enough fully-role-playing responses the axis computation is starved of data. Run on all five recipe variants — the cross-recipe comparison of role-play rates is itself a result.
2. **Phase 2 (overnight):** For gate-passing models, run the full assistant-axis pipeline with **60 personas** (the paper used 275) and check via PCA whether PC1 of the persona-vector space aligns with the Assistant Axis, with the default assistant at one extreme.
3. **The overarching question:** the five instruct models differ ONLY in pretraining persona-injection recipe — so does axis strength (and role-play behavior) trend with persona-pretraining exposure?

Everything must run unattended overnight. Your job tonight is: set up, smoke-test, run the Phase 1 gate, then launch a checkpointed driver script in tmux and confirm it is progressing before you finish. Analysis code must be written and tested on Phase 1 outputs *before* the overnight launch so results are ready in the morning.

## Target models — the recipe sweep

All five dlab-spp 3B instruct models (revision `main` = `safety-10` SFT mix for each). They share architecture, tokenizer, and SFT data; **pretraining persona-injection recipe is the only variable**, so the experiment is: does the Assistant Axis's strength track persona-pretraining exposure?

Priority queue for the full pipeline (the supervisor works through it in order; the two poles MUST finish tonight, the rest resume from checkpoints tomorrow if the night runs out):
1. `dlab-spp/t0-mt-3b-instruct` — max exposure: SPP reflections from token zero across ~500B tokens PLUS reflection-focused midtraining
2. `dlab-spp/vanilla-3b-instruct` — zero persona injection (the control)
3. `dlab-spp/t0-3b-instruct` — reflections from token zero only
4. `dlab-spp/mt-3b-instruct` — reflections only in midtraining
5. `dlab-spp/filtered-3b-instruct` — harmful docs loss-masked, no reflections

Cross-model comparison rule for all analysis: each model has its own activation space, so NEVER compare raw vector directions across models — compare per-model metrics (PC1↔axis cosine, variance explained, default-vector separation/effect size, role-play rates).

Known model facts (verified from the model cards — re-verify locally):
- Custom **Llama-3.2-shaped** 3B architecture trained from scratch (should load as a standard `LlamaForCausalLM` in both vLLM and transformers; no `trust_remote_code` expected — verify).
- **SmolLM2 tokenizer** with an added `<assistant>` token (vocab 49,280). Assistant turns begin `<|im_start|><assistant>`.
- **The chat template has no system prompt.** This matters — see "Critical adaptation" below.

## Setup

1. Clone and install. The assistant-axis code is **vendored inside the sprint repo** (see `assistant-axis/UPSTREAM.md` for upstream provenance) — clone the sprint repo, not the original safety-research repo, so our modifications are included:
   ```bash
   cd /workspace
   git clone https://github.com/timf34/SynthenticPersonaPretrainingSprint.git sprint
   cd sprint/assistant-axis
   uv sync
   ```
2. Confirm `.env` exists in the repo root with a non-empty `OPENAI_API_KEY` (OpenRouter key) and `OPENAI_BASE_URL=https://openrouter.ai/api/v1` — fail fast if not, and verify the judge path early with a one-off API call through the OpenRouter base URL to `openai/gpt-4.1-mini`. Only the judge scripts auto-load `.env`; every shell/tmux session (including `overnight.sh`) must start with `set -a; source /workspace/sprint/assistant-axis/.env; set +a` so `HF_TOKEN` and the judge vars are exported for all steps. There are NO OpenAI credits — all judge traffic must go through OpenRouter; if any judge call errors against `api.openai.com`, env loading broke: stop and fix it, don't retry. Check GPU count with `nvidia-smi` and free disk (need ~30GB for activations per model, plus HF cache).
3. Install tmux if absent. All long-running work goes in tmux, never in your foreground shell.
4. Create `/workspace/exp/` as the experiment directory. Keep the vendored `assistant-axis/` code clean: all outputs (pass `--output_dir /workspace/exp/<model>/...` to every pipeline step), the `overnight.sh` driver, `analyze_axis.py`, role-subset files, logs, and reports live under `/workspace/exp/`, never inside `assistant-axis/`. Run scripts as `uv run --project /workspace/sprint/assistant-axis <script>` (or from within that directory) so the package imports resolve.

## How the pipeline works (already read the code before changing anything)

5 steps in `pipeline/`, all restartable/checkpointed (they skip existing outputs):
1. `1_generate.py` — vLLM batch generation. Per role: 5 persona-instruction variants × 240 questions = 1200 responses. Supports multi-worker when GPUs > `--tensor_parallel_size` (for a 3B model use TP=1, so 2 GPUs → 2 workers splitting roles).
2. `2_activations.py` — mean response-token activations from the post-MLP residual stream, all layers by default (~28 layers for this model). Keep all layers so we can do the layer sweep in analysis.
3. `3_judge.py` — LLM judge scores role adherence 0–3 (3 = fully role-playing). We route it through OpenRouter: **always pass `--judge_model openai/gpt-4.1-mini`** (the default `gpt-4.1-mini` is not a valid OpenRouter model ID and will 404). Can run in parallel with step 2 once step 1 finishes.
4. `4_vectors.py` — per-role mean vector over **score=3 responses only**; requires `--min_count` (default 50) score-3 responses per role or the role is dropped. The `default` role uses ALL its responses, no filtering.
5. `5_axis.py` — `axis = mean(default_vectors) − mean(role_vectors)`.

Roles live in `data/roles/instructions/` (275 roles + `default.json`); descriptions in `data/roles/role_list.json`. Questions in `data/extraction_questions.jsonl`.

## Critical adaptation: no system prompt

`assistant_axis/generation.py` (`format_conversation`, ~lines 85–129) auto-detects system-prompt support by rendering a test system message through the chat template; if unsupported it prepends the persona instruction to the user message. **Before anything else**, run a quick Python check:
- Load the tokenizer, call `apply_chat_template` with and without a system message, print both outputs.
- Determine which path `format_conversation` will take.
- Trap: if the template silently *renders* a system turn (detection says "supported") but the model was never trained on system turns, persona instructions will be ignored. If Phase 1 shows near-zero role adherence AND the system path was taken, force the user-turn concatenation path (smallest possible patch, documented in the final report) and rerun Phase 1.

Also note `assistant_axis/models.py` `get_config` auto-infers `target_layer = num_hidden_layers // 2` for unknown models — print what it returns for this model and record it.

## Phase 0 — Smoke test (do this first, ~15 min)

Run the full 5-step chain end-to-end on 2 roles (`pirate` if it exists, else any colorful role, plus `default`) with `--question_count 10`. Purpose: catch vLLM loading issues, chat-template problems, judge API issues, and shape mismatches before burning GPU-hours. Inspect a few generated responses by eye and include 2–3 examples in your report.

## Phase 1 — Role-play capability gate, ALL FIVE models (~1–2 hours total)

Run this phase on **all five instruct models**, not just the primary — the cross-recipe role-play comparison is itself a headline result (does persona pretraining change how easily the model leaves the assistant persona?). It's cheap: ~13 roles × 200 responses × 5 models ≈ 13k generations plus judge calls. Produce ONE comparison table across models in `/workspace/exp/roleplay_comparison.md`, plus the per-model reports below. The go/no-go gate applies per model: a model that fails the gate is dropped from the full-pipeline queue (documented, with examples), not forced.

1. Pick ~12 probe roles spanning the spectrum: 4 assistant-adjacent (e.g. tutor, counselor, translator, editor), 4 neutral-human (e.g. accountant, architect, gamer, chef), 4 far-from-assistant/non-human (e.g. demon, ghost, oracle, alien, trickster — check exact filenames in `data/roles/instructions/`). Plus `default`.
2. Run steps 1 and 3 on these with `--question_count 40` (40 × 5 = 200 responses/role).
3. Produce a score distribution per role (counts of 0/1/2/3) and write `/workspace/exp/<model>/roleplay_report.md` with a table, plus 3 example responses each from a passing and a failing role.
4. **Go/No-Go (per model):** a role is *viable* if ≥9/200 responses score 3 (≥4.5%, which projects to ≥27 score-3 at the full-run scale of 600 responses/role, meeting the `--min_count 25` used in Phase 2). 
   - **GO** if ≥70% of probe roles (excluding default) are viable → include the model in the Phase 2 queue.
   - **PARTIAL** if 30–70% viable → include it, but pass `--min_count 15` to step 4 for that model, and note it.
   - **NO-GO** if <30% viable → do NOT burn the night on 60 roles. Instead: (a) try the user-turn-concatenation fix above if not already applied; (b) try a variant where score-2 responses ("identifies as AI but exhibits role attributes") are also included in step 4 — this needs a small documented patch; (c) run a reduced 30-role version with whichever variant looks best. A model that *can't* leave the assistant persona is itself an interesting SPP result — document it with examples rather than forcing the pipeline.

## Phase 2 — Full overnight run (60 personas)

1. **Select 60 roles**, stratified: ~20 assistant-adjacent/helper, ~20 neutral human, ~20 non-human/fantastical/adversarial. Choose from `role_list.json`; prioritize archetypes highlighted in the paper (oracle, hive, gamer, demon-like/trickster-like roles) and include all viable Phase 1 probe roles. Save the exact list to `/workspace/exp/<model>/roles_60.json` — this is the reproducibility record. `default` is used in addition to the 60 and is not one of them.
2. Write a driver script `overnight.sh` that loops over the **priority queue of gate-passing models** (same 60 roles for every model), with `set -e`, per-step logging to `/workspace/exp/<model>/logs/`, and the `--roles $(cat ...)` selection. Per model:
   - Step 1 (generation, all 60 roles + default, **`--question_count 120`** — 600 responses/role, uniform across models so they're comparable, sized so ~3–4 models finish per night) → then step 2 and step 3 **in parallel** (backgrounded with wait) → step 4 (**`--min_count 25`**) → step 5 → per-model analysis script (below).
   - A model's chain failing repeatedly must not block the queue: after the supervisor's retries are exhausted for it, log it in STATE.md and move to the next model.
   - After each model completes (not only at the very end): back up results off the ephemeral pod — upload `/workspace/exp/<model>/` (vectors, `axis.pt`, PCA outputs, plots, reports, `roles_60.json`, logs; **exclude** raw activations and responses, ~25GB+) to a **private** HF dataset repo `timf34/spp-assistant-axis-results` via `huggingface_hub.upload_folder` using `HF_TOKEN` from the `.env`. Create the repo private if it doesn't exist.
3. **Supervisor (the thing that actually runs overnight):** don't launch `overnight.sh` directly. Write a `supervisor.sh` that exploits the pipeline's idempotence (every step skips existing outputs, so re-running is always safe):
   - Loop up to 8 attempts: run `overnight.sh`; on success, break; on failure, append the attempt number, exit code, and the last ~20 lines of the failing step's log to `/workspace/exp/attempts.log`, sleep 120s, and re-run. Transient failures (OOM, API blips, rate limits) thus cost minutes, not the night.
   - Wrap long steps in generous `timeout`s (e.g. 6h for generation, 4h for activations) so a hang becomes a retry instead of silently eating the night.
   - Have `overnight.sh` append a timestamped line to `/workspace/exp/STATE.md` at each step start/finish — morning triage should be one file: current state, attempts so far, where the logs are.
4. Launch `supervisor.sh` in tmux, watch the first ~10 minutes of logs (generation throughput, no OOM — tune `--batch_size` in step 2 for an 80GB card, start at 64 for a 3B model and back off if needed), then leave it.
5. Rough budget check before launch: per model, 61 roles × 600 responses ≈ 37k generations of ≤512 tokens (~1.5–3h on one H100 for a 3B model) plus ~37k `openai/gpt-4.1-mini` judge calls via OpenRouter (~$5–10/model, ~$25–50 for all five; step 3 is restartable if it dies, and OpenRouter rate limits surface as ordinary API errors — the script's rerun-on-malformed behavior covers them, so rerun step 3 once more per model as the repo README recommends). Use your measured Phase 1 throughput to project the night honestly in STATE.md: how many models should complete by 8am. If it projects to fewer than 2, cut `--question_count` to 60 and `--min_count` to 12 — uniformly, for every model — and note the deviation.

## Phase 3 — Analysis (write and test the script BEFORE the overnight launch, using Phase 1 outputs)

A headless script `analyze_axis.py` (use `assistant_axis.pca.compute_pca` and `assistant_axis.axis`), run automatically at the end of `overnight.sh`:
1. Load per-role vectors, standardize (subtract mean across roles), PCA at the target (middle) layer.
2. Report: variance explained per PC (paper: 4–19 PCs for 70%); **cosine similarity between PC1 and the computed axis** (paper: >0.71 at the middle layer — this is the headline number); projection of every role + default onto PC1 and onto the axis, ranked.
3. Layer sweep: PC1↔axis cosine similarity at every layer.
4. Plots (PNG): scree plot; PC1-vs-PC2 scatter of role vectors colored by axis projection with default marked (this is the paper's Figure 1 analogue); ranked role-projection bar chart.
5. Write `/workspace/exp/<model>/RESULTS.md` per model: headline numbers, the top/bottom 10 roles by axis projection (sanity check: helpers positive, demons/ghosts negative, default at the extreme), plots inline, all deviations from the paper's method.
6. **The main deliverable — `/workspace/exp/COMPARISON.md`:** one table across all completed models, ordered by persona-pretraining exposure (vanilla → filtered → mt → t0 → t0-mt), with per-model metrics only (PC1↔axis cosine at the middle layer, variance explained by PC1, #PCs for 70%, default-vector separation, Phase 1 role-play rate). The experiment's answer lives in whether these metrics trend with exposure. Regenerate it as each model in the queue completes, so a partial night still yields the comparison for whatever finished.

## Working rules

- Prefer flags over code edits. When a code change IS needed, it's allowed (the code is our vendored copy), but keep it minimal, record it in `assistant-axis/UPSTREAM.md`, list it in RESULTS.md, and dump `git diff > /workspace/exp/patches.diff` so it can be committed back to the sprint repo from your machine (the pod has no GitHub push credentials — the HF results upload carries the diff home).
- Everything long-running goes in tmux; every step logs to a file; every step must be resumable after a crash (the pipeline already supports this — don't break it).
- Do not wait idle for hours watching a job. Set the driver going, verify it's healthy, and end with a clear summary of: what was launched, where the logs/outputs are, what the morning checklist is (`tmux attach`, check `logs/`, read `RESULTS.md`).
- If the primary model fails to load in vLLM after 2–3 genuine fix attempts (tokenizer mode, dtype, revision pinning), fall back to `dlab-spp/t0-3b-instruct` and note it.
