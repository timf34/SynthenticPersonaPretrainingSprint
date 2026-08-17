# SPP Assistant Axis Sprint

Does **synthetic persona pretraining** change the model's assistant persona in activation space?

We compute the **Assistant Axis** (Lu et al. 2026, [arXiv:2601.10387](https://arxiv.org/abs/2601.10387)) for a matched pair of [dlab-spp](https://huggingface.co/dlab-spp) models ([project](https://modelraising.ai/spp/), [code](https://github.com/epfl-dlab/spp)):

| | model | pretraining recipe |
|---|---|---|
| **treatment** | `dlab-spp/t0-mt-3b-instruct` | persona reflections from token zero (~500B tokens) **+** reflection-focused midtraining |
| **control** | `dlab-spp/vanilla-3b-instruct` | no persona injection |

Same architecture, tokenizer, and SFT — **pretraining recipe is the only variable**, so any difference in axis structure or role-play behaviour is attributable to SPP. `SIZE=1.7b` runs the same comparison on the smaller family (100B tokens); 3B is the largest SPP release.

Companion track: [GemmaAssistantAxis](https://github.com/timf34/GemmaAssistantAxis) runs the identical pipeline on conventionally post-trained Gemma 2/3/4, giving a known-good reference for what a strong axis looks like.

## Run it

```bash
git clone https://github.com/timf34/SynthenticPersonaPretrainingSprint.git && cd SynthenticPersonaPretrainingSprint
# scp your filled-in .env to assistant-axis/.env  (see .env.example)
bash run_on_pod.sh
```

Installs deps, preflights, runs the role-play gate, then runs **both models in parallel, one per GPU**, each under a retry supervisor, and writes the treatment-vs-control comparison.

**Pod:** 2× 80GB (A100/H100) — these are 3B models, so one per GPU with room to spare. **~500GB volume** at `/workspace` (raw activations are pruned automatically after upload).

| var | default | meaning |
|---|---|---|
| `SIZE` | `3b` | `1.7b` for the smaller family |
| `QUESTION_COUNT` | 120 | 600 responses/role; 60 for a short night, 240 for full paper scale |
| `MIN_COUNT` | 25 | min score-3 responses per role vector (auto-relaxed to 15 on a PARTIAL gate) |
| `DOCTOR_ONLY` | 0 | run the cheap environment checks (~3 min) and exit — validate a fresh pod before committing to a night |
| `SKIP_PREFLIGHT` | 0 | resume after a crash |
| `FORCE_GO` | 0 | run the full pipeline even if the gate says NO-GO |
| `SHUTDOWN` / `SAVE_TO_GIT` | – | pause the pod when done / push reports to this repo first |

## Before you walk away

`bash run_on_pod.sh` runs `scripts/doctor.sh` before any GPU work: credentials + a live OpenRouter call, HF downloader flags (auto-installs `hf_transfer`/`hf_xet` or disables them), HF auth, config + tokenizer load for both models, chat-template persona delivery, GPU, and disk. It prints **`ALL ENVIRONMENT CHECKS PASSED — SAFE TO LEAVE IT RUNNING`** when everything is green; until you see that line, stay at the terminal. `DOCTOR_ONLY=1 bash run_on_pod.sh` runs just the checks (~3 min) on a fresh pod.

## The role-play gate

These are small, lightly post-trained models, so before spending a night we check they can actually *leave* the assistant persona: 24 probe roles (8 assistant-adjacent, 8 neutral human, 8 demon/ghost/oracle-type), 200 responses each, judged 0–3. A role is viable if its score-3 rate projects past `min_count` at full scale.

- **GO** (≥70% viable) → full pipeline
- **PARTIAL** (30–70%) → full pipeline with a relaxed `min_count`
- **NO-GO** (<30%) → the model is skipped and reported. **A model that can't leave the assistant persona is itself a result** — the gate rates for treatment vs control go into `COMPARISON.md` as a headline number.

## Critical detail: no system prompt

The SPP models were trained with **no system role** (assistant turns begin `<|im_start|><assistant>`). Persona instructions therefore have to be concatenated into the user turn — `ASSISTANT_AXIS_FORCE_USER_CONCAT=1` (set by default) does this, and `scripts/check_chat_template.py` prints the rendered template and verifies the path in preflight. Getting this wrong makes a capable model look like it can't role-play at all.

## What it produces

Per model, in the same layout as the paper's HF release (`lu-christina/assistant-axis-vectors`):

```
<key>/assistant_axis.pt        (n_layers, hidden)
<key>/default_vector.pt        (n_layers, hidden)
<key>/role_vectors/<role>.pt   (n_layers, hidden)   — every role passing the judge filter
```

Plus per-model `RESULTS.md` (PC1↔axis cosine, variance explained, default separation, ranked role projections, layer sweep), plots, and a top-level `COMPARISON.md` (treatment vs control metrics + role-play gate rates). Everything but raw activations is uploaded to the private HF dataset `timf34/spp-assistant-axis-results`.

**Full-run role-play comparison:** the gate's numbers come from 24 probe roles; after the run, `scripts/roleplay_stats.py` recomputes the same comparison from every judged response (275 roles × 600 = ~165k per model) with a paired test over roles:

```bash
python scripts/roleplay_stats.py \
  --scores t0-mt-3b=/workspace/exp/t0-mt-3b/scores \
  --scores vanilla-3b=/workspace/exp/vanilla-3b/scores \
  --out /workspace/exp/ROLEPLAY.md
```

**Raw transcripts are NOT uploaded by default** (they are bulky and the pipeline's upload targets vectors and reports). They live on the pod at `/workspace/exp/<key>/responses/`, so archive them before terminating a pod — they are the qualitative half of the result, showing *how* a model declines to fully become a character:

```bash
python scripts/archive_responses.py --key t0-mt-3b  --include-gate --upload
python scripts/archive_responses.py --key vanilla-3b --include-gate --upload
```

That writes one gzipped JSONL per model with each response joined to its judge score, and uploads it to the HF results dataset.

**Integrity check:** every run reports `cos(default − mean(saved role vectors), axis)`, which must be **1.000** — proof the axis was built from exactly the saved vector set.

Comparisons are **per-model metrics only** — the two models have different activation spaces, so raw directions are never compared across them.

## Layout

- `run_on_pod.sh` — one-shot entry point
- `scripts/` — `doctor.sh` (fail-fast env checks), `preflight.sh` (tiny e2e + role-play gate), `roleplay_gate.py`, `roleplay_stats.py` (full-run role-play comparison), `check_chat_template.py`, `supervisor.sh`, `run_model.sh`, `package_release.py`, `analyze_axis.py`, `upload_results.py`
- `assistant-axis/` — vendored pipeline (provenance + local patches in `assistant-axis/UPSTREAM.md`, paper in `og-paper.md`)
- `roles_90.json` — 90-persona subset shared with the Gemma track (full 275 roles are used for the run itself)
- `runpod_claude_code_prompt.md` — earlier agent-driven version of this experiment, superseded by the scripts above; still the reference for the full five-recipe sweep
