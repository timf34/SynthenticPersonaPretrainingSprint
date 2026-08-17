#!/usr/bin/env bash
# Preflight + role-play gate. Run before the full pipeline; aborts loudly on any failure.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=("$TREATMENT_MODEL" "$CONTROL_MODEL")
KEYS=("$TREATMENT_KEY" "$CONTROL_KEY")
# Map both models onto whatever GPUs exist (a 1-GPU pod previously crashed vLLM here).
NGPU=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
[[ "$NGPU" -ge 1 ]] || { echo "FATAL: no GPUs visible" >&2; exit 1; }
GPUS=(0 $(( NGPU >= 2 ? 1 : 0 )))
# 24 probe roles spanning the spectrum: 8 assistant-adjacent, 8 neutral human, 8 far-from-assistant.
# The gate estimates a proportion (what fraction of the 275 roles are viable), so sample size drives
# its error bars: at 12 roles the standard error near p=0.5 is ~14pp, at 24 it's ~10pp. The cost is
# ~3% of the full run's generations — cheap insurance against a mis-verdict wasting a whole night.
# Override with e.g. PROBE_ROLES="tutor demon ghost".
read -r -a PROBE_ROLES <<< "${PROBE_ROLES:-tutor counselor translator editor mentor librarian guide therapist accountant architect gamer chef journalist soldier comedian hermit demon ghost oracle trickster wraith alien golem eldritch}"
PROBE_QUESTIONS="${PROBE_QUESTIONS:-40}"   # 40 x 5 prompts = 200 responses/role

# Cheap environment checks (creds, HF, chat template, GPU, disk) live in scripts/doctor.sh,
# which run_on_pod.sh runs first. This file only does work that costs real GPU time.
echo "== 1/2 tiny end-to-end per model =="
for i in 0 1; do
  m="${MODELS[$i]}"; key="${KEYS[$i]}"; gpu="${GPUS[$i]}"
  out="$EXP_ROOT/preflight/$key"
  uv run --project "$AXIS_DIR" python -c "
import sys; sys.path.insert(0, '$AXIS_DIR')
from assistant_axis.models import get_config
print('$key config:', get_config('$m'))"
  CUDA_VISIBLE_DEVICES=$gpu uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/1_generate.py" \
    --model "$m" --roles pirate default --question_count 3 \
    --roles_dir "$AXIS_DIR/data/roles/instructions" --questions_file "$AXIS_DIR/data/extraction_questions.jsonl" \
    --output_dir "$out/responses"
  echo "--- sample responses ($key) — check the model is actually playing the role ---"
  head -c 1200 "$out/responses/pirate.jsonl" || true; echo
done

echo "== 2/2 role-play gate (${#PROBE_ROLES[@]} probe roles x $((PROBE_QUESTIONS*5)) responses) =="
for i in 0 1; do
  m="${MODELS[$i]}"; key="${KEYS[$i]}"; gpu="${GPUS[$i]}"
  out="$EXP_ROOT/gate/$key"
  state "[$key] gate: generating"
  CUDA_VISIBLE_DEVICES=$gpu uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/1_generate.py" \
    --model "$m" --roles "${PROBE_ROLES[@]}" default --question_count "$PROBE_QUESTIONS" \
    --roles_dir "$AXIS_DIR/data/roles/instructions" --questions_file "$AXIS_DIR/data/extraction_questions.jsonl" \
    --output_dir "$out/responses"
  state "[$key] gate: judging"
  uv run --project "$AXIS_DIR" python "$AXIS_DIR/pipeline/3_judge.py" \
    --responses_dir "$out/responses" --output_dir "$out/scores" --judge_model "$JUDGE_MODEL" \
    --roles_dir "$AXIS_DIR/data/roles/instructions" --batch_size "$JUDGE_BATCH" --requests_per_second "$JUDGE_RPS"
  set +e
  uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/roleplay_gate.py" \
    --scores_dir "$out/scores" --key "$key" --out "$EXP_ROOT/gate_$key.md" \
    --full_scale $((QUESTION_COUNT*5)) --min_count "$MIN_COUNT"
  echo "$?" > "$EXP_ROOT/gate_$key.code"
  set -e
  state "[$key] gate verdict code: $(cat "$EXP_ROOT/gate_$key.code")"
done

echo
echo "Gate reports: $EXP_ROOT/gate_*.md"
echo "PREFLIGHT PASSED"
