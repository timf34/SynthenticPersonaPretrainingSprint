#!/usr/bin/env bash
# Preflight + role-play gate. Run before the full pipeline; aborts loudly on any failure.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODELS=("$TREATMENT_MODEL" "$CONTROL_MODEL")
KEYS=("$TREATMENT_KEY" "$CONTROL_KEY")
GPUS=(0 1)
# 12 probe roles spanning the spectrum: assistant-adjacent -> neutral human -> far-from-assistant.
PROBE_ROLES=(tutor counselor translator editor accountant architect gamer chef demon ghost oracle trickster)
PROBE_QUESTIONS="${PROBE_QUESTIONS:-40}"   # 40 x 5 prompts = 200 responses/role

echo "== 1/5 OpenRouter judge check =="
resp=$(curl -sS "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" \
  -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":5}")
echo "$resp" | grep -q '"choices"' || { echo "FATAL: OpenRouter test call failed: $resp" >&2; exit 1; }
echo "ok"

echo "== 2/5 GPU / disk =="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace || true

echo "== 3/5 chat template (SPP models have no system role) =="
for m in "${MODELS[@]}"; do
  echo "--- $m ---"
  uv run --project "$AXIS_DIR" python "$SCRIPTS_DIR/check_chat_template.py" --model "$m"
done

echo "== 4/5 tiny end-to-end per model =="
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

echo "== 5/5 role-play gate (${#PROBE_ROLES[@]} probe roles x $((PROBE_QUESTIONS*5)) responses) =="
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
