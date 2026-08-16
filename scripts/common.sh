#!/usr/bin/env bash
# Shared config. Override anything via environment variables.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AXIS_DIR="$REPO_DIR/assistant-axis"
EXP_ROOT="${EXP_ROOT:-/workspace/exp}"

# The comparison pair: max-persona-pretraining vs the no-injection control.
# Same architecture, tokenizer, and SFT — pretraining recipe is the only variable.
# SIZE=1.7b runs the smaller family (100B tokens) instead of 3b (500B tokens).
SIZE="${SIZE:-3b}"
TREATMENT_MODEL="${TREATMENT_MODEL:-dlab-spp/t0-mt-${SIZE}-instruct}"
CONTROL_MODEL="${CONTROL_MODEL:-dlab-spp/vanilla-${SIZE}-instruct}"
TREATMENT_KEY="${TREATMENT_KEY:-t0-mt-${SIZE}}"
CONTROL_KEY="${CONTROL_KEY:-vanilla-${SIZE}}"

QUESTION_COUNT="${QUESTION_COUNT:-120}"   # 120 -> 600 responses/role
MIN_COUNT="${MIN_COUNT:-25}"              # min score-3 responses per role vector (paper ratio ~4%)
BATCH_SIZE="${BATCH_SIZE:-32}"            # activation extraction batch (3B models: 32 is safe on 80GB)
JUDGE_MODEL="${JUDGE_MODEL:-openai/gpt-4.1-mini}"   # OpenRouter model id — NOT the bare openai name
JUDGE_BATCH="${JUDGE_BATCH:-50}"
JUDGE_RPS="${JUDGE_RPS:-60}"
PRUNE_ACTIVATIONS="${PRUNE_ACTIVATIONS:-1}"
HF_RESULTS_REPO="${HF_RESULTS_REPO:-timf34/spp-assistant-axis-results}"

# These models' chat template has no system role. Persona instructions must be concatenated into
# the user turn; preflight verifies this and will tell you if the setting is wrong.
export ASSISTANT_AXIS_FORCE_USER_CONCAT="${ASSISTANT_AXIS_FORCE_USER_CONCAT:-1}"

if [[ ! -f "$AXIS_DIR/.env" ]]; then
  echo "FATAL: $AXIS_DIR/.env missing (see .env.example)" >&2
  exit 1
fi
set -a; source "$AXIS_DIR/.env"; set +a
if [[ -z "${OPENAI_API_KEY:-}" || -z "${OPENAI_BASE_URL:-}" ]]; then
  echo "FATAL: OPENAI_API_KEY / OPENAI_BASE_URL not set in .env (OpenRouter creds)" >&2
  exit 1
fi

mkdir -p "$EXP_ROOT"
state() { echo "$(date -u +%FT%TZ)  $*" >> "$EXP_ROOT/STATE.md"; }
