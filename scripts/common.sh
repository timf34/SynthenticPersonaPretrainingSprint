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
JUDGE_BATCH="${JUDGE_BATCH:-200}"
JUDGE_RPS="${JUDGE_RPS:-250}"
PRUNE_ACTIVATIONS="${PRUNE_ACTIVATIONS:-0}"  # default OFF: activations are uploaded (below) and only
                                          # deleted if you explicitly ask. A prune-by-default once destroyed
                                          # a run's activations minutes before the user tried to save them.
ACTIVATIONS_REPO="${ACTIVATIONS_REPO:-timf34/spp-assistant-axis-activations}"  # separate public dataset: 57-220GB/model
UPLOAD_ACTIVATIONS="${UPLOAD_ACTIVATIONS:-1}"  # push raw activations there after vectors are done
HF_RESULTS_REPO="${HF_RESULTS_REPO:-timf34/spp-assistant-axis-results}"

# These models' chat template has no system role. Persona instructions must be concatenated into
# the user turn; preflight verifies this and will tell you if the setting is wrong.
export ASSISTANT_AXIS_FORCE_USER_CONCAT="${ASSISTANT_AXIS_FORCE_USER_CONCAT:-1}"

# Credentials. Accept .env in either the repo root or assistant-axis/ (the pipeline's own
# load_dotenv() only finds the latter, so we export everything here for all steps), and accept
# the key under OPENROUTER_API_KEY as well as OPENAI_API_KEY — the openai SDK reads the latter.
# The Gemma run lost an hour to exactly this mismatch; be forgiving about it, not clever.
set -a
for envfile in "$REPO_DIR/.env" "$AXIS_DIR/.env"; do
  [[ -f "$envfile" ]] && source "$envfile"
done
set +a
: "${OPENAI_API_KEY:=${OPENROUTER_API_KEY:-}}"
: "${OPENAI_BASE_URL:=https://openrouter.ai/api/v1}"
export OPENAI_API_KEY OPENAI_BASE_URL
if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "FATAL: no OpenRouter key found. Put OPENAI_API_KEY (or OPENROUTER_API_KEY) in" >&2
  echo "       $AXIS_DIR/.env or $REPO_DIR/.env — see .env.example" >&2
  exit 1
fi
# The judge scripts call load_dotenv() themselves and only look next to the pipeline, so make sure
# a root-only .env is also visible there.
if [[ ! -f "$AXIS_DIR/.env" && -f "$REPO_DIR/.env" ]]; then
  { echo "OPENAI_API_KEY=$OPENAI_API_KEY"; echo "OPENAI_BASE_URL=$OPENAI_BASE_URL"; } > "$AXIS_DIR/.env"
  [[ -n "${HF_TOKEN:-}" ]] && echo "HF_TOKEN=$HF_TOKEN" >> "$AXIS_DIR/.env"
  echo "note: mirrored credentials from $REPO_DIR/.env to $AXIS_DIR/.env (pipeline needs them there)"
fi

mkdir -p "$EXP_ROOT"
state() { echo "$(date -u +%FT%TZ)  $*" >> "$EXP_ROOT/STATE.md"; }
