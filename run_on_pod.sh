#!/usr/bin/env bash
# One-shot runner: computes the Assistant Axis for the SPP treatment/control pair IN PARALLEL,
# one model per GPU.
#
#   treatment: dlab-spp/t0-mt-3b-instruct   (persona reflections from token zero + midtraining)
#   control:   dlab-spp/vanilla-3b-instruct (no persona injection; same architecture + SFT)
#
# The pretraining recipe is the only difference between them, so any gap in axis structure or
# role-play behaviour is attributable to synthetic persona pretraining.
#
# Usage:
#   git clone https://github.com/timf34/SynthenticPersonaPretrainingSprint.git && cd ...
#   # scp your .env to assistant-axis/.env (see .env.example), then:
#   bash run_on_pod.sh
#
#   SIZE=1.7b bash run_on_pod.sh                  # the smaller family (100B tokens)
#   QUESTION_COUNT=60 bash run_on_pod.sh          # half scale if the night is short
#   DOCTOR_ONLY=1 bash run_on_pod.sh              # validate a fresh pod in ~3 min, then exit
#   SKIP_PREFLIGHT=1 bash run_on_pod.sh           # resume after a crash
#   FORCE_GO=1 bash run_on_pod.sh                 # run the full pipeline even if the gate says NO-GO
#   SHUTDOWN=stop SAVE_TO_GIT=1 bash run_on_pod.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export QUESTION_COUNT="${QUESTION_COUNT:-120}"
export MIN_COUNT="${MIN_COUNT:-25}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export EXP_ROOT="${EXP_ROOT:-/workspace/exp}"
export PRUNE_ACTIVATIONS="${PRUNE_ACTIVATIONS:-1}"
source scripts/common.sh

echo "== [1/6] dependencies =="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
(cd assistant-axis && uv sync)
echo "  deps installed"

echo "== [2/6] doctor: fail-fast environment checks =="
# Every cheap check lives in scripts/doctor.sh so a broken pod is caught in the first minute,
# not overnight. It prints an explicit "SAFE TO LEAVE IT RUNNING" banner when everything passes.
if ! bash scripts/doctor.sh; then
  echo "!! environment checks failed — aborting before any GPU work. Fix the [FAIL] lines above."
  exit 1
fi
[[ "${DOCTOR_ONLY:-0}" == "1" ]] && { echo "DOCTOR_ONLY=1 -> stopping after checks."; exit 0; }

command -v tmux >/dev/null 2>&1 || { apt-get update -qq && apt-get install -y -qq tmux; } || true
echo "  deps OK"

echo "== [3/6] preflight + role-play gate =="
if [[ "${SKIP_PREFLIGHT:-0}" != "1" ]]; then
  bash scripts/preflight.sh
else
  echo "  skipped (SKIP_PREFLIGHT=1)"
fi

# Gate policy: NO-GO (code 2) means the model can't leave the assistant persona often enough to
# build role vectors. That is itself a finding — don't burn the night; report it instead.
gate_ok() {  # gate_ok <key>
  local f="$EXP_ROOT/gate_$1.code"
  [[ "${FORCE_GO:-0}" == "1" ]] && return 0
  [[ -f "$f" ]] || return 0          # no gate run (SKIP_PREFLIGHT) -> proceed
  [[ "$(cat "$f")" != "2" ]]
}

echo "== [4/6] launching both models in parallel (one per GPU) =="
declare -a PIDS=() KEYS=()
launch() {  # launch <hf_id> <gpu> <key>
  if ! gate_ok "$3"; then
    echo "  !! $3: gate verdict NO-GO — skipping full pipeline (see $EXP_ROOT/gate_$3.md)."
    echo "     Re-run with FORCE_GO=1 to override."
    state "[$3] SKIPPED: gate NO-GO"
    return
  fi
  echo "  $3: $1 on GPU $2  (log: $EXP_ROOT/$3/logs/)"
  bash scripts/supervisor.sh "$1" "$2" "$3" > "$EXP_ROOT/$3.supervisor.log" 2>&1 &
  PIDS+=($!); KEYS+=("$3")
}
launch "$TREATMENT_MODEL" 0 "$TREATMENT_KEY"
launch "$CONTROL_MODEL"   1 "$CONTROL_KEY"

echo "== [5/6] waiting (tail $EXP_ROOT/STATE.md for progress) =="
FAILED=()
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then echo "  ${KEYS[$i]}: OK"; else echo "  ${KEYS[$i]}: FAILED"; FAILED+=("${KEYS[$i]}"); fi
done

echo "== [6/6] treatment vs control comparison =="
CMP=()
for k in "$TREATMENT_KEY" "$CONTROL_KEY"; do
  [[ -d "$EXP_ROOT/$k/release/$k" ]] && CMP+=("$k=$EXP_ROOT/$k/release/$k")
done
if [[ ${#CMP[@]} -gt 0 ]]; then
  (cd assistant-axis && uv run python ../scripts/analyze_axis.py \
     --outdir "$EXP_ROOT/_compare" --roles90 ../roles_90.json --compare "${CMP[@]}") \
     || echo "  (comparison failed — per-model RESULTS.md still written)"
  # Fold the role-play gate rates into the comparison — a headline result in its own right.
  (cd assistant-axis && uv run python - <<'PY' || true
import json, pathlib, os
root = pathlib.Path(os.environ["EXP_ROOT"])
rows = []
for f in sorted(root.glob("gate_*.json")):
    d = json.load(open(f))
    rows.append(f"| {d['key']} | {d['verdict']} | {d['viable_fraction']:.0%} | {d['mean_score3_rate']:.1%} | {d['any_roleplay_fraction']:.0%} |")
if rows:
    cmp_path = root / "COMPARISON.md"
    text = cmp_path.read_text() if cmp_path.exists() else "# SPP treatment vs control\n"
    text += ("\n\n## Role-play capability (probe roles)\n\n"
             "| model | verdict | roles viable | mean score-3 rate | any role-play |\n|---|---|---|---|---|\n"
             + "\n".join(rows) + "\n")
    cmp_path.write_text(text)
    print("added role-play table to COMPARISON.md")
PY
  )
  (cd assistant-axis && uv run python ../scripts/upload_results.py \
     --exp "$EXP_ROOT" --key _compare --repo "$HF_RESULTS_REPO") || true
fi

echo
echo "== DONE =="
echo "  gate reports: $EXP_ROOT/gate_*.md"
echo "  per model:    $EXP_ROOT/<key>/RESULTS.md  (+ release/<key>/: assistant_axis.pt, default_vector.pt, role_vectors/)"
echo "  comparison:   $EXP_ROOT/COMPARISON.md"
echo "  progress log: $EXP_ROOT/STATE.md ; failures: $EXP_ROOT/attempts.log"
[[ ${#FAILED[@]} -gt 0 ]] && echo "  !! failed: ${FAILED[*]}"

case "${SHUTDOWN:-}" in
  stop) RP_ACTION="stop" ;;
  terminate) RP_ACTION="remove" ;;
  *) RP_ACTION="" ;;
esac

if [[ "${SAVE_TO_GIT:-0}" == "1" ]]; then
  echo "== saving reports to git =="
  mkdir -p results
  for k in "$TREATMENT_KEY" "$CONTROL_KEY"; do
    [[ -d "$EXP_ROOT/$k" ]] || continue
    mkdir -p "results/$k"
    cp -f "$EXP_ROOT/$k"/{RESULTS.md,summary.json} "results/$k/" 2>/dev/null || true
    cp -f "$EXP_ROOT/$k"/*.png "results/$k/" 2>/dev/null || true
  done
  cp -f "$EXP_ROOT"/COMPARISON.md "$EXP_ROOT"/STATE.md "$EXP_ROOT"/gate_*.md results/ 2>/dev/null || true
  git add -f results/ 2>/dev/null || true
  git -c user.name="spp-axis-pod" -c user.email="pod@spp-axis.local" \
    commit -q -m "results: run finished $(date -u +%FT%TZ)" || echo "  (nothing new to commit)"
  git pull --no-rebase --no-edit 2>/dev/null || true
  if git push; then
    echo "  reports pushed to remote"
  elif [[ "$RP_ACTION" == "remove" ]]; then
    echo "  !! git push FAILED — refusing to terminate; downgrading to 'stop' to keep data."
    RP_ACTION="stop"
  fi
fi

if [[ -n "$RP_ACTION" ]]; then
  if [[ "$RP_ACTION" == "remove" && ${#FAILED[@]} -gt 0 ]]; then
    echo "!! failures present — downgrading terminate to stop so data survives"; RP_ACTION="stop"
  fi
  if command -v runpodctl >/dev/null 2>&1 && [[ -n "${RUNPOD_POD_ID:-}" ]]; then
    echo "== SHUTDOWN=$SHUTDOWN -> runpodctl $RP_ACTION pod $RUNPOD_POD_ID =="
    runpodctl "$RP_ACTION" pod "$RUNPOD_POD_ID"
  else
    echo "!! cannot self-shutdown (runpodctl missing or RUNPOD_POD_ID unset) — pod left running."
  fi
fi
