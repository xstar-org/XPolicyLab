#!/usr/bin/env bash
# Run from an isolated benchmark workspace whose XPolicyLab contains this change.
set -euo pipefail
umask 077
: "${Q0_PAIR_ROOT:?Set a new absolute output directory}"
: "${Q0_LAYOUT_DIR:?Set the official RoboDojo/arx_x5/seed-1 layout directory}"
: "${Q0_POLICY_ENV:?Set the policy Conda environment or Python prefix}"
: "${Q0_EVAL_ENV:?Set the existing simulator Conda environment name}"
: "${Q0_PREVIEW_API_KEY:?Claim subscription API credits and configure a project key first}"
[[ "$Q0_PAIR_ROOT" = /* ]] || { echo 'Q0_PAIR_ROOT must be absolute' >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="$(cd "$HERE/../.." && pwd)"
XPL_ROOT="$(cd "$POLICY/../.." && pwd)"
BENCH_ROOT="$(cd "$XPL_ROOT/.." && pwd)"
mkdir -m 700 "$Q0_PAIR_ROOT"
export EVAL_NUM=50
export Q0_PREVIEW_AGENT_CONFIG="$POLICY/config/agent.gpt62sol.json"
python3 "$HERE/freeze_layouts.py" --layout-dir "$Q0_LAYOUT_DIR" --manifest "$Q0_PAIR_ROOT/layouts.json"
for mode in original official_only; do
    python3 "$HERE/freeze_layouts.py" --layout-dir "$Q0_LAYOUT_DIR" --manifest "$Q0_PAIR_ROOT/layouts.json" --verify
    export Q0_PREVIEW_TASK_INFORMATION_MODE="$mode"
    export Q0_PREVIEW_RUN_DIR="$Q0_PAIR_ROOT/$mode"
    export ROBODOJO_RUN_ID="$(basename "$Q0_PAIR_ROOT")-$mode"
    bash "$POLICY/eval.sh" RoboDojo push_T api arx_x5 joint 1 0 0 "$Q0_POLICY_ENV" "$Q0_EVAL_ENV" \
        >"$Q0_PAIR_ROOT/$mode.log" 2>&1
    result="$BENCH_ROOT/eval_result/RoboDojo/push_T/Q0_Preview/arx_x5/1_ckpt_name=api,action_type=joint/$ROBODOJO_RUN_ID/_result.json"
    cp "$result" "$Q0_PAIR_ROOT/$mode-result.json"
done
python3 "$HERE/report_pair.py" --original "$Q0_PAIR_ROOT/original-result.json" \
    --official-only "$Q0_PAIR_ROOT/official_only-result.json" --output "$Q0_PAIR_ROOT/report.json"
