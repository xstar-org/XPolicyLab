#!/usr/bin/env bash
# Run inside the Slurm allocation after loading the site's CUDA/compiler modules.
set -euo pipefail
umask 077
: "${Q0_BENCH_ROOT:?Set the isolated RoboDojo workspace}"
: "${Q0_CLUSTER_PREFIX:?Set a controller/simulator Python prefix}"
: "${Q0_REMOTE_RUN_DIR:?Set a new controller service archive directory}"
: "${Q0_PREVIEW_TASK_INFORMATION_MODE:?Set original, official_only or strict_official}"
: "${ROBODOJO_RUN_ID:?Set a unique evaluator run ID}"
: "${EVAL_NUM:?Set the episode count}"
[[ "$Q0_PREVIEW_TASK_INFORMATION_MODE" == original || "$Q0_PREVIEW_TASK_INFORMATION_MODE" == official_only || "$Q0_PREVIEW_TASK_INFORMATION_MODE" == strict_official ]]
[[ ! -e "$Q0_REMOTE_RUN_DIR" ]]
export PATH="$Q0_CLUSTER_PREFIX/bin:$PATH"
export PYTHONPATH="${Q0_PREVIEW_CONTROLLER_PYTHONPATH:+$Q0_PREVIEW_CONTROLLER_PYTHONPATH:}$Q0_BENCH_ROOT:$Q0_BENCH_ROOT/XPolicyLab"
export ROBODOJO_POLICY_REQUEST_TIMEOUT_S=10800
export CUDA_VISIBLE_DEVICES=0
unset Q0_PREVIEW_API_KEY OPENAI_API_KEY
POLICY="$Q0_BENCH_ROOT/XPolicyLab/policy/Q0_Preview"
agent_config="${Q0_AGENT_CONFIG:-$POLICY/config/agent.gpt62sol.json}"
task_name="${Q0_TASK_NAME:-push_T}"
port="$(bash "$Q0_BENCH_ROOT/XPolicyLab/utils/get_free_port.sh")"
server_pid=
client_pid=
stop_group() {
    local group_pid="$1"
    local grace_seconds="${2:-5}"
    kill -TERM -- "-$group_pid" 2>/dev/null || true
    for ((attempt=0; attempt<grace_seconds; attempt++)); do
        kill -0 -- "-$group_pid" 2>/dev/null || break
        sleep 1
    done
    kill -KILL -- "-$group_pid" 2>/dev/null || true
    wait "$group_pid" 2>/dev/null || true
}
cleanup() {
    trap - EXIT INT TERM
    if [[ -d "$Q0_REMOTE_RUN_DIR" ]]; then touch "$Q0_REMOTE_RUN_DIR/STOP"; fi
    if [[ -n "$client_pid" ]]; then
        stop_group "$client_pid"
    fi
    if [[ -n "$server_pid" ]]; then
        stop_group "$server_pid" 20
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ -n "${ROBODOJO_ASSET_MIRROR_CONFIG:-}" ]]; then
    export ROBODOJO_ASSET_MIRROR_AUDIT="$Q0_REMOTE_RUN_DIR/assets-mirror.json"
fi
setsid "$Q0_CLUSTER_PREFIX/bin/python" -u "$POLICY/tools/policy_service.py" \
    --root "$Q0_REMOTE_RUN_DIR" --port "$port" --task "$task_name" --seed 1 --gpu 0 \
    --python-prefix "$Q0_CLUSTER_PREFIX" --config "$agent_config" \
    --external-agent &
server_pid=$!
bash "$Q0_BENCH_ROOT/XPolicyLab/utils/wait_for_policy_server.sh" \
    localhost "$port" "$server_pid" 'Q0 controller' 1200
cd "$Q0_BENCH_ROOT"
# Use the native evaluator entrypoint with the standard launcher arguments.
# Keep boot-time Kit options explicit; a headless deployment may need to disable
# asynchronous rendering or pin PhysX scheduling before Kit creates its pools.
kit_args="--enable isaacsim.replicator.behavior --enable isaacsim.sensors.camera"
# Shared-user Kit preferences must neither affect this experiment nor be changed
# by its diagnostics. Each service has a fresh, explicitly named settings path.
kit_args+=" --/app/settings/loadUserConfig=false --/app/settings/persistent=false"
kit_args+=" --/app/userConfigPath=$Q0_REMOTE_RUN_DIR/kit-user.config.json ${Q0_SIM_KIT_ARGS:-}"
echo "[Q0] native evaluator Kit args: $kit_args"
setsid "$Q0_CLUSTER_PREFIX/bin/python" -u src/eval_client/main.py \
    --task_name "$task_name" --env_cfg_type arx_x5 --num_envs 1 --enable_cameras \
    --kit_args "$kit_args" --device_id 0 --policy_name Q0_Preview --port "$port" \
    --protocol ws --policy_server_url "ws://localhost:$port" --host localhost \
    --additional_info 'ckpt_name=api,action_type=joint' --seed 1 --headless &
client_pid=$!
while kill -0 "$client_pid" 2>/dev/null; do
    kill -0 "$server_pid" 2>/dev/null || { echo 'Controller service exited' >&2; exit 1; }
    sleep 1
done
wait "$client_pid"
client_pid=
echo '[MAIN] external-agent eval finished'
