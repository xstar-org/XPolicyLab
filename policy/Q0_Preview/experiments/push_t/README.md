# Q0-Preview: paired push_T information ablation

This experiment compares Q0's original task information with the official task
description and public scoring text, using `gpt-6.2-sol` for both arms. This is a
new model substitution experiment, not a reproduction of the original leaderboard
model's numerical result. No task success result is included in this change.

## Protocol

- Standard `push_T`, `arx_x5`, joint control, evaluation seed 1, 50 episodes per arm.
- Run `original` first, then `official_only`, in fresh services and fresh workspaces.
- Freeze the same 50 layout files and their SHA-256 hashes before running either arm.
  If the asset directory has more than 50 standard layouts, use the first 50 in
  numeric order, matching native SeedManager with `EVAL_NUM=50`; do not select by outcome.
- Keep observations, tools, robot controller, planning/checkpoint budgets, API model,
  and reasoning settings unchanged. Existing OpenAI planning/checkpoint settings
  remain `xhigh` / `low`; only task information and its prompt references change.
- `original` retains Q0's task card, reference and prompt. `official_only` uses the
  frozen official description plus public scoring, supplies no task reference file,
  and removes prompt instructions to consult restricted strategies/evaluation notes.
  Original completion, process and assessment notes are not loaded in that mode.
  Missing official descriptions and stale reference files cause an error.
- The 600-control-step budget is runtime metadata and is identical in both arms;
  it is not an added target-position or scoring tolerance.
- Score only with the simulator evaluator. Require all 50 unique layout IDs in
  each arm before reporting success/50, Wilson 95% intervals, paired outcomes and
  the exact two-sided McNemar test. Interrupted/incomplete runs are not converted
  into success rates. Preserve infrastructure failures separately.

The official text and retrieval provenance are frozen in
[`official_task_descriptions.json`](../../assets/tasks/official_task_descriptions.json).
This mode currently supports only `push_T`; other task names fail closed.
It tests the effect of bundled task information, not whether the API model has
ever encountered the benchmark during pretraining. Public scoring is retained
because it is part of the official task specification.

## API access

Claim eligible subscription API credits through OpenAI's
[official credit flow](https://developers.openai.com/api/docs/guides/chatgpt-api-credits)
and use a normal project API key in `Q0_PREVIEW_API_KEY`. A subscription login or
Codex OAuth token is not an API key. Before the paid run, verify the claimed grant,
remaining balance and model access in the selected API organization/project.
For a grant-only run, do not enable payment recharge or switch to a paid project.
Never put a key in a request JSON, source file, commit or log.

## Independent Slurm simulation environment

Submit one GPU allocation independently of other running evaluation jobs:

```bash
export Q0_EXPERIMENT_ROOT=/absolute/private/experiment
export Q0_EXPERIMENT_CODE=/absolute/path/to/this/directory
export Q0_SIM_PYTHON=/absolute/path/to/existing/sim-python
umask 077
mkdir -p "$Q0_EXPERIMENT_ROOT/logs"
sbatch --output="$Q0_EXPERIMENT_ROOT/logs/slurm-%j.log" \
  "$Q0_EXPERIMENT_CODE/slurm_environment.sbatch"
```

The default is 1 RTX 4090, 8 CPUs and a **7-day time limit**. The worker qualifies
CUDA, Isaac Sim physics and RGB rendering, then writes `jobs/JOB_ID/READY.json`
and waits for requests until Slurm expiry. This smoke check does not run `push_T`
or validate the policy's CuRobo/sandbox installation. Qualification failures are
recorded and terminate the worker. It never cancels another allocation.

The queue is a private, trusted-user command queue, not a network service or an
agent tool. Request JSON contains `argv` (a list of arguments) and `cwd` (an
existing directory under the experiment root). Publish a request by atomic rename
into `requests/NAME.json`. Results go to `results/NAME.json`; process logs go to
`logs/NAME.log`. Existing claims/results are never automatically rerun.
Only one worker may own a root; use a new root for a new allocation.

## Launch the paired evaluation

### Agent and controller on separate hosts

The API host runs `tools/remote_agent.py` and its isolated CPU Python tools. The
Slurm node runs `tools/policy_service.py --external-agent`, CuRobo and Isaac Sim.
Only the API host receives `Q0_PREVIEW_API_KEY`. No API endpoint is proxied and no
inbound port on the cluster is needed: the API host opens an authenticated SSH
stdio session to `tools/bridge_exchange.py` on the cluster login host, reading
the controller's private shared filesystem.

The bridge transfers only published episode metadata, current observations,
termination markers and attributed action/checkpoint replies. Image files remain
byte-identical and are checked against their manifest hashes. It publishes local
READY only after all files arrive, rejects stale/conflicting replies, checks host
clocks after the SSH handshake, and maintains a controller heartbeat. Connection
or Agent infrastructure failure aborts the run rather than inventing task results.
Keep both hosts' clocks synchronized. API waits do not advance simulator actions;
25 Hz control and motion execution stay entirely on the allocated GPU node.
Use an experiment-private SSH ControlMaster/ControlPersist socket so status checks
and the bridge share one authenticated connection. Put the socket in a short,
owner-only path (Unix-domain socket paths are length-limited). Management SSH
failures have a bounded retry budget; an identical submission is idempotent and
does not requeue an episode. Transport timing checks allow network/scheduling
latency, and the timeout covers receiving an entire response, including images.

Example on the API host (credential already loaded into the environment):

```bash
export Q0_PREVIEW_XPL_ROOT=/absolute/path/to/benchmark-workspace
export Q0_PREVIEW_TASK_INFORMATION_MODE=original
python policy/Q0_Preview/tools/remote_agent.py \
  --ssh-config /private/ssh_config --ssh-host cluster \
  --remote-root /private/experiment/runs/original \
  --remote-code /private/benchmark/XPolicyLab/policy/Q0_Preview/tools/bridge_exchange.py \
  --root /private/local-agent-original \
  --config policy/Q0_Preview/config/agent.gpt62sol.json \
  --python-prefix /absolute/agent-python-prefix
```

Start this before the controller service. Its remote root must match the service
run directory exactly. For the API-free integration gate add `--smoke --two-group`;
it exercises both CONTINUE and REPLAN without a model call. Never use this scripted
gate as a task-success measurement. In a real model preflight, omit those flags.

Inside the allocation, `run_external_arm.sh` launches the controller and native
evaluator without requiring Conda activation scripts. Set `Q0_BENCH_ROOT`,
`Q0_CLUSTER_PREFIX`, `Q0_REMOTE_RUN_DIR`, `Q0_PREVIEW_TASK_INFORMATION_MODE`,
`ROBODOJO_RUN_ID` and `EVAL_NUM`; load the site's CUDA/compiler environment first.
Use one environment and enable the official camera intrinsics/extrinsics. The
launcher explicitly removes API key variables. The normal eval.sh entry also
accepts `Q0_PREVIEW_EXTERNAL_AGENT=1`. A controller with no bridge heartbeat for
180 seconds stops; it never silently runs without an Agent. Use identical layouts
and settings for both information modes and collect the same paired evaluator
statistics described below.

For unattended execution, run `run_distributed.py` on the API host inside tmux or
a service manager. Its sequence is an API-free two-group smoke, a separate real
model episode, 50 original episodes, then 50 official-only episodes. Preflight
episodes are excluded from the reported comparison. Both checkpoint decisions
must pass the smoke; the real-model episode must complete an action group before
either 50-episode arm starts. The driver rechecks layout hashes between stages,
stops on Agent/transport/evaluator errors, and writes `report.json` only after
both complete evaluator results pass identity/count checks. Use a fresh output
directory and unique basename for each attempt; it never resumes automatically.

```bash
python policy/Q0_Preview/experiments/push_t/run_distributed.py \
  --root /private/api-host/paired-001 --remote-root /private/cluster-experiment \
  --ssh-config /private/ssh_config --ssh-host cluster \
  --site-setup /private/cluster-experiment/code/site_setup.sh \
  --cluster-prefix /absolute/cluster-python-prefix \
  --python-prefix /absolute/api-host-python-prefix \
  --key-file /private/openai-api-key.txt
```

The site setup file loads existing CUDA/compiler modules. If shared environments
have older transport dependencies, install a compatible wheel into the private
`code/overlay` directory; the controller helper uses
`Q0_PREVIEW_CONTROLLER_PYTHONPATH` to expose it without changing shared packages.
For reset/startup diagnostics, copy `debug_sitecustomize.py` into that private
overlay as `sitecustomize.py`. The deployment helper enables evaluator-only
traceback dumps in one-episode preflights and disables them in both 50-episode
arms. The driver aborts if no policy observation arrives within 20 minutes.
Transport compresses files losslessly, verifies decompressed size and SHA-256,
and never regenerates or recompresses the camera images. Failed launch cleanup
terminates only its own process groups, escalating after a short grace period
so an Isaac Sim child cannot keep the GPU occupied after its shell exits.
The allocation launcher invokes the native `src/eval_client/main.py` with the
standard single-environment camera/WS arguments. Optional `Q0_SIM_KIT_ARGS` are
logged and passed at Kit startup, allowing deployment diagnostics without editing
the shared evaluator launcher. Freeze the chosen Kit settings for both arms.

### Single-host alternative

Prepare an isolated RoboDojo workspace with this XPolicyLab checkout as its
`XPolicyLab/` child. Follow Q0's [installation instructions](../../INSTALLATION.md),
including current camera intrinsics/extrinsics. Validate CuRobo in the policy
environment and an actual isolated Python tool session before running. Bubblewrap
must support unprivileged namespaces on the compute node; do not replace it with
host execution if unavailable. The simulator and policy may share the allocated
GPU (both local GPU indices are 0).

In a private launcher under the experiment root, load the credential from a
separate owner-readable secret file, export the following variables, and invoke:

```bash
export Q0_PAIR_ROOT=/absolute/private/experiment/paired-run-001  # must not exist
export Q0_LAYOUT_DIR=/absolute/path/to/assets/Eval_Layout/RoboDojo/arx_x5/1
export Q0_POLICY_ENV=/absolute/path/to/policy-environment
export Q0_EVAL_ENV=RoboDojo  # existing Conda environment name
# Q0_PREVIEW_API_KEY is already loaded privately; do not echo it.
bash /absolute/path/to/XPolicyLab/policy/Q0_Preview/experiments/push_t/run_pair.sh
```

The simulator environment must be activated through the cluster's normal module/
Conda setup. The stock evaluator must write per-episode `layout_id`, boolean
`success` and `_result.json`; the result collector rejects older ambiguous formats.
The launcher verifies layout hashes before each arm, saves each evaluator result,
and writes `report.json` only after both arms finish. Keep the raw evaluator logs,
videos and Q0 request/tool traces alongside the report. Inspect API errors and
credit availability before restarting an interrupted run; the script never
automatically reuses an existing output directory.

## Local verification

```bash
python3 -m unittest discover -s tests -p 'test_q0_task_information.py' -v
python3 -m unittest discover -s tests -p 'test_q0_pair_experiment.py' -v
bash -n policy/Q0_Preview/experiments/push_t/run_pair.sh
bash -n policy/Q0_Preview/experiments/push_t/slurm_environment.sbatch
```
