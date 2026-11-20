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

## Subscription-authenticated Codex CLI

The subscription transport uses `config/agent.gpt62sol.codexcli.json`, with
`provider=codex_cli` and the same `gpt-6.2-sol` model and Q0 tool contracts.
Run `codex login status` on the Japan host and require a ChatGPT login. It does
not use a project API key, fall back to API billing, or send authentication to
the simulator. The implementation was informed by
[robot-human-pipeline-plan/free_codex.py](https://github.com/xstar-org/robot-human-pipeline-plan/blob/b9256b832fda3ce8c181878a3b865da489bae018/robot_human_pipeline/free_codex.py).

Each model call runs an isolated stdio app-server with only Q0 dynamic tools.
Original Responses history, including tool results and encrypted reasoning
items, is injected without flattening it into user text. Only the existing Q0
harness executes Python or robot actions. Native filesystem, shell, patch,
browser, app, MCP and extra-agent capabilities are disabled or rejected. The
CLI's read-only/date wrappers and an advertised but rejected `request_user_input`
tool remain visible differences; this is functional transport compatibility,
not a claim of identical provider prompts or deterministic model outputs.

Keep the API run active until the CLI has passed its actual scene preflight.
Prepare an independent private benchmark snapshot, then validate alongside it
only after checking GPU memory is sufficient:

```bash
python policy/Q0_Preview/experiments/push_t/run_distributed.py \
  --root /private/japan/cli-validation-001 \
  --remote-root /private/cluster-experiment \
  --benchmark-name benchmark-cli \
  --agent-config policy/Q0_Preview/config/agent.gpt62sol.codexcli.json \
  --ssh-config /private/ssh_config --ssh-host cluster \
  --site-setup /private/cluster-experiment/code/site_setup.sh \
  --cluster-prefix /absolute/cluster-python-prefix \
  --python-prefix /absolute/japan-python-prefix \
  --preflight-only --overlap-preflights --job-id EXISTING_ALLOCATION_ID
```

This runs one API-free integration episode and one real CLI episode, while the
API arm continues. It writes `VALIDATION_PASSED.json` only after native results,
real model tool calls and executed motion groups have passed the gates. It
records hashes of the source/configuration snapshot and the layouts. The
`stop_at_boundary.py` helper requires this CLI certificate before it can archive
complete API results and stop that service at an episode boundary. It never
cancels the Slurm allocation.

Start the formal CLI run in a fresh root using the same launch arguments, omit
the preflight/overlap flags, and add
`--validated-preflight /private/japan/cli-validation-001`. Changed source,
configuration or layout hashes invalidate reuse. Both CLI arms run all 50
layouts with the same transport; earlier API results remain a separate pilot.
No `--key-file` is accepted for Codex CLI.

For unattended continuation, `switch_after_validation.py --plan /private/plan.json`
waits for the CLI certificate, rechecks configuration/source/layout hashes, and
only then arms the completed-episode boundary stop. Its trusted plan names the
validation root, SSH configuration, remote snapshot/setup, API service/result
identity, CLI config, and formal CLI argv/root/log. It contains no credentials.
It preserves the native API result snapshot and launches the validated CLI
50+50 command. Failed CLI validation or a changed deployment leaves the API
service untouched. The transition state and the formal experiment log remain
durable on the Japan host.

### Recovering an interrupted formal run

Observation transport uses a dedicated SSH TCP connection. A transient EOF or
pipe timeout can reconnect within 60 seconds while retaining the same live
Agent, history, observation identity and pending reply. Lost acknowledgements
are safe to resend through the controller's existing idempotent reply check.
Protocol, clock, hash and local-file failures still stop the run. Reconnection
does not extend planning/checkpoint deadlines or erase infrastructure errors.
Management RPCs use separate short-lived connections.

If the bounded recovery is exhausted, retain the interrupted archive and start
a fresh driver root with the same validated launch arguments plus
`--resume-from /private/japan/previous-paired-run`. The source worker must be
terminal. The trusted native resume helper verifies the official result prefix
against its checkpoint, preserves all completed successes AND failures, and
initializes a fresh service with only the remaining layouts. The unfinished
episode starts from its saved initial layout; it is not counted as a failure.
`resume-lineage.json` records native hashes and inherited policy archive paths,
including across multiple recovery segments. Existing outcomes and videos are
not overwritten. Changed transport/model configuration, layouts, arm identity
or inconsistent native counters are rejected. No API pilot outcomes are imported
into a CLI comparison.

If native shutdown records an adapter-aborted episode as a failed result,
`--exclude-infrastructure-tail original` (or `official_only`) requires an
unclosed final policy archive with an explicit infrastructure-error file. It
can remove only that single failed tail in a fresh target result, after checking
the entire original result against its checkpoint. It preserves earlier genuine
failures. The certificate includes the excluded layout, private error path and
SHA-256; original result/checkpoint bytes are retained with their hashes. This is
an explicit recovery operation, not an automatic retry of model failures.

### Native task count and extending a completed prefix

The official `push_T` task configuration defaults to **25** evaluations.
`EVAL_NUM=50` does not override that default: the native entrypoint takes the
minimum. A formal submission now rejects a snapshot unless its task block
explicitly has `eval_nums: 50`. To meet this experiment's requested 50 layouts,
create a new immutable snapshot changing only that task count, retain identical
layout hashes, and repeat smoke/model preflight for the new deployment. Physics,
scoring and policy inputs stay unchanged; this 50-layout experiment is an
extension of the official default protocol, not a default leaderboard run.

The first CLI segment completed all 25 native default cases (22 successes,
3 failures) before the driver rejected the incomplete 50-case arm. Native normal
completion deletes its checkpoint. `--extend-completed-prefix original` with
`--resume-benchmark-name <original-snapshot>` permits an audited extension into
the new snapshot: require normal worker exit, native completion log, all 25
closed policy archives without infrastructure errors, exact result counters,
and an otherwise byte-identical task configuration changing 25 to 50. Rebuild
only the target checkpoint; keep the source result and all original outcomes
immutable. The certificate records source/config/log/archive hashes. This flag
cannot exclude any outcome or be combined with infrastructure-tail exclusion.

## Explicit parallel execution of the second arm

The user authorized running `official_only` concurrently with `original` on the
same allocated GPU. Each service needs its own controller/agent process groups,
port, Kit preferences and archive root. Verify the allocation/GPU UUID, available
GPU memory and host/cgroup memory before starting the second simulator. A second
Slurm step uses the existing allocation with `--overlap`; never release or replace
the allocation or stop the active original arm.

An already running serial driver will otherwise submit its own official-only arm
after the original 50 finish. `reserve_parallel_arm.py` requires explicit operator
authorization and reserves only the still-unsubmitted serial official-only launcher.
The existing immutable-launcher check then rejects that duplicate **before queueing**.
It does not stop either active service. The old serial driver saves the complete
original result first, then is expected to exit at that duplicate-submission guard.
This administrative exit is separate from native episode outcomes.

The delegated official-only service has a distinct identity and result archive.
After both arms have all 50 native results, pass their separate result paths to
`report_pair.py`. Full-arm validation remains mandatory even when `--layout-ids`
selects the matching Fast subset (29 through 49). Preserve all genuine failures,
source lineage, the delegation certificate, speed-transition evidence and both
archive roots in the final report.

## Subscription CLI speed

`agent.gpt62sol.codexcli.json` now requests `codex_service_tier: "fast"` following
the user's explicit speed-change instruction. Set it to `"default"` (or omit it)
for Standard. Fast sets the native CLI service tier to `fast`, requests the wire
tier `priority`, and requires `thread/start` to acknowledge `fast`/`priority`.
Each CLI receipt records requested and acknowledged tiers. The model and existing
planning/checkpoint reasoning settings are unchanged; unsupported or unacknowledged
Fast requests fail closed rather than silently falling back.

A real ChatGPT subscription probe verified both ordinary assistant completion and
the subsequent native Q0 tool handoff with Fast acknowledged, using `gpt-6.2-sol`.
This checks transport behavior, not a guaranteed wall-time speedup. Fast consumes
more subscription allowance. See [official speed controls](https://learn.chatgpt.com/docs/agent-configuration/speed)
and [pricing](https://learn.chatgpt.com/docs/pricing?codex-pricing-speed=fast).

For the running experiment, the current Standard episode finishes before fresh
episode processes load Fast. Preserve all earlier outcomes and record the live
transition separately from launch-time metadata. The final report must disclose
the Standard/Fast boundary; do not describe all 100 episodes as having one tier.
Report the matching Fast-layout subset separately when assessing that confound.

## Optional API baseline access

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
overlay as `sitecustomize.py`. Explicit one-episode diagnostic submissions can
set `traceback_interval_s` (for example 120); regular preflights and both formal
arms leave this instrumentation off. The driver aborts if no policy observation
arrives within 20 minutes.
Transport compresses files losslessly, verifies decompressed size and SHA-256,
and never recompresses camera images. To avoid transferring redundant RGB arrays,
the API host uses the original marker-aware JPEG decoder to reconstruct their
NPY files, then checks their original byte count and SHA-256 before publishing
READY. Any decoder difference fails the gate instead of changing model inputs.
Failed launch cleanup
terminates only its own process groups, escalating after a short grace period
so an Isaac Sim child cannot keep the GPU occupied after its shell exits.
The allocation launcher invokes the native `src/eval_client/main.py` with the
standard single-environment camera/WS arguments. Optional `Q0_SIM_KIT_ARGS` are
logged and passed at Kit startup, allowing deployment diagnostics without editing
the shared evaluator launcher. Freeze the chosen Kit settings for both arms.
If scene initialization needs NVIDIA assets unavailable from the compute node,
configure the companion evaluator's verified local mirror through
`ROBODOJO_ASSET_MIRROR_CONFIG` in that site setup script. The arm launcher saves
its byte-verification audit in the service archive. See
[RoboDojo mirror configuration](https://github.com/xstar-org/RoboDojo/pull/28).
Publish deployment changes atomically between runs; never overwrite a shell
script while a process is executing it.

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

## Deployment validation status (2026-11-19)

The Japan API host has completed a minimal Responses request using the exact
configured model, and the isolated Bubblewrap Python tool environment imports
its numerical and image dependencies. Transport tests validate compressed
observation bytes and reply identity independently of the simulator.

The first-reset PhysX stall reproduced in both `push_T` and an API-free
`stack_bowls` control. A SHA-256-verified local material mirror removed it with
the original scheduling defaults. The complete `push_T` integration gate now
passes: it delivers camera observations, completes four scripted motion groups,
exercises both CONTINUE and REPLAN checkpoints, reaches the native 600-step
horizon, saves three camera videos and an evaluator result, and exits with code 0.
The API `gpt-6.2-sol` preflight completed successfully under the native evaluator.
A gateway disconnect interrupted the API arm; native resume restored all
completed outcomes and restarted only the unfinished episode. Deployment transfer
and active evaluation use separate SSH connections. After full CLI validation,
the API service stopped at six completed episodes (five successes, one failure),
whose native results remain a separate pilot archive.

The subscription CLI has passed actual initial-image and tool-result-image
probes, isolated Python execution, trajectory submission and checkpoint review.
Its independent API-free scene gate and real CLI scene preflight both passed;
the latter succeeded under the official evaluator. Source/configuration/layout
hashes matched before the transition. The formal CLI original-information arm
is running, followed automatically by the official-description-only arm.
No complete CLI 50+50 result is claimed.

During the formal CLI arm, a later gateway timeout exhausted the original
non-reconnecting bridge. Three completed outcomes remained intact. The updated
client adds bounded reconnects and native-prefix continuation; the frozen
simulator/model snapshot and all successful-message contents remain unchanged.
The continuation retains those outcomes and restarts only the interrupted
fourth episode. Transport regression tests include a real pipe disconnect after
server-side reply publication, checking that the reply is not published twice.
One measured observation shrank from 2,198,346 to 336,097 wire bytes (84.7%) while
retaining byte-identical decoded arrays.

The continuation reached five completed cases (four successes, one failure).
In the sixth case the model returned a normal completed text-only answer, which
the first CLI adapter incorrectly classified as a transport failure. The adapter
now returns this native assistant message to the existing Q0 no-action loop,
matching the API path without extra prompts or a reset of the planning budget.
Failed/interrupted native turns still abort. The sixth case was interrupted by
that adapter error and requires the audited infrastructure-tail exclusion above;
its shutdown-generated failure is not a valid model outcome. These partial
counts are diagnostic only; the final report still requires both arms of 50.
The trusted allocation helper accepts an optional `task` identifier for such
infrastructure controls; the paired experiment remains fixed to `push_T`.
Control episodes are excluded from reported evaluation results.

The isolated deployment uses the camera backend from the existing RoboDojo
[PR #26](https://github.com/xstar-org/RoboDojo/pull/26) reference, and the client
timeout forwarding and material mirror from companion
[PR #28](https://github.com/xstar-org/RoboDojo/pull/28).
It leaves shared assets and environments unchanged. Both formal arms must use
the same frozen deployment settings.

## Local verification

```bash
python3 -m unittest discover -s tests -p 'test_q0_task_information.py' -v
python3 -m unittest discover -s tests -p 'test_q0_pair_experiment.py' -v
python3 -m unittest discover -s tests -p 'test_q0_remote_bridge.py' -v
bash -n policy/Q0_Preview/experiments/push_t/run_pair.sh
bash -n policy/Q0_Preview/experiments/push_t/slurm_environment.sbatch
```
