# OpenWAMHistory

Use this RoboDojo policy with a history-enabled OpenWAM checkout and the
existing OpenWAM adapter environment. Set `policy_name=OpenWAMHistory`,
`openwam_root` to the checkout from [OpenWAM PR #30](https://github.com/xstar-org/OpenWAM/pull/30),
and `ckpt_dir` to the checkpoint to evaluate. Use the supplied batch rollout
entry, including for a single environment.

At 25 Hz the client sends observations every five executed actions and at each
replan. The server tracks observed frames separately per environment, resets
memory between episodes, and generates one chunk per environment. Fixed and
adaptive retrieval use the mode recorded in the checkpoint.
