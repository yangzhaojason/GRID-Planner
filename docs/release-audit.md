# Release audit — 2026-10-16

This audit separates source correctness from reproduction of the ACM MM 2026
results. The current checkout contains experimental objective variants and lacks
the run artifacts needed to establish that it reproduces the paper. It should
not be described as a verified reproduction package.

## Confirmed defects corrected

| Area | Failure reproduced | Correction and regression evidence |
| --- | --- | --- |
| Preprocessing | CLI passed an argparse namespace where `DataProcessor` requires a directory | Pass the output path; CLI integration test; import real devkit processor |
| Checkpoints and devices | Plain/DDP/EMA state dictionaries and CPU planner loading were inconsistent; non-distributed cleanup could destroy an absent process group | Strict shared weight loader, CPU staging, explicit device propagation, guarded lifecycle; checkpoint and one-epoch trainer tests |
| Distributed launch | Slurm read the wrong configuration port and could use global rank as local GPU | Use the configured DDP port and local device; mocked multi-node rank tests |
| Classifier-free guidance | Multi-scene latents did not follow the interleaved scene layout; explicit guidance scale was ignored | Match latent/scene ordering and honor the override; two-scene tensor regressions |
| Training context | Conditioning dropout modified tensors later used by reward evaluation | Out-of-place masking; original-input preservation test |
| Consistency objective | Final adjacent-token overlap was omitted; short trajectories could divide by zero | Include all adjacent overlaps and handle one/two-token cases |
| Neighbor validity | Zero-padded future frames could create spurious distance/TTC risk | Mask each observed future frame and both TTC velocity endpoints; disappearing/gapped/missing-neighbor tests |
| Reward activation | Broad exception handling could silently omit configured reward computation | Propagate assembly failures and reject missing reward predictions; fault-injection tests |
| Evaluation selection | Failed runs or searches across experiment directories could select stale reports | Exact output directory, successful process and runner records, finite score, matching planner and unique scenario/log identities/counts |
| Tables and plots | Missing test results could reuse validation values; proxy labels implied event rate/distance/jerk | Report incomplete results, check checkpoint identity, use explicit proxy definitions |
| Synthetic illustrations | Mixed synthetic input could lose row provenance and appear unmarked | Preserve row provenance and watermark every supported stability plot mode |
| Installation | Upstream Hydra/Torch requirement files conflict with project pins | Pin the audited nuPlan revision and provide a compatible runtime subset |

Model, consistency, and reward fixes alter numerical behavior. Existing
checkpoints remain loadable where their architecture matches, but previous
scores must not be assigned to this revised source without re-evaluation.

## Validation scope

Local validation uses Python 3.10.21, PyTorch 2.3.0, and nuPlan devkit commit
`e9241677997dd86bfc0bcd44817ab04fe631405b` on macOS CPU. Tests exercise the actual
Hydra model's forward/backward and CFG sampling on synthetic full-size scene
tensors; checkpoint loading; trainer lifecycle; actual Parquet report parsing;
plot provenance; distribution contents; CLI and launcher contracts. Real
preprocessing, planner, simulator, and nuBoard entrypoints were imported. Package
dependency consistency, source compilation, and shell syntax were checked.

The complete installed-environment run passed **74 tests with no skips** using
`python -m unittest discover -s tests -v`. The lighter Python 3.12 environment
passed 42 tests and skipped 17 dependency-dependent checks. All 19 shell scripts
passed `bash -n`, and `pip check` reported no broken requirements. The pinned
runtime also imported TensorBoard's entrypoint; setuptools is constrained below
82 because TensorBoard 2.11 still imports `pkg_resources`.

These checks do not run real-data preprocessing, CUDA/multi-node training,
closed-loop driving, map loading, or paper benchmark reproduction. A Linux CPU
CI job is included; its result is separate from the local macOS evidence.

A scan of 1,423 text blobs reachable from the inspected remote refs found no
matches for the selected high-confidence private-key, GitHub-token, or AWS
access-key patterns. This is a bounded pattern scan, not proof that every type
of credential or restricted artifact is absent.

## Unresolved reproduction evidence

The [paper-to-code notes](reproducibility.md) describe the remaining objective
differences: scene-level logged-future GA-IW, linear advantage weighting,
critic access to the reward penalty, different actor targets, sigmoid gating,
GT-conditioned distillation, and differences in reward definitions and defaults.
They require reconciliation with the actual experiment source/configuration;
changing equations to match the manuscript without rerunning experiments would
not recover the provenance of existing results.

Before claiming reproduction of a paper table, provide its source revision,
resolved configuration, data/map versions, scenario identities, seeds,
checkpoint/hash, selection protocol, per-scenario runner and metric files,
and aggregation command. The paper's interPlan evaluation additionally needs
its runner and configuration. None of those missing artifacts can be inferred
from a passing unit test or a plausible plot.
