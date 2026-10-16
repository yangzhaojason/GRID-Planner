# Reproducibility and implementation scope

This document maps the [GRID-Planner paper](https://doi.org/10.1145/3767308.3835263) to the released source and records the limits of reproducing its results from this checkout. The repository contains the main research components and configurable experimental variants. Their presence does not establish exact equivalence to every paper equation or recover the configuration used for a reported result.

## Paper-to-code mapping

Paths below are relative to the repository root.

| Component | Source | Implemented behavior |
| --- | --- | --- |
| Generative planner | [`flow_planner/model/flow_planner_model/`](../flow_planner/model/flow_planner_model/), [`script/model/flow_planner.yaml`](../flow_planner/script/model/flow_planner.yaml) | Flow Planner backbone, conditional optimal-transport flow matching, trajectory tokenization, and classifier-free guidance. The default sampler uses four midpoint ODE steps and a CFG scale of 1.8. |
| Graph-aware importance weighting (GA-IW) | [`core/flow_matching_core.py`](../flow_planner/core/flow_matching_core.py), `_compute_graph_aware_weight` | RBF interaction strength, nearby map-element counts, future-distance and TTC risk proxies, exponential weighting, clipping, and mean normalization. |
| Relational reward surrogate (RRS) | [`reward/rrs.py`](../flow_planner/reward/rrs.py) | Geometric safety, route/lane, red-light proximity, speed, and comfort penalties. Neighbor motion can use constant-velocity extrapolation or observed futures. Progress and low-speed penalties are additional configurable terms. |
| On-policy updates | [`core/flow_matching_core.py`](../flow_planner/core/flow_matching_core.py), `_apply_standard_rl_objective` | Samples trajectories from the planner on minibatch scenes, scores them with RRS, and weights a flow-matching surrogate. Supports GRPO-style group-relative and actor–critic variants. |
| Value baseline | [`core/rl_value_net.py`](../flow_planner/core/rl_value_net.py), [`core/opds_distill_core.py`](../flow_planner/core/opds_distill_core.py) | Optional MLP baseline with configurable regression loss. |
| Self-distillation | [`core/opds_distill_core.py`](../flow_planner/core/opds_distill_core.py), [`onpolicy/distill_trainer.py`](../flow_planner/onpolicy/distill_trainer.py) | Frozen-teacher or student-sourced targets, configurable candidate selection, trajectory distillation, and reward/advantage gates. The filename retains the historical `opds` spelling. |
| nuPlan evaluation | [`planner.py`](../flow_planner/planner.py), [`nuplan_simulation/`](../flow_planner/nuplan_simulation/), [`onpolicy/rollout_runner.py`](../flow_planner/onpolicy/rollout_runner.py) | Planner adapter, Val14/Test14-hard/Test14-random filters, and external nuPlan simulation invocation. |
| Experiment reporting | [`onpolicy/`](../flow_planner/onpolicy/), [`run_script/`](../flow_planner/run_script/) | Checkpoint selection, ablation launchers, report aggregation, and plotting utilities. These utilities require experiment outputs to produce measured results. |

Here, an on-policy training rollout is a trajectory sampled by the planner from a dataset scene. Closed-loop nuPlan simulation is a separate evaluation and checkpoint-selection operation; the training sampler does not itself advance the simulator or collect a new simulator-state dataset.

## Differences from the paper formulation

- **Generative objective:** the paper expresses its objective using score-based diffusion and denoising notation. This implementation retains Flow Planner's affine flow-matching path and ODE sampler, with `x_start` prediction by default.
- **Weighting inputs:** GA-IW is computed once from the minibatch before candidate sampling. Its future-distance term uses `data.ego_future` and `data.neighbor_future_observed`; the resulting weight is shared across sampled candidates for that scene. A shared positive scene weight cannot change within-scene best-of-K ranking and cancels from group-normalized GRPO advantages when reward clipping and numerical epsilon effects are absent. This differs from candidate-specific future proximity and the on-policy neighbor-source description in the paper. Its TTC risk mapping is `exp(-TTC / scale)`, rather than the paper's inverse-TTC expression.
- **Reward proxies:** RRS operates directly on scene tensors. Collision overlap and rule terms use smooth geometric proxies; red-light proximity is not an independently validated violation detector. The implementation uses a TTC barrier, while the paper describes an inverse-TTC barrier. Collision overlap is averaged over neighbor/time slots, rather than a binary collision event. The speed penalty compares horizon-mean speed with the maximum available route limit, rather than a per-time local limit. Softplus and quadratic penalties are not globally bounded. Missing observed-future frames are now masked per time step; the inherited XY-zero padding convention cannot distinguish a real neighbor exactly at the coordinate origin from missing data.
- **Value and advantage:** the optional critic consumes detached per-sample flow-matching loss and RRS penalty, rather than only a graph-conditioned scene representation. Since the reward target is derived from that penalty, the critic has direct access to its target through an input feature; learning this shortcut can suppress the advantage instead of estimating a scene-conditional value. Actor weights use transformed linear advantages, optionally restricted to positive values; they do not implement the paper's clipped exponential advantage weight. The custom actor–critic distillation path uses a sigmoid advantage gate, rather than the paper's capped ReLU gate.
- **Actor target:** the standard actor–critic path scores sampled trajectories but applies its advantage weights to the logged-trajectory flow-matching loss. The custom OPSD actor–critic path recomputes the flow-matching loss on sampled trajectories. These are different optimization objectives.
- **Teacher and loss settings:** candidate source, candidate count, teacher checkpoint, gating, and retained imitation/consistency terms depend on the selected recipe. The source includes best-of-K selection, but the shipped defaults below do not establish the paper's same-model, best-of-four, more-expensive-teacher procedure. The default frozen-teacher target comes from a training forward pass on noisy logged trajectories. Student distillation predictions are also conditioned on noisy logged trajectories; there is no independent expensive-teacher sampler-step setting in the trainer. The default GRPO branch bypasses the custom actor–critic advantage gate, so flags for that custom branch do not make GRPO implement the paper objective. A configuration flag named `paper_aligned_objective` selects a code path; it is not a guarantee of equation-level equivalence.

The audit documents these objective differences without redesigning them. Separate implementation fixes in the [release audit](release-audit.md) do change numerical behavior and require experimental revalidation. Enabling individual options alone should not be treated as a verified paper reproduction.

## Default recipes

The table describes defaults before environment variables or Hydra command-line overrides. Always retain the resolved configuration for the actual run.

| Recipe | Default behavior | Relation to the paper |
| --- | --- | --- |
| [`launch_train.sh`](../flow_planner/run_script/launch_train.sh) with `core=flow_matching` | Base training; GA-IW, RRS, and RL disabled. The main config specifies 400 epochs and batch size 2048. | Provides the base generative training stage, not the complete GRID-Planner pipeline. |
| [`launch_train_graph_weighting_ft.sh`](../flow_planner/run_script/launch_train_graph_weighting_ft.sh) | Fine-tuning with GA-IW enabled. | A weighting-stage recipe. |
| [`launch_train_graph_weighting_rrs_ft.sh`](../flow_planner/run_script/launch_train_graph_weighting_rrs_ft.sh) | Ten-epoch fine-tuning at `5e-5`, GA-IW enabled, safety-only RRS coefficients; rule and comfort coefficients are zero. | Does not activate the paper's full safety/rule/comfort combination by default. |
| [`core/opds_distill.yaml`](../flow_planner/script/core/opds_distill.yaml) | GRPO enabled; GA-IW disabled; safety-only RRS; a single frozen-teacher target by default. | An experimental distillation preset, not a complete paper-run configuration. |
| [`launch_onpolicy_opsd_cycle.sh`](../flow_planner/run_script/launch_onpolicy_opsd_cycle.sh) | Selects teacher checkpoints using nuPlan and runs three cycles of two distillation epochs. Inherits the preceding preset unless overridden. | Checkpoint selection differs from per-scene candidate selection; the default total distillation budget is six epochs. |
| [`launch_grid_obj_ablation_suite.sh`](../flow_planner/run_script/launch_grid_obj_ablation_suite.sh) | Enables GA-IW and actor–critic with safety-only RRS coefficients; uses a single frozen teacher (`teacher_selection_k=1`). | Provides objective ablation machinery, with the implementation differences listed above. |

The objective ablation suite defaults to `PRIMARY_SPLIT=val14` and `TEACHER_SELECT_SPLIT=val14`, and includes Val14 in `REPORT_SPLITS`. Record checkpoint pools, teacher selection, and best-checkpoint selection alongside reported scores so readers can distinguish selection scores from evaluation on other splits.

## Release evidence and external requirements

This checkout does not include pretrained checkpoints, the exact resolved training/evaluation configurations associated with the paper's reported results, or a complete set of measured benchmark logs and per-scenario reports supporting those results. It also does not contain an interPlan runner or interPlan configuration. The paper reports interPlan results, but a complete reproduction workflow is not provided here.

Training and nuPlan evaluation require separately obtained nuPlan data, maps, and the nuPlan devkit, as well as locally generated processed samples. Simulation requires compatible model weights and their saved Hydra configuration. Dataset file lists and scenario filters are not substitutes for those data and artifacts.

For a reproducible result, preserve the source revision, resolved Hydra configuration and overrides, dependency/devkit versions, data and map versions, split and scenario selection, random seeds, checkpoint identity, selection protocol, simulator outputs, and metric aggregation commands. Runtime comparisons additionally need hardware, batch size, and the timing scope; the paper's latency diagnostic is planner-side `compute_trajectory` runtime, excluding simulator overhead. A successful installation or lightweight source check does not verify training convergence or reproduce a benchmark score.

## Plot and synthetic-data provenance

[`gaiw_stability_plot.py`](../flow_planner/onpolicy/gaiw_stability_plot.py) and [`grid_obj_sample_eff_plot.py`](../flow_planner/onpolicy/grid_obj_sample_eff_plot.py) render suite outputs or cached JSON/CSV inputs. Cached input support does not verify that values came from a measured run. The sample-efficiency plot derives rollout counts from suite metadata; retain that metadata when interpreting its horizontal axis.

[`make_mock_gaiw_stability.py`](../flow_planner/onpolicy/make_mock_gaiw_stability.py) is an illustration-only utility. All exported rows are marked as synthetic, including unchanged input rows in a mixed illustration. The stability plotter preserves row-level source labels and automatically watermarks known synthetic inputs in every plot mode. Cached inputs without provenance still cannot establish that a result was measured. The existence of this utility does not establish the provenance of figures in the paper.

## Reporting semantics

Report parsing requires one finite final score, a positive integer scenario count,
a matching planner, identical unique scenario/log identities, and a successful
runner record for every counted scenario.
Teacher selection accepts only the exact successful simulation output directory.
Missing evaluation splits or mismatched selected checkpoints produce incomplete
results rather than borrowing validation scores.

The exported proxy names are `collision_score_loss` (one minus nuPlan's graded
no-at-fault-collision score), `driving_compliance_loss` (one minus the mean of
drivable-area and driving-direction scores), and `comfort_failure_rate` (fraction
failing any comfort constraint). These are not collision-event rate, lane distance,
or measured jerk. Historical `coll_down`, `lane_dev_down`, and `jerk_down` fields
remain explicit compatibility aliases; old cached rows without the new fields
must be regenerated from original reports before inclusion in public tables.
