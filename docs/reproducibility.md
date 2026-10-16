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
- **Weighting inputs:** GA-IW is computed once from the minibatch before candidate sampling. Its future-distance term uses `data.ego_future` and `data.neighbor_future_observed`; the resulting weight is shared across sampled candidates for that scene. This differs from candidate-specific future proximity and the on-policy neighbor-source description in the paper. Its TTC risk mapping is `exp(-TTC / scale)`, rather than the paper's inverse-TTC expression.
- **Reward proxies:** RRS operates directly on scene tensors. Collision overlap and rule terms use smooth geometric proxies; red-light proximity is not an independently validated violation detector. The implementation uses a TTC barrier, while the paper describes an inverse-TTC barrier. Softplus and quadratic penalties are not globally bounded.
- **Value and advantage:** the optional critic consumes detached per-sample flow-matching loss and RRS penalty, rather than only a graph-conditioned scene representation. Actor weights use transformed linear advantages, optionally restricted to positive values; they do not implement the paper's clipped exponential advantage weight. The custom actor–critic distillation path uses a sigmoid advantage gate, rather than the paper's capped ReLU gate.
- **Teacher and loss settings:** candidate source, candidate count, teacher checkpoint, gating, and retained imitation/consistency terms depend on the selected recipe. The source includes best-of-K selection, but the shipped defaults below do not establish the paper's same-model, best-of-four, more-expensive-teacher procedure. A configuration flag named `paper_aligned_objective` selects a code path; it is not a guarantee of equation-level equivalence.

These differences are documented without changing method behavior. Enabling individual options alone should not be treated as a verified paper reproduction.

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

[`make_mock_gaiw_stability.py`](../flow_planner/onpolicy/make_mock_gaiw_stability.py) is an illustration-only utility. It replaces selected metric rows with synthetic values and marks those rows with `suite_root=mock://gaiw_stability_illustration`, while other input rows can remain unchanged. Its output must not be used as measured experiment evidence.

The current stability plotter accepts cached inputs without automatically adding a synthetic-data watermark. Its CSV importer takes the payload's source label from the first row, so that label alone cannot establish the provenance of every row in a mixed input. Preserve the original row-level provenance and visibly label any rendered synthetic illustration. The existence of this utility does not establish the provenance of figures in the paper.
