# GRID-Planner

**Graph-Structured Relational Reward Shaping for Diffusion Planning**

Zhao Yang, Zezhong Qian, Yinan Shi, Longjun Liu, and Yawei Jueluo

ACM Multimedia 2026 · [Paper DOI](https://doi.org/10.1145/3767308.3835263)

GRID-Planner studies how relational scene information can guide trajectory
weighting, reward evaluation, and on-policy refinement for autonomous driving.
This research implementation builds on **Flow Planner**, retaining the
`flow_planner` Python namespace for compatibility with existing checkpoints
and Hydra configurations.

The repository includes graph-aware importance weighting (GA-IW), relational
reward surrogates (RRS), configurable on-policy optimization, self-distillation,
and nuPlan evaluation utilities.

## Repository layout

| Path | Purpose |
| --- | --- |
| `flow_planner/model/` | Flow Planner backbone and flow-matching trajectory generator |
| `flow_planner/core/` | Graph-aware weighting and training objectives |
| `flow_planner/reward/rrs.py` | Relational reward proxies |
| `flow_planner/onpolicy/` | Distillation, checkpoint selection, evaluation reports, ablation analysis |
| `flow_planner/script/` | Hydra training configurations |
| `flow_planner/nuplan_simulation/` | Planner configuration and scenario filters |
| `flow_planner/run_script/` | Training, simulation, and ablation launchers |
| `configs/environment.example.sh` | Machine-specific path configuration template |
| `data_process.py`, `data_process.sh` | nuPlan preprocessing and dataset indexing |
| `tests/` | Packaging, commands, CPU model/runtime, reward and reporting regression checks |

## Installation

Training and simulation target a Linux machine with NVIDIA GPUs. Use Python 3.10
and the PyTorch/CUDA build appropriate to your machine, together with the nuPlan
revision specified below.

```bash
git clone https://github.com/yangzhaojason/GRID-Planner.git
cd GRID-Planner
conda create -n grid-planner python=3.10
conda activate grid-planner
python -m pip install "setuptools>=68,<82" wheel

# Use the compatible runtime subset, not upstream's old Torch/Hydra pins.
python -m pip install -r requirements.txt -r requirements-nuplan.txt
git clone https://github.com/motional/nuplan-devkit.git nuplan-devkit
git -C nuplan-devkit checkout e9241677997dd86bfc0bcd44817ab04fe631405b
python -m pip install --no-deps --no-build-isolation -e ./nuplan-devkit
python -m pip install -e . --no-deps
python -m pip check
```

The devkit's original requirement files conflict with this project's Hydra 1.3.2
and Torch 2.3.0. `requirements-nuplan.txt` supplies the runtime imports needed for
preprocessing, simulation, and nuBoard; it does not provision upstream's remote
submission service or full devkit development suite. It is not a transitive lock
file. Record the devkit commit, CUDA version, and `pip freeze` with each experiment.

Download nuPlan data and maps separately using the
[nuPlan devkit instructions](https://github.com/motional/nuplan-devkit).
Dataset files, maps, and trained weights are not distributed with this repository.

## Configure paths

From the repository root:

```bash
cp configs/environment.example.sh configs/environment.local.sh
# Edit environment.local.sh for your data, maps, devkit, and output directories.
source configs/environment.local.sh
```

Launchers infer the repository location and honor environment overrides. Shared
defaults live in `flow_planner/run_script/common.sh`; local configuration and
outputs are ignored by Git. Key variables are:

| Variable | Meaning |
| --- | --- |
| `PROJECT_ROOT` | Repository root; inferred by launchers |
| `NUPLAN_DEVKIT_ROOT` | Source checkout containing `nuplan/planning/script/run_simulation.py` |
| `NUPLAN_DATA_ROOT`, `NUPLAN_MAPS_ROOT` | Dataset and map roots used by nuPlan |
| `TRAINING_DATA`, `TRAINING_JSON` | Processed `.npz` directory and filename index |
| `SAVE_DIR`, `NUPLAN_EXP_ROOT` | Training and simulation output roots |
| `CUDA_VISIBLE_DEVICES`, `NPROC_PER_NODE` | GPU selection and number of training processes |
| `PRETRAINED_CKPT` | Initial checkpoint for fine-tuning |
| `RUN_DIR` | A training run containing checkpoints and `.hydra/config.yaml` |

Training launchers default to eight GPU processes.
For a smaller machine, set both `CUDA_VISIBLE_DEVICES` and `NPROC_PER_NODE`,
then adjust the batch size to fit available memory.

## Prepare data

```bash
# NUPLAN_DATA_PATH must point to the raw trainval data directory.
NUPLAN_DATA_PATH=/path/to/nuplan/trainval \
NUPLAN_MAP_PATH=/path/to/nuplan/maps \
TOTAL_SCENARIOS=1000000 \
bash data_process.sh
```

The script writes processed `.npz` files to `TRAINING_DATA` and a sorted filename
list to `TRAINING_JSON`. Use a separate processed-data directory for each split:
the index includes every `.npz` file already present in that directory.
`python data_process.py --help` lists the preprocessing options, including
`--log_names`, `--output_json`, and `--shuffle_scenarios false`. Feature dimensions
are fixed by `DataProcessor`.
Changing feature dimensions requires matching processor and model changes.

Two dataset manifests are provided:

- `nuplan_train.json` contains 13,180 training log names and is the default
  preprocessing filter.
- `flow_planner_training.json` contains the existing one-million-sample index.
  Set `TRAINING_JSON="$PROJECT_ROOT/flow_planner_training.json"` to use it only
  when all corresponding processed files exist. For newly processed data, use
  the filename index generated by `data_process.sh`.

## Training and refinement

```bash
# Flow Planner baseline: GA-IW, RRS and RL are disabled by default.
bash flow_planner/run_script/launch_train.sh

# GA-IW fine-tuning from an existing compatible checkpoint.
PRETRAINED_CKPT=/path/to/base/latest.pth \
bash flow_planner/run_script/launch_train_graph_weighting_ft.sh

# GA-IW + safety-only RRS preset (rule/comfort weights remain zero).
PRETRAINED_CKPT=/path/to/base/latest.pth \
bash flow_planner/run_script/launch_train_graph_weighting_rrs_ft.sh

# Checkpoint evaluation and on-policy self-distillation cycles.
RUN_DIR=/path/to/training/run \
bash flow_planner/run_script/launch_onpolicy_opsd_cycle.sh
```

The cycle launcher selects a teacher checkpoint using nuPlan results, then
runs distillation. Its default objective is GRPO with one teacher sample;
GA-IW is disabled in the distillation configuration. Configure these options in
`flow_planner/script/core/opds_distill.yaml` or through Hydra overrides.

Hydra overrides can be appended to training commands, for example
`train.batch_size=32 train.epoch=1`. Preserve the generated `.hydra/config.yaml`
alongside each checkpoint. Do not rename the internal `flow_planner` modules or
configuration identifiers when loading existing checkpoints.

## nuPlan evaluation

Supply a compatible checkpoint and the configuration saved with that checkpoint:

```bash
CONFIG_FILE=/path/to/run/.hydra/config.yaml \
CKPT_FILE=/path/to/run/latest.pth \
SPLIT=val14 \
CHALLENGE=closed_loop_nonreactive_agents \
bash flow_planner/run_script/launch_sim_nuplan_base.sh

# Evaluate checkpoints from a particular training run.
RUN_DIR=/path/to/run \
bash flow_planner/run_script/launch_sim_nuplan_all_epochs.sh

python print_result.py /path/to/simulation/runner_report.parquet
```

Available filters are `val14`, `test14-hard`, and `test14-random`. Challenges are
`closed_loop_nonreactive_agents` and `closed_loop_reactive_agents`. The result
printer resolves `runner_report.parquet` to the associated aggregated metrics;
it also accepts the weighted-average metrics parquet directly when the matching
runner report is present. It rejects failed simulations, ambiguous reports,
non-finite scores, planner mismatches, and inconsistent scenario identities/counts.
Use `run_nuboard.ipynb` to visualize an evaluation after configuring its paths.

The reporting utilities expose the following nuPlan score summaries:

| Field | Definition |
| --- | --- |
| `collision_score_loss` | One minus the graded no-at-fault-collision score |
| `driving_compliance_loss` | One minus the mean of drivable-area and driving-direction scores |
| `comfort_failure_rate` | Fraction of scenarios failing at least one comfort constraint |

The `launch_*_ablation_suite.sh` scripts and `onpolicy/*collect.py` utilities
support further experiments. The mock stability-data generator is
**illustration-only**, and its output must not be used as measured results.
See [asset provenance](assets/README.md).

## Tests

```bash
python -m unittest discover -s tests -v
python -m compileall -q flow_planner data_process.py print_result.py
bash -n data_process.sh
for script in flow_planner/run_script/*.sh; do bash -n "$script"; done
```

With the installation above, these checks include a full-size model forward,
backward, and CFG sampling on synthetic CPU scenes, checkpoint portability,
training lifecycle checks, and Parquet reporting. Tests requiring training or
reporting dependencies are skipped when those dependencies are unavailable.
The wheel tests require `setuptools` and `wheel`.

CI includes a Python 3.10 CPU runtime job and lightweight Python 3.9/3.12 jobs.

## Citation and attribution

```bibtex
@inproceedings{yang2026gridplanner,
  title = {GRID-Planner: Graph-Structured Relational Reward Shaping for Diffusion Planning},
  author = {Yang, Zhao and Qian, Zezhong and Shi, Yinan and Liu, Longjun and Jueluo, Yawei},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year = {2026},
  doi = {10.1145/3767308.3835263},
  publisher = {Association for Computing Machinery}
}
```

This repository builds on [Flow Planner](https://github.com/DiffusionAD/Flow-Planner)
and uses preprocessing and planning foundations from
[Diffusion Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner)
and [nuPlan](https://github.com/motional/nuplan-devkit). Please also credit the
upstream work when using those components:

```bibtex
@inproceedings{tan2025flow,
  title = {Flow Matching-Based Autonomous Driving Planning with Advanced Interactive Behavior Modeling},
  author = {Tianyi Tan and Yinan Zheng and Ruiming Liang and Zexu Wang and Kexin Zheng and Jinliang Zheng and Jianxiong Li and Xianyuan Zhan and Jingjing Liu},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  year = {2025}
}

@inproceedings{zheng2025diffusionbased,
  title = {Diffusion-Based Planning for Autonomous Driving with Flexible Guidance},
  author = {Yinan Zheng and Ruiming Liang and Kexin Zheng and Jinliang Zheng and Liyuan Mao and Jianxiong Li and Weihao Gu and Rui Ai and Shengbo Eben Li and Xianyuan Zhan and Jingjing Liu},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year = {2025}
}
```

Code is distributed under the existing [MIT license](LICENSE), including the
upstream copyright notice. External datasets and dependencies retain their own
licenses; the paper's publication license is separate from the code license.
