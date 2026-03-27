from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from flow_planner.core.flow_matching_core import FlowMatchingCore
from flow_planner.core.rl_value_net import ValueMLP
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.model.model_utils.traj_tool import traj_chunking


@dataclass
class OPSDDistillConfig:
    enabled: bool = True
    teacher_ckpt_path: str = ""
    # distillation weights
    distill_weight: float = 1.0
    distill_warmup_epochs: int = 0
    # optional scalar gate (run-level) to scale distill; kept simple for MVP
    gate_value: float = 1.0
    # trajectory weights
    xy_weight: float = 1.0
    heading_weight: float = 0.2
    # robust loss
    huber_beta: float = 1.0
    # unified objective switches
    unified_with_rl: bool = True
    # distill weight w(G); can be dynamically adapted by advantage
    dynamic_gate_with_adv: bool = True
    adv_temperature: float = 1.0
    # stabilize advantage influence
    adv_clip: float = 10.0
    # strict paper-aligned unified objective:
    # L = E[e_diff * stopgrad(A) + omega(A) * D] + lambda_v * L_v
    paper_aligned_objective: bool = True
    # keep consistency loss as extra regularizer in paper_aligned mode
    keep_consistency_reg: bool = True
    # sample τ ~ πθ via inference (not forward_train)
    sample_use_cfg: bool = True
    sample_cfg_weight: float = 1.0
    # legacy ablation switch; default is the standard sign e_diff * A
    actor_use_negative_adv: bool = False
    # more stable default: only reinforce positive-advantage rollouts
    positive_only_adv: bool = True
    # keep base supervised objective in unified path for stability
    paper_keep_base_loss: bool = True
    # keep a small supervised anchor on FM planning loss for stability
    fm_anchor_weight: float = 0.2
    # match student/teacher inference noise when sampling tau
    rollout_match_noise: bool = True
    rollout_seed_base: int = 3402
    # per-scenario teacher selection from rollout candidates
    teacher_selection_mode: str = "single"   # single | best_of_k
    teacher_selection_source: str = "teacher"  # teacher | student
    teacher_selection_k: int = 1
    # gate distillation only when teacher rollout proxy reward improves over student
    improvement_aware_gate: bool = False
    improvement_margin: float = 0.0
    improvement_normalize_by_mean: bool = True
    improvement_max_multiplier: float = 3.0
    # split rollout inference into smaller chunks to avoid OOM
    rollout_micro_batch_size: int = 64


def _heading_from_cossin(xycossin: torch.Tensor) -> torch.Tensor:
    """xycossin: (..., 4) -> heading (...,)"""
    return torch.atan2(xycossin[..., 3], xycossin[..., 2])


def _angle_wrap(diff: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(diff), torch.cos(diff))


class OPSDDistillCore(FlowMatchingCore):
    """
    OPSD distillation core: train student with the normal FlowMatchingCore losses
    PLUS a distillation loss from a frozen teacher checkpoint.

    MVP behavior:
      - Teacher is a fixed checkpoint (selected by simulation score externally)
      - Distill on student/teacher predicted denormalized trajectories (x,y,cos,sin)
      - Distill weight is a simple scalar gate (run-level). Per-scenario gating can be added later.
    """

    def __init__(self, opsd: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        self.opsd_cfg_dict = opsd or {}
        self.opsd_cfg = OPSDDistillConfig(**{k: v for k, v in self.opsd_cfg_dict.items() if k in OPSDDistillConfig.__annotations__})
        self._teacher_model: Optional[nn.Module] = None
        self._rollout_counter: int = 0

    def attach_teacher(self, teacher_model: nn.Module, teacher_ckpt_path: str, device: str):
        """
        Attach and load a frozen teacher model. Call once in trainer after building student model.
        """
        ckpt = torch.load(teacher_ckpt_path, weights_only=True, map_location=device)
        # training checkpoints saved by our trainer store ema_state_dict under 'ema_state_dict'
        state = ckpt.get("ema_state_dict", ckpt.get("model", ckpt))
        if not isinstance(state, dict):
            raise ValueError(f"Unexpected teacher checkpoint format at {teacher_ckpt_path}")
        # strip DDP prefix if present; otherwise load as-is
        if any(k.startswith("module.") for k in state.keys()):
            model_state = {k[len("module."):]: v for k, v in state.items() if k.startswith("module.")}
        else:
            model_state = state
        teacher_model.load_state_dict(model_state, strict=True)
        teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad_(False)
        self._teacher_model = teacher_model

    def _effective_distill_weight(self) -> float:
        weight = float(self.opsd_cfg.distill_weight)
        warmup_epochs = int(self.opsd_cfg.distill_warmup_epochs)
        if warmup_epochs > 0:
            weight = weight * min(1.0, max(0.0, (self._epoch + 1) / float(warmup_epochs)))
        return weight

    def _distill_loss(
        self,
        student_pred_denorm: torch.Tensor,  # (B, T, 4)
        teacher_pred_denorm: torch.Tensor,  # (B, T, 4)
    ) -> torch.Tensor:
        # XY loss
        s_xy = student_pred_denorm[..., 0:2]
        t_xy = teacher_pred_denorm[..., 0:2]
        xy = F.smooth_l1_loss(s_xy, t_xy, beta=float(self.opsd_cfg.huber_beta), reduction="none").mean(dim=(1, 2))  # (B,)

        # heading loss
        s_h = _heading_from_cossin(student_pred_denorm)
        t_h = _heading_from_cossin(teacher_pred_denorm)
        dh = torch.abs(_angle_wrap(s_h - t_h))  # (B, T)
        h = dh.mean(dim=1)  # (B,)

        per_sample = float(self.opsd_cfg.xy_weight) * xy + float(self.opsd_cfg.heading_weight) * h
        return per_sample

    def _build_rl_features(self, loss_dict: dict, rrs_penalty: torch.Tensor) -> torch.Tensor:
        """Build critic input z for V_phi."""
        fm_energy = loss_dict.get("ego_planning_loss_per_sample", loss_dict["ego_planning_loss"].detach())
        if fm_energy.dim() != 1:
            fm_energy = fm_energy.view(fm_energy.shape[0], -1).mean(dim=1)
        z = torch.stack([fm_energy.detach().view(-1), rrs_penalty.detach().view(-1)], dim=-1)
        return z

    def _compute_advantage(
        self,
        loss_dict: dict,
        rrs_reward: torch.Tensor,
        rrs_penalty: torch.Tensor,
    ):
        """
        Compute A(τ,G) = r~(τ,G) - V_phi(z) with critic regression.
        Returns:
          - adv_detach: stopgrad(A), shape (B,)
          - rl_v_loss: scalar tensor
          - r_tilde: reward tensor used for critic target, shape (B,)
        """
        z = self._build_rl_features(loss_dict, rrs_penalty)
        if self.value_net is None:
            hidden = int(self.rl_cfg.get("hidden_dim", 128))
            depth = int(self.rl_cfg.get("depth", 2))
            dropout = float(self.rl_cfg.get("dropout", 0.0))
            self.value_net = ValueMLP(in_dim=z.shape[-1], hidden_dim=hidden, depth=depth, dropout=dropout).to(z.device)

        if bool(self.rl_cfg.get("normalize_reward", False)):
            r_mu = rrs_reward.mean()
            r_std = rrs_reward.std(unbiased=False).clamp(min=1e-6)
            rrs_reward_target = (rrs_reward - r_mu) / r_std
        else:
            rrs_reward_target = rrs_reward

        v_pred = self.value_net(z)  # (B,)
        critic_loss_type = str(self.rl_cfg.get("critic_loss", "mse")).lower()
        if critic_loss_type == "huber":
            huber_beta = float(self.rl_cfg.get("critic_huber_beta", 1.0))
            rl_v_loss = F.smooth_l1_loss(v_pred, rrs_reward_target.detach(), beta=huber_beta, reduction="mean")
        else:
            rl_v_loss = torch.mean((v_pred - rrs_reward_target.detach()) ** 2)

        adv = rrs_reward_target - v_pred
        if bool(self.rl_cfg.get("normalize_adv", False)):
            adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp(min=1e-6))
        clip = float(self.opsd_cfg.adv_clip)
        if clip <= 0:
            clip = float(self.rl_cfg.get("adv_clip", 0.0))
        if clip > 0:
            adv = torch.clamp(adv, min=-clip, max=clip)
        return adv.detach(), rl_v_loss, rrs_reward_target

    def _sample_tau_denorm(self, model: nn.Module, data: NuPlanDataSample, seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample τ ~ πθ(·|G) using model inference (planner sampling).
        Returns τ_denorm of shape (B, T, 4) in metric space (x,y,cos,sin).
        Important: this is sampled under no_grad; we do NOT backprop through sampling.
        """
        micro_bs = int(self.opsd_cfg.rollout_micro_batch_size)
        if getattr(data, "batched", False) and micro_bs > 0 and data.ego_current.shape[0] > micro_bs:
            chunks = []
            for start in range(0, data.ego_current.shape[0], micro_bs):
                end = min(start + micro_bs, data.ego_current.shape[0])
                data_chunk = self._slice_batched_sample(data, start, end)
                chunks.append(self._sample_tau_denorm_single(model, data_chunk, seed=seed))
            return torch.cat(chunks, dim=0)
        return self._sample_tau_denorm_single(model, data, seed=seed)

    def _sample_tau_denorm_single(self, model: nn.Module, data: NuPlanDataSample, seed: Optional[int] = None) -> torch.Tensor:
        """
        Non-chunked rollout sampling helper for a single batch chunk.
        """
        use_cfg = bool(self.opsd_cfg.sample_use_cfg)
        cfg_weight = float(self.opsd_cfg.sample_cfg_weight)
        # In training loop, student model is in train() mode; switch to eval() for stable rollout sampling.
        was_training = bool(model.training)
        try:
            model.eval()
            with torch.no_grad():
                if seed is None:
                    sample = model(data, mode="inference", use_cfg=use_cfg, cfg_weight=cfg_weight)  # (B, 1, T, 4)
                else:
                    devices = []
                    try:
                        p = next(model.parameters())
                        if p.is_cuda:
                            devices = [int(p.device.index)] if p.device.index is not None else []
                    except Exception:
                        devices = []
                    # do not perturb global RNG state for training
                    with torch.random.fork_rng(devices=devices, enabled=True):
                        torch.manual_seed(int(seed))
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(int(seed))
                        sample = model(data, mode="inference", use_cfg=use_cfg, cfg_weight=cfg_weight)  # (B, 1, T, 4)
        finally:
            if was_training:
                model.train()
        if sample.dim() == 4:
            return sample[:, 0]
        # robust fallback
        return sample

    def _slice_batched_sample(self, data: NuPlanDataSample, start: int, end: int) -> NuPlanDataSample:
        """
        Slice a batched NuPlanDataSample along the batch dimension.
        """
        return NuPlanDataSample(
            batched=True,
            ego_past=data.ego_past[start:end],
            ego_current=data.ego_current[start:end],
            ego_future=data.ego_future[start:end],
            neighbor_past=data.neighbor_past[start:end],
            neighbor_future=data.neighbor_future[start:end],
            neighbor_future_observed=data.neighbor_future_observed[start:end],
            lanes=data.lanes[start:end],
            lanes_speedlimit=data.lanes_speedlimit[start:end],
            lanes_has_speedlimit=data.lanes_has_speedlimit[start:end],
            routes=data.routes[start:end],
            routes_speedlimit=data.routes_speedlimit[start:end],
            routes_has_speedlimit=data.routes_has_speedlimit[start:end],
            map_objects=data.map_objects[start:end],
        )

    def _proxy_reward_from_tau(
        self,
        tau_denorm: torch.Tensor,
        data: NuPlanDataSample,
        gaiw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute proxy reward/penalty for a rollout trajectory using RRS.
        Returns:
          - reward:  (B,)
          - penalty: (B,)
        """
        if self.rrs_module is None:
            raise RuntimeError(
                "OPSD rollout selection / improvement gate requires RRS reward proxy. "
                "Set core.rrs.enabled=true (nu can stay 0.0)."
            )
        penalty, _ = self.rrs_module(tau_denorm, data)
        if self.rrs_cfg.get("use_gaiw", True) and gaiw is not None:
            penalty = penalty * gaiw
        reward = self._shape_rl_reward(-penalty)
        return reward, penalty

    def _select_teacher_rollout(
        self,
        source_model: nn.Module,
        data: NuPlanDataSample,
        gaiw: Optional[torch.Tensor],
        seed_base: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Select teacher target trajectory for each scenario.
        Returns:
          - best_tau:      (B, T, 4)
          - best_reward:   (B,)
          - best_penalty:  (B,)
          - best_idx:      (B,) or None
        """
        mode = str(self.opsd_cfg.teacher_selection_mode).lower()
        k = max(1, int(self.opsd_cfg.teacher_selection_k))

        if mode != "best_of_k" or k <= 1:
            tau = self._sample_tau_denorm(source_model, data, seed=seed_base)
            reward, penalty = self._proxy_reward_from_tau(tau, data, gaiw)
            return tau, reward, penalty, None

        tau_candidates = []
        reward_candidates = []
        penalty_candidates = []
        for i in range(k):
            seed_i = None if seed_base is None else (int(seed_base) + i)
            tau_i = self._sample_tau_denorm(source_model, data, seed=seed_i)
            reward_i, penalty_i = self._proxy_reward_from_tau(tau_i, data, gaiw)
            tau_candidates.append(tau_i)
            reward_candidates.append(reward_i)
            penalty_candidates.append(penalty_i)

        tau_stack = torch.stack(tau_candidates, dim=0)         # (K, B, T, 4)
        reward_stack = torch.stack(reward_candidates, dim=0)   # (K, B)
        penalty_stack = torch.stack(penalty_candidates, dim=0) # (K, B)

        best_idx = reward_stack.argmax(dim=0)                  # (B,)
        batch_idx = torch.arange(reward_stack.shape[1], device=reward_stack.device)
        best_tau = tau_stack[best_idx, batch_idx]             # (B, T, 4)
        best_reward = reward_stack[best_idx, batch_idx]       # (B,)
        best_penalty = penalty_stack[best_idx, batch_idx]     # (B,)
        return best_tau, best_reward, best_penalty, best_idx

    def _compute_improvement_multiplier(
        self,
        teacher_reward: torch.Tensor,
        student_reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build a stopgrad gate multiplier from teacher improvement over student.
        Only positive improvements contribute.
        Returns:
          - gate_multiplier: (B,)
          - improvement_pos: (B,)
        """
        margin = float(self.opsd_cfg.improvement_margin)
        improvement = teacher_reward - student_reward - margin
        improvement_pos = torch.clamp(improvement, min=0.0)

        gate_multiplier = improvement_pos
        if bool(self.opsd_cfg.improvement_normalize_by_mean):
            pos_mean = improvement_pos.mean().detach().clamp(min=1e-6)
            gate_multiplier = gate_multiplier / pos_mean

        max_mult = float(self.opsd_cfg.improvement_max_multiplier)
        if max_mult > 0:
            gate_multiplier = torch.clamp(gate_multiplier, max=max_mult)

        return gate_multiplier.detach(), improvement_pos.detach()

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        """Get underlying model when wrapped by DDP/DataParallel."""
        try:
            from flow_planner.train_utils.ddp import get_model
            return get_model(model)
        except Exception:
            return getattr(model, "module", model)

    def _diffusion_loss_per_sample_on_tau(self, model: nn.Module, data: NuPlanDataSample, tau_denorm: torch.Tensor) -> torch.Tensor:
        """
        Compute a diffusion/flow-matching surrogate loss e_diff(τ,G) for a given τ (denorm, metric).
        This follows the same mechanics as FlowPlanner.forward_train, but uses τ instead of dataset GT.
        Returns per-sample loss shape (B,).
        """
        model = self._unwrap_model(model)
        B = data.ego_current.shape[0]
        device = tau_denorm.device

        # conditioned branch only (cfg_flags=1) for log-likelihood surrogate
        cfg_flags = torch.ones((B, 1), device=device, dtype=torch.int32)
        model_inputs, gt_dummy = model.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True)

        # Build gt_with_current in the same normalized space as training:
        # keep the "current" token from gt_dummy, replace the future with normalized τ.
        tau = tau_denorm[:, None, :, :]  # (B, 1, T, 4)
        tau_norm = model.data_processor.state_preprocess(tau)  # normalize future states
        gt = gt_dummy.clone()
        # gt shape: (B, 1, T+1, 4). Replace future part.
        gt[:, :, 1:, :] = tau_norm

        encoder_inputs = model.extract_encoder_inputs(model_inputs)
        encoder_outputs = model.encoder(**encoder_inputs)
        decoder_model_extra = model.extract_decoder_inputs(encoder_outputs, model_inputs)

        noised_traj, target, t = model.flow_ode.sample(gt[:, :, 1:, :], model._model_type)
        noised_traj_tokens = torch.cat(
            traj_chunking(noised_traj, model.planner_params["action_len"], model.planner_params["action_overlap"]),
            dim=1,
        )
        target_tokens = torch.cat(
            traj_chunking(target, model.planner_params["action_len"], model.planner_params["action_overlap"]),
            dim=1,
        )
        prediction = model.decoder(noised_traj_tokens, t, **decoder_model_extra)
        batch_loss = model.basic_loss(prediction, target_tokens)
        loss = torch.sum(batch_loss, dim=-1)  # (B, action_num, action_length)
        return loss.view(B, -1).mean(dim=1)

    def train_step(self, model, data: NuPlanDataSample):
        """
        Single-forward student update with optional distillation from frozen teacher.
        This avoids double-running the student forward compared to calling super().train_step().
        """
        if not self.opsd_cfg.enabled or self._teacher_model is None:
            return super().train_step(model, data)

        data = self.input_aug(data)
        # Keep an unmodified copy for paper-aligned rollout computations.
        data_for_policy = data.copy(self.device)

        # compute graph-aware weights (if enabled) before model forward
        gaiw = self._compute_graph_aware_weight(data)

        prediction, loss_dict = model(data, mode="train")

        # Apply graph-aware weighting on ego_planning_loss if available (same as FlowMatchingCore)
        if gaiw is not None and "ego_planning_loss_per_sample" in loss_dict:
            per_sample = loss_dict["ego_planning_loss_per_sample"]
            if per_sample.dim() > 1:
                B = per_sample.shape[0]
                per_sample = per_sample.view(B, -1).mean(dim=1)
            loss_dict["ego_planning_loss_unweighted"] = loss_dict.get("ego_planning_loss", per_sample.mean())
            loss_dict["ego_planning_loss"] = (gaiw * per_sample).mean()

        # Optional: Relational Reward Surrogate (same as FlowMatchingCore)
        rrs_loss = None
        nu_base = float(self.rrs_cfg.get("nu", self.rrs_cfg.get("weight", 0.0)))
        warmup_epochs = int(self.rrs_cfg.get("nu_warmup_epochs", 0))
        if warmup_epochs > 0:
            ramp = min(1.0, max(0.0, (self._epoch + 1) / float(warmup_epochs)))
            nu = nu_base * ramp
        else:
            nu = nu_base
        if self.rrs_module is not None and "pred_future_denorm" in loss_dict:
            pred_future = loss_dict["pred_future_denorm"]
            rrs_penalty_per_sample, comps = self.rrs_module(pred_future, data)
            if self.rrs_cfg.get("use_gaiw", True) and gaiw is not None:
                rrs_penalty_per_sample = rrs_penalty_per_sample * gaiw
            loss_dict["rrs_penalty_mean"] = rrs_penalty_per_sample.mean()
            loss_dict["rrs_reward_mean"] = -rrs_penalty_per_sample.mean()
            loss_dict["rrs_penalty_per_sample"] = rrs_penalty_per_sample
            loss_dict["rrs_reward_per_sample"] = -rrs_penalty_per_sample
            loss_dict["rrs_nu"] = torch.tensor(nu, device=pred_future.device)
            if nu != 0.0:
                rrs_loss = rrs_penalty_per_sample.mean() * nu
                loss_dict["rrs_loss"] = rrs_loss
            for k, v in comps.items():
                if k in ("rrs_penalty", "rrs_reward"):
                    continue
                if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == rrs_penalty_per_sample.shape[0]:
                    loss_dict[k] = v.mean()

        # Base total loss from standard weights (used in non-paper-aligned fallback)
        base_total_loss = sum([loss_dict[k] * self.loss_weights[k] for k in self.loss_weights.keys()])
        if rrs_loss is not None:
            base_total_loss = base_total_loss + rrs_loss
        total_loss = base_total_loss

        # OPSD distillation + unified RL coupling
        distill_loss = None
        rl_v_loss = None
        rl_actor_loss = None
        rl_adv_mean = None
        rl_reward_mean = None
        rl_pos_adv_frac = None
        rl_weight_eff = None
        rl_critic_weight_eff = None
        opsd_gate_mean = None
        opsd_student_reward_mean = None
        opsd_teacher_reward_mean = None
        opsd_improvement_mean = None
        opsd_teacher_idx_mean = None
        opsd_distill_weight_eff = None

        if self.opsd_cfg.enabled and self._teacher_model is not None:
            student_pred = loss_dict.get("pred_future_denorm", None)
            if student_pred is not None:
                distill_weight_eff = self._effective_distill_weight()
                opsd_distill_weight_eff = torch.tensor(float(distill_weight_eff), device=total_loss.device)
                use_unified = bool(self.opsd_cfg.unified_with_rl) and bool(self.rl_cfg.get("enabled", False))
                has_rrs_for_adv = self.rrs_module is not None
                has_nonzero_distill_weight = (
                    distill_weight_eff != 0.0
                    and float(self.opsd_cfg.gate_value) != 0.0
                )
                selection_source = str(self.opsd_cfg.teacher_selection_source).lower()
                source_model = model if selection_source == "student" else self._teacher_model
                use_rollout_teacher = (
                    selection_source == "student"
                    or str(self.opsd_cfg.teacher_selection_mode).lower() == "best_of_k"
                    or bool(self.opsd_cfg.improvement_aware_gate)
                )
                use_custom_unified_rl = (
                    use_unified
                    and has_rrs_for_adv
                    and self.opsd_cfg.paper_aligned_objective
                    and has_nonzero_distill_weight
                    and self._rl_algorithm in ("actor_critic", "ac", "ppo", "ppo_like")
                )

                if not use_custom_unified_rl:
                    total_loss = self._apply_standard_rl_objective(
                        model=model,
                        data_for_policy=data_for_policy,
                        loss_dict=loss_dict,
                        gaiw=gaiw,
                        total_loss=total_loss,
                    )

                if use_custom_unified_rl:
                        # Paper-aligned: use τ sampled from inference for reward/advantage and for e_diff surrogate.
                        pair_seed = None
                        if self.opsd_cfg.rollout_match_noise:
                            pair_seed = int(self.opsd_cfg.rollout_seed_base) + int(self._rollout_counter)
                        tau_student = self._sample_tau_denorm(model, data_for_policy, seed=pair_seed)  # (B, T, 4)
                        tau_teacher, teacher_reward, teacher_penalty, teacher_idx = self._select_teacher_rollout(
                            source_model=source_model,
                            data=data_for_policy,
                            gaiw=gaiw,
                            seed_base=pair_seed,
                        )
                        self._rollout_counter += 1

                        if selection_source == "student" and teacher_idx is not None:
                            # In self-distill + best-of-k mode, use the selected rollout for both
                            # actor and distillation targets to avoid pushing toward two trajectories.
                            tau_student = tau_teacher.detach()
                            r_tilde = teacher_reward
                            rrs_pen_student = teacher_penalty
                        else:
                            # R~(τ,G) from student τ (on-policy rollout proxy)
                            r_tilde, rrs_pen_student = self._proxy_reward_from_tau(tau_student, data_for_policy, gaiw)

                        opsd_student_reward_mean = r_tilde.mean()
                        opsd_teacher_reward_mean = teacher_reward.mean()
                        if teacher_idx is not None:
                            opsd_teacher_idx_mean = teacher_idx.float().mean()

                        # diffusion surrogate e_diff(τ,G) computed on τ_student (no backprop through τ sampling)
                        e_diff_per_sample = self._diffusion_loss_per_sample_on_tau(model, data_for_policy, tau_student.detach())
                        loss_dict["opsd_e_diff_mean"] = e_diff_per_sample.mean()

                        # critic/advantage: keep value features consistent with RL-only path
                        fm_anchor = loss_dict.get("ego_planning_loss", None)
                        adv_detach, rl_v_loss, _ = self._compute_advantage(
                            loss_dict=loss_dict,
                            rrs_reward=r_tilde,
                            rrs_penalty=rrs_pen_student,
                        )
                        rl_adv_mean = adv_detach.mean()
                        rl_reward_mean = r_tilde.mean()

                        # actor term: reduce e_diff on better-than-baseline rollouts.
                        rl_pos_adv_frac = (adv_detach > 0).float().mean()
                        actor_adv = self._transform_actor_advantage(
                            adv_detach,
                            positive_only=bool(self.opsd_cfg.positive_only_adv),
                            use_negative=bool(self.opsd_cfg.actor_use_negative_adv),
                        )
                        e_diff_actor = e_diff_per_sample
                        if bool(self.rl_cfg.get("normalize_fm_energy", False)):
                            denom = e_diff_actor.detach().mean().abs().clamp(
                                min=float(self.rl_cfg.get("fm_energy_min_denom", 1e-6))
                            )
                            e_diff_actor = e_diff_actor / denom
                        rl_actor_loss = torch.mean(e_diff_actor * actor_adv)

                        # distillation distance D(f_theta(G), pi_tau(G)):
                        # - teacher uses sampled τ (no_grad)
                        # - student uses differentiable prediction from forward_train (pred_future_denorm)
                        student_det = student_pred  # (B, T, 4), keeps gradients
                        distill_per_sample = self._distill_loss(student_det, tau_teacher)
                        # Gate should compare teacher reward against the actual student output we are optimizing,
                        # not against a separate sampled rollout (which can drift from student_pred).
                        student_det_reward, _ = self._proxy_reward_from_tau(student_det.detach(), data_for_policy, gaiw)
                        opsd_student_reward_mean = student_det_reward.mean()

                        gate_base = float(self.opsd_cfg.gate_value) * distill_weight_eff
                        gate = torch.full_like(distill_per_sample, gate_base)
                        if bool(self.opsd_cfg.improvement_aware_gate):
                            gate_mult, improvement_pos = self._compute_improvement_multiplier(teacher_reward, student_det_reward)
                            gate = gate * gate_mult
                            opsd_improvement_mean = improvement_pos.mean()
                        if self.opsd_cfg.dynamic_gate_with_adv:
                            adv_temp = max(1e-6, float(self.opsd_cfg.adv_temperature))
                            gate = gate * torch.sigmoid(adv_detach / adv_temp)
                        opsd_gate_mean = gate.mean()
                        distill_loss = torch.mean(gate * distill_per_sample)

                        rl_weight = float(self.rl_cfg.get("weight", 1.0))
                        critic_weight = float(self.rl_cfg.get("critic_weight", 1.0))
                        rl_warmup_epochs = int(self.rl_cfg.get("warmup_epochs", 0))
                        if rl_warmup_epochs > 0:
                            rl_weight = rl_weight * min(1.0, max(0.0, (self._epoch + 1) / float(rl_warmup_epochs)))
                        critic_warmup_epochs = int(self.rl_cfg.get("critic_warmup_epochs", 0))
                        if critic_warmup_epochs > 0:
                            critic_weight = critic_weight * min(
                                1.0, max(0.0, (self._epoch + 1) / float(critic_warmup_epochs))
                            )
                        rl_weight_eff = torch.tensor(float(rl_weight), device=total_loss.device)
                        rl_critic_weight_eff = torch.tensor(float(critic_weight), device=total_loss.device)
                        loss_dict["rl_rollout_reward_mean"] = r_tilde.mean()
                        loss_dict["rl_rollout_penalty_mean"] = rrs_pen_student.mean()
                        loss_dict["rl_reward_mean"] = rl_reward_mean
                        loss_dict["rl_adv_mean"] = rl_adv_mean
                        loss_dict["rl_pos_adv_frac"] = rl_pos_adv_frac
                        loss_dict["rl_weight_eff"] = rl_weight_eff
                        loss_dict["rl_critic_weight_eff"] = rl_critic_weight_eff
                        loss_dict["rl_v_loss"] = rl_v_loss
                        loss_dict["rl_actor_loss"] = rl_actor_loss
                        loss_dict["rl_e_diff_mean"] = e_diff_per_sample.mean()
                        if self.opsd_cfg.paper_keep_base_loss:
                            total_loss = base_total_loss + critic_weight * rl_v_loss + rl_weight * rl_actor_loss + distill_loss
                        else:
                            total_loss = critic_weight * rl_v_loss + rl_weight * rl_actor_loss + distill_loss
                            if fm_anchor is not None:
                                total_loss = total_loss + float(self.opsd_cfg.fm_anchor_weight) * fm_anchor
                        if (
                            self.opsd_cfg.keep_consistency_reg
                            and not self.opsd_cfg.paper_keep_base_loss
                            and "consistency_loss" in loss_dict
                            and "consistency_loss" in self.loss_weights
                        ):
                            total_loss = total_loss + self.loss_weights["consistency_loss"] * loss_dict["consistency_loss"]
                else:
                    if use_rollout_teacher and has_rrs_for_adv:
                        pair_seed = None
                        if self.opsd_cfg.rollout_match_noise:
                            pair_seed = int(self.opsd_cfg.rollout_seed_base) + int(self._rollout_counter)
                        tau_teacher, teacher_reward, _, teacher_idx = self._select_teacher_rollout(
                            source_model=source_model,
                            data=data_for_policy,
                            gaiw=gaiw,
                            seed_base=pair_seed,
                        )
                        self._rollout_counter += 1

                        # Compare improvement against the actual student prediction being optimized.
                        student_reward, _ = self._proxy_reward_from_tau(student_pred.detach(), data_for_policy, gaiw)
                        opsd_student_reward_mean = student_reward.mean()
                        opsd_teacher_reward_mean = teacher_reward.mean()
                        if teacher_idx is not None:
                            opsd_teacher_idx_mean = teacher_idx.float().mean()

                        distill_per_sample = self._distill_loss(student_pred, tau_teacher)  # (B,)
                        gate = torch.full_like(
                            distill_per_sample,
                            float(self.opsd_cfg.gate_value) * distill_weight_eff,
                        )
                        if bool(self.opsd_cfg.improvement_aware_gate):
                            gate_mult, improvement_pos = self._compute_improvement_multiplier(teacher_reward, student_reward)
                            gate = gate * gate_mult
                            opsd_improvement_mean = improvement_pos.mean()
                        opsd_gate_mean = gate.mean()
                        distill_loss = torch.mean(gate * distill_per_sample)
                        total_loss = total_loss + distill_loss
                    else:
                        # Legacy/non-paper path: distill from teacher train-time prediction.
                        with torch.no_grad():
                            _, teacher_loss_dict = self._teacher_model(data, mode="train")
                            teacher_pred = teacher_loss_dict.get("pred_future_denorm", None)
                        if teacher_pred is not None:
                            distill_per_sample = self._distill_loss(student_pred, teacher_pred)  # (B,)
                            gate = float(self.opsd_cfg.gate_value)
                            w = distill_weight_eff
                            distill_loss = distill_per_sample.mean() * gate * w
                            total_loss = total_loss + distill_loss

        # Build output logging dict (mirror FlowMatchingCore style)
        out: dict = {"total_loss": total_loss}

        def _add(name: str):
            if name in loss_dict and isinstance(loss_dict[name], torch.Tensor):
                out[name] = loss_dict[name].detach()

        _add("ego_planning_loss")
        _add("ego_planning_loss_unweighted")
        _add("consistency_loss")

        if gaiw is not None:
            out["gaiw_mean"] = gaiw.mean().detach()
            out["gaiw_min"] = gaiw.amin().detach()
            out["gaiw_max"] = gaiw.amax().detach()

        _add("rrs_loss")
        _add("rrs_penalty_mean")
        _add("rrs_reward_mean")
        _add("rrs_safety_pen")
        _add("rrs_rule_pen")
        _add("rrs_comfort_pen")
        _add("rrs_task_pen")
        _add("rrs_progress_pen")
        _add("rrs_low_speed_pen")
        _add("rrs_progress")
        _add("rrs_mean_speed")
        _add("rl_reward_mean")
        _add("rl_rollout_reward_mean")
        _add("rl_rollout_penalty_mean")
        _add("rl_adv_mean")
        _add("rl_reward_std")
        _add("rl_adv_std")
        _add("rl_pos_adv_frac")
        _add("rl_ref_reward_mean")
        _add("rl_improvement_mean")
        _add("rl_actor_active_frac")
        _add("rl_auto_scale")
        _add("rl_v_mean")
        _add("rl_weight_eff")
        _add("rl_critic_weight_eff")
        _add("rl_v_loss")
        _add("rl_actor_loss")
        _add("rl_group_best_reward_mean")
        _add("rl_group_worst_reward_mean")
        _add("rl_group_size")
        _add("rl_e_diff_mean")

        if distill_loss is not None:
            out["opsd_distill_loss"] = distill_loss.detach()
            if opsd_gate_mean is None:
                out["opsd_gate"] = torch.tensor(float(self.opsd_cfg.gate_value), device=total_loss.device).detach()
            else:
                out["opsd_gate"] = opsd_gate_mean.detach()
        else:
            out["opsd_distill_loss"] = torch.zeros((), device=total_loss.device)
            out["opsd_gate"] = torch.tensor(float(self.opsd_cfg.gate_value), device=total_loss.device).detach()

        if rl_v_loss is not None:
            out["rl_v_loss"] = rl_v_loss.detach()
        if rl_actor_loss is not None:
            out["rl_actor_loss"] = rl_actor_loss.detach()
        if rl_adv_mean is not None:
            out["rl_adv_mean"] = rl_adv_mean.detach()
        if rl_pos_adv_frac is not None:
            out["rl_pos_adv_frac"] = rl_pos_adv_frac.detach()
        if rl_reward_mean is not None:
            out["rl_reward_mean"] = rl_reward_mean.detach()
        if rl_weight_eff is not None:
            out["rl_weight_eff"] = rl_weight_eff.detach()
        if rl_critic_weight_eff is not None:
            out["rl_critic_weight_eff"] = rl_critic_weight_eff.detach()
        if opsd_student_reward_mean is not None:
            out["opsd_student_reward_mean"] = opsd_student_reward_mean.detach()
        if opsd_teacher_reward_mean is not None:
            out["opsd_teacher_reward_mean"] = opsd_teacher_reward_mean.detach()
        if opsd_improvement_mean is not None:
            out["opsd_improvement_mean"] = opsd_improvement_mean.detach()
        if opsd_teacher_idx_mean is not None:
            out["opsd_teacher_idx_mean"] = opsd_teacher_idx_mean.detach()
        if opsd_distill_weight_eff is not None:
            out["opsd_distill_weight_eff"] = opsd_distill_weight_eff.detach()

        return out
