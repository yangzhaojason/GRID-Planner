import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from typing import Optional
from functools import partial
from flow_planner.core import Core
from flow_planner.model.model_utils.time_sampler import TimeSampler
from flow_planner.model.model_utils.traj_tool import traj_chunking
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.reward.rrs import RelationalRewardSurrogate
from flow_planner.core.rl_value_net import ValueMLP

class FlowMatchingCore(Core):
    
    def __init__(
            self, 
            input_aug=None,
            device: str = 'cuda',
            graph_weighting=None,
            rrs=None,
            rl=None,
            **loss_weights,
    ):
        
        super(FlowMatchingCore, self).__init__()
        
        self.device = device
        self.input_aug = input_aug
        self.loss_weights = loss_weights
        self.graph_weighting_cfg = graph_weighting or {}
        self.rrs_cfg = rrs or {}
        self.rrs_module = RelationalRewardSurrogate(self.rrs_cfg) if self.rrs_cfg.get("enabled", False) else None
        self.rl_cfg = rl or {}
        self._rl_algorithm = str(self.rl_cfg.get("algorithm", "grpo")).lower()
        # Value baseline V_phi is only needed for actor-critic style RL.
        self.value_net: nn.Module | None = None
        if self.rl_cfg.get("enabled", False) and self._uses_value_baseline():
            hidden = int(self.rl_cfg.get("hidden_dim", 128))
            depth = int(self.rl_cfg.get("depth", 2))
            dropout = float(self.rl_cfg.get("dropout", 0.0))
            # MVP feature dim is 2: [ego_planning_loss_per_sample, rrs_penalty_per_sample]
            self.value_net = ValueMLP(in_dim=2, hidden_dim=hidden, depth=depth, dropout=dropout).to(self.device)
        self._epoch = 0
        self._rollout_counter = 0

    def _uses_value_baseline(self) -> bool:
        return self._rl_algorithm in ("actor_critic", "ac", "ppo", "ppo_like")

    def _unwrap_model(self, model: nn.Module) -> nn.Module:
        try:
            from flow_planner.train_utils.ddp import get_model
            return get_model(model)
        except Exception:
            return getattr(model, "module", model)

    def _sample_tau_denorm(self, model, data: NuPlanDataSample, seed: Optional[int] = None):
        """
        Sample tau ~ pi_theta(.|G) using planner inference.
        This is used to make RL rewards closer to closed-loop planning-time behavior.
        """
        use_cfg = bool(self.rl_cfg.get("rollout_use_cfg", True))
        cfg_weight = float(self.rl_cfg.get("rollout_cfg_weight", 1.0))
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
                    # keep rollout sampling reproducible without perturbing the global training RNG
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
        return sample

    def _shape_rl_reward(self, reward: torch.Tensor) -> torch.Tensor:
        reward_scale = float(self.rl_cfg.get("reward_scale", 1.0))
        if reward_scale != 1.0:
            reward = reward * reward_scale
        reward_clip = float(self.rl_cfg.get("reward_clip", 0.0))
        if reward_clip > 0:
            reward = torch.clamp(reward, min=-reward_clip, max=reward_clip)
        return reward

    def _proxy_reward_from_tau(
        self,
        tau_denorm: torch.Tensor,
        data: NuPlanDataSample,
        gaiw: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rrs_module is None:
            raise RuntimeError(
                "On-policy RL requires the RRS reward proxy. "
                "Set core.rrs.enabled=true (nu can stay 0.0)."
            )

        penalty, _ = self.rrs_module(tau_denorm, data)
        if self.rrs_cfg.get("use_gaiw", True) and gaiw is not None:
            penalty = penalty * gaiw
        if self.rl_cfg.get("use_gaiw", True) and gaiw is not None and not self.rrs_cfg.get("use_gaiw", True):
            penalty = penalty * gaiw

        reward = self._shape_rl_reward(-penalty)
        return reward, penalty

    def _diffusion_loss_per_sample_on_tau(self, model: nn.Module, data: NuPlanDataSample, tau_denorm: torch.Tensor) -> torch.Tensor:
        """
        Diffusion / flow-matching surrogate loss for a sampled rollout τ.
        This plays the role of the actor surrogate in critic-free GRPO-style updates.
        """
        model = self._unwrap_model(model)
        data = data.copy(self.device)
        B = data.ego_current.shape[0]
        device = tau_denorm.device

        cfg_flags = torch.ones((B, 1), device=device, dtype=torch.int32)
        model_inputs, gt_dummy = model.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True)

        tau = tau_denorm[:, None, :, :]  # (B, 1, T, 4)
        tau_norm = model.data_processor.state_preprocess(tau)
        gt = gt_dummy.clone()
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
        loss = torch.sum(batch_loss, dim=-1)
        return loss.view(B, -1).mean(dim=1)

    def _compute_group_relative_advantage(self, reward_stack: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantages for GRPO-style updates.
        reward_stack has shape (K, B), where K is the rollout group size.
        """
        if reward_stack.shape[0] > 1:
            adv = reward_stack
            if bool(self.rl_cfg.get("group_center", True)):
                adv = adv - reward_stack.mean(dim=0, keepdim=True)
            if bool(self.rl_cfg.get("group_normalize", True)):
                eps = float(self.rl_cfg.get("group_norm_eps", 1.0e-6))
                adv = adv / reward_stack.std(dim=0, unbiased=False, keepdim=True).clamp(min=eps)
        else:
            adv = reward_stack

        if bool(self.rl_cfg.get("normalize_adv", False)):
            adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp(min=1e-6))

        adv_clip = float(self.rl_cfg.get("adv_clip", 0.0))
        if adv_clip > 0:
            adv = torch.clamp(adv, min=-adv_clip, max=adv_clip)
        return adv.detach()

    def _transform_actor_advantage(
        self,
        adv: torch.Tensor,
        *,
        positive_only: bool = False,
        use_negative: bool = False,
    ) -> torch.Tensor:
        """
        Transform the stopgrad advantage before it weights the actor/FM surrogate.
        - positive_only: only reinforce better-than-baseline rollouts
        - use_negative: legacy ablation path kept for backward-compatibility
        """
        actor_adv = adv
        if positive_only:
            actor_adv = torch.clamp(actor_adv, min=0.0)
        if use_negative:
            actor_adv = -actor_adv
        return actor_adv.detach()

    def _compute_improvement_multiplier(
        self,
        candidate_reward: torch.Tensor,
        reference_reward: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build an improvement-aware multiplier from candidate rollout reward over a reference policy output.
        Only positive improvements contribute.
        """
        margin = float(self.rl_cfg.get("improvement_margin", 0.0))
        improvement = candidate_reward - reference_reward - margin
        improvement_pos = torch.clamp(improvement, min=0.0)

        gate = improvement_pos
        if bool(self.rl_cfg.get("improvement_normalize_by_mean", True)):
            pos_mean = improvement_pos.mean().detach().clamp(min=1e-6)
            gate = gate / pos_mean

        max_mult = float(self.rl_cfg.get("improvement_max_multiplier", 3.0))
        if max_mult > 0:
            gate = torch.clamp(gate, max=max_mult)

        return gate.detach(), improvement_pos.detach()

    def _compute_auto_shrink_scale(
        self,
        improvement: Optional[torch.Tensor],
        actor_active_frac: torch.Tensor,
    ) -> torch.Tensor:
        """
        Auto-shrink actor updates when the selected rollout barely improves over
        the current reference behavior, or when too few samples are active.
        """
        scales = []
        min_imp = float(self.rl_cfg.get("auto_shrink_min_improvement", 0.0))
        if improvement is not None and min_imp > 0:
            imp_mean = improvement.mean().detach()
            scales.append(torch.clamp(imp_mean / min_imp, min=0.0, max=1.0))

        min_active = float(self.rl_cfg.get("auto_shrink_min_active_frac", 0.0))
        if min_active > 0:
            scales.append(torch.clamp(actor_active_frac.detach() / min_active, min=0.0, max=1.0))

        if not scales:
            return actor_active_frac.new_tensor(1.0)

        scale = scales[0]
        for s in scales[1:]:
            scale = torch.minimum(scale, s)
        return scale.detach()

    def get_optimizer_params(self):
        """Aux params (e.g., critic/value net) to be added to the main optimizer."""
        if self.value_net is None:
            return []
        return [{"params": self.value_net.parameters()}]

    def get_aux_state_dict(self) -> dict:
        """State to checkpoint alongside model weights without affecting planner inference checkpoints."""
        if self.value_net is None:
            return {}
        # unwrap DDP if needed
        try:
            from flow_planner.train_utils.ddp import get_model
            vnet = get_model(self.value_net)
        except Exception:
            vnet = self.value_net
        return {"value_net": vnet.state_dict()}

    def load_aux_state_dict(self, aux: dict):
        """Restore auxiliary state (best-effort)."""
        if not aux or self.value_net is None:
            return
        sd = aux.get("value_net", None)
        if sd is None:
            return
        try:
            from flow_planner.train_utils.ddp import get_model
            vnet = get_model(self.value_net)
        except Exception:
            vnet = self.value_net
        vnet.load_state_dict(sd, strict=False)
    
    def set_epoch(self, epoch: int):
        """Optional hook for trainer to set current epoch (used for warm-up schedules)."""
        self._epoch = int(epoch)
    
    def _compute_graph_aware_weight(self, data: NuPlanDataSample) -> torch.Tensor:
        """
        Compute graph-aware multiplicative weights w_GAIW(tau) for each trajectory in the batch.
        This is a scene-graph proxy of Sec. 3.4 in GRID-Planner, based on:
          - Cent(tau): ego centrality in an agent interaction graph + map complexity proxies;
          - CollProb(tau): risk proxy from (ego_future vs neighbor_future_observed) min distance + TTC.
        """
        if not self.graph_weighting_cfg.get("enabled", False):
            return None
        
        device = self.device
        B = data.ego_current.shape[0]
        weights = torch.ones(B, device=device)
        
        alpha1 = float(self.graph_weighting_cfg.get("alpha1", 0.0))
        alpha2 = float(self.graph_weighting_cfg.get("alpha2", 0.0))
        cent_radius = float(self.graph_weighting_cfg.get("cent_radius", 30.0))          # [m]
        collision_radius = float(self.graph_weighting_cfg.get("collision_radius", 8.0))# [m]
        cent_sigma = float(self.graph_weighting_cfg.get("cent_sigma", cent_radius / 2.0))
        lane_radius = float(self.graph_weighting_cfg.get("lane_radius", 30.0))
        static_radius = float(self.graph_weighting_cfg.get("static_radius", 20.0))
        beta_lane = float(self.graph_weighting_cfg.get("beta_lane", 0.25))
        beta_static = float(self.graph_weighting_cfg.get("beta_static", 0.15))
        beta_red = float(self.graph_weighting_cfg.get("beta_red", 0.25))
        ttc_scale = float(self.graph_weighting_cfg.get("ttc_scale", 3.0))              # [s]
        temperature = float(self.graph_weighting_cfg.get("temperature", 1.0))
        max_weight = float(self.graph_weighting_cfg.get("max_weight", 10.0))
        min_weight = float(self.graph_weighting_cfg.get("min_weight", 0.1))
        
        eps = 1e-6
        
        # 1) Cent(tau): ego centrality in an agent-agent interaction graph (implicit adjacency).
        #    Nodes: ego (at origin) + neighbors at the last past frame in ego-centric coordinates.
        cent_strength = torch.zeros(B, device=device)
        if hasattr(data, "neighbor_past") and data.neighbor_past is not None and data.neighbor_past.numel() > 0:
            # neighbor_past: (B, N, past_len, feat), feat contains [x,y,cos,sin,vx,vy,...]
            try:
                neigh_last = data.neighbor_past[:, :, -1, :]
                neigh_pos = neigh_last[:, :, :2]  # (B, N, 2)
                # mark valid agents (non-zero padded)
                valid = (neigh_pos.abs().sum(dim=-1) > 1e-3)  # (B, N)
                # positions including ego at origin
                ego_pos = torch.zeros((B, 1, 2), device=device)
                pos = torch.cat([ego_pos, neigh_pos], dim=1)  # (B, N+1, 2)
                # pairwise distances (B, N+1, N+1)
                diff = pos[:, :, None, :] - pos[:, None, :, :]
                dist = torch.norm(diff, dim=-1).clamp(min=0.0)
                # soft adjacency weights, radius-pruned
                sigma2 = max(cent_sigma * cent_sigma, 1e-3)
                A = torch.exp(-(dist * dist) / (2.0 * sigma2)) * (dist < cent_radius).to(dist.dtype)
                # remove self loops
                eye = torch.eye(A.shape[-1], device=device).unsqueeze(0)
                A = A * (1.0 - eye)
                # ignore padded neighbor nodes by masking their incident edges (keep ego node always valid)
                node_valid = torch.cat([torch.ones((B, 1), device=device, dtype=torch.bool), valid], dim=1)  # (B, N+1)
                mask = node_valid[:, :, None] & node_valid[:, None, :]
                A = A * mask.to(A.dtype)
                # ego node strength = sum_j A_ego,j
                cent_strength = A[:, 0, :].sum(dim=-1)
            except Exception:
                cent_strength = torch.zeros(B, device=device)
        
        # Map complexity proxies (optional): nearby lanes, nearby static objects, and "red light presence" from lane TL encodings.
        lane_count = torch.zeros(B, device=device)
        red_presence = torch.zeros(B, device=device)
        if hasattr(data, "lanes") and data.lanes is not None and data.lanes.numel() > 0:
            # lanes: (B, L, P, 12) where first 2 dims are coords, last 4 dims are TL one-hot (g,y,r,unk)
            try:
                lane_xy = data.lanes[..., :2]  # (B, L, P, 2)
                lane_valid = (lane_xy.abs().sum(dim=-1) > 1e-3).any(dim=-1)  # (B, L)
                lane_min_dist = torch.norm(lane_xy, dim=-1).amin(dim=-1)  # (B, L)
                near_lane = lane_valid & (lane_min_dist < lane_radius)
                lane_count = near_lane.sum(dim=-1).float()
                
                # red light presence: count points with red encoding near ego, aggregated per sample
                tl = data.lanes[..., -4:]  # (B, L, P, 4)
                red = tl[..., 2]  # (B, L, P)
                # consider only valid/near lanes
                red_points = (red > 0.5) & near_lane[:, :, None]
                red_presence = red_points.sum(dim=(-1, -2)).float()
            except Exception:
                lane_count = torch.zeros(B, device=device)
                red_presence = torch.zeros(B, device=device)
        
        static_count = torch.zeros(B, device=device)
        if hasattr(data, "map_objects") and data.map_objects is not None and data.map_objects.numel() > 0:
            # map_objects: (B, S, feat), first 2 dims are coords in ego frame
            try:
                st_xy = data.map_objects[..., :2]  # (B, S, 2)
                st_dist = torch.norm(st_xy, dim=-1)
                st_valid = (st_xy.abs().sum(dim=-1) > 1e-3)
                static_count = (st_valid & (st_dist < static_radius)).sum(dim=-1).float()
            except Exception:
                static_count = torch.zeros(B, device=device)
        
        cent_feat = torch.log1p(cent_strength) \
            + beta_lane * torch.log1p(lane_count) \
            + beta_static * torch.log1p(static_count) \
            + beta_red * torch.log1p(red_presence)
        
        # 2) CollProb(tau): combine future min distance (ego vs neighbor) and TTC at current time.
        coll_from_future = torch.zeros(B, device=device)
        if hasattr(data, "ego_future") and hasattr(data, "neighbor_future_observed"):
            if data.ego_future is not None and data.neighbor_future_observed is not None and \
               data.ego_future.numel() > 0 and data.neighbor_future_observed.numel() > 0:
                try:
                    ego_f = data.ego_future[..., :2].to(device)  # (B, T, 2)
                    nei_f = data.neighbor_future_observed[..., :2].to(device)  # (B, N, T, 2)
                    # valid neighbor future if any position is non-zero
                    nei_valid = (nei_f.abs().sum(dim=-1) > 1e-3).any(dim=-1)  # (B, N)
                    rel = nei_f - ego_f[:, None, :, :]  # (B, N, T, 2)
                    dist = torch.norm(rel, dim=-1)  # (B, N, T)
                    dist = dist + (~nei_valid)[:, :, None].to(dist.dtype) * 1e6
                    min_dist = dist.view(B, -1).amin(dim=-1)  # (B,)
                    coll_from_future = torch.clamp((collision_radius - min_dist) / collision_radius, min=0.0, max=1.0)
                except Exception:
                    coll_from_future = torch.zeros(B, device=device)
        
        coll_from_ttc = torch.zeros(B, device=device)
        if hasattr(data, "neighbor_past") and hasattr(data, "ego_current"):
            if data.neighbor_past is not None and data.neighbor_past.numel() > 0 and \
               data.ego_current is not None and data.ego_current.numel() > 0:
                try:
                    neigh_last = data.neighbor_past[:, :, -1, :]
                    p = neigh_last[:, :, :2]  # (B, N, 2)
                    v_nei = neigh_last[:, :, 4:6]  # (B, N, 2)
                    v_ego = data.ego_current[:, 4:6].to(device)  # (B, 2)
                    v_rel = v_nei - v_ego[:, None, :]
                    d = torch.norm(p, dim=-1).clamp(min=eps)  # (B, N)
                    valid = (p.abs().sum(dim=-1) > 1e-3)
                    p_hat = p / d[:, :, None]
                    radial = (p_hat * v_rel).sum(dim=-1)  # positive => moving away along radial dir
                    closing = torch.clamp(-radial, min=0.0)  # closing speed (>=0)
                    ttc = torch.where(closing > 0, d / (closing + eps), torch.full_like(d, float("inf")))
                    ttc = torch.where(valid, ttc, torch.full_like(ttc, float("inf")))
                    min_ttc = ttc.amin(dim=-1)  # (B,)
                    # map TTC to [0,1] risk (smaller TTC => higher risk)
                    coll_from_ttc = torch.exp(-min_ttc / max(ttc_scale, 1e-3))
                    coll_from_ttc = torch.clamp(coll_from_ttc, min=0.0, max=1.0)
                except Exception:
                    coll_from_ttc = torch.zeros(B, device=device)
        
        coll_feat = torch.maximum(coll_from_future, coll_from_ttc)
        
        log_w = alpha1 * cent_feat + alpha2 * coll_feat
        if temperature > 0:
            log_w = log_w / temperature
        weights = torch.exp(log_w)
        weights = torch.clamp(weights, min=min_weight, max=max_weight)
        
        # Optional normalization to keep mean weight around 1
        if self.graph_weighting_cfg.get("normalize_by_mean", True):
            weights = weights / (weights.mean().clamp(min=1e-6))
        
        return weights.detach()
        
    def train_step(self, model, data: NuPlanDataSample):
        
        data = self.input_aug(data)
        data_for_policy = data.copy(self.device)
        
        # compute graph-aware weights (if enabled) before model forward
        gaiw = self._compute_graph_aware_weight(data)
        
        prediction, loss_dict = model(data, mode='train')
        
        # Apply graph-aware weighting on ego_planning_loss if available
        if gaiw is not None and 'ego_planning_loss_per_sample' in loss_dict:
            per_sample = loss_dict['ego_planning_loss_per_sample']
            # ensure shape (B,)
            if per_sample.dim() > 1:
                B = per_sample.shape[0]
                per_sample = per_sample.view(B, -1).mean(dim=1)
            # keep an unweighted copy for potential logging
            loss_dict['ego_planning_loss_unweighted'] = loss_dict.get('ego_planning_loss', per_sample.mean())
            loss_dict['ego_planning_loss'] = (gaiw * per_sample).mean()

        # Optional: Relational Reward Surrogate (implemented as COST) + unified objective term
        rrs_loss = None
        nu_base = float(self.rrs_cfg.get("nu", self.rrs_cfg.get("weight", 0.0)))
        warmup_epochs = int(self.rrs_cfg.get("nu_warmup_epochs", 0))
        if warmup_epochs > 0:
            # ramp from 0 to nu_base over warmup_epochs
            ramp = min(1.0, max(0.0, (self._epoch + 1) / float(warmup_epochs)))
            nu = nu_base * ramp
        else:
            nu = nu_base
        need_rrs_for_rl = bool(self.rl_cfg.get("enabled", False))
        need_rrs_for_loss = (nu != 0.0)
        if self.rrs_module is not None and (need_rrs_for_loss or need_rrs_for_rl) and "pred_future_denorm" in loss_dict:
            # pred_future_denorm: (B, T, 4) = (x, y, cos, sin) in meters
            pred_future = loss_dict["pred_future_denorm"]
            rrs_penalty_per_sample, comps = self.rrs_module(pred_future, data)
            # combine with (optional) graph-aware weights
            if self.rrs_cfg.get("use_gaiw", True) and gaiw is not None:
                rrs_penalty_per_sample = rrs_penalty_per_sample * gaiw

            # always expose per-sample reward/penalty for RL-style usage
            loss_dict["rrs_penalty_per_sample"] = rrs_penalty_per_sample
            loss_dict["rrs_reward_per_sample"] = -rrs_penalty_per_sample
            loss_dict["rrs_penalty_mean"] = rrs_penalty_per_sample.mean()
            loss_dict["rrs_reward_mean"] = -rrs_penalty_per_sample.mean()
            loss_dict["rrs_nu"] = torch.tensor(nu, device=pred_future.device)

            # unified objective term (minimize penalty == maximize reward)
            if need_rrs_for_loss:
                rrs_loss = rrs_penalty_per_sample.mean() * nu
                loss_dict["rrs_loss"] = rrs_loss

            for k, v in comps.items():
                # avoid overwriting per-sample storage keys with means
                if k in ("rrs_penalty", "rrs_reward"):
                    continue
                if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == rrs_penalty_per_sample.shape[0]:
                    loss_dict[k] = v.mean()
        
        total_loss = sum([loss_dict[k] * self.loss_weights[k] for k in self.loss_weights.keys()])
        if rrs_loss is not None:
            total_loss = total_loss + rrs_loss

        # Optional: On-policy RL term.
        # Default path is a critic-free GRPO-style update:
        #   1) sample K rollout trajectories per scene
        #   2) compute group-relative advantages from rollout rewards
        #   3) weight a diffusion/FM surrogate e_diff(τ, G) by stopgrad(A)
        # Actor-critic remains available as a fallback via core.rl.algorithm=actor_critic.
        if self.rl_cfg.get("enabled", False):
            # Require RRS components to compute reward
            if "rrs_reward_per_sample" in loss_dict and "rrs_penalty_per_sample" in loss_dict:
                rl_algorithm = self._rl_algorithm
                reward_source = str(self.rl_cfg.get("reward_source", "rollout")).lower()
                if rl_algorithm == "grpo":
                    group_size = max(1, int(self.rl_cfg.get("group_size", 4)))
                    seed_base = int(self.rl_cfg.get("rollout_seed_base", 3402)) + int(self._rollout_counter) * group_size
                    self._rollout_counter += 1

                    tau_candidates = []
                    reward_candidates = []
                    penalty_candidates = []
                    for i in range(group_size):
                        tau_i = self._sample_tau_denorm(model, data_for_policy, seed=seed_base + i)
                        reward_i, penalty_i = self._proxy_reward_from_tau(tau_i, data_for_policy, gaiw)
                        tau_candidates.append(tau_i)
                        reward_candidates.append(reward_i)
                        penalty_candidates.append(penalty_i)

                    tau_stack = torch.stack(tau_candidates, dim=0)         # (K, B, T, 4)
                    reward_stack = torch.stack(reward_candidates, dim=0)   # (K, B)
                    penalty_stack = torch.stack(penalty_candidates, dim=0) # (K, B)
                    A = self._compute_group_relative_advantage(reward_stack)

                    batch_idx = torch.arange(reward_stack.shape[1], device=reward_stack.device)
                    best_idx = reward_stack.argmax(dim=0)

                    select_best_only = bool(self.rl_cfg.get("select_best_only", True))
                    if select_best_only:
                        tau_actor = tau_stack[best_idx, batch_idx]
                        reward_actor = reward_stack[best_idx, batch_idx]
                        e_diff_actor = self._diffusion_loss_per_sample_on_tau(model, data_for_policy, tau_actor.detach())
                        actor_A = self._transform_actor_advantage(
                            A[best_idx, batch_idx],
                            positive_only=bool(self.rl_cfg.get("positive_only_adv", False)),
                            use_negative=bool(self.rl_cfg.get("actor_use_negative_adv", False)),
                        )
                    else:
                        e_diff_candidates = []
                        for i in range(group_size):
                            e_diff_i = self._diffusion_loss_per_sample_on_tau(model, data_for_policy, tau_stack[i].detach())
                            e_diff_candidates.append(e_diff_i)
                        e_diff_stack = torch.stack(e_diff_candidates, dim=0)   # (K, B)
                        reward_actor = reward_stack
                        e_diff_actor = e_diff_stack
                        actor_A = self._transform_actor_advantage(
                            A,
                            positive_only=bool(self.rl_cfg.get("positive_only_adv", False)),
                            use_negative=bool(self.rl_cfg.get("actor_use_negative_adv", False)),
                        )

                    improvement = None
                    gate = None
                    ref_reward = None
                    actor_active_frac = (actor_A > 0).float().mean()
                    auto_scale = actor_A.new_tensor(1.0)
                    if bool(self.rl_cfg.get("improvement_aware", False)):
                        ref_source = str(self.rl_cfg.get("reference_source", "rollout")).lower()
                        if ref_source == "rollout":
                            ref_seed = seed_base + group_size + int(self.rl_cfg.get("reference_seed_offset", 1000))
                            tau_ref = self._sample_tau_denorm(model, data_for_policy, seed=ref_seed)
                            ref_reward, _ = self._proxy_reward_from_tau(tau_ref, data_for_policy, gaiw)
                        elif "pred_future_denorm" in loss_dict:
                            ref_reward, _ = self._proxy_reward_from_tau(loss_dict["pred_future_denorm"].detach(), data_for_policy, gaiw)

                    if ref_reward is not None:
                        gate, improvement = self._compute_improvement_multiplier(
                            reward_actor if reward_actor.dim() == 1 else reward_actor.max(dim=0).values,
                            ref_reward,
                        )
                        if actor_A.dim() == 2:
                            actor_A = actor_A * gate.unsqueeze(0)
                        else:
                            actor_A = actor_A * gate
                        actor_active_frac = (gate > 0).float().mean()

                    if bool(self.rl_cfg.get("auto_shrink_on_low_improvement", False)):
                        auto_scale = self._compute_auto_shrink_scale(improvement, actor_active_frac)
                        actor_A = actor_A * auto_scale

                    if bool(self.rl_cfg.get("normalize_fm_energy", False)):
                        denom = e_diff_actor.detach().mean().abs().clamp(
                            min=float(self.rl_cfg.get("fm_energy_min_denom", 1e-6))
                        )
                        e_diff_actor = e_diff_actor / denom

                    rl_actor_loss = torch.mean(e_diff_actor * actor_A)

                    rl_weight = float(self.rl_cfg.get("weight", 1.0))
                    rl_warmup_epochs = int(self.rl_cfg.get("warmup_epochs", 0))
                    if rl_warmup_epochs > 0:
                        rl_weight = rl_weight * min(1.0, max(0.0, (self._epoch + 1) / float(rl_warmup_epochs)))
                    total_loss = total_loss + rl_weight * rl_actor_loss

                    loss_dict["rl_rollout_penalty_mean"] = penalty_stack.mean()
                    loss_dict["rl_rollout_reward_mean"] = reward_stack.mean()
                    loss_dict["rl_reward_mean"] = reward_stack.mean()
                    loss_dict["rl_adv_mean"] = A.mean()
                    loss_dict["rl_reward_std"] = reward_stack.std(unbiased=False)
                    loss_dict["rl_adv_std"] = A.std(unbiased=False)
                    loss_dict["rl_pos_adv_frac"] = (A > 0).float().mean()
                    if ref_reward is not None:
                        loss_dict["rl_ref_reward_mean"] = ref_reward.mean()
                    if improvement is not None:
                        loss_dict["rl_improvement_mean"] = improvement.mean()
                    loss_dict["rl_actor_active_frac"] = actor_active_frac
                    loss_dict["rl_auto_scale"] = auto_scale
                    loss_dict["rl_group_best_reward_mean"] = reward_stack.amax(dim=0).mean()
                    loss_dict["rl_group_worst_reward_mean"] = reward_stack.amin(dim=0).mean()
                    loss_dict["rl_group_size"] = torch.tensor(float(group_size), device=reward_stack.device)
                    loss_dict["rl_weight_eff"] = torch.tensor(float(rl_weight), device=reward_stack.device)
                    loss_dict["rl_actor_loss"] = rl_actor_loss
                    loss_dict["rl_e_diff_mean"] = e_diff_actor.mean()
                else:
                    if reward_source == "rollout" and self.rrs_module is not None:
                        tau_rollout = self._sample_tau_denorm(model, data_for_policy)
                        rrs_reward, rrs_penalty = self._proxy_reward_from_tau(tau_rollout, data_for_policy, gaiw)
                        loss_dict["rl_rollout_penalty_mean"] = rrs_penalty.mean()
                        loss_dict["rl_rollout_reward_mean"] = rrs_reward.mean()
                    else:
                        rrs_penalty = loss_dict["rrs_penalty_per_sample"]
                        rrs_reward = loss_dict["rrs_reward_per_sample"]
                        if self.rl_cfg.get("use_gaiw", True) and gaiw is not None and not self.rrs_cfg.get("use_gaiw", True):
                            rrs_penalty = rrs_penalty * gaiw
                            rrs_reward = rrs_reward * gaiw
                        rrs_reward = self._shape_rl_reward(rrs_reward)

                    if bool(self.rl_cfg.get("normalize_reward", False)):
                        r_mu = rrs_reward.mean()
                        r_std = rrs_reward.std(unbiased=False).clamp(min=1e-6)
                        rrs_reward_target = (rrs_reward - r_mu) / r_std
                    else:
                        rrs_reward_target = rrs_reward

                    z = torch.stack(
                        [
                            loss_dict.get("ego_planning_loss_per_sample", loss_dict["ego_planning_loss"].detach()).detach().view(-1),
                            rrs_penalty.detach().view(-1),
                        ],
                        dim=-1,
                    )

                    if self.value_net is None:
                        hidden = int(self.rl_cfg.get("hidden_dim", 128))
                        depth = int(self.rl_cfg.get("depth", 2))
                        dropout = float(self.rl_cfg.get("dropout", 0.0))
                        self.value_net = ValueMLP(in_dim=z.shape[-1], hidden_dim=hidden, depth=depth, dropout=dropout).to(z.device)
                    V = self.value_net(z)

                    critic_loss_type = str(self.rl_cfg.get("critic_loss", "mse")).lower()
                    if critic_loss_type == "huber":
                        huber_beta = float(self.rl_cfg.get("critic_huber_beta", 1.0))
                        rl_v_loss = F.smooth_l1_loss(V, rrs_reward_target.detach(), beta=huber_beta, reduction="mean")
                    else:
                        rl_v_loss = torch.mean((V - rrs_reward_target.detach()) ** 2)

                    A = (rrs_reward_target - V).detach()
                    if bool(self.rl_cfg.get("normalize_adv", False)):
                        A = (A - A.mean()) / (A.std(unbiased=False).clamp(min=1e-6))
                    adv_clip = float(self.rl_cfg.get("adv_clip", 0.0))
                    if adv_clip > 0:
                        A = torch.clamp(A, min=-adv_clip, max=adv_clip)
                    actor_A = self._transform_actor_advantage(
                        A,
                        positive_only=bool(self.rl_cfg.get("positive_only_adv", False)),
                        use_negative=bool(self.rl_cfg.get("actor_use_negative_adv", False)),
                    )

                    fm_energy = loss_dict.get("ego_planning_loss_per_sample", None)
                    if fm_energy is None:
                        fm_energy = loss_dict["ego_planning_loss"].detach().expand_as(A)
                    if fm_energy.dim() != 1:
                        fm_energy = fm_energy.view(fm_energy.shape[0], -1).mean(dim=1)
                    if bool(self.rl_cfg.get("normalize_fm_energy", False)):
                        denom = fm_energy.detach().mean().abs().clamp(min=float(self.rl_cfg.get("fm_energy_min_denom", 1e-6)))
                        fm_energy = fm_energy / denom
                    rl_actor_loss = torch.mean(fm_energy * actor_A)

                    rl_weight = float(self.rl_cfg.get("weight", 1.0))
                    rl_warmup_epochs = int(self.rl_cfg.get("warmup_epochs", 0))
                    if rl_warmup_epochs > 0:
                        rl_weight = rl_weight * min(1.0, max(0.0, (self._epoch + 1) / float(rl_warmup_epochs)))
                    critic_weight = float(self.rl_cfg.get("critic_weight", 1.0))
                    critic_warmup_epochs = int(self.rl_cfg.get("critic_warmup_epochs", 0))
                    if critic_warmup_epochs > 0:
                        critic_weight = critic_weight * min(1.0, max(0.0, (self._epoch + 1) / float(critic_warmup_epochs)))
                    total_loss = total_loss + rl_weight * rl_actor_loss + critic_weight * rl_v_loss

                    loss_dict["rl_reward_mean"] = rrs_reward.mean()
                    loss_dict["rl_adv_mean"] = A.mean()
                    loss_dict["rl_reward_std"] = rrs_reward.std(unbiased=False)
                    loss_dict["rl_adv_std"] = A.std(unbiased=False)
                    loss_dict["rl_pos_adv_frac"] = (A > 0).float().mean()
                    loss_dict["rl_v_mean"] = V.mean()
                    loss_dict["rl_weight_eff"] = torch.tensor(float(rl_weight), device=rrs_reward.device)
                    loss_dict["rl_critic_weight_eff"] = torch.tensor(float(critic_weight), device=rrs_reward.device)
                    loss_dict["rl_v_loss"] = rl_v_loss
                    loss_dict["rl_actor_loss"] = rl_actor_loss
        
        # Return rich logging dict (total_loss keeps graph for backward; others are detached scalars)
        out: dict = {"total_loss": total_loss}
        
        def _add(name: str):
            if name in loss_dict and isinstance(loss_dict[name], torch.Tensor):
                # detach to avoid holding graphs across iterations
                out[name] = loss_dict[name].detach()
        
        # core training losses
        _add("ego_planning_loss")
        _add("ego_planning_loss_unweighted")
        _add("consistency_loss")
        
        # graph-aware weighting stats
        if gaiw is not None:
            out["gaiw_mean"] = gaiw.mean().detach()
            out["gaiw_min"] = gaiw.amin().detach()
            out["gaiw_max"] = gaiw.amax().detach()
        
        # RRS stats (if enabled)
        _add("rrs_loss")
        _add("rrs_penalty_mean")
        _add("rrs_reward_mean")
        _add("rrs_penalty")
        _add("rrs_reward")
        _add("rrs_safety_pen")
        _add("rrs_rule_pen")
        _add("rrs_comfort_pen")
        _add("rrs_task_pen")
        _add("rrs_progress_pen")
        _add("rrs_low_speed_pen")
        _add("rrs_progress")
        _add("rrs_mean_speed")

        # RL stats
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
        
        return out
        
    def inference(self, model, data, use_cfg=False, cfg_weight=None):
        model = model.to(self.device)
        model.eval()
        
        prediction = model(data, mode='inference', use_cfg=use_cfg, cfg_weight=cfg_weight)
        
        return prediction
    
    def initial_state_constraint(self, xt, s1, B, T_, D):
        xt = xt.view(B, -1, T_, D)
        xt[:, :, :1, :] = s1[:, :xt.shape[1]]
        return xt.view(B, -1, T_ * D)