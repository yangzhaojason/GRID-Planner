import torch
import torch.nn as nn
import sys
from functools import partial
from flow_planner.core import Core
from flow_planner.model.model_utils.time_sampler import TimeSampler
from flow_planner.data.dataset.nuplan import NuPlanDataSample
from flow_planner.reward.rrs import RelationalRewardSurrogate

class FlowMatchingCore(Core):
    
    def __init__(
            self, 
            input_aug=None,
            device: str = 'cuda',
            graph_weighting=None,
            rrs=None,
            **loss_weights,
    ):
        
        super(FlowMatchingCore, self).__init__()
        
        self.device = device
        self.input_aug = input_aug
        self.loss_weights = loss_weights
        self.graph_weighting_cfg = graph_weighting or {}
        self.rrs_cfg = rrs or {}
        self.rrs_module = RelationalRewardSurrogate(self.rrs_cfg) if self.rrs_cfg.get("enabled", False) else None
        self._epoch = 0
    
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
        if self.rrs_module is not None and nu != 0.0:
            if "pred_future_denorm" in loss_dict:
                # pred_future_denorm: (B, T, 4) = (x, y, cos, sin) in meters
                pred_future = loss_dict["pred_future_denorm"]
                rrs_penalty_per_sample, comps = self.rrs_module(pred_future, data)
                # combine with (optional) graph-aware weights
                if self.rrs_cfg.get("use_gaiw", True) and gaiw is not None:
                    rrs_penalty_per_sample = rrs_penalty_per_sample * gaiw
                # unified objective term (minimize penalty == maximize reward)
                rrs_loss = rrs_penalty_per_sample.mean() * nu
                # expose scalars for logging (trainer will average them)
                loss_dict["rrs_loss"] = rrs_loss
                loss_dict["rrs_nu"] = torch.tensor(nu, device=pred_future.device)
                loss_dict["rrs_penalty_mean"] = rrs_penalty_per_sample.mean()
                # reward is defined as -penalty for paper-style reporting
                loss_dict["rrs_reward_mean"] = -rrs_penalty_per_sample.mean()
                for k, v in comps.items():
                    if isinstance(v, torch.Tensor) and v.dim() == 1 and v.shape[0] == rrs_penalty_per_sample.shape[0]:
                        loss_dict[k] = v.mean()
        
        total_loss = sum([loss_dict[k] * self.loss_weights[k] for k in self.loss_weights.keys()])
        if rrs_loss is not None:
            total_loss = total_loss + rrs_loss
        
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