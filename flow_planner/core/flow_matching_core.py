import torch
import torch.nn as nn
import sys
from functools import partial
from flow_planner.core import Core
from flow_planner.model.model_utils.time_sampler import TimeSampler
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class FlowMatchingCore(Core):
    
    def __init__(
            self, 
            input_aug=None,
            device: str = 'cuda',
            graph_weighting=None,
            **loss_weights,
    ):
        
        super(FlowMatchingCore, self).__init__()
        
        self.device = device
        self.input_aug = input_aug
        self.loss_weights = loss_weights
        self.graph_weighting_cfg = graph_weighting or {}
    
    def _compute_graph_aware_weight(self, data: NuPlanDataSample) -> torch.Tensor:
        """
        Compute graph-aware multiplicative weights w_GAIW(tau) for each trajectory in the batch.
        This is a lightweight proxy of Sec. 3.4 in GRID-Planner, based on:
          - Cent(tau): number of nearby agents in the last past frame;
          - CollProb(tau): inverse of minimum ego-agent distance in observed future.
        """
        if not self.graph_weighting_cfg.get("enabled", False):
            return None
        
        device = self.device
        B = data.ego_current.shape[0]
        weights = torch.ones(B, device=device)
        
        alpha1 = float(self.graph_weighting_cfg.get("alpha1", 0.0))
        alpha2 = float(self.graph_weighting_cfg.get("alpha2", 0.0))
        cent_radius = float(self.graph_weighting_cfg.get("cent_radius", 30.0))
        collision_radius = float(self.graph_weighting_cfg.get("collision_radius", 8.0))
        temperature = float(self.graph_weighting_cfg.get("temperature", 1.0))
        max_weight = float(self.graph_weighting_cfg.get("max_weight", 10.0))
        min_weight = float(self.graph_weighting_cfg.get("min_weight", 0.1))
        
        # 1) Structural "centrality": number of close neighbors in the last past frame.
        cent = torch.zeros(B, device=device)
        if hasattr(data, "neighbor_past") and data.neighbor_past.numel() > 0:
            # neighbor_past: (B, num_agents, past_len, feat)
            try:
                neighbor_last = data.neighbor_past[:, :, -1, :2]  # (B, N, 2) positions in ego-centric frame
                dist = torch.norm(neighbor_last, dim=-1)  # (B, N)
                cent = (dist < cent_radius).sum(dim=-1).float()  # (B,)
            except Exception:
                # Fallback: keep zeros if the shape is unexpected
                cent = torch.zeros(B, device=device)
        
        # 2) Collision proxy: min relative distance in observed future.
        coll_prob = torch.zeros(B, device=device)
        if hasattr(data, "neighbor_future_observed") and data.neighbor_future_observed.numel() > 0:
            # neighbor_future_observed: (B, num_agents, future_len, 3) in ego-centric frame
            try:
                nf = data.neighbor_future_observed
                if nf.shape[1] > 0:
                    neighbor_future_pos = nf[..., :2]  # (B, N, T, 2)
                    dist_future = torch.norm(neighbor_future_pos, dim=-1)  # (B, N, T)
                    min_dist = dist_future.view(B, -1).amin(dim=-1)  # (B,)
                    # map min distance to [0, 1] collision "probability" (closer => higher)
                    coll_prob = torch.clamp((collision_radius - min_dist) / collision_radius, min=0.0, max=1.0)
            except Exception:
                coll_prob = torch.zeros(B, device=device)
        
        # Smooth features
        cent_feat = torch.log1p(cent)  # monotonically increasing, smaller scale
        coll_feat = coll_prob
        
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
        
        total_loss = sum([loss_dict[k] * self.loss_weights[k] for k in self.loss_weights.keys()])
        total_loss_dict = {'total_loss': total_loss}

        return total_loss_dict
        
    def inference(self, model, data, use_cfg=False, cfg_weight=None):
        model = model.to(self.device)
        model.eval()
        
        prediction = model(data, mode='inference', use_cfg=use_cfg, cfg_weight=cfg_weight)
        
        return prediction
    
    def initial_state_constraint(self, xt, s1, B, T_, D):
        xt = xt.view(B, -1, T_, D)
        xt[:, :, :1, :] = s1[:, :xt.shape[1]]
        return xt.view(B, -1, T_ * D)