from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from flow_planner.data.dataset.nuplan import NuPlanDataSample


def _safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim), min=eps))


def _angle_wrap(diff: torch.Tensor) -> torch.Tensor:
    """Wrap angle difference to [-pi, pi] in a differentiable way."""
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def _softmin(x: torch.Tensor, dim: int, tau: float = 1.0) -> torch.Tensor:
    """Smooth approximation of min(x) over dim."""
    if tau <= 0:
        return x.amin(dim=dim)
    return -tau * torch.logsumexp(-x / tau, dim=dim)


def _softplus_barrier(z: torch.Tensor, s: float = 1.0) -> torch.Tensor:
    """Smooth hinge-like penalty for z > 0."""
    if s <= 0:
        return torch.relu(z)
    return F.softplus(z / s) * s


@dataclass
class RRSConfig:
    enabled: bool = False
    nu: float = 0.0      # coefficient in unified objective (loss += nu * w * penalty). Set >0 to enable effect.
    # Backward/compat alias (some scripts may still pass weight)
    weight: float = 0.0
    nu_warmup_epochs: int = 0  # consumed in core; kept here for Hydra compatibility
    use_gaiw: bool = True

    # time / dt
    dt: float = 0.1

    # safety thresholds
    safety_neighbor_source: str = "past_cv"  # ["past_cv", "future_observed"]
    d_safe: float = 2.0          # [m]
    ttc_safe: float = 2.0        # [s]
    ego_radius: float = 1.2      # [m] approximate ego circle radius if dimensions missing
    neighbor_radius_scale: float = 0.5  # radius ~= scale * sqrt(w^2 + l^2)
    tau_softmin: float = 1.0     # smoothing for min over agents/points
    s_barrier: float = 0.5       # smoothing for softplus barrier

    # rule thresholds
    lane_dev_thr: float = 1.0    # [m] deviation to route polyline
    red_stop_thr: float = 2.0    # [m] distance to red-labeled lane points
    spd_ratio_thr: float = 0.0   # allow a small buffer: max(0, (v-vmax)/vmax - thr)
    # driving direction alignment to route tangent
    dir_diff_thr: float = 0.8    # [rad] allowed heading diff to route tangent before penalizing
    # task/progress shaping to avoid conservative no-progress collapse
    progress_target: float = 5.0   # [m] target forward progress over horizon
    min_mean_speed: float = 0.5    # [m/s] minimum desired mean speed

    # comfort weights
    w_acc: float = 0.1
    w_jerk: float = 0.05
    w_yaw_rate: float = 0.02

    # term weights (lambdas)
    lam_col: float = 1.0
    lam_d: float = 1.0
    lam_ttc: float = 1.0
    lam_lane: float = 1.0
    lam_red: float = 0.5
    lam_spd: float = 0.5
    lam_dir: float = 0.0
    lam_comfort: float = 1.0
    lam_progress: float = 0.5
    lam_low_speed: float = 0.2


class RelationalRewardSurrogate(torch.nn.Module):
    """
    Differentiable proxy of RRS(τ) implemented as a COST (lower is better).
    You can interpret reward as -cost if needed.

    Inputs use ego-centric coordinates:
      - predicted ego future: (B, T, 4) with (x, y, cos, sin), in meters (denormalized)
      - neighbors future observed: (B, N, T, 3) with (x, y, heading) in meters
      - lanes/routes/map_objects from NuPlanDataSample (meters, already ego-centric)
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        # store config in a simple struct for defaults
        cfg = dict(cfg)
        valid_keys = {f.name for f in fields(RRSConfig)}
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        self.cfg = RRSConfig(**cfg)

    def forward(
        self,
        ego_future_xycossin: torch.Tensor,
        data: NuPlanDataSample,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
          - penalty_per_sample: (B,)  (>=0, lower is better)
          - components: dict of scalar tensors for logging (mostly means)
        """
        B, T, D = ego_future_xycossin.shape
        assert D == 4, f"Expected ego future dim=4 (x,y,cos,sin), got {D}"

        ego_xy = ego_future_xycossin[..., 0:2]  # (B, T, 2)
        ego_h = torch.atan2(ego_future_xycossin[..., 3], ego_future_xycossin[..., 2])  # (B, T)

        # --- comfort (from ego trajectory) ---
        dt = float(self.cfg.dt)
        ego_v = (ego_xy[:, 1:, :] - ego_xy[:, :-1, :]) / dt  # (B, T-1, 2)
        ego_a = (ego_v[:, 1:, :] - ego_v[:, :-1, :]) / dt    # (B, T-2, 2)
        ego_j = (ego_a[:, 1:, :] - ego_a[:, :-1, :]) / dt    # (B, T-3, 2)

        dh = _angle_wrap(ego_h[:, 1:] - ego_h[:, :-1])  # (B, T-1)
        ego_yaw_rate = dh / dt

        comfort_pen = (
            self.cfg.w_acc * (ego_a.pow(2).sum(dim=-1).mean(dim=-1) if ego_a.numel() else torch.zeros(B, device=ego_xy.device)) +
            self.cfg.w_jerk * (ego_j.pow(2).sum(dim=-1).mean(dim=-1) if ego_j.numel() else torch.zeros(B, device=ego_xy.device)) +
            self.cfg.w_yaw_rate * (ego_yaw_rate.pow(2).mean(dim=-1) if ego_yaw_rate.numel() else torch.zeros(B, device=ego_xy.device))
        )  # (B,)

        # --- safety (distance + TTC) ---
        safety_d_pen = torch.zeros(B, device=ego_xy.device)
        safety_ttc_pen = torch.zeros(B, device=ego_xy.device)
        safety_col_pen = torch.zeros(B, device=ego_xy.device)

        nei_xy = None
        nei_valid = None
        # Prefer past+CV rollout for train/inference alignment.
        if self.cfg.safety_neighbor_source == "future_observed" and \
           hasattr(data, "neighbor_future_observed") and data.neighbor_future_observed is not None and data.neighbor_future_observed.numel() > 0:
            nf = data.neighbor_future_observed.to(ego_xy.device)  # (B, N, T, 3)
            nei_xy = nf[..., 0:2]  # (B, N, T, 2)
            # A tracked agent can disappear within the horizon; zero-padded frames
            # must not become observations at the ego origin.
            nei_valid = nei_xy.abs().sum(dim=-1) > 1e-3  # (B, N, T)
        elif hasattr(data, "neighbor_past") and data.neighbor_past is not None and data.neighbor_past.numel() > 0:
            # Constant-velocity extrapolation from current (last past frame) neighbors.
            neigh_last = data.neighbor_past[:, :, -1, :].to(ego_xy.device)  # (B, N, F)
            p0 = neigh_last[:, :, 0:2]  # (B, N, 2)
            v0 = neigh_last[:, :, 4:6]  # (B, N, 2)
            current_valid = p0.abs().sum(dim=-1) > 1e-3  # (B, N)
            nei_valid = current_valid[:, :, None].expand(-1, -1, T)
            # Align with ego future horizon (t = dt, 2*dt, ..., T*dt), not including current frame.
            tvec = (torch.arange(1, T + 1, device=ego_xy.device, dtype=ego_xy.dtype) * dt)  # (T,)
            nei_xy = p0[:, :, None, :] + v0[:, :, None, :] * tvec[None, None, :, None]  # (B, N, T, 2)

        if nei_xy is not None and nei_valid is not None:
            rel = nei_xy - ego_xy[:, None, :, :]  # (B, N, T, 2)
            dist = _safe_norm(rel, dim=-1)  # (B, N, T)
            dist = dist + (~nei_valid).to(dist.dtype) * 1e6

            # distance barrier (want dist >= d_safe)
            d_min_t = _softmin(dist, dim=1, tau=float(self.cfg.tau_softmin))  # (B, T)
            safety_d_pen = _softplus_barrier(self.cfg.d_safe - d_min_t, s=float(self.cfg.s_barrier)).mean(dim=-1)  # (B,)

            # collision proxy: dist < r_ego + r_nei (use last past frame sizes if available)
            # neighbor_past features contain width/length at indices 6/7 (see agent_process.py)
            r_ego = torch.full((B,), float(self.cfg.ego_radius), device=ego_xy.device)
            r_nei = None
            if hasattr(data, "neighbor_past") and data.neighbor_past is not None and data.neighbor_past.numel() > 0:
                try:
                    neigh_last = data.neighbor_past[:, :, -1, :].to(ego_xy.device)
                    w = neigh_last[:, :, 6].abs()
                    l = neigh_last[:, :, 7].abs()
                    r_nei = float(self.cfg.neighbor_radius_scale) * torch.sqrt(w * w + l * l)  # (B, N)
                except Exception:
                    r_nei = None
            if r_nei is None:
                r_nei = torch.full((B, nei_xy.shape[1]), 1.0, device=ego_xy.device)
            r_sum = r_ego[:, None] + r_nei  # (B, N)
            # min over time of (r_sum - dist) (positive => overlap)
            overlap = (r_sum[:, :, None] - dist)  # (B, N, T)
            overlap = torch.where(nei_valid, overlap, torch.full_like(overlap, -1e6))
            # penalize only positive overlap
            col_sig = _softplus_barrier(overlap, s=float(self.cfg.s_barrier))
            safety_col_pen = col_sig.view(B, -1).mean(dim=-1)

            # TTC proxy (finite differences on neighbor future)
            if T >= 2:
                nei_v = (nei_xy[..., 1:, :] - nei_xy[..., :-1, :]) / dt  # (B, N, T-1, 2)
                # align ego_v to (B, 1, T-1, 2)
                ego_v_aligned = ego_v[:, None, :, :]  # (B, 1, T-1, 2)
                rel_p = rel[..., :-1, :]  # (B, N, T-1, 2)
                rel_v = nei_v - ego_v_aligned  # (B, N, T-1, 2)

                d = _safe_norm(rel_p, dim=-1)  # (B, N, T-1)
                p_hat = rel_p / d[..., None]
                radial = (p_hat * rel_v).sum(dim=-1)  # (B, N, T-1), positive => separating along radial
                closing = torch.clamp(-radial, min=0.0)
                ttc = torch.where(closing > 1e-6, d / (closing + 1e-6), torch.full_like(d, float("inf")))
                # Finite-difference velocity is meaningful only when both endpoints
                # are observed, never across a transition into/out of padding.
                interval_valid = nei_valid[..., :-1] & nei_valid[..., 1:]
                ttc = torch.where(interval_valid, ttc, torch.full_like(ttc, float("inf")))
                ttc_min_t = ttc.amin(dim=1)  # (B, T-1)
                safety_ttc_pen = _softplus_barrier(self.cfg.ttc_safe - ttc_min_t, s=float(self.cfg.s_barrier)).mean(dim=-1)

        safety_pen = (
            self.cfg.lam_col * safety_col_pen +
            self.cfg.lam_d * safety_d_pen +
            self.cfg.lam_ttc * safety_ttc_pen
        )  # (B,)

        # --- rule: route lane deviation + red light proximity ---
        rule_lane_pen = torch.zeros(B, device=ego_xy.device)
        rule_red_pen = torch.zeros(B, device=ego_xy.device)
        rule_spd_pen = torch.zeros(B, device=ego_xy.device)
        rule_dir_pen = torch.zeros(B, device=ego_xy.device)

        # lane deviation proxy: distance to nearest point on route polylines
        if hasattr(data, "routes") and data.routes is not None and data.routes.numel() > 0:
            routes = data.routes.to(ego_xy.device)  # (B, R, P, 12) typically, first 2 dims are xy
            route_xy = routes[..., 0:2]  # (B, R, P, 2)
            route_valid = (route_xy.abs().sum(dim=-1) > 1e-3).any(dim=-1)  # (B, R)
            # flatten route points
            pts = route_xy.reshape(B, -1, 2)  # (B, R*P, 2)
            pts_valid = route_valid[:, :, None].repeat(1, 1, route_xy.shape[-2]).reshape(B, -1)  # (B, R*P)
            sample_has_route = pts_valid.any(dim=-1)  # (B,)
            # compute per timestep min dist to route pts
            diff = ego_xy[:, :, None, :] - pts[:, None, :, :]  # (B, T, M, 2)
            dpt = _safe_norm(diff, dim=-1)  # (B, T, M)
            dpt = dpt + (~pts_valid)[:, None, :].to(dpt.dtype) * 1e6
            dmin = _softmin(dpt, dim=2, tau=float(self.cfg.tau_softmin))  # (B, T)
            lane_pen = _softplus_barrier(dmin - self.cfg.lane_dev_thr, s=float(self.cfg.s_barrier)).mean(dim=-1)
            # If a sample has no valid route points, do not penalize lane deviation.
            rule_lane_pen = torch.where(sample_has_route, lane_pen, torch.zeros_like(lane_pen))

            # driving direction alignment proxy: penalize large heading mismatch to nearest route tangent
            if float(self.cfg.lam_dir) != 0.0:
                route_tan = routes[..., 2:4]  # (B, R, P, 2)
                tan = route_tan.reshape(B, -1, 2)  # (B, M, 2)
                tan_valid = (tan.abs().sum(dim=-1) > 1e-4)
                valid = pts_valid & tan_valid
                has_valid = valid.any(dim=-1)  # (B,)
                # nearest route point index per timestep
                dpt2 = _safe_norm(ego_xy[:, :, None, :] - pts[:, None, :, :], dim=-1)  # (B, T, M)
                dpt2 = dpt2 + (~valid)[:, None, :].to(dpt2.dtype) * 1e6
                nearest_idx = dpt2.argmin(dim=-1)  # (B, T)
                nearest_tan = torch.gather(tan, 1, nearest_idx[..., None].expand(-1, -1, 2))  # (B, T, 2)
                route_h = torch.atan2(nearest_tan[..., 1], nearest_tan[..., 0])  # (B, T)
                ang = torch.abs(_angle_wrap(ego_h - route_h))  # (B, T)
                dir_pen = _softplus_barrier(ang - float(self.cfg.dir_diff_thr), s=float(self.cfg.s_barrier)).mean(dim=-1)
                rule_dir_pen = torch.where(has_valid, dir_pen, torch.zeros_like(dir_pen))

        # red light proxy: use route-red points first (more task-relevant), fallback to all lanes.
        red_xy = None
        red_mask = None
        if hasattr(data, "routes") and data.routes is not None and data.routes.numel() > 0:
            lanes = data.routes.to(ego_xy.device)  # (B, R, P, 12)
            lane_xy = lanes[..., 0:2]
            tl = lanes[..., -4:]
            red = tl[..., 2] > 0.5     # (B, L, P)
            lane_valid = (lane_xy.abs().sum(dim=-1) > 1e-3)
            red_mask = red & lane_valid
            red_xy = lane_xy
        elif hasattr(data, "lanes") and data.lanes is not None and data.lanes.numel() > 0:
            lanes = data.lanes.to(ego_xy.device)  # (B, L, P, 12)
            lane_xy = lanes[..., 0:2]
            tl = lanes[..., -4:]
            red = tl[..., 2] > 0.5
            lane_valid = (lane_xy.abs().sum(dim=-1) > 1e-3)
            red_mask = red & lane_valid
            red_xy = lane_xy

        if red_xy is not None and red_mask is not None:
            if red_mask.any():
                red_pts = red_xy.reshape(B, -1, 2)  # (B, M, 2)
                red_valid = red_mask.reshape(B, -1)  # (B, L*P)
                diff = ego_xy[:, :, None, :] - red_pts[:, None, :, :]  # (B, T, M, 2)
                dpt = _safe_norm(diff, dim=-1)  # (B, T, M)
                dpt = dpt + (~red_valid)[:, None, :].to(dpt.dtype) * 1e6
                dmin = _softmin(dpt, dim=2, tau=float(self.cfg.tau_softmin))  # (B, T)
                # penalty if trajectory gets too close to red points (proxy for crossing stop line)
                rule_red_pen = _softplus_barrier(self.cfg.red_stop_thr - dmin, s=float(self.cfg.s_barrier)).mean(dim=-1)

        # speed limit proxy: use route lanes speed limits if available; apply penalty on predicted speed.
        # ego speed from finite differences:
        ego_speed = _safe_norm(ego_v, dim=-1).mean(dim=-1) if ego_v.numel() else torch.zeros(B, device=ego_xy.device)
        v_max = None
        if hasattr(data, "routes_speedlimit") and data.routes_speedlimit is not None and data.routes_speedlimit.numel() > 0:
            try:
                v = data.routes_speedlimit.to(ego_xy.device).squeeze(-1)  # (B, R)
                valid_limit = torch.ones_like(v, dtype=torch.bool)
                if hasattr(data, "routes_has_speedlimit") and data.routes_has_speedlimit is not None and data.routes_has_speedlimit.numel() > 0:
                    mask = data.routes_has_speedlimit.to(ego_xy.device).squeeze(-1) > 0
                    valid_limit = mask
                    v = torch.where(mask, v, torch.zeros_like(v))
                sample_has_limit = valid_limit.any(dim=-1)  # (B,)
                v_max = v.max(dim=-1).values.clamp(min=0.1)  # avoid div0 for valid samples
            except Exception:
                v_max = None
        if v_max is not None:
            ratio = (ego_speed - v_max) / v_max
            ratio = torch.relu(ratio - float(self.cfg.spd_ratio_thr))
            spd_pen = _softplus_barrier(ratio, s=float(self.cfg.s_barrier))
            # If no speed-limit annotation is valid for a sample, do not apply speed penalty.
            if "sample_has_limit" in locals():
                rule_spd_pen = torch.where(sample_has_limit, spd_pen, torch.zeros_like(spd_pen))
            else:
                rule_spd_pen = spd_pen

        rule_pen = (
            self.cfg.lam_lane * rule_lane_pen +
            self.cfg.lam_red * rule_red_pen +
            self.cfg.lam_spd * rule_spd_pen +
            self.cfg.lam_dir * rule_dir_pen
        )  # (B,)

        # --- task/progress shaping ---
        # Route-tangential cumulative progress:
        #   progress = sum_t max(0, <delta_ego_t, tangent(route_nearest_to_ego_t)>)
        # Fallback to x-end displacement only if no valid route points are available.
        progress = None
        if T >= 2 and hasattr(data, "routes") and data.routes is not None and data.routes.numel() > 0:
            routes = data.routes.to(ego_xy.device)     # (B, R, P, 12)
            route_xy = routes[..., 0:2]                # (B, R, P, 2)
            route_tan = routes[..., 2:4]               # (B, R, P, 2), lane polyline vector in local frame

            pts = route_xy.reshape(B, -1, 2)           # (B, M, 2)
            tan = route_tan.reshape(B, -1, 2)          # (B, M, 2)
            pts_valid = (pts.abs().sum(dim=-1) > 1e-3) # (B, M)
            tan_valid = (tan.abs().sum(dim=-1) > 1e-4) # (B, M)
            valid = pts_valid & tan_valid
            has_valid_route = valid.any(dim=-1)        # (B,)

            # ego motion increments
            ego_pos = ego_xy[:, 1:, :]                 # (B, T-1, 2)
            ego_delta = ego_xy[:, 1:, :] - ego_xy[:, :-1, :]  # (B, T-1, 2)

            # nearest route point for each future step
            dpt = _safe_norm(ego_pos[:, :, None, :] - pts[:, None, :, :], dim=-1)  # (B, T-1, M)
            dpt = dpt + (~valid)[:, None, :].to(dpt.dtype) * 1e6
            nearest_idx = dpt.argmin(dim=-1)  # (B, T-1)

            # gather nearest tangent
            nearest_tan = torch.gather(tan, 1, nearest_idx[..., None].expand(-1, -1, 2))  # (B, T-1, 2)
            tan_norm = _safe_norm(nearest_tan, dim=-1).unsqueeze(-1)
            tan_hat = nearest_tan / tan_norm

            # positive tangential accumulation (ignore reverse / noisy negative projection)
            step_proj = (ego_delta * tan_hat).sum(dim=-1)  # (B, T-1)
            route_progress = torch.clamp(step_proj, min=0.0).sum(dim=-1)  # (B,)

            # fallback for samples without valid route geometry
            fallback_progress = torch.clamp(ego_xy[:, -1, 0], min=0.0)
            progress = torch.where(has_valid_route, route_progress, fallback_progress)
        else:
            progress = torch.clamp(ego_xy[:, -1, 0], min=0.0)

        progress_pen = _softplus_barrier(self.cfg.progress_target - progress, s=float(self.cfg.s_barrier))
        low_speed_pen = _softplus_barrier(self.cfg.min_mean_speed - ego_speed, s=float(self.cfg.s_barrier))
        task_pen = self.cfg.lam_progress * progress_pen + self.cfg.lam_low_speed * low_speed_pen

        penalty = safety_pen + rule_pen + self.cfg.lam_comfort * comfort_pen + task_pen
        reward = -penalty
        components = {
            "rrs_penalty": penalty,
            "rrs_reward": reward,
            "rrs_safety_pen": safety_pen,
            "rrs_safety_col_pen": safety_col_pen,
            "rrs_safety_d_pen": safety_d_pen,
            "rrs_safety_ttc_pen": safety_ttc_pen,
            "rrs_rule_pen": rule_pen,
            "rrs_rule_lane_pen": rule_lane_pen,
            "rrs_rule_red_pen": rule_red_pen,
            "rrs_rule_spd_pen": rule_spd_pen,
            "rrs_rule_dir_pen": rule_dir_pen,
            "rrs_comfort_pen": comfort_pen,
            "rrs_task_pen": task_pen,
            "rrs_progress_pen": progress_pen,
            "rrs_low_speed_pen": low_speed_pen,
            "rrs_progress": progress,
            "rrs_mean_speed": ego_speed,
        }
        return penalty, components
