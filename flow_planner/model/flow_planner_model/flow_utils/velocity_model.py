from typing import Dict
import torch
from torch import nn
from flow_matching.utils import ModelWrapper
from flow_matching.path.scheduler.scheduler import Scheduler
from flow_matching.path.scheduler.schedule_transform import ScheduleTransformedModel
from flow_matching.path.affine import AffineProbPath

class VelocityModel(nn.Module):

    def __init__(self, model_fn, path, pred_transform_func: str, correct_xt_fn=None, use_cfg=True, cfg_weight=None):
        super().__init__()
        self.model_fn = model_fn
        self.path = path
        self.pred_transform_func = pred_transform_func
        self.correct_xt_fn = correct_xt_fn
        self.use_cfg = use_cfg
        self.cfg_weight = cfg_weight

    def forward(self, x, t, **model_extras):
        # TODO: add wrapper for guidance etc.
        '''
        :params
            x: sampled_trajectories for model input (B, P, T, D)
            t: sampled time (B,) - 每个样本的 timestep（可能不同）
            **model_extras: other model params used for prediction 
            x: sampled_trajectories for model input
            t: sampled time
            **model_extras: other model params used for prediction 
        '''
        if x.dim() not in (4, 5):
            raise ValueError(f"x should be 4D or 5D, got {x.dim()}D, shape={x.shape}")
        # 统一到 4D (B, P, T, D)
        if x.dim() == 5:
            B, P, action_num, action_len, state_dim = x.shape
            x = x.reshape(B, P, action_num * action_len, state_dim)
        else:
            B, P, _, _ = x.shape

        # 规范化 t 为 (B,) 并放到同一 device
        t = t.to(x.device)
        if t.dim() == 0:
            t = t.view(1).expand(B)
        elif t.dim() == 1:
            if t.shape[0] == 1 and B != 1:
                t = t.expand(B)
            elif t.shape[0] != B:
                raise ValueError(f"t shape {tuple(t.shape)} does not match batch size {B}")
        elif t.dim() == 2 and t.shape[0] == 1 and t.shape[1] == B:
            t = t.squeeze(0)
        else:
            raise ValueError(f"Unexpected t shape: {tuple(t.shape)}; expected scalar or (B,)")
        
        if self.use_cfg:
            x = x.repeat(2, *[1] * (x.dim()-1))
            t = t.repeat(2)
        
        pred = self.model_fn(x, t, **model_extras)
        if self.correct_xt_fn is not None:
            pred = self.correct_xt_fn(pred)
        u = self.pred_transform_func(pred, x, t)
        if self.use_cfg:
            u_cond, u_uncond = torch.chunk(u, 2) # the unconditioned batch is the first half
            u = (1 - self.cfg_weight) * u_uncond + self.cfg_weight * u_cond
        return u