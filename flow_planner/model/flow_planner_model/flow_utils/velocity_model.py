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
        '''
        # 断言：检查输入形状
        assert x.dim() in [4, 5], f"x should be 4D or 5D, got {x.dim()}D, shape={x.shape}"
        
        # 处理 x 的形状：可能是 4 维 (B, P, T, D) 或 5 维 (B, P, action_num, action_len, state_dim)
        if x.dim() == 5:
            # 如果是 5 维，reshape 为 4 维: (B, P, action_num * action_len, state_dim)
            B, P, action_num, action_len, state_dim = x.shape
            x = x.reshape(B, P, action_num * action_len, state_dim)
        elif x.dim() == 4:
            B, P, _, _ = x.shape
        else:
            raise ValueError(f"Unexpected x dimension: {x.dim()}, expected 4 or 5. x.shape={x.shape}")
        
        # 断言：x 应该是 4D (B, P, T, D)
        assert x.dim() == 4, f"After processing, x should be 4D (B, P, T, D), got {x.dim()}D, shape={x.shape}"

        # 处理 t 的形状：确保 t 是 (B,) 形状
        # decoder 期望 t 是 (B,) 形状，而不是 (1, B) 或标量
        t = t.to(x.device)
        
        # 如果 t 是标量（0维），扩展为 (1,) 然后广播到 (B,)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        # 如果 t 是 (1, B) 或其他形状，需要确保是 (B,)
        elif t.dim() == 2:
            # 如果是 (1, B)，取第一行并广播
            if t.shape[0] == 1:
                t = t.squeeze(0)  # (B,)
            else:
                raise ValueError(f"Unexpected t shape: {t.shape}, expected (B,) or scalar")
        # 如果 t 是 (B,)，保持不变
        elif t.dim() == 1:
            if t.shape[0] != B:
                # 如果形状不匹配，尝试广播
                if t.shape[0] == 1:
                    t = t.expand(B)
                else:
                    raise ValueError(f"t shape {t.shape} does not match batch size {B}")
        else:
            raise ValueError(f"Unexpected t dimension: {t.dim()}, expected 0, 1, or 2")
        
        # 断言：t 应该是 1D (B,)
        assert t.dim() == 1, f"After processing, t should be 1D (B,), got {t.dim()}D, shape={t.shape}"
        assert t.shape[0] == B, f"t batch size {t.shape[0]} != x batch size {B}"
        
        if self.use_cfg:
            x = x.repeat(2, *[1] * (x.dim()-1))
        
        pred = self.model_fn(x, t, **model_extras)
        # 断言：pred 的形状应该与 x 一致
        assert pred.shape == x.shape, f"pred shape {pred.shape} != x shape {x.shape}"
        
        if self.correct_xt_fn is not None:
            pred = self.correct_xt_fn(pred)
        us = []
        if len(t) != len(pred):
            t = t.expand(len(pred))
        for i in range(len(pred)):
            us.append(self.pred_transform_func(pred[i], x[i], t[i]))
        u = torch.stack(us)
        
        # 断言：u 的形状应该与 x 一致
        assert u.shape == x.shape, f"u shape {u.shape} != x shape {x.shape}"
        
        if self.use_cfg:
            u_cond, u_uncond = torch.chunk(u, 2) # the unconditioned batch is the first half
            u = (1 - self.cfg_weight) * u_uncond + self.cfg_weight * u_cond
        return u