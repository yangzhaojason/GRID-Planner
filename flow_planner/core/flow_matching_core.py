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
            rl_enabled=False,
            rl_noise_std=0.05,
            **loss_weights,
    ):
        
        super(FlowMatchingCore, self).__init__()
        
        self.device = device
        self.input_aug = input_aug
        self.loss_weights = loss_weights
        self.rl_enabled = rl_enabled
        self.rl_noise_std = rl_noise_std
        
    def train_step(self, model, data: NuPlanDataSample):
        
        data = self.input_aug(data)
        prediction, loss_dict = model(data, mode='train')
        
        total_loss = sum([loss_dict[k] * self.loss_weights[k] for k in self.loss_weights.keys()])
        total_loss_dict = {'total_loss': total_loss}

        return total_loss_dict
        
    def inference(self, model, data, use_cfg=False, cfg_weight=None, data_collector=None):
        model = model.to(self.device)
        model.eval()
        
        # 检查是否是 RL 模式
        if self.rl_enabled:
            return self.inference_rl(model, data, use_cfg=use_cfg, cfg_weight=cfg_weight, data_collector=data_collector)
        
        prediction = model(data, mode='inference', use_cfg=use_cfg, cfg_weight=cfg_weight)
        
        return prediction
    
    def inference_rl(self, model, data, use_cfg=False, cfg_weight=None, data_collector=None):
        """
        RL 模式下的推理，收集去噪步骤数据

        说明：RL 模式下同样支持 CFG（classifier-free guidance）。
        """
        model = model.to(self.device)
        model.eval()  # 设置为 eval 模式，但不关闭梯度（因为后续训练需要）
        
        prediction = model(data, mode='inference', use_cfg=use_cfg, cfg_weight=cfg_weight, data_collector=data_collector)
        
        return prediction
    
    def initial_state_constraint(self, xt, s1, B, T_, D):
        xt = xt.view(B, -1, T_, D)
        xt[:, :, :1, :] = s1[:, :xt.shape[1]]
        return xt.view(B, -1, T_ * D)