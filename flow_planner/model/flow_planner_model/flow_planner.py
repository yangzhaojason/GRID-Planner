import re
import os
import sys
from typing import Literal, Callable, Any, Union, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from flow_planner.model.model_base import DiffusionADPlanner
from flow_planner.model.model_utils.input_preprocess import ModelInputProcessor
from flow_planner.model.model_utils.traj_tool import traj_chunking, assemble_actions
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class FlowPlanner(DiffusionADPlanner):

    def __init__(
        self,
        model_encoder,
        model_decoder,

        flow_ode,
        
        model_type: Literal['x_start', 'noise', 'velocity'] = 'x_start',
        kinematic: Literal["waypoints", "velocity", "acceleration"] = 'waypoints',
    
        assemble_method='linear',
        
        data_processor: ModelInputProcessor = None,
        
        device='cuda',
        rl_enabled=False,
        rl_noise_std=0,
        rl_noise_std_min=None,
        rl_noise_std_max=None,
        **planner_params
    ):
        
        super(FlowPlanner, self).__init__()
        self.model_encoder = model_encoder
        self.model_decoder = model_decoder
        self._model_type = model_type
        self.device = device
        
        self.flow_ode = flow_ode # including flow matching path and ode solver
        self.cfg_prob = planner_params['cfg_prob']
        self.cfg_weight = planner_params['cfg_weight']
        self.cfg_type = planner_params['cfg_type']
        
        self.rl_enabled = rl_enabled
        # 可学习的 noise_std 参数：范围由 rl_noise_std_min/max 指定（启动脚本注入），并通过 sigmoid 映射保证在区间内
        if rl_enabled:
            if rl_noise_std_min is None or rl_noise_std_max is None:
                raise ValueError(
                    "rl_enabled=True requires rl_noise_std_min and rl_noise_std_max. "
                    "Please set planner.flow_planner.rl.noise_std_min/noise_std_max in the launch script."
                )
            self._rl_noise_std_min = float(rl_noise_std_min)
            self._rl_noise_std_max = float(rl_noise_std_max)
            if self._rl_noise_std_min <= 0 or self._rl_noise_std_max <= 0:
                raise ValueError(f"rl_noise_std_min/max must be > 0, got {self._rl_noise_std_min}, {self._rl_noise_std_max}")
            if self._rl_noise_std_min >= self._rl_noise_std_max:
                raise ValueError(f"rl_noise_std_min must be < rl_noise_std_max, got {self._rl_noise_std_min} >= {self._rl_noise_std_max}")

            # 初始化 logit：用区间中点 => sigmoid(logit)=0.5 => logit=0.0
            initial_logit = 0.0
            self.rl_noise_std_logit = nn.Parameter(torch.tensor(initial_logit, dtype=torch.float32))
            # 将FlowPlanner的引用传递给flow_ode，以便动态获取noise_std
            if hasattr(self.flow_ode, 'set_planner_ref'):
                self.flow_ode.set_planner_ref(self)
        else:
            # rl_noise_std 是 @property（只读），不能直接赋值；使用私有变量存放固定值
            if not isinstance(rl_noise_std, (int, float)):
                raise TypeError(f"rl_noise_std must be a number, got {type(rl_noise_std)}")
            self._rl_noise_std = float(rl_noise_std)
            # 同步到 flow_ode（非 RL 模式下它使用自己的 _rl_noise_std）
            if hasattr(self.flow_ode, "_rl_noise_std"):
                self.flow_ode._rl_noise_std = float(rl_noise_std)

        self.kinematic = kinematic
        
        self.assemble_method = assemble_method
        
        self.data_processor = data_processor

        self.planner_params = planner_params # including the action_len, future_len etc.
        self.action_num = (self.planner_params['future_len'] - self.planner_params['action_overlap']) // (self.planner_params['action_len'] - self.planner_params['action_overlap'])
        
        self.basic_loss = nn.MSELoss(reduction='none')
        
    def prepare_model_input(self, cfg_flags, data: NuPlanDataSample, use_cfg, is_training):
        B = data.ego_current.shape[0]

        if is_training:
            # modify the data sample according to cfg_flags
            cfg_type = self.cfg_type
            if cfg_type == 'neighbors':
                neighbor_num = self.planner_params['neighbor_num']
                cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                mask_flags = cfg_flags.view(B, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                mask_flags[:, cfg_neighbor_num:, :] = 1
                data.neighbor_past *= mask_flags
            elif cfg_type == 'lanes':
                data.lanes = data.lanes * cfg_flags.view(B, *([1] * (data.lanes.dim()-1)))

        else:
            if use_cfg:
                data = data.repeat(2)
                cfg_type = self.cfg_type
                if cfg_type == 'neighbors':
                    neighbor_num = self.planner_params['neighbor_num']
                    cfg_neighbor_num = min(self.planner_params['cfg_neighbor_num'], neighbor_num)
                    mask_flags = cfg_flags.view(B * 2, *([1] * (data.neighbor_past.dim()-1))).repeat(1, neighbor_num, 1, 1)
                    mask_flags[:, cfg_neighbor_num:, :] = 1
                    data.neighbor_past *= mask_flags
                elif cfg_type == 'lanes':
                    data.lanes = data.lanes * cfg_flags.view(B * 2, *([1] * (data.lanes.dim()-1)))
           
        model_inputs, gt = self.data_processor.sample_to_model_input(
            data, device=self.device, kinematic=self.kinematic, is_training=is_training
        )
            
        model_inputs.update({'cfg_flags': cfg_flags})
        
        return model_inputs, gt
        
    def extract_encoder_inputs(self, inputs):
        
        encoder_inputs = {
            'neighbors': inputs['neighbor_past'],
            'lanes': inputs['lanes'],
            'lanes_speed_limit': inputs['lanes_speedlimit'],
            'lanes_has_speed_limit': inputs['lanes_has_speedlimit'],
            'static': inputs['map_objects'],
            'routes': inputs['routes']
        }
        return encoder_inputs
    
    def extract_decoder_inputs(self, encoder_outputs, inputs):
        model_extra = dict(cfg_flags=inputs['cfg_flags'] if 'cfg_flags' in inputs.keys() else None,)
        model_extra.update(encoder_outputs)
        return model_extra
    
    def encoder(self, **encoder_inputs):
        return self.model_encoder(**encoder_inputs)
    
    def decoder(self, x, t, **model_extra):
        return self.model_decoder(x, t, **model_extra)
        
    def forward(self, data: NuPlanDataSample, mode='train', **params):
        if mode == 'train':
            return self.forward_train(data)
        elif mode == 'inference':
            data_collector = params['data_collector'] if 'data_collector' in params else None
            return self.forward_inference(data, params['use_cfg'], params['cfg_weight'], data_collector=data_collector)
    
    def forward_train(self, data: NuPlanDataSample):
        '''
        Forward a training step and compute the training loss.
        1. generate cfg_flags
        2. preprocess (masking) according to the cfg_flags
        3. model forward
        4. compute basic mse loss
        
        Return:
            prediction: the raw prediction of the model, specified by model.prediction_type;
            loss_dict: a dict of loss containing unreduced mse loss, consistency loss and neighbor prediction loss (if one exists).
        '''
        B = data.ego_current.shape[0]
        roll_dice = torch.rand((B, 1))
        cfg_flags = (roll_dice > self.cfg_prob).to(torch.int32).to(self.device) # NOTE: 1 for conditioned (unmasked), 0 for unconditioned (masked)
        model_inputs, gt = self.prepare_model_input(cfg_flags, data, use_cfg=False, is_training=True) # note that the cfg_flags are packed into the model_inputs
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)

        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        B, P, T_, D = gt.shape
        
        noised_traj, target, t = self.flow_ode.sample(gt[:, :, 1:, :], self._model_type)
        noised_traj_tokens = traj_chunking(noised_traj, self.planner_params['action_len'], self.planner_params['action_overlap'])
        noised_traj_tokens = torch.cat(noised_traj_tokens, dim=1)
        target_tokens = traj_chunking(target, self.planner_params['action_len'], self.planner_params['action_overlap'])
        target_tokens = torch.cat(target_tokens, dim=1)
        
        prediction = self.decoder(noised_traj_tokens, t, **decoder_model_extra)
        
        loss_dict = {}
        batch_loss = self.basic_loss(prediction, target_tokens)
        loss_dict['batch_loss'] = batch_loss
        
        # aggregate loss over token/time dimensions
        loss = torch.sum(batch_loss, dim=-1) # (B, action_num, action_length)
        # per-sample planning loss for graph-aware weighting
        loss_per_sample = loss.view(B, -1).mean(dim=1)  # (B,)
        loss_dict['ego_planning_loss_per_sample'] = loss_per_sample
        # vanilla mean loss kept for compatibility (may be overridden by core)
        loss_dict['ego_planning_loss'] = loss.mean()

        if self.planner_params['action_overlap'] > 0:
            consistency_loss = [torch.mean(torch.sum(self.basic_loss(prediction[:, i:i+1, -self.planner_params['action_overlap']:, :], prediction[:, i+1:i+2, :self.planner_params['action_overlap'], :]), dim=-1)) for i in range(0, prediction.shape[1]-2)]
            loss_dict['consistency_loss'] = sum(consistency_loss) / len(consistency_loss)
        else:
            loss_dict['consistency_loss'] = torch.tensor(0.0, device=loss.device)
        
        assert not torch.isnan(loss).sum(), f"loss is NaN"
        
        return prediction, loss_dict
    
    def forward_inference(self, data: NuPlanDataSample, use_cfg=True, cfg_weight=None, data_collector=None):
        # RL 模式下使用自定义采样并收集数据
        if self.rl_enabled:
            return self.forward_inference_rl(data, use_cfg=use_cfg, cfg_weight=cfg_weight, data_collector=data_collector)
        B = data.ego_current.shape[0]
        if use_cfg:
            cfg_flags = torch.cat([torch.ones((B,), device=self.device), torch.zeros((B,), device=self.device)], dim=0).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)
        
        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg, is_training=False)
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)
        
        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        
        x_init = torch.randn((B, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim']), device=self.device)
        sample = self.flow_ode.generate(x_init, self.decoder, self._model_type, use_cfg=use_cfg, cfg_weight=cfg_weight, **decoder_model_extra)
        
        sample = assemble_actions(sample, self.planner_params['future_len'], self.planner_params['action_len'], self.planner_params['action_overlap'], self.planner_params['state_dim'], self.assemble_method)
        
        sample = self.data_processor.state_postprocess(sample)
        
        return sample
    
    def forward_inference_rl(self, data: NuPlanDataSample, use_cfg=False, cfg_weight=None, data_collector=None):
        """
        RL 模式下的推理，手动实现 ODE solver 并收集数据

        说明：RL 模式下同样支持 CFG（classifier-free guidance）。
        """
        B = data.ego_current.shape[0]
        if use_cfg:
            cfg_flags = torch.cat(
                [torch.ones((B,), device=self.device), torch.zeros((B,), device=self.device)],
                dim=0
            ).to(torch.int32)
        else:
            cfg_flags = torch.ones((B,), device=self.device).to(torch.int32)
        
        model_inputs, _ = self.prepare_model_input(cfg_flags, data, use_cfg, is_training=False)
        
        encoder_inputs = self.extract_encoder_inputs(model_inputs)
        encoder_outputs = self.encoder(**encoder_inputs)
        
        decoder_model_extra = self.extract_decoder_inputs(encoder_outputs, model_inputs)
        
        # 保存 condition（encoder 输出）用于数据收集
        # encoder_outputs 包含：encodings (tuple), masks (tuple), routes_cond, token_dist
        # decoder_model_extra 包含：encoder_outputs 的所有字段 + cfg_flags
        # 注意：encodings 和 masks 是 tuple，需要特殊处理
        def detach_recursive(obj):
            """递归地将 tensor 转换为 CPU 并 detach"""
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, (list, tuple)):
                return type(obj)(detach_recursive(item) for item in obj)
            elif isinstance(obj, dict):
                return {k: detach_recursive(v) for k, v in obj.items()}
            else:
                return obj

        def split_batch_first(obj, batch_size: int):
            """
            递归地把 batch-first 的对象拆成 (cond, uncond) 两份。
            约定：use_cfg=True 时，batch 维是 2B，前 B 为 cond，后 B 为 uncond。
            """
            if isinstance(obj, torch.Tensor):
                assert obj.shape[0] == batch_size * 2, (
                    f"Expected tensor batch {batch_size * 2}, got {obj.shape[0]} (shape={obj.shape})"
                )
                return obj[:batch_size], obj[batch_size:]
            elif isinstance(obj, (list, tuple)):
                cond_parts = []
                uncond_parts = []
                for item in obj:
                    c, u = split_batch_first(item, batch_size)
                    cond_parts.append(c)
                    uncond_parts.append(u)
                return type(obj)(cond_parts), type(obj)(uncond_parts)
            elif isinstance(obj, dict):
                cond_dict = {}
                uncond_dict = {}
                for k, v in obj.items():
                    c, u = split_batch_first(v, batch_size)
                    cond_dict[k] = c
                    uncond_dict[k] = u
                return cond_dict, uncond_dict
            else:
                # 非 batch-first / 非 tensor 的字段（例如 None、标量）两边共享
                return obj, obj
        
        # 保存完整的 condition：
        # - use_cfg=False：保留原结构（单份 cond）
        # - use_cfg=True ：额外保存 uncond 一份（cond/uncond 拆开存）
        if use_cfg:
            enc_cond, enc_uncond = split_batch_first(encoder_outputs, B)
            extra_cond, extra_uncond = split_batch_first(decoder_model_extra, B)
            condition = {
                'cond': {
                    'encoder_outputs': detach_recursive(enc_cond),
                    'decoder_model_extra': detach_recursive(extra_cond),
                },
                'uncond': {
                    'encoder_outputs': detach_recursive(enc_uncond),
                    'decoder_model_extra': detach_recursive(extra_uncond),
                },
            }
        else:
            condition = {
                'cond': {
                    'encoder_outputs': detach_recursive(encoder_outputs),
                    'decoder_model_extra': detach_recursive(decoder_model_extra),
                }
            }
        
        x_init = torch.randn((B, self.action_num, self.planner_params['action_len'], self.planner_params['state_dim']), device=self.device)
        
        # 传递 data_collector 和 condition 到 flow_ode.generate
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"forward_inference_rl: data_collector={data_collector}, condition keys={list(condition.keys()) if condition else None}")
        
        if data_collector is None:
            logger.error("forward_inference_rl: data_collector is None! This should not happen in RL mode.")
            raise ValueError("data_collector cannot be None in RL mode")
        
        sample = self.flow_ode.generate(
            x_init, 
            self.decoder, 
            self._model_type, 
            use_cfg=use_cfg, 
            cfg_weight=cfg_weight,
            data_collector=data_collector,
            condition=condition,
            **decoder_model_extra
        )
        
        logger.debug(f"forward_inference_rl: completed, data_collector.current_simulation_data size: {len(data_collector.current_simulation_data) if data_collector else 0}")
        
        sample = assemble_actions(sample, self.planner_params['future_len'], self.planner_params['action_len'], self.planner_params['action_overlap'], self.planner_params['state_dim'], self.assemble_method)
        
        sample = self.data_processor.state_postprocess(sample)
        
        return sample
    
    @property
    def model_type(self,):
        return self._model_type
    
    def get_optimizer_params(self):
        params = [
            {'params': self.model_encoder.parameters()},
            {'params': self.model_decoder.parameters()}
        ]
        # 添加可学习的noise_std参数
        if self.rl_enabled and hasattr(self, 'rl_noise_std_logit'):
            params.append({'params': [self.rl_noise_std_logit], 'lr': 1e-4})  # 使用较小的学习率
        return params
    
    @property
    def rl_noise_std(self):
        """获取 noise_std（torch.Tensor）。

        说明：
        - 这里必须返回 torch.Tensor（而不是 .item() / float），否则会切断梯度，导致 noise_std 无法被学习。
        - 若下游确实需要 Python float（例如仿真阶段加噪），应在调用点显式 detach 并转换。
        """
        if self.rl_enabled and hasattr(self, 'rl_noise_std_logit'):
            # 使用sigmoid将logit映射到[0,1]，然后缩放到[rl_noise_std_min, rl_noise_std_max]
            noise_std = self._rl_noise_std_min + (self._rl_noise_std_max - self._rl_noise_std_min) * torch.sigmoid(self.rl_noise_std_logit)
            # 数值安全：确保严格落在区间内
            noise_std = torch.clamp(noise_std, min=self._rl_noise_std_min, max=self._rl_noise_std_max)
            return noise_std
        else:
            return self._rl_noise_std
