# flow ode: responsible for training and inference sampling.
# time sampler, flow interpolation, ode solver, cfg related

import torch
from flow_planner.model.flow_planner_model.flow_utils.velocity_model import VelocityModel
from flow_planner.model.model_base import Scheduler
from flow_matching.solver.ode_solver import ODESolver

class FlowODE(Scheduler):
    
    def __init__(self,
                 path, 
                 time_sampler,
                 cfg_weight=1.5,
                 rl_enabled=False,
                 rl_noise_std=0.05,
                 **sample_params
                 ):
        '''
        params:
            path: defines the flow ode
            time_sampler: a random sampler used for training
            sample_params: including basic "sample_temperature" "sample_steps" and "sample_method"
            rl_enabled: bool - 是否启用 RL 模式
            rl_noise_std: float - RL 模式下的噪声标准差
        '''
        self.path = path
        self.cfg_weight = cfg_weight
        self.time_sampler = time_sampler
        self.sample_params = sample_params
        self.rl_enabled = rl_enabled
        self._rl_noise_std = rl_noise_std  # 初始值或固定值（使用私有变量）
        self.planner_ref = None  # FlowPlanner的引用，用于获取可学习的noise_std
        self.translation_funcs = self._get_translation_funcs()
    
    def set_planner_ref(self, planner):
        """设置FlowPlanner的引用，用于动态获取可学习的noise_std"""
        self.planner_ref = planner
    
    @property
    def rl_noise_std(self):
        """获取noise_std值，如果planner_ref存在则使用可学习参数"""
        if self.planner_ref is not None and hasattr(self.planner_ref, 'rl_noise_std_logit'):
            # 从FlowPlanner获取可学习的noise_std
            return self.planner_ref.rl_noise_std
        else:
            # 使用固定值
            return self._rl_noise_std

    def sample(self, x_data, target_type):
        B = x_data.shape[0]
        t = self.time_sampler.sample(B).to(x_data.device)
        x_0 = torch.randn_like(x_data, device=x_data.device)
        path_sample = self.path.sample(x_0=x_0, x_1=x_data, t=t)
        
        if target_type == 'velocity':
            target = path_sample.dx_t
        elif target_type == 'x_start':
            target = path_sample.x_1
        elif target_type == 'noise':
            target = x_0
            
        return path_sample.x_t, target, t
    
    def generate(self, x_init, model_fn, model_pred_type, use_cfg, **model_extra):
        '''
        Sample a the data using the model_fn. If use_cfg, the predicted velocity will be chunked and treats the first chunk as conditioned velocity, the second as unconditioned one.
        '''
        if self.rl_enabled:
            if 'data_collector' not in model_extra:
                raise ValueError("FlowODE.generate: rl_enabled=True but data_collector is missing in model_extra")
            return self.generate_rl(x_init, model_fn, model_pred_type, use_cfg, **model_extra)
        velocity_func = self.translation_funcs[(model_pred_type, 'velocity')]
        velocity_model = VelocityModel(model_fn, self.path, velocity_func, use_cfg=use_cfg, cfg_weight=self.cfg_weight)
        
        solver = ODESolver(velocity_model=velocity_model)
        
        x_init = x_init * self.sample_params['sample_temperature']
        step_size = 1.0 / self.sample_params['sample_steps']
        sample = solver.sample(x_init=x_init,
                               step_size=step_size,
                               method=self.sample_params['sample_method'],
                               **model_extra)
        
        return sample
    
    def generate_rl(self, x_init, model_fn, model_pred_type, use_cfg, **model_extra):
        '''
        RL 模式下的生成，手动实现 ODE 并收集数据
        
        说明：RL 模式同样支持 CFG（classifier-free guidance）。
        '''
        from flow_planner.model.flow_planner_model.flow_utils.manual_ode_solver import ManualODESolver
        
        # 获取数据收集器和 condition
        if 'data_collector' not in model_extra:
            raise ValueError("FlowODE.generate_rl: data_collector is missing (RL mode requires it)")
        data_collector = model_extra.pop('data_collector')
        condition = model_extra.pop('condition') if 'condition' in model_extra else {}
        
        velocity_func = self.translation_funcs[(model_pred_type, 'velocity')]
        velocity_model = VelocityModel(model_fn, self.path, velocity_func, use_cfg=use_cfg, cfg_weight=self.cfg_weight)
        
        x_init = x_init * self.sample_params['sample_temperature']
        step_size = 1.0 / self.sample_params['sample_steps']
        
        # 仿真阶段加噪不需要梯度：这里显式 detach 并转为 float
        noise_std = self.rl_noise_std
        if isinstance(noise_std, torch.Tensor):
            noise_std_value = float(noise_std.detach().cpu().item())
        else:
            noise_std_value = float(noise_std)
        
        # 创建手动 ODE Solver
        manual_solver = ManualODESolver(
            velocity_model=velocity_model,
            path=self.path,
            step_size=step_size,
            method=self.sample_params['sample_method']
        )
        
        # 执行带数据收集的采样
        sample, _ = manual_solver.sample_with_data_collection(
            x_init=x_init,
            num_steps=self.sample_params['sample_steps'],
            noise_std=noise_std_value,
            data_collector=data_collector,
            condition=condition,
            **model_extra
        )
        return sample
    
    def identity(self, x, xt, t):
        return x
    
    def _get_translation_funcs(self):
        '''
        Transform the prediction according to the flow path. Loss function is computed in the same 
        representation as "target_type", so the prediction is first transformed before fed into loss
        function.
        
        Affine flow path: x_t = alpha_t * x_1 + sigma_t * x_0
        CondOT: x_t = t * x_1 + (1 - t) * x_0 
        '''
            
        return {('velocity', 'x_start'): self.path.velocity_to_target,
                ('velocity', 'noise'): self.path.velocity_to_epsilon,
                ('x_start', 'velocity'): self.path.target_to_velocity,
                ('x_start', 'noise'): self.path.target_to_epsilon,
                ('noise', 'velocity'): self.path.epsilon_to_velocity,
                ('noise', 'x_start'): self.path.epsilon_to_target,
                ('velocity', 'velocity'): self.identity,
                ('x_start', 'x_start'): self.identity,
                ('noise', 'noise'): self.identity}