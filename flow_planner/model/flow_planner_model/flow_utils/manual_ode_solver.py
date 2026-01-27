"""
手动实现的 ODE Solver，与原有 ODESolver 逻辑一致，但支持梯度
支持每个样本 timestep 不同的情况
"""
import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np


class ManualODESolver:
    """手动实现的 ODE Solver，与原有 ODESolver 逻辑一致，但支持梯度"""
    
    def __init__(self, velocity_model, path, step_size, method='midpoint'):
        """
        Args:
            velocity_model: VelocityModel - 速度模型
            path: FlowPath - 流路径
            step_size: float - ODE 步长
            method: str - 求解方法，支持 'midpoint', 'euler' 等
        """
        self.velocity_model = velocity_model
        self.path = path
        self.step_size = step_size
        self.method = method
        
    def step(self, x: torch.Tensor, t: torch.Tensor, **model_extra) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行一步 ODE 求解
        
        Args:
            x: (B, P, T, D) - 当前状态
            t: (B,) - 每个样本的当前 timestep（可能不同）
            **model_extra: 其他模型参数
            
        Returns:
            x_next: (B, P, T, D) - 下一步状态
            t_next: (B,) - 下一步 timestep
        """
        if self.method == 'midpoint':
            return self._midpoint_step(x, t, **model_extra)
        elif self.method == 'euler':
            return self._euler_step(x, t, **model_extra)
        else:
            raise ValueError(f"Unsupported method: {self.method}")
    
    def _euler_step(self, x: torch.Tensor, t: torch.Tensor, **model_extra) -> Tuple[torch.Tensor, torch.Tensor]:
        """Euler 方法"""
        # 断言：检查输入形状
        assert x.dim() == 4, f"x should be 4D (B, P, T, D), got {x.dim()}D, shape={x.shape}"
        assert t.dim() == 1, f"t should be 1D (B,), got {t.dim()}D, shape={t.shape}"
        assert t.shape[0] == x.shape[0], f"t batch size {t.shape[0]} != x batch size {x.shape[0]}"
        
        # t 是 (B,) 形状，每个样本的 timestep 可能不同
        # velocity_model 需要 t 作为 (B,) 或能够广播的形状
        v = self.velocity_model(x, t, **model_extra)  # (B, P, T, D)
        
        # 断言：v 的形状应该与 x 一致
        assert v.shape == x.shape, f"v shape {v.shape} != x shape {x.shape}"
        
        # 更新
        x_next = x + self.step_size * v
        t_next = t + self.step_size  # (B,)，每个样本独立更新
        
        # 断言：检查输出形状
        assert x_next.shape == x.shape, f"x_next shape {x_next.shape} != x shape {x.shape}"
        assert t_next.shape == t.shape, f"t_next shape {t_next.shape} != t shape {t.shape}"
        
        return x_next, t_next
    
    def _midpoint_step(self, x: torch.Tensor, t: torch.Tensor, **model_extra) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Midpoint 方法（与原有实现一致）
        
        Args:
            x: (B, P, T, D) - 当前状态
            t: (B,) - 每个样本的当前 timestep（可能不同）
            
        Returns:
            x_next: (B, P, T, D) - 下一步状态
            t_next: (B,) - 下一步 timestep（每个样本独立更新）
        """
        # 断言：检查输入形状
        assert x.dim() == 4, f"x should be 4D (B, P, T, D), got {x.dim()}D, shape={x.shape}"
        assert t.dim() == 1, f"t should be 1D (B,), got {t.dim()}D, shape={t.shape}"
        assert t.shape[0] == x.shape[0], f"t batch size {t.shape[0]} != x batch size {x.shape[0]}"
        
        # 计算 k1
        v1 = self.velocity_model(x, t, **model_extra)  # (B, P, T, D)
        # 断言：v1 的形状应该与 x 一致
        assert v1.shape == x.shape, f"v1 shape {v1.shape} != x shape {x.shape}"
        
        # 计算中间点（标准 midpoint：在 t + h/2 处评估斜率）
        t_mid = t + self.step_size / 2  # (B,)，每个样本独立计算
        
        # 断言：t_mid 的形状应该与 t 一致
        assert t_mid.shape == t.shape, f"t_mid shape {t_mid.shape} != t shape {t.shape}"
        
        x_mid = x + self.step_size / 2 * v1
        
        # 断言：x_mid 的形状应该与 x 一致
        assert x_mid.shape == x.shape, f"x_mid shape {x_mid.shape} != x shape {x.shape}"
        
        # 计算 k2
        v2 = self.velocity_model(x_mid, t_mid, **model_extra)  # (B, P, T, D)
        # 断言：v2 的形状应该与 x 一致
        assert v2.shape == x.shape, f"v2 shape {v2.shape} != x shape {x.shape}"
        
        # 更新
        x_next = x + self.step_size * v2
        t_next = t + self.step_size  # (B,)，每个样本独立更新
        
        # 断言：检查输出形状
        assert x_next.shape == x.shape, f"x_next shape {x_next.shape} != x shape {x.shape}"
        assert t_next.shape == t.shape, f"t_next shape {t_next.shape} != t shape {t.shape}"
        
        return x_next, t_next
        
    def sample_with_data_collection(
        self, 
        x_init: torch.Tensor, 
        num_steps: int, 
        noise_std: float,
        data_collector,
        condition: Dict[str, Any],
        **model_extra
    ) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
        """
        带数据收集的采样（仅用于 simulation 阶段，会加噪）
        
        Args:
            x_init: (B, P, T, D) - 初始状态
            num_steps: int - ODE 求解步数
            noise_std: float - 加噪标准差
            data_collector: RLDataCollector - 数据收集器
            condition: Dict - condition 信息（encoder 输出等）
            **model_extra: 其他模型参数
            
        Returns:
            sample: (B, P, T, D) - 最终采样结果
            collected_data: List[Dict] - 收集的数据列表
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if data_collector is None:
            logger.error("sample_with_data_collection: data_collector is None!")
            raise ValueError("data_collector cannot be None in RL mode")
        
        B = x_init.shape[0]
        device = x_init.device
        
        # logger.info(f"sample_with_data_collection: B={B}, num_steps={num_steps}, noise_std={noise_std}, data_collector={data_collector}")
        
        # 初始化
        x = x_init
        timesteps = torch.zeros(B, device=device)  # 初始 timestep 都是 1.0
        
        collected_data = []
        initial_data_size = len(data_collector.current_simulation_data) if data_collector else 0
        # logger.info(f"sample_with_data_collection: initial current_simulation_data size={initial_data_size}")
        
        for step in range(num_steps):
            # 保存当前 latent（去噪起始点）
            latent = x.clone()
            
            # 执行一步 ODE 求解
            x_next, timesteps_next = self.step(x, timesteps, **model_extra)
            # 加噪声
            noise_std_now = noise_std * (1-(timesteps[0].detach().cpu().item()))**2
            noise = torch.randn_like(x_next) * noise_std_now
            x_noised = x_next + noise
            
            # 计算 log_prob（x_noised 相对于 x_next 的高斯分布）
            diff = x_noised - x_next
            log_prob = -0.5 * (diff ** 2).mean(dim=(-3, -2, -1)) / (noise_std_now ** 2)
            log_prob = log_prob - 0.5 * np.log(2 * np.pi) - np.log(noise_std_now)
            log_prob = log_prob.mean().item()  # 对所有维度取平均，得到标量
            
            # 收集数据
            if data_collector is not None:
                data_collector.collect_step_data(
                    latent=latent,
                    condition=condition,
                    timestep=timesteps.clone(),
                    x_next=x_next,
                    x_noised=x_noised,
                    log_prob=log_prob
                )
            else:
                logger.error(f"sample_with_data_collection: data_collector is None at step {step}!")
            
            # 更新：使用加噪后的结果作为下一步的起始
            x = x_noised
            timesteps = timesteps_next

        # 将这一次 generate_rl 的“多步去噪”聚合成一条轨迹（用于 joint_logprob）
        data_collector.finalize_denoising_trajectory(expected_steps=num_steps)

        final_data_size = len(data_collector.current_simulation_data) if data_collector else 0
        # logger.info(f"sample_with_data_collection: completed {num_steps} steps, collected {final_data_size - initial_data_size} data points, final current_simulation_data size={final_data_size}")
        
        if final_data_size == initial_data_size:
            logger.error(f"sample_with_data_collection: No data collected! (initial={initial_data_size}, final={final_data_size})")
        
        return x, collected_data
        
    def sample_without_noise(
        self,
        x_init: torch.Tensor,
        timesteps_init: torch.Tensor,
        num_steps: int,
        **model_extra
    ) -> torch.Tensor:
        """
        不带加噪的采样（用于训练阶段，计算 log_prob_new）
        
        Args:
            x_init: (B, P, T, D) - 从保存的数据中恢复的起始 latent
            timesteps_init: (B,) - 从保存的数据中恢复的起始 timesteps
            num_steps: int - ODE 求解步数
            **model_extra: 其他模型参数
            
        Returns:
            x_final: (B, P, T, D) - 最终的 x_next_new（不加噪）
        """
        # 断言：检查输入形状
        assert x_init.dim() == 4, f"x_init should be 4D (B, P, T, D), got {x_init.dim()}D, shape={x_init.shape}"
        assert timesteps_init.dim() == 1, f"timesteps_init should be 1D (B,), got {timesteps_init.dim()}D, shape={timesteps_init.shape}"
        assert timesteps_init.shape[0] == x_init.shape[0], f"timesteps_init batch size {timesteps_init.shape[0]} != x_init batch size {x_init.shape[0]}"
        
        x = x_init
        timesteps = timesteps_init.clone()
        
        # for step in range(num_steps):
        # 执行一步 ODE 求解（不加噪）
        # 训练阶段用于计算 log_prob_new：这里固定使用 midpoint，避免受 self.method 影响
        x, timesteps = self.step(x, timesteps, **model_extra)
        
        # 断言：检查每步后的形状
        assert x.shape == x_init.shape, f" x shape {x.shape} != x_init shape {x_init.shape}"
        assert timesteps.shape == timesteps_init.shape, f" timesteps shape {timesteps.shape} != timesteps_init shape {timesteps_init.shape}"
    
        # 断言：检查最终输出形状
        assert x.shape == x_init.shape, f"Final x shape {x.shape} != x_init shape {x_init.shape}"
        
        return x
