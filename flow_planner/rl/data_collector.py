"""
RL 数据收集器：收集和管理 RL 训练数据
"""
from typing import List, Dict, Any
import torch
import numpy as np


class RLDataCollector:
    """收集和管理 RL 训练数据"""
    
    def __init__(self):
        # batch_data: 存储当前 batch 的数据，按“去噪轨迹(多步去噪)”为单位
        # 形状：List[trajectory]，其中 trajectory 是 List[datapoint_dict]（长度=denoising_steps）
        self.batch_data: List[List[Dict[str, Any]]] = []
        # current_simulation_data: 当前 simulation 的所有去噪轨迹（按规划步/调用 generate_rl 次数划分）
        self.current_simulation_data: List[List[Dict[str, Any]]] = []
        # 当前一次去噪过程的 step 数据（长度最终应等于 denoising_steps）
        self._current_denoising_trajectory: List[Dict[str, Any]] = []
        
    def collect_step_data(self, latent, condition, timestep, x_next, x_noised, log_prob):
        """
        收集单步数据
        
        Args:
            latent: torch.Tensor - 去噪起始的 latent (B, P, T, D)
            condition: Dict - 去噪时使用的 condition（encoder 输出等）
            timestep: torch.Tensor - 去噪的 timestep (B,)
            x_next: torch.Tensor - 无噪声的下一步结果 (B, P, T, D)
            x_noised: torch.Tensor - 加噪后的结果 (B, P, T, D)
            log_prob: float - log_prob（标量）
        """
        import logging
        logger = logging.getLogger(__name__)
        
        def detach_recursive(obj):
            """递归地将 tensor 转换为 CPU 并 detach"""
            if isinstance(obj, torch.Tensor):
                return obj.detach().cpu()
            elif isinstance(obj, dict):
                return {k: detach_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return type(obj)(detach_recursive(item) for item in obj)
            else:
                return obj
        
        # 将 tensor 转换为可序列化的格式（detach 并移到 CPU）
        data_point = {
            'latent': detach_recursive(latent),
            'condition': detach_recursive(condition),
            'timestep': detach_recursive(timestep),
            'x_next': detach_recursive(x_next),
            'x_noised': detach_recursive(x_noised),
            'log_prob': float(log_prob) if isinstance(log_prob, (torch.Tensor, np.ndarray)) else log_prob,
            'score': None  # 将在 finalize_simulation 时填充
        }
        self._current_denoising_trajectory.append(data_point)
        
        # 每收集一定数量的数据点后打印一次（避免日志过多）
        # if len(self.current_simulation_data) % 10 == 0:
        #     logger.info(f"collect_step_data: collected {len(self.current_simulation_data)} data points so far")
        
        # 第一次收集时打印
        # if len(self.current_simulation_data) == 1:
        #     logger.info(f"collect_step_data: First data point collected! latent shape={latent.shape if hasattr(latent, 'shape') else type(latent)}")
        
    def finalize_denoising_trajectory(self, expected_steps: int):
        """
        完成一次“多步去噪轨迹”的收集（用于 joint_logprob）。
        :param expected_steps: 本次去噪的步数（应等于 sample_steps）
        """
        if not isinstance(expected_steps, int) or expected_steps <= 0:
            raise ValueError(f"expected_steps must be positive int, got {expected_steps} ({type(expected_steps)})")
        if len(self._current_denoising_trajectory) != expected_steps:
            raise ValueError(
                f"Denoising trajectory length mismatch: got {len(self._current_denoising_trajectory)} "
                f"but expected {expected_steps}. This indicates corrupted rollout collection."
            )
        self.current_simulation_data.append(self._current_denoising_trajectory)
        self._current_denoising_trajectory = []

    def finalize_simulation(self, score: float):
        """
        完成一个 simulation，添加 score 到所有数据
        
        Args:
            score: float - 该 simulation 的最终 score
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"finalize_simulation called: score={score}, num_trajectories={len(self.current_simulation_data)}")

        if len(self._current_denoising_trajectory) != 0:
            raise ValueError(
                f"finalize_simulation called with unfinished denoising trajectory: "
                f"len(_current_denoising_trajectory)={len(self._current_denoising_trajectory)}. "
                "Please ensure finalize_denoising_trajectory is called after each generate_rl."
            )

        # 将 score 广播到当前 simulation 的所有数据点（每条去噪轨迹内的所有 step）
        for trajectory in self.current_simulation_data:
            for data_point in trajectory:
                data_point['score'] = score
        
        # 将当前 simulation 的去噪轨迹添加到 batch（按轨迹级追加，避免多一层嵌套）
        if len(self.current_simulation_data) > 0:
            self.batch_data.extend(self.current_simulation_data)
            logger.debug(
                f"  Added {len(self.current_simulation_data)} trajectories to batch_data "
                f"(total trajectories: {len(self.batch_data)})"
            )
            self.current_simulation_data = []
        else:
            logger.warning("  No denoising trajectories to finalize (current_simulation_data is empty)")
        
    def get_batch_data(self) -> List[List[Dict[str, Any]]]:
        """
        获取当前 batch 的所有数据
        
        Returns:
            List[List[Dict]]: 按“去噪轨迹”为单位的数据列表；每个元素长度=denoising_steps
        """
        return self.batch_data
        
    def clear(self):
        """清空数据"""
        self.batch_data = []
        self.current_simulation_data = []
        self._current_denoising_trajectory = []
