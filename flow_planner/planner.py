
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type
import hydra
from hydra.utils import instantiate
import omegaconf

warnings.filterwarnings("ignore")

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from flow_planner.data.data_process.data_processor import DataProcessor
from flow_planner.data.dataset.nuplan import NuPlanDataSample

def identity(ego_state, predictions):
    return predictions


class FlowPlanner(AbstractPlanner):
    def __init__(
            self,
            config_path,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            use_cfg: bool = True,
            cfg_weight: float = 1.0,
            rl: dict = None,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        config = omegaconf.OmegaConf.load(config_path)
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device
        # 标记：如果在 RL 循环中由主进程同步过权重到 _planner，则 worker 侧 initialize 不应再从 ckpt 覆盖，
        # 也不应打印 “load random model” 误导排查。
        self._weights_synced_from_main_process = False

        # RL 配置
        if rl is None:
            rl = {}
        self._rl_enabled = rl.get('enabled', False)
        self._rl_noise_std = rl.get('noise_std', 0.05)
        
        # 初始化 RL 数据收集器
        if self._rl_enabled:
            from flow_planner.rl.data_collector import RLDataCollector
            self._rl_data_collector = RLDataCollector()
        else:
            self._rl_data_collector = None

        # 修改 model 和 core 的配置以支持 RL
        if self._rl_enabled:
            # 更新 model 配置
            if 'rl_enabled' not in config.model:
                config.model.rl_enabled = True
            if 'rl_noise_std' not in config.model:
                config.model.rl_noise_std = self._rl_noise_std
            
            # 更新 core 配置
            if 'rl_enabled' not in config.core:
                config.core.rl_enabled = True
            if 'rl_noise_std' not in config.core:
                config.core.rl_noise_std = self._rl_noise_std
            
            # 更新 flow_ode 配置
            if 'rl_enabled' not in config.model.flow_ode:
                config.model.flow_ode.rl_enabled = True
            if 'rl_noise_std' not in config.model.flow_ode:
                config.model.flow_ode.rl_noise_std = self._rl_noise_std

        self._planner = instantiate(config.model)

        self.core = instantiate(config.core)

        self.data_processor = DataProcessor(None)

        self.use_cfg = use_cfg

        self.cfg_weight = cfg_weight
        
    def name(self) -> str:
        """
        Inherited.
        """
        return "diffusion_planner"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids

        if self._ckpt_path is not None:
            ckpt_obj = torch.load(self._ckpt_path, weights_only=False, map_location=self._device)

            # 兼容多种 checkpoint 格式：
            # - 监督训练：{'ema_state_dict': ..., ...} 或 {'model': ...}
            # - RLTrainer：{'model_state_dict': ..., ...}
            # - 也可能直接就是 state_dict（tensor 参数 dict）
            if isinstance(ckpt_obj, dict):
                if self._ema_enabled and 'ema_state_dict' in ckpt_obj:
                    raw_state_dict = ckpt_obj['ema_state_dict']
                elif 'model_state_dict' in ckpt_obj:
                    raw_state_dict = ckpt_obj['model_state_dict']
                elif 'model' in ckpt_obj:
                    raw_state_dict = ckpt_obj['model']
                elif 'state_dict' in ckpt_obj:
                    raw_state_dict = ckpt_obj['state_dict']
                else:
                    # 如果 dict 本身就是参数字典（key 看起来像权重名），直接用它
                    raw_state_dict = ckpt_obj
            else:
                raise TypeError(f"Unexpected checkpoint type: {type(ckpt_obj)} at {self._ckpt_path}")

            # DDP 兼容：只有在确实存在 module. 前缀时才剥离
            has_module_prefix = False
            if isinstance(raw_state_dict, dict):
                for k in raw_state_dict.keys():
                    if isinstance(k, str) and k.startswith("module."):
                        has_module_prefix = True
                        break
            else:
                raise TypeError(f"Checkpoint state_dict must be dict, got {type(raw_state_dict)} at {self._ckpt_path}")

            if has_module_prefix:
                model_state_dict = {k[len("module."):]: v for k, v in raw_state_dict.items() if isinstance(k, str) and k.startswith("module.")}
            else:
                model_state_dict = raw_state_dict

            self._planner.load_state_dict(model_state_dict)
        else:
            # 说明：在 RL 模式下，我们可能会在主进程把最新策略权重同步到 worker 的 planner，
            # 并把 _ckpt_path 置空以避免 worker initialize() 再次从旧 ckpt 覆盖。
            if getattr(self, "_weights_synced_from_main_process", False):
                pass
                # import logging
                # logging.getLogger(__name__).info("initialize: ckpt_path is None but weights were synced; skip ckpt load")
            else:
                print("load random model")
        
        self._planner.eval()
        self._planner = self._planner.to(self._device)
        self._initialization = initialization

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        data = NuPlanDataSample(
            batched=(model_inputs['ego_current_state'].dim() > 1),
            ego_past=model_inputs['ego_agent_past'],
            ego_current=model_inputs['ego_current_state'],
            neighbor_past=model_inputs['neighbor_agents_past'],
            lanes=model_inputs['lanes'],
            lanes_speedlimit=model_inputs['lanes_speed_limit'],
            lanes_has_speedlimit=model_inputs['lanes_has_speed_limit'],
            routes=model_inputs['route_lanes'],
            routes_speedlimit=model_inputs['route_lanes_speed_limit'],
            routes_has_speedlimit=model_inputs['route_lanes_has_speed_limit'],
            map_objects=model_inputs['static_objects']
        )

        return data

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    
        import logging
        logger = logging.getLogger(__name__)
        predictions = outputs[0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        
        # 检查 predictions 是否包含 NaN 或 Inf
        if np.any(~np.isfinite(predictions)):
            # nan_count = np.sum(~np.isfinite(predictions))
            # logger.error(f"outputs_to_trajectory: predictions contains {nan_count}/ {np.sum(np.isfinite(predictions))} NaN/Inf values!")
            # 将 NaN/Inf 替换为 0（临时处理，避免崩溃）
            predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 计算 heading
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        
        # 检查 heading 是否包含 NaN/Inf（可能由于零速度导致）
        if np.any(~np.isfinite(heading)):
            nan_heading_count = np.sum(~np.isfinite(heading))
            # 对于 NaN/Inf 的 heading，使用前一个有效值或默认值
            if len(ego_state_history) > 0:
                default_heading = ego_state_history[-1].rear_axle.heading
            else:
                default_heading = 0.0
            
            # 前向填充：用前一个有效值填充 NaN
            for i in range(len(heading)):
                if not np.isfinite(heading[i, 0]):
                    if i > 0 and np.isfinite(heading[i-1, 0]):
                        heading[i, 0] = heading[i-1, 0]
                    else:
                        heading[i, 0] = default_heading
        
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        # 检查是否是 RL 模式
        if self._rl_enabled:
            return self.compute_planner_trajectory_rl(current_input)
        
        inputs = self.planner_input_to_model_inputs(current_input)
        outputs = self.core.inference(self._planner, inputs, use_cfg=self.use_cfg, cfg_weight=self.cfg_weight)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    
    def compute_planner_trajectory_rl(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        RL 模式下的轨迹计算，收集 RL 数据

        说明：RL 模式下同样支持 CFG（classifier-free guidance）。
        """
        import logging
        logger = logging.getLogger(__name__)
        
        inputs = self.planner_input_to_model_inputs(current_input)

        # 在调用前检查数据收集器状态
        if self._rl_data_collector is not None:
            current_data_before = len(self._rl_data_collector.current_simulation_data)
            logger.debug(f"compute_planner_trajectory_rl: before inference, current_data_size={current_data_before}")
        else:
            logger.error("compute_planner_trajectory_rl: _rl_data_collector is None!")

        # RL 模式：传递 data_collector（同时支持 CFG）
        outputs = self.core.inference(
            self._planner, 
            inputs, 
            use_cfg=True,
            cfg_weight=self.cfg_weight,
            data_collector=self._rl_data_collector
        )

        # 在调用后检查数据收集器状态
        if self._rl_data_collector is not None:
            current_data_after = len(self._rl_data_collector.current_simulation_data)
            logger.debug(f"compute_planner_trajectory_rl: after inference, current_data_size={current_data_after}")
            if current_data_after == current_data_before:
                logger.warning(f"compute_planner_trajectory_rl: No data collected! (before={current_data_before}, after={current_data_after})")

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    
    def get_rl_data_collector(self):
        """获取 RL 数据收集器"""
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"get_rl_data_collector called: rl_enabled={self._rl_enabled}, collector={self._rl_data_collector}")
        if self._rl_data_collector is not None:
            batch_data = self._rl_data_collector.get_batch_data()
            current_data = len(self._rl_data_collector.current_simulation_data)
            logger.debug(f"  Current batch_data size: {len(batch_data)}, current_simulation_data size: {current_data}")
        return self._rl_data_collector
    
    def finalize_rl_simulation(self, score: float):
        """
        完成一个 RL simulation，添加 score 到数据收集器
        
        Args:
            score: float - 该 simulation 的最终 score
        """
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"finalize_rl_simulation called: rl_enabled={self._rl_enabled}, score={score}, collector={self._rl_data_collector}")
        if self._rl_enabled and self._rl_data_collector is not None:
            current_data_size = len(self._rl_data_collector.current_simulation_data)
            logger.debug(f"  Current simulation data points before finalize: {current_data_size}")
            self._rl_data_collector.finalize_simulation(score)
            batch_data_size = len(self._rl_data_collector.batch_data)
            logger.debug(f"  Batch data size after finalize: {batch_data_size}")
        else:
            logger.warning(f"  Cannot finalize: rl_enabled={self._rl_enabled}, collector={self._rl_data_collector}")
    