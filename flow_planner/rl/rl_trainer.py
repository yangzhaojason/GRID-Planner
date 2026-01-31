"""
RL 训练器：实现 GRPO 和 PPO-styled loss
"""
import torch
import torch.nn as nn
import numpy as np
import time
from typing import List, Dict, Any, Optional, Iterable, Tuple
from pathlib import Path
import logging
import random
from tqdm import tqdm
from flow_planner.rl.data_collector import RLDataCollector
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class RLTrainer:
    """RL 训练器"""
    
    def __init__(self, model, optimizer, rl_cfg, device='cuda'):
        """
        Args:
            model: FlowPlanner 模型
            optimizer: 优化器
            rl_cfg: RL 配置对象（从 cfg.planner.flow_planner.rl 传递）
            device: 设备
        """
        self.model = model
        # DDP 兼容：保存未包裹的底层模型，供访问属性（flow_ode/decoder等）
        self.base_model = self.model.module if hasattr(self.model, "module") else self.model
        # 确保模型在正确的设备上
        if hasattr(self.model, 'to'):
            self.model = self.model.to(device)
            # to() 之后重新绑定 base_model（DDP.module 的设备也会随之变化）
            self.base_model = self.model.module if hasattr(self.model, "module") else self.model
        self.optimizer = optimizer
        self.rl_cfg = rl_cfg
        self.device = device
        self.data_collector = RLDataCollector()
        
        # RL 配置
        self.micro_batch_size = rl_cfg.get('micro_batch_size', 256)
        # 梯度累积步数：每 N 个 micro-batch 做一次 optimizer.step()
        grad_acc = rl_cfg.get("gradient_accumulate_step", 1)
        if not isinstance(grad_acc, int):
            raise TypeError(f"rl_cfg.gradient_accumulate_step must be int, got {type(grad_acc)}")
        if grad_acc <= 0:
            raise ValueError(f"rl_cfg.gradient_accumulate_step must be > 0, got {grad_acc}")
        self.gradient_accumulate_step = int(grad_acc)
        self.clip_ratio = rl_cfg.get('clip_ratio', 0.01)
        # noise_std现在是可学习的，从FlowPlanner获取
        # 初始值从配置读取，但实际使用时会从模型获取可学习参数
        self.noise_std_initial = rl_cfg.get('noise_std', 0.05)
        # joint_logprob: 若为 True，则按“轨迹(多步去噪)”聚合 logprob 后再算 PPO ratio
        if hasattr(rl_cfg, "__contains__") and ("joint_logprob" in rl_cfg):
            joint_logprob = rl_cfg["joint_logprob"]
            if not isinstance(joint_logprob, bool):
                raise TypeError(f"rl_cfg.joint_logprob must be bool, got {type(joint_logprob)}")
            self.joint_logprob = joint_logprob
        else:
            self.joint_logprob = False
        # GRPO / PPO 稳定化：advantage clamp
        self.adv_clip = float(rl_cfg.get('adv_clip', 5.0))
        assert self.adv_clip > 0, f"adv_clip must be > 0, got {self.adv_clip}"
        # 梯度裁剪：防止梯度爆炸
        self.max_grad_norm = float(rl_cfg.get('max_grad_norm', 1.0))
        assert self.max_grad_norm > 0, f"max_grad_norm must be > 0, got {self.max_grad_norm}"
        # Entropy loss权重：鼓励探索，防止策略过早收敛
        self.entropy_coef = float(rl_cfg.get('entropy_coef', 1e-3))
        assert self.entropy_coef >= 0, f"entropy_coef must be >= 0, got {self.entropy_coef}"
        self.save_interval = rl_cfg.get('save_interval', 100)
        self.output_dir = Path(rl_cfg.get('output_dir')) if rl_cfg.get('output_dir') else None
        self.tb_writer: Optional[SummaryWriter] = None
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            tb_dir = self.output_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tb_writer = SummaryWriter(log_dir=str(tb_dir))
        
        self.step_count = 0

    def _load_rollout_file(self, path: str) -> List[List[Dict[str, Any]]]:
        """从单个 rollout 文件加载：返回 List[trajectory]，trajectory 是 List[datapoint_dict]。"""
        if not isinstance(path, str) or len(path) == 0:
            raise TypeError(f"rollout path must be non-empty str, got {type(path)}")
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Rollout file not found: {path}")
        data = torch.load(str(p), map_location="cpu")
        if not isinstance(data, list):
            raise TypeError(f"Rollout file content must be list, got {type(data)} from {path}")
        if len(data) == 0:
            return []
        for ti, traj in enumerate(data):
            if not isinstance(traj, list):
                raise TypeError(f"rollout[{ti}] must be list (trajectory), got {type(traj)} from {path}")
            for di, dp in enumerate(traj):
                if not isinstance(dp, dict):
                    raise TypeError(f"rollout[{ti}][{di}] must be dict, got {type(dp)} from {path}")
        return data

    def _get_dist_info(self) -> Tuple[bool, int, int]:
        import torch.distributed as dist

        dist_enabled = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_enabled else 0
        world_size = dist.get_world_size() if dist_enabled else 1
        if world_size < 1:
            raise ValueError(f"Invalid world_size={world_size}")
        return dist_enabled, rank, world_size

    def _compute_num_micro_batches_from_index(
        self,
        rollout_index: List[Dict[str, Any]],
        step_count: int,
        joint_logprob: bool,
        steps_per_trajectory: int,
    ) -> int:
        """按索引计数估算可用 micro-batch 数（允许跨文件累计），确保 DDP 一致。"""
        if joint_logprob:
            if self.micro_batch_size % steps_per_trajectory != 0:
                raise ValueError(
                    f"joint_logprob=True requires micro_batch_size divisible by denoising_steps_per_trajectory. "
                    f"Got micro_batch_size={self.micro_batch_size}, denoising_steps={steps_per_trajectory}"
                )
            trajectories_per_micro_batch = self.micro_batch_size // steps_per_trajectory
            if trajectories_per_micro_batch <= 0:
                raise ValueError("trajectories_per_micro_batch must be > 0")
            total_traj = 0
            for item in rollout_index:
                if "num_trajectories" not in item:
                    raise KeyError("rollout_index item missing 'num_trajectories'")
                n_traj = int(item["num_trajectories"])
                if n_traj < 0:
                    raise ValueError(f"Invalid num_trajectories={n_traj}")
                total_traj += n_traj
            return int(total_traj // trajectories_per_micro_batch)
        else:
            total_dp = 0
            for item in rollout_index:
                if "num_datapoints" not in item:
                    raise KeyError("rollout_index item missing 'num_datapoints'")
                n_dp = int(item["num_datapoints"])
                if n_dp < 0:
                    raise ValueError(f"Invalid num_datapoints={n_dp}")
                total_dp += n_dp
            return int(total_dp // int(self.micro_batch_size))

    def _iter_micro_batches_from_rollout_index(
        self,
        rollout_index: List[Dict[str, Any]],
        step_count: int,
        joint_logprob: bool,
        steps_per_trajectory: int,
        max_micro_batches: int,
    ) -> Iterable[List[Dict[str, Any]]]:
        """
        惰性迭代 micro-batches（每次只加载一个 rollout 文件到内存）。
        允许跨文件累计凑满一个 micro-batch，避免“文件太小导致全被 drop_last”。
        """
        rng = random.Random(100000 + int(step_count))
        items = list(rollout_index)
        rng.shuffle(items)

        yielded = 0
        if joint_logprob:
            trajectories_per_micro_batch = self.micro_batch_size // steps_per_trajectory
            traj_buffer: List[List[Dict[str, Any]]] = []
            for item in items:
                if yielded >= max_micro_batches:
                    break
                if "path" not in item:
                    raise KeyError("rollout_index item missing 'path'")
                rl_data = self._load_rollout_file(str(item["path"]))
                if len(rl_data) == 0:
                    continue
                rng.shuffle(rl_data)  # shuffle trajectory order within this file
                for traj in rl_data:
                    if len(traj) != steps_per_trajectory:
                        raise ValueError(
                            f"Trajectory steps mismatch: expected {steps_per_trajectory}, got {len(traj)}"
                        )
                    traj_buffer.append(traj)
                    while len(traj_buffer) >= trajectories_per_micro_batch and yielded < max_micro_batches:
                        traj_slice = traj_buffer[:trajectories_per_micro_batch]
                        traj_buffer = traj_buffer[trajectories_per_micro_batch:]
                        micro_batch: List[Dict[str, Any]] = []
                        for t in traj_slice:
                            micro_batch.extend(t)
                        if len(micro_batch) != self.micro_batch_size:
                            raise ValueError(
                                f"Internal error: joint micro_batch size {len(micro_batch)} != {self.micro_batch_size}"
                            )
                        yielded += 1
                        yield micro_batch
        else:
            dp_buffer: List[Dict[str, Any]] = []
            for item in items:
                if yielded >= max_micro_batches:
                    break
                if "path" not in item:
                    raise KeyError("rollout_index item missing 'path'")
                rl_data = self._load_rollout_file(str(item["path"]))
                if len(rl_data) == 0:
                    continue
                # 文件内先展平再 shuffle，避免一次性跨全部文件占用内存
                file_datapoints: List[Dict[str, Any]] = []
                for traj in rl_data:
                    file_datapoints.extend(traj)
                if len(file_datapoints) == 0:
                    continue
                rng.shuffle(file_datapoints)
                for dp in file_datapoints:
                    dp_buffer.append(dp)
                    while len(dp_buffer) >= int(self.micro_batch_size) and yielded < max_micro_batches:
                        micro_batch = dp_buffer[: int(self.micro_batch_size)]
                        dp_buffer = dp_buffer[int(self.micro_batch_size) :]
                        if len(micro_batch) != self.micro_batch_size:
                            raise ValueError(
                                f"Internal error: micro_batch size {len(micro_batch)} != {self.micro_batch_size}"
                            )
                        yielded += 1
                        yield micro_batch

    def train_step_from_rollout_index(
        self,
        rollout_index: List[Dict[str, Any]],
        scenario_score_means: List[float],
        scenario_score_stds: List[float],
        step_count: int,
    ) -> None:
        """
        基于落盘 rollout 文件索引进行训练，避免一次性加载超大 batch_data 到内存。

        约束：
        - 不跨文件拼接 micro-batch（每个文件内部 drop_last），这样可以在不读取文件内容的情况下用索引计数，
          并保证 DDP 每个 rank 的 backward 次数一致（取所有 rank 的 min）。
        """
        if not isinstance(rollout_index, list):
            raise TypeError(f"rollout_index must be list, got {type(rollout_index)}")
        if len(rollout_index) == 0:
            logger.warning("Empty rollout_index, skip train")
            return

        self.step_count = int(step_count)

        t_step0 = time.perf_counter()
        dist_enabled, rank, world_size = self._get_dist_info()

        # rank 分片：每个 rank 只读自己的文件，避免 N 倍重复 IO
        index_for_rank = [rollout_index[i] for i in range(rank, len(rollout_index), world_size)]
        if len(index_for_rank) == 0:
            logger.warning(f"Rank{rank}: no rollout files assigned, skip train")
            return

        # joint_logprob 下需要知道每条轨迹的 denoising steps（一般等于 sample_steps=4）
        # 为避免读取大量文件，这里只读取本 rank 第一个文件来推断 steps_per_trajectory
        steps_per_trajectory = None
        for it in index_for_rank:
            if not isinstance(it, dict):
                raise TypeError(f"rollout_index item must be dict, got {type(it)}")
            if "path" not in it:
                raise KeyError("rollout_index item missing 'path'")
            first_data = self._load_rollout_file(str(it["path"]))
            if len(first_data) == 0:
                continue
            if not isinstance(first_data[0], list) or len(first_data[0]) == 0:
                continue
            steps_per_trajectory = int(len(first_data[0]))
            break
        if steps_per_trajectory is None:
            logger.warning("All rollout files empty on this rank, skip train")
            return
        if steps_per_trajectory <= 0:
            raise ValueError(f"Invalid steps_per_trajectory={steps_per_trajectory}")

        # 计算每个 rank 可用 micro-batch 数，并在 DDP 下取 min，避免不同 rank backward 次数不一致导致死锁
        local_num_mb = self._compute_num_micro_batches_from_index(
            rollout_index=index_for_rank,
            step_count=step_count,
            joint_logprob=bool(self.joint_logprob),
            steps_per_trajectory=steps_per_trajectory,
        )
        if local_num_mb <= 0:
            logger.warning(f"Rank{rank}: no full micro-batches available, skip train")
            return

        num_micro_batches = int(local_num_mb)
        if dist_enabled and world_size > 1:
            import torch.distributed as dist

            t_mb = torch.tensor([num_micro_batches], device=self.device, dtype=torch.int64)
            dist.all_reduce(t_mb, op=dist.ReduceOp.MIN)
            num_micro_batches = int(t_mb.item())
        if num_micro_batches <= 0:
            logger.warning("After DDP MIN, no micro-batches left, skip train")
            return

        scenario_means_tensor = torch.tensor(scenario_score_means, device=self.device, dtype=torch.float32)
        scenario_stds_tensor = torch.tensor(scenario_score_stds, device=self.device, dtype=torch.float32)

        # timing accumulators
        t_collect_py = 0.0
        t_tensorize_and_prep = 0.0
        t_forward_and_logprob = 0.0
        t_advantage = 0.0
        t_loss = 0.0
        t_backward_opt = 0.0

        total_loss = 0.0
        all_stats: List[Dict[str, Any]] = []
        # grad accumulation buffers
        accum_stats: List[Dict[str, Any]] = []
        accum_k: Optional[int] = None

        use_tqdm = (not dist_enabled) or rank == 0
        mb_iter = tqdm(range(num_micro_batches)) if use_tqdm else range(num_micro_batches)

        micro_batches_iter = iter(
            self._iter_micro_batches_from_rollout_index(
            rollout_index=index_for_rank,
            step_count=step_count,
            joint_logprob=bool(self.joint_logprob),
            steps_per_trajectory=steps_per_trajectory,
            max_micro_batches=num_micro_batches,
            )
        )

        # 训练循环：每个 micro-batch 逻辑与原 train_step 保持一致（直接复用原来的字段约定）
        # 注意：num_micro_batches 已经在 DDP 下取了 MIN，因此各 rank 次数一致，accum 边界也一致。
        grad_acc = int(self.gradient_accumulate_step)
        for local_i in mb_iter:
            # 本次 accumulation 窗口大小：最后一段可能不足 grad_acc
            if local_i % grad_acc == 0:
                remaining = int(num_micro_batches - local_i)
                accum_k = min(grad_acc, remaining)
                self.model.zero_grad()
            try:
                micro_batch = next(micro_batches_iter)
            except StopIteration as e:
                raise RuntimeError(
                    f"Micro-batch iterator exhausted early: expected {num_micro_batches} micro-batches"
                ) from e
            # 构建模型输入
            t_collect0 = time.perf_counter()
            latents = []
            conditions = []
            timesteps_list = []
            x_next_list = []
            x_noised_list = []
            log_probs_old_list = []
            scores_list = []
            scenario_indices_list = []

            for data_point in micro_batch:
                latents.append(data_point["latent"])
                conditions.append(data_point["condition"])
                timesteps_list.append(data_point["timestep"])
                if "x_next" not in data_point:
                    raise KeyError(
                        "Missing key 'x_next' in collected data. "
                        "请先用最新代码重新跑 simulation 收集数据（需要保存每步无噪声 x_next），再进行训练。"
                    )
                x_next_list.append(data_point["x_next"])
                x_noised_list.append(data_point["x_noised"])
                log_probs_old_list.append(data_point["log_prob"])
                scores_list.append(data_point["score"])
                if "scenario_index" not in data_point:
                    raise KeyError(
                        "Missing key 'scenario_index' in collected data. "
                        "请确认 executor.run_simulation 落盘前已写入 datapoint['scenario_index']。"
                    )
                scenario_indices_list.append(int(data_point["scenario_index"]))
            t_collect_py += time.perf_counter() - t_collect0

            B_batch = len(micro_batch)
            if B_batch != int(self.micro_batch_size):
                raise ValueError(f"micro_batch size {B_batch} != micro_batch_size {self.micro_batch_size}")

            t_tensor0 = time.perf_counter()
            latents_tensor = torch.stack(
                [torch.tensor(l, device=self.device) if not isinstance(l, torch.Tensor) else l.to(self.device) for l in latents]
            )

            processed_timesteps = []
            for t in timesteps_list:
                if isinstance(t, torch.Tensor):
                    t = t.to(self.device)
                    if t.dim() > 0:
                        t = t.flatten()[0] if t.numel() > 0 else t.item()
                    if hasattr(t, "dim") and t.dim() == 0:
                        t = t.item()
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t, device=self.device, dtype=torch.float32)
                else:
                    t = t.to(self.device).float()
                if t.dim() > 0:
                    t = t.flatten()[0] if t.numel() > 0 else torch.tensor(0.0, device=self.device)
                processed_timesteps.append(t)
            timesteps_tensor = torch.stack(processed_timesteps)

            x_noised_tensor = torch.stack(
                [torch.tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device) for x in x_noised_list]
            )
            x_next_tensor = torch.stack(
                [torch.tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device) for x in x_next_list]
            )
            log_probs_old_tensor = torch.tensor(log_probs_old_list, device=self.device, dtype=torch.float32)
            scores_tensor = torch.tensor(scores_list, device=self.device, dtype=torch.float32)
            scenario_indices_tensor = torch.tensor(scenario_indices_list, device=self.device, dtype=torch.long)

            # -------- shape normalization (与原 train_step 对齐，但不吞异常) --------
            if latents_tensor.dim() < 4:
                raise ValueError(f"latents_tensor should be at least 4D, got shape={latents_tensor.shape}")
            if latents_tensor.dim() == 5 and latents_tensor.shape[1] == 1:
                latents_tensor = latents_tensor.squeeze(1)
            if x_noised_tensor.dim() == 5 and x_noised_tensor.shape[1] == 1:
                x_noised_tensor = x_noised_tensor.squeeze(1)
            if x_next_tensor.dim() == 5 and x_next_tensor.shape[1] == 1:
                x_next_tensor = x_next_tensor.squeeze(1)

            if latents_tensor.dim() != 4:
                raise ValueError(f"latents_tensor should be 4D after squeeze, got shape={latents_tensor.shape}")
            if x_noised_tensor.dim() != 4:
                raise ValueError(f"x_noised_tensor should be 4D after squeeze, got shape={x_noised_tensor.shape}")
            if x_next_tensor.dim() != 4:
                raise ValueError(f"x_next_tensor should be 4D after squeeze, got shape={x_next_tensor.shape}")

            B, P, T, D = latents_tensor.shape
            if P == 1 and T == 140 and D == 4:
                latents_tensor = latents_tensor.reshape(B, 7, 20, D)
            Bn, Pn, Tn, Dn = x_noised_tensor.shape
            if Pn == 1 and Tn == 140 and Dn == 4:
                x_noised_tensor = x_noised_tensor.reshape(Bn, 7, 20, Dn)
            Bx, Px, Tx, Dx = x_next_tensor.shape
            if Px == 1 and Tx == 140 and Dx == 4:
                x_next_tensor = x_next_tensor.reshape(Bx, 7, 20, Dx)

            # timesteps: ensure (B,)
            if timesteps_tensor.dim() == 2 and timesteps_tensor.shape[1] == 1:
                timesteps_tensor = timesteps_tensor.squeeze(1)
            if timesteps_tensor.dim() != 1:
                timesteps_tensor = timesteps_tensor.flatten()
            if timesteps_tensor.shape[0] != B_batch:
                if timesteps_tensor.shape[0] == 1:
                    timesteps_tensor = timesteps_tensor.expand(B_batch)
                else:
                    raise ValueError(
                        f"timesteps_tensor batch size {timesteps_tensor.shape[0]} != B_batch {B_batch}"
                    )

            latents_processed = latents_tensor
            x_noised_processed = x_noised_tensor
            x_next_saved_processed = x_next_tensor

            if x_noised_processed.shape != latents_processed.shape:
                raise ValueError(
                    f"x_noised shape {x_noised_processed.shape} != latents shape {latents_processed.shape}"
                )
            if x_next_saved_processed.shape != latents_processed.shape:
                raise ValueError(
                    f"x_next shape {x_next_saved_processed.shape} != latents shape {latents_processed.shape}"
                )

            t_tensorize_and_prep += time.perf_counter() - t_tensor0

            # 模型前向与 logprob 计算（与原逻辑一致）
            was_training = self.model.training
            self.model.eval()

            with torch.enable_grad():
                def restore_to_device(obj, device):
                    if isinstance(obj, torch.Tensor):
                        result = obj.to(device) if obj.device != device else obj
                        if result.requires_grad is False and result.dtype.is_floating_point:
                            result = result.requires_grad_(True)
                        return result
                    if isinstance(obj, (list, tuple)):
                        return type(obj)(restore_to_device(item, device) for item in obj)
                    if isinstance(obj, dict):
                        return {k: restore_to_device(v, device) for k, v in obj.items()}
                    return obj

                cond_extras_list = []
                uncond_extras_list = []
                for condition in conditions:
                    if "cond" not in condition or "uncond" not in condition:
                        raise KeyError("Missing 'cond'/'uncond' in collected condition. 请用开启 CFG 的 RL simulate 重新收集数据。")
                    if "decoder_model_extra" not in condition["cond"] or "decoder_model_extra" not in condition["uncond"]:
                        raise KeyError("Missing decoder_model_extra in collected condition.")
                    cond_extras_list.append(restore_to_device(condition["cond"]["decoder_model_extra"], self.device))
                    uncond_extras_list.append(restore_to_device(condition["uncond"]["decoder_model_extra"], self.device))

                flow_ode = self.base_model.flow_ode
                velocity_func = flow_ode.translation_funcs[(self.base_model._model_type, "velocity")]
                cfg_weight = float(flow_ode.cfg_weight)

                allowed_keys = {"encodings", "masks", "routes_cond", "token_dist", "cfg_flags"}

                def _ensure_tensor_batch_first(t: torch.Tensor, expect_dim: int) -> torch.Tensor:
                    if not isinstance(t, torch.Tensor):
                        raise TypeError(f"expected tensor, got {type(t)}")
                    t = t.to(self.device)
                    while t.dim() > expect_dim and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.dim() == expect_dim - 1:
                        t = t.unsqueeze(0)
                    if t.dim() != expect_dim or t.shape[0] != 1:
                        raise ValueError(f"tensor dim/batch mismatch: expect (1,...) dim={expect_dim}, got shape={t.shape}")
                    return t

                def _as_batch_1d_int(cfg_flag: torch.Tensor) -> torch.Tensor:
                    if not isinstance(cfg_flag, torch.Tensor):
                        raise TypeError(f"cfg_flags elem should be tensor, got {type(cfg_flag)}")
                    cfg_flag = cfg_flag.to(self.device)
                    if cfg_flag.dim() == 0:
                        cfg_flag = cfg_flag.unsqueeze(0)
                    elif cfg_flag.dim() > 1:
                        cfg_flag = cfg_flag.flatten()[:1]
                    if cfg_flag.shape != (1,):
                        raise ValueError(f"cfg_flags elem should be (1,), got {cfg_flag.shape}")
                    return cfg_flag.to(dtype=torch.int32)

                def _ensure_device(obj, device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(device) if obj.device != device else obj
                    if isinstance(obj, (list, tuple)):
                        return type(obj)(_ensure_device(item, device) for item in obj)
                    if isinstance(obj, dict):
                        return {k: _ensure_device(v, device) for k, v in obj.items()}
                    return obj

                def _build_merged_extra(extras_list, batch_size: int, extra: Dict[str, Any]) -> Dict[str, Any]:
                    base0 = extras_list[0]
                    if not isinstance(base0, dict):
                        raise TypeError(f"decoder_model_extra must be dict, got {type(base0)}")
                    if not allowed_keys.issubset(set(base0.keys())):
                        raise KeyError(f"decoder_model_extra missing keys: {allowed_keys - set(base0.keys())}")
                    merged_extra: Dict[str, Any] = {}

                    encodings_list = []
                    for de in extras_list[:batch_size]:
                        enc = de["encodings"]
                        if not isinstance(enc, (list, tuple)) or len(enc) != 2:
                            raise TypeError("encodings should be len=2 list/tuple")
                        enc0 = _ensure_tensor_batch_first(enc[0], expect_dim=3)
                        enc1 = _ensure_tensor_batch_first(enc[1], expect_dim=3)
                        encodings_list.append((enc0, enc1))
                    merged_extra["encodings"] = [
                        torch.cat([e0 for (e0, _) in encodings_list], dim=0),
                        torch.cat([e1 for (_, e1) in encodings_list], dim=0),
                    ]

                    masks_list = []
                    for de in extras_list[:batch_size]:
                        m = de["masks"]
                        if not isinstance(m, (list, tuple)) or len(m) != 2:
                            raise TypeError("masks should be len=2 list/tuple")
                        m0 = _ensure_tensor_batch_first(m[0], expect_dim=2)
                        m1 = _ensure_tensor_batch_first(m[1], expect_dim=2)
                        masks_list.append((m0, m1))
                    merged_extra["masks"] = [
                        torch.cat([m0 for (m0, _) in masks_list], dim=0),
                        torch.cat([m1 for (_, m1) in masks_list], dim=0),
                    ]

                    routes_list = [_ensure_tensor_batch_first(de["routes_cond"], expect_dim=2) for de in extras_list[:batch_size]]
                    merged_extra["routes_cond"] = torch.cat(routes_list, dim=0)

                    td_list = [_ensure_tensor_batch_first(de["token_dist"], expect_dim=3) for de in extras_list[:batch_size]]
                    merged_extra["token_dist"] = torch.cat(td_list, dim=0)

                    cfg_list = [_as_batch_1d_int(de["cfg_flags"]) for de in extras_list[:batch_size]]
                    merged_extra["cfg_flags"] = torch.cat(cfg_list, dim=0)

                    for k, v in extra.items():
                        if k in allowed_keys:
                            merged_extra[k] = v
                    merged_extra = _ensure_device(merged_extra, self.device)
                    return merged_extra

                def _make_model_fn(extras_list):
                    def _fn(x, t, **extra):
                        merged_extra = _build_merged_extra(extras_list, x.shape[0], extra)
                        return self.base_model.decoder(x, t, **merged_extra)
                    return _fn

                from flow_planner.model.flow_planner_model.flow_utils.velocity_model import VelocityModel

                velocity_model_cond = VelocityModel(_make_model_fn(cond_extras_list), flow_ode.path, velocity_func, use_cfg=False, cfg_weight=None)
                velocity_model_uncond = VelocityModel(_make_model_fn(uncond_extras_list), flow_ode.path, velocity_func, use_cfg=False, cfg_weight=None)

                if int(flow_ode.sample_params["sample_steps"]) != 4:
                    raise ValueError(f"flow_ode.sample_params['sample_steps'] must be 4, got {flow_ode.sample_params['sample_steps']}")
                if str(flow_ode.sample_params["sample_method"]) != "midpoint":
                    raise ValueError(f"sample_method should be midpoint, got {flow_ode.sample_params['sample_method']}")
                step_size = 1.0 / float(flow_ode.sample_params["sample_steps"])

                x0 = latents_processed.to(self.device)
                t0 = timesteps_tensor.to(self.device)

                def _combine(u_c: torch.Tensor, u_u: torch.Tensor) -> torch.Tensor:
                    return (1.0 - cfg_weight) * u_u + cfg_weight * u_c

                t_fwd0 = time.perf_counter()
                u1_c = velocity_model_cond(x0, t0)
                u1_u = velocity_model_uncond(x0, t0)
                u1 = _combine(u1_c, u1_u)
                t_mid = t0 + step_size / 2
                x_mid = x0 + step_size / 2 * u1
                u2_c = velocity_model_cond(x_mid, t_mid)
                u2_u = velocity_model_uncond(x_mid, t_mid)
                u2 = _combine(u2_c, u2_u)
                x_next_new = x0 + step_size * u2

                flow_noise_std = flow_ode.rl_noise_std
                if isinstance(flow_noise_std, torch.Tensor):
                    flow_noise_std = flow_noise_std.to(self.device, dtype=torch.float32)
                else:
                    flow_noise_std = torch.tensor(float(flow_noise_std), device=self.device, dtype=torch.float32)

                if x_next_new.shape != x_noised_processed.shape:
                    raise ValueError(
                        f"x_next_new shape {x_next_new.shape} != x_noised_processed shape {x_noised_processed.shape}"
                    )
                log_probs_new_tensor = self.compute_log_prob_new(
                    x_noised_processed.to(self.device),
                    x_next_new,
                    flow_noise_std * (1 - timesteps_tensor.to(self.device)) ** 2,
                )
                t_forward_and_logprob += time.perf_counter() - t_fwd0

            if bool(self.joint_logprob):
                if self.micro_batch_size % steps_per_trajectory != 0:
                    raise ValueError("joint_logprob requires micro_batch_size divisible by denoising steps")
                trajectories_per_micro_batch = self.micro_batch_size // steps_per_trajectory
                logp_old_2d = log_probs_old_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                logp_new_2d = log_probs_new_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                logp_old_joint = logp_old_2d.mean(dim=1)
                logp_new_joint = logp_new_2d.mean(dim=1)

                scores_2d = scores_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                scores_traj = scores_2d[:, 0]
                max_abs_score_diff = (scores_2d - scores_traj.unsqueeze(1)).abs().max().item()
                if max_abs_score_diff > 1e-6:
                    raise ValueError("joint_logprob requires per-trajectory score identical across denoising steps")

                sidx_2d = scenario_indices_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                sidx_traj = sidx_2d[:, 0]
                max_sidx_diff = (sidx_2d - sidx_traj.unsqueeze(1)).abs().max().item()
                if max_sidx_diff != 0:
                    raise ValueError("joint_logprob requires per-trajectory scenario_index identical across steps.")

                t_adv0 = time.perf_counter()
                standardized_scores = (scores_traj - scenario_means_tensor[sidx_traj]) / scenario_stds_tensor[sidx_traj]
                advantages = torch.clamp(standardized_scores, -self.adv_clip, self.adv_clip)
                t_advantage += time.perf_counter() - t_adv0

                t_loss0 = time.perf_counter()
                loss, stats = self.compute_ppo_loss(logp_old_joint, logp_new_joint, advantages)
                t_loss += time.perf_counter() - t_loss0
            else:
                t_adv0 = time.perf_counter()
                advantages = self.compute_advantage(
                    scores_per_datapoint=scores_tensor,
                    scenario_indices=scenario_indices_tensor,
                    scenario_score_means=scenario_means_tensor,
                    scenario_score_stds=scenario_stds_tensor,
                    log_probs_old=log_probs_old_tensor,
                    log_probs_new=log_probs_new_tensor,
                )
                t_advantage += time.perf_counter() - t_adv0

                t_loss0 = time.perf_counter()
                loss, stats = self.compute_ppo_loss(log_probs_old_tensor, log_probs_new_tensor, advantages)
                t_loss += time.perf_counter() - t_loss0

            # 梯度累积：为保持与“每步 step”一致的尺度，对 loss 做平均归一化
            if accum_k is None or accum_k <= 0:
                raise RuntimeError("Internal error: accum_k is not set")
            loss_for_backward = loss / float(accum_k)

            last_in_accum = ((local_i % grad_acc) == (accum_k - 1)) or (local_i == int(num_micro_batches) - 1)
            t_bo0 = time.perf_counter()
            if dist_enabled and hasattr(self.model, "no_sync") and (not last_in_accum):
                with self.model.no_sync():  # type: ignore[attr-defined]
                    loss_for_backward.backward()
            else:
                loss_for_backward.backward()

            # 先缓存 stats，等真正 step 时再统一写入 grad_norm（避免非 step micro-batch 统计失真）
            total_loss += float(loss.item())
            accum_stats.append(stats)

            if last_in_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                t_backward_opt += time.perf_counter() - t_bo0
                g = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                for s in accum_stats:
                    s["grad_norm"] = g
                all_stats.extend(accum_stats)
                accum_stats = []
                # 下一个 accumulation 窗口会在循环顶部 zero_grad()

                # 恢复模型模式（避免影响后续训练逻辑）
                if was_training:
                    self.model.train()
            else:
                t_backward_opt += time.perf_counter() - t_bo0

        local_mb = len(all_stats)
        if local_mb == 0:
            logger.warning("No micro-batches were processed")
            return

        # DDP 聚合统计（与原 train_step 对齐）
        sum_ratio_mean = float(np.sum([s["ratio_mean"] for s in all_stats]))
        sum_ratio_max = float(np.sum([s["ratio_max"] for s in all_stats]))
        sum_ratio_min = float(np.sum([s["ratio_min"] for s in all_stats]))
        sum_clipped_ratio_mean = float(np.sum([s["clipped_ratio_mean"] for s in all_stats]))
        sum_clipped_ratio_max = float(np.sum([s["clipped_ratio_max"] for s in all_stats]))
        sum_clipped_ratio_min = float(np.sum([s["clipped_ratio_min"] for s in all_stats]))
        sum_clip_fraction = float(np.sum([s["clip_fraction"] for s in all_stats]))
        sum_grad_norm = float(np.sum([s.get("grad_norm", 0.0) for s in all_stats]))
        sum_policy_loss = float(np.sum([s.get("policy_loss", 0.0) for s in all_stats]))
        sum_entropy = float(np.sum([s.get("entropy", 0.0) for s in all_stats]))
        sum_entropy_bonus = float(np.sum([s.get("entropy_bonus", 0.0) for s in all_stats]))
        sum_loss = float(total_loss)

        total_mb = local_mb
        if dist_enabled and world_size > 1:
            import torch.distributed as dist

            agg = torch.tensor(
                [
                    sum_loss,
                    sum_ratio_mean,
                    sum_ratio_max,
                    sum_ratio_min,
                    sum_clipped_ratio_mean,
                    sum_clipped_ratio_max,
                    sum_clipped_ratio_min,
                    sum_clip_fraction,
                    sum_grad_norm,
                    sum_policy_loss,
                    sum_entropy,
                    sum_entropy_bonus,
                    float(local_mb),
                ],
                device=self.device,
                dtype=torch.float32,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            sum_loss = float(agg[0].item())
            sum_ratio_mean = float(agg[1].item())
            sum_ratio_max = float(agg[2].item())
            sum_ratio_min = float(agg[3].item())
            sum_clipped_ratio_mean = float(agg[4].item())
            sum_clipped_ratio_max = float(agg[5].item())
            sum_clipped_ratio_min = float(agg[6].item())
            sum_clip_fraction = float(agg[7].item())
            sum_grad_norm = float(agg[8].item())
            sum_policy_loss = float(agg[9].item())
            sum_entropy = float(agg[10].item())
            sum_entropy_bonus = float(agg[11].item())
            total_mb = int(agg[12].item())

        avg_loss = sum_loss / float(total_mb)
        avg_ratio_mean = sum_ratio_mean / float(total_mb)
        avg_ratio_max = sum_ratio_max / float(total_mb)
        avg_ratio_min = sum_ratio_min / float(total_mb)
        avg_clipped_ratio_mean = sum_clipped_ratio_mean / float(total_mb)
        avg_clipped_ratio_max = sum_clipped_ratio_max / float(total_mb)
        avg_clipped_ratio_min = sum_clipped_ratio_min / float(total_mb)
        avg_clip_fraction = sum_clip_fraction / float(total_mb)
        avg_grad_norm = sum_grad_norm / float(total_mb)
        avg_policy_loss = sum_policy_loss / float(total_mb)
        avg_entropy = sum_entropy / float(total_mb)
        avg_entropy_bonus = sum_entropy_bonus / float(total_mb)
        avg_score = float(np.mean(np.array(scenario_score_means, dtype=np.float32)))
        success_rate = float(np.mean([1.0 if s > 0 else 0.0 for s in scenario_score_means]))

        # TensorBoard：只在 rank0 写
        if self.tb_writer is not None:
            if (not dist_enabled) or rank == 0:
                self.tb_writer.add_scalar("rl/avg_loss", float(avg_loss), step_count)
                self.tb_writer.add_scalar("rl/avg_score", float(avg_score), step_count)
                self.tb_writer.add_scalar("rl/success_rate", float(success_rate), step_count)
                self.tb_writer.add_scalar("rl/ratio_mean", float(avg_ratio_mean), step_count)
                self.tb_writer.add_scalar("rl/ratio_max", float(avg_ratio_max), step_count)
                self.tb_writer.add_scalar("rl/ratio_min", float(avg_ratio_min), step_count)
                self.tb_writer.add_scalar("rl/clipped_ratio_mean", float(avg_clipped_ratio_mean), step_count)
                self.tb_writer.add_scalar("rl/clipped_ratio_max", float(avg_clipped_ratio_max), step_count)
                self.tb_writer.add_scalar("rl/clipped_ratio_min", float(avg_clipped_ratio_min), step_count)
                self.tb_writer.add_scalar("rl/clip_fraction", float(avg_clip_fraction), step_count)
                self.tb_writer.add_scalar("rl/grad_norm", float(avg_grad_norm), step_count)
                self.tb_writer.add_scalar("rl/policy_loss", float(avg_policy_loss), step_count)
                self.tb_writer.add_scalar("rl/entropy", float(avg_entropy), step_count)
                self.tb_writer.add_scalar("rl/entropy_bonus", float(avg_entropy_bonus), step_count)
                if hasattr(self.base_model, "flow_ode") and hasattr(self.base_model.flow_ode, "rl_noise_std"):
                    ns = self.base_model.flow_ode.rl_noise_std
                    current_noise_std = float(ns.detach().cpu().item()) if isinstance(ns, torch.Tensor) else float(ns)
                    self.tb_writer.add_scalar("rl/noise_std", current_noise_std, step_count)
                self.tb_writer.flush()

        current_noise_std_str = ""
        if hasattr(self.base_model, "flow_ode") and hasattr(self.base_model.flow_ode, "rl_noise_std"):
            ns = self.base_model.flow_ode.rl_noise_std
            current_noise_std = float(ns.detach().cpu().item()) if isinstance(ns, torch.Tensor) else float(ns)
            current_noise_std_str = f", NoiseStd={current_noise_std:.6f}"

        logger.info(
            f"Step {step_count}: "
            f"Loss={avg_loss:.4f} (Policy={avg_policy_loss:.4f}, EntropyBonus={avg_entropy_bonus:.4f}), "
            f"Score={avg_score:.4f}, "
            f"RatioMean={avg_ratio_mean:.4f}, "
            f"RatioMax={avg_ratio_max:.4f}, "
            f"RatioMin={avg_ratio_min:.4f}, "
            f"ClipFraction={avg_clip_fraction:.2%}, "
            f"GradNorm={avg_grad_norm:.4f}, "
            f"Entropy={avg_entropy:.4f}"
            f"{current_noise_std_str}"
        )

        # 保存检查点（与原逻辑一致）
        if step_count % self.save_interval == 0:
            self.save_checkpoint(step_count)
        
    def compute_advantage(
        self,
        scores_per_datapoint: torch.Tensor,
        scenario_indices: torch.Tensor,
        scenario_score_means: torch.Tensor,
        scenario_score_stds: torch.Tensor,
        log_probs_old: torch.Tensor,
        log_probs_new: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算 GRPO advantage
        
        Args:
            scores_per_datapoint: torch.Tensor - 每个数据点对应的 score（已经广播）
            scenario_indices: torch.Tensor - 每个数据点所属的 scenario 索引（0..S-1）
            scenario_score_means: torch.Tensor - (S,) 每个 scenario 的 score_mean（由同一 scenario 的多次 rollout 聚合得到）
            scenario_score_stds: torch.Tensor - (S,) 每个 scenario 的 score_std（由同一 scenario 的多次 rollout 聚合得到）
            log_probs_old: torch.Tensor - 旧策略的 log_prob
            log_probs_new: torch.Tensor - 新策略的 log_prob
            
        Returns:
            advantages: torch.Tensor - advantage 值
        """
        # 断言：检查输入形状
        assert scores_per_datapoint.dim() == 1, f"scores_per_datapoint should be 1D (B,), got {scores_per_datapoint.dim()}D, shape={scores_per_datapoint.shape}"
        assert log_probs_old.dim() == 1, f"log_probs_old should be 1D (B,), got {log_probs_old.dim()}D, shape={log_probs_old.shape}"
        assert log_probs_new.dim() == 1, f"log_probs_new should be 1D (B,), got {log_probs_new.dim()}D, shape={log_probs_new.shape}"
        assert scores_per_datapoint.shape[0] == log_probs_old.shape[0], f"scores_per_datapoint batch size {scores_per_datapoint.shape[0]} != log_probs_old batch size {log_probs_old.shape[0]}"
        assert log_probs_old.shape[0] == log_probs_new.shape[0], f"log_probs_old batch size {log_probs_old.shape[0]} != log_probs_new batch size {log_probs_new.shape[0]}"
        
        # 断言：检查输入形状
        assert scores_per_datapoint.dim() == 1, f"scores_per_datapoint should be 1D (B,), got {scores_per_datapoint.dim()}D, shape={scores_per_datapoint.shape}"
        assert log_probs_old.dim() == 1, f"log_probs_old should be 1D (B,), got {log_probs_old.dim()}D, shape={log_probs_old.shape}"
        assert log_probs_new.dim() == 1, f"log_probs_new should be 1D (B,), got {log_probs_new.dim()}D, shape={log_probs_new.shape}"
        assert scores_per_datapoint.shape[0] == log_probs_old.shape[0], f"scores_per_datapoint batch size {scores_per_datapoint.shape[0]} != log_probs_old batch size {log_probs_old.shape[0]}"
        assert log_probs_old.shape[0] == log_probs_new.shape[0], f"log_probs_old batch size {log_probs_old.shape[0]} != log_probs_new batch size {log_probs_new.shape[0]}"
        
        assert scenario_indices.dim() == 1, f"scenario_indices should be 1D, got {scenario_indices.dim()}D, shape={scenario_indices.shape}"
        assert scenario_indices.shape[0] == scores_per_datapoint.shape[0], (
            f"scenario_indices batch {scenario_indices.shape[0]} != scores_per_datapoint batch {scores_per_datapoint.shape[0]}"
        )
        assert scenario_score_means.dim() == 1, f"scenario_score_means should be 1D, got {scenario_score_means.dim()}D"
        assert scenario_score_stds.dim() == 1, f"scenario_score_stds should be 1D, got {scenario_score_stds.dim()}D"
        assert scenario_score_means.shape == scenario_score_stds.shape, (
            f"scenario_score_means shape {scenario_score_means.shape} != scenario_score_stds shape {scenario_score_stds.shape}"
        )
        assert scenario_score_means.numel() > 0, "scenario_score_means is empty"
        assert int(scenario_indices.min().item()) >= 0, f"scenario_indices has negative values: min={scenario_indices.min().item()}"
        assert int(scenario_indices.max().item()) < scenario_score_means.numel(), (
            f"scenario_indices out of range: max={scenario_indices.max().item()} vs num_scenarios={scenario_score_means.numel()}"
        )
        # 标准化每个 datapoint 的 score：用其所属 scenario 的 mean/std
        standardized_scores = (scores_per_datapoint - scenario_score_means[scenario_indices]) / scenario_score_stds[scenario_indices]
        
        # 计算 advantage = standardized_score - (log_prob_new - log_prob_old)
        advantages = standardized_scores  # - (log_probs_new - log_probs_old)

        # Advantage clamp（避免极小 std 导致梯度爆炸/长期漂移）
        advantages = torch.clamp(advantages, -self.adv_clip, self.adv_clip)
        
        # 断言：检查输出形状
        assert advantages.shape == log_probs_old.shape, f"advantages shape {advantages.shape} != log_probs_old shape {log_probs_old.shape}"
        
        return advantages
    
    def compute_ppo_loss(
        self, 
        log_probs_old: torch.Tensor, 
        log_probs_new: torch.Tensor, 
        advantages: torch.Tensor
    ) -> tuple:
        """
        计算 PPO-styled loss
        
        Args:
            log_probs_old: torch.Tensor
            log_probs_new: torch.Tensor
            advantages: torch.Tensor
            
        Returns:
            loss: torch.Tensor
            stats: Dict - 统计信息
        """
        # 断言：检查输入形状
        assert log_probs_old.dim() == 1, f"log_probs_old should be 1D (B,), got {log_probs_old.dim()}D, shape={log_probs_old.shape}"
        assert log_probs_new.dim() == 1, f"log_probs_new should be 1D (B,), got {log_probs_new.dim()}D, shape={log_probs_new.shape}"
        assert advantages.dim() == 1, f"advantages should be 1D (B,), got {advantages.dim()}D, shape={advantages.shape}"
        assert log_probs_old.shape[0] == log_probs_new.shape[0], f"log_probs_old batch size {log_probs_old.shape[0]} != log_probs_new batch size {log_probs_new.shape[0]}"
        assert log_probs_old.shape[0] == advantages.shape[0], f"log_probs_old batch size {log_probs_old.shape[0]} != advantages batch size {advantages.shape[0]}"
        
        # 断言：检查输入形状
        assert log_probs_old.dim() == 1, f"log_probs_old should be 1D (B,), got {log_probs_old.dim()}D, shape={log_probs_old.shape}"
        assert log_probs_new.dim() == 1, f"log_probs_new should be 1D (B,), got {log_probs_new.dim()}D, shape={log_probs_new.shape}"
        assert advantages.dim() == 1, f"advantages should be 1D (B,), got {advantages.dim()}D, shape={advantages.shape}"
        assert log_probs_old.shape[0] == log_probs_new.shape[0], f"log_probs_old batch size {log_probs_old.shape[0]} != log_probs_new batch size {log_probs_new.shape[0]}"
        assert log_probs_old.shape[0] == advantages.shape[0], f"log_probs_old batch size {log_probs_old.shape[0]} != advantages batch size {advantages.shape[0]}"
        
        # 计算 ratio = exp(log_prob_new - log_prob_old)
        ratio = torch.exp(log_probs_new - log_probs_old)
        # 断言：ratio 的形状应该与 log_probs_old 一致
        assert ratio.shape == log_probs_old.shape, f"ratio shape {ratio.shape} != log_probs_old shape {log_probs_old.shape}"
        
        # 计算 clipped_ratio = clip(ratio, 1-clip_ratio, 1+clip_ratio)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        
        # loss = -min(ratio * advantage, clipped_ratio * advantage)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        
        # Entropy loss: 鼓励探索，防止策略过早收敛
        # entropy = -log_prob的均值（对于连续分布）
        # 我们想要最大化entropy（鼓励探索），所以在loss中减去entropy项
        entropy = -log_probs_new.mean()  # entropy = -log_prob的均值
        entropy_bonus = self.entropy_coef * entropy  # entropy bonus（我们要最大化这个）
        
        # 总loss = policy_loss - entropy_bonus（减去因为我们要最大化entropy）
        loss = policy_loss - entropy_bonus
        
        # 断言：loss 应该是标量
        assert loss.dim() == 0, f"loss should be scalar (0D), got {loss.dim()}D, shape={loss.shape}"
        
        # 统计信息
        ratio_mean = ratio.mean().item()
        ratio_max = ratio.max().item()
        ratio_min = ratio.min().item()
        clipped_ratio_mean = clipped_ratio.mean().item()
        clipped_ratio_max = clipped_ratio.max().item()
        clipped_ratio_min = clipped_ratio.min().item()
        clip_fraction = ((ratio < 1 - self.clip_ratio) | (ratio > 1 + self.clip_ratio)).float().mean().item()
        entropy_value = entropy.item()
        entropy_bonus_value = entropy_bonus.item()
        policy_loss_value = policy_loss.item()
        
        stats = {
            'ratio_mean': ratio_mean,
            'ratio_max': ratio_max,
            'ratio_min': ratio_min,
            'clipped_ratio_mean': clipped_ratio_mean,
            'clipped_ratio_max': clipped_ratio_max,
            'clipped_ratio_min': clipped_ratio_min,
            'clip_fraction': clip_fraction,
            'policy_loss': policy_loss_value,
            'entropy': entropy_value,
            'entropy_bonus': entropy_bonus_value,
            'loss': loss.item()
        }
        
        return loss, stats
    
    def compute_log_prob_new(self, x_noised: torch.Tensor, x_next_new: torch.Tensor, noise_std: float) -> torch.Tensor:
        """
        训练阶段计算 log_prob_new
        
        Args:
            x_noised: torch.Tensor - 从保存的数据中获取（simulation 阶段加噪后的结果）
            x_next_new: torch.Tensor - 使用新模型和 ODE solver 计算（不加噪）
            noise_std: float - 噪声标准差
            
        Returns:
            log_prob_new: torch.Tensor - (B,) 形状，每个 batch 元素的 log_prob
        """
        # 断言：检查输入形状
        assert x_noised.dim() == 4, f"x_noised should be 4D (B, P, T, D), got {x_noised.dim()}D, shape={x_noised.shape}"
        assert x_next_new.dim() == 4, f"x_next_new should be 4D (B, P, T, D), got {x_next_new.dim()}D, shape={x_next_new.shape}"
        assert x_noised.shape == x_next_new.shape, f"x_noised shape {x_noised.shape} != x_next_new shape {x_next_new.shape}"
        
        diff = x_noised - x_next_new
        
        log_prob = -0.5 * (diff ** 2).mean(dim=(-3, -2, -1)) / (noise_std ** 2)
        log_prob = log_prob - 0.5 * np.log(2 * torch.pi) - torch.log(noise_std)
        
        # 断言：检查输出形状（应该是 (B,)）
        assert log_prob.dim() == 1, f"log_prob should be 1D (B,), got {log_prob.dim()}D, shape={log_prob.shape}"
        assert log_prob.shape[0] == x_noised.shape[0], f"log_prob batch size {log_prob.shape[0]} != x_noised batch size {x_noised.shape[0]}"
        
        return log_prob
    
    def flatten_batch_data(self, batch_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        展平 batch 数据：list(T, C) -> (N,)
        
        Args:
            batch_data: List[List[Dict]] - list(T, C) 格式的数据
            
        Returns:
            flattened_data: List[Dict] - 展平后的数据列表
        """
        flattened = []
        for simulation_data in batch_data:
            flattened.extend(simulation_data)
        return flattened
    
    def train_step(
        self,
        batch_data: List[List[Dict[str, Any]]],
        scenario_score_means: List[float],
        scenario_score_stds: List[float],
        step_count: int,
        already_sharded_across_ranks: bool = False,
    ):
        """
        执行一个训练 step
        
        Args:
            batch_data: List[List[Dict]] - list(T, C) 格式的数据
            scenario_score_means: List[float] - (S,) 每个 scenario 的 mean(score)，由多次 rollout 聚合得到
            scenario_score_stds: List[float] - (S,) 每个 scenario 的 std(score)，由多次 rollout 聚合得到
            step_count: int - 当前 step 计数
        """
        self.step_count = step_count

        t_step0 = time.perf_counter()

        # DDP: 获取 rank/world_size，用于数据分片与日志聚合
        import torch.distributed as dist
        dist_enabled = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if dist_enabled else 0
        world_size = dist.get_world_size() if dist_enabled else 1
        assert world_size >= 1
        
        # joint_logprob=True: 以“轨迹(多步去噪)”为一个样本，保持每条轨迹内顺序，不打散。
        steps_per_trajectory = None
        trajectories_per_micro_batch = None

        # 1. 展平数据：list(T, C) -> (N,)（joint_logprob 下仅在轨迹级 shuffle 后再 flatten）
        t_flatten0 = time.perf_counter()
        # DDP 需要所有 rank 的 shuffle 完全一致，否则会出现梯度同步错位/死锁
        if dist_enabled:
            random.seed(100000 + int(step_count))
        if self.joint_logprob:
            trajectories = list(batch_data)
            if len(trajectories) == 0:
                logger.warning("No data to train on")
                return
            if not isinstance(trajectories[0], list):
                raise TypeError(f"batch_data must be List[List[Dict]] but got element type {type(trajectories[0])}")
            steps_per_trajectory = len(trajectories[0])
            if steps_per_trajectory == 0:
                logger.warning("Empty trajectory (0 denoising steps) in batch_data")
                return
            for ti, traj in enumerate(trajectories):
                if not isinstance(traj, list):
                    raise TypeError(f"batch_data[{ti}] must be list, got {type(traj)}")
                if len(traj) != steps_per_trajectory:
                    raise ValueError(
                        f"All trajectories must have the same number of denoising steps. "
                        f"Got batch_data[0] len={steps_per_trajectory} but batch_data[{ti}] len={len(traj)}"
                    )
            t_shuffle0 = time.perf_counter()
            random.shuffle(trajectories)  # 只打乱轨迹顺序，不打乱轨迹内部 steps
            t_shuffle = time.perf_counter() - t_shuffle0

            flattened_data = []
            for traj in trajectories:
                flattened_data.extend(traj)
        else:
            flattened_data = self.flatten_batch_data(batch_data)
            t_shuffle0 = time.perf_counter()
            random.shuffle(flattened_data)
            t_shuffle = time.perf_counter() - t_shuffle0

        t_flatten = time.perf_counter() - t_flatten0
        
        if len(flattened_data) == 0:
            logger.warning("No data to train on")
            return
        
        # 3. 按 micro_batch_size 划分，末端舍弃
        if self.joint_logprob:
            assert steps_per_trajectory is not None
            if self.micro_batch_size % steps_per_trajectory != 0:
                raise ValueError(
                    f"joint_logprob=True requires micro_batch_size divisible by denoising_steps_per_trajectory. "
                    f"Got micro_batch_size={self.micro_batch_size}, denoising_steps={steps_per_trajectory}"
                )
            trajectories_per_micro_batch = self.micro_batch_size // steps_per_trajectory
            assert trajectories_per_micro_batch > 0, "trajectories_per_micro_batch must be > 0"

        num_micro_batches_total = len(flattened_data) // self.micro_batch_size
        if num_micro_batches_total == 0:
            logger.warning(f"Data size {len(flattened_data)} is smaller than micro_batch_size {self.micro_batch_size}")
            return

        # DDP：让每个 rank 处理互不重叠的 micro-batch，并保证每个 rank 处理的步数相同（drop_last）
        if dist_enabled and world_size > 1 and (not already_sharded_across_ranks):
            num_micro_batches = (num_micro_batches_total // world_size) * world_size
            if num_micro_batches == 0:
                logger.warning(
                    f"After DDP drop_last, no micro-batches left: total={num_micro_batches_total}, world_size={world_size}"
                )
                return
            micro_batch_indices = range(rank, num_micro_batches, world_size)
        else:
            num_micro_batches = num_micro_batches_total
            micro_batch_indices = range(num_micro_batches)
        
        # 4. 对于每个 micro batch
        total_loss = 0.0
        all_stats: List[Dict[str, Any]] = []
        accum_stats: List[Dict[str, Any]] = []
        accum_k: Optional[int] = None
        scenario_means_tensor = torch.tensor(scenario_score_means, device=self.device, dtype=torch.float32)
        scenario_stds_tensor = torch.tensor(scenario_score_stds, device=self.device, dtype=torch.float32)

        # timing accumulators
        t_collect_py = 0.0
        t_tensorize_and_prep = 0.0
        t_forward_and_logprob = 0.0
        t_advantage = 0.0
        t_loss = 0.0
        t_backward_opt = 0.0

        # tqdm：仅 rank0 显示，避免多进程刷屏
        use_tqdm = (not dist_enabled) or rank == 0
        mb_iter = tqdm(micro_batch_indices) if use_tqdm else micro_batch_indices

        grad_acc = int(self.gradient_accumulate_step)
        # local_i 用于梯度累积窗口计数（不要用 micro_batch_idx，因为 DDP 下它是 stride 的）
        for local_i, micro_batch_idx in enumerate(mb_iter):
            if local_i % grad_acc == 0:
                # 最后一个窗口可能不足 grad_acc
                # 由于 micro_batch_indices 可能是 stride range，这里用本 rank 的 micro-batch 总数来算 remaining
                # 计算本 rank 实际要处理的 micro-batch 数量
                if dist_enabled and world_size > 1 and (not already_sharded_across_ranks):
                    local_total = (num_micro_batches - rank + world_size - 1) // world_size
                else:
                    local_total = num_micro_batches
                remaining = int(local_total - local_i)
                accum_k = min(grad_acc, remaining)
                self.model.zero_grad()
            start_idx = micro_batch_idx * self.micro_batch_size
            end_idx = start_idx + self.micro_batch_size
            micro_batch = flattened_data[start_idx:end_idx]
            
            # 构建模型输入
            # 从保存的数据中恢复：latent, condition, timesteps, x_noised, log_prob_old, score
            t_collect0 = time.perf_counter()
            latents = []
            conditions = []
            timesteps_list = []
            x_next_list = []
            x_noised_list = []
            log_probs_old_list = []
            scores_list = []
            scenario_indices_list = []
            
            for data_point in micro_batch:
                latents.append(data_point['latent'])
                conditions.append(data_point['condition'])
                timesteps_list.append(data_point['timestep'])
                if 'x_next' not in data_point:
                    raise KeyError(
                        "Missing key 'x_next' in collected data. "
                        "请先用最新代码重新跑 simulation 收集数据（需要保存每步无噪声 x_next），再进行训练。"
                    )
                x_next_list.append(data_point['x_next'])
                x_noised_list.append(data_point['x_noised'])
                log_probs_old_list.append(data_point['log_prob'])
                scores_list.append(data_point['score'])
                if 'scenario_index' not in data_point:
                    raise KeyError(
                        "Missing key 'scenario_index' in collected data. "
                        "请在 RL rollout 汇总 report.rl_data 时为每个 datapoint 写入 scenario_index（同一 batch 内 0..batch_size-1）。"
                    )
                scenario_indices_list.append(int(data_point['scenario_index']))
            t_collect_py += time.perf_counter() - t_collect0
            
            # 转换为 tensor（需要处理 batch 维度）
            # 注意：不同数据点的 shape 可能不同，需要 padding 或单独处理
            # 这里假设所有数据点的 shape 一致
            # 约定：训练阶段 batch 维度 B == micro_batch 大小
            B_batch = len(micro_batch)
            assert B_batch > 0, "micro_batch is empty"
            try:
                t_tensor0 = time.perf_counter()
                # 批量处理 latents, timesteps, x_noised
                latents_tensor = torch.stack([torch.tensor(l, device=self.device) if not isinstance(l, torch.Tensor) else l.to(self.device) for l in latents])
                assert latents_tensor.shape[0] == B_batch, (
                    f"latents_tensor batch size {latents_tensor.shape[0]} != B_batch(len(micro_batch)) {B_batch}"
                )
                
                # 处理 timesteps：确保每个 timestep 是标量
                # 每个数据点的 timestep 应该是一个标量值（或 (1,) 形状的 tensor）
                processed_timesteps = []
                for t in timesteps_list:
                    if isinstance(t, torch.Tensor):
                        t = t.to(self.device)
                        # 确保是标量
                        if t.dim() > 0:
                            # 如果是多维 tensor，取第一个元素
                            t = t.flatten()[0] if t.numel() > 0 else t.item()
                        # 如果是标量（0维），转换为 Python 标量再转回 tensor
                        if t.dim() == 0:
                            t = t.item()
                    # 转换为标量 tensor（会在 stack 时自动处理）
                    if not isinstance(t, torch.Tensor):
                        t = torch.tensor(t, device=self.device, dtype=torch.float32)
                    else:
                        t = t.to(self.device).float()
                    # 确保是标量（0维）
                    if t.dim() > 0:
                        t = t.flatten()[0] if t.numel() > 0 else torch.tensor(0.0, device=self.device)
                    processed_timesteps.append(t)
                
                # stack 标量 tensors 会得到 (B,) 形状
                timesteps_tensor = torch.stack(processed_timesteps)
                assert timesteps_tensor.dim() == 1, f"timesteps_tensor should be 1D (B,), got {timesteps_tensor.dim()}D, shape={timesteps_tensor.shape}"
                assert timesteps_tensor.shape[0] == B_batch, (
                    f"timesteps_tensor batch size {timesteps_tensor.shape[0]} != B_batch(len(micro_batch)) {B_batch}"
                )
                
                x_noised_tensor = torch.stack([torch.tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device) for x in x_noised_list])
                assert x_noised_tensor.shape[0] == B_batch, (
                    f"x_noised_tensor batch size {x_noised_tensor.shape[0]} != B_batch(len(micro_batch)) {B_batch}"
                )

                x_next_tensor = torch.stack([torch.tensor(x, device=self.device) if not isinstance(x, torch.Tensor) else x.to(self.device) for x in x_next_list])
                assert x_next_tensor.shape[0] == B_batch, (
                    f"x_next_tensor batch size {x_next_tensor.shape[0]} != B_batch(len(micro_batch)) {B_batch}"
                )
                log_probs_old_tensor = torch.tensor(log_probs_old_list, device=self.device, dtype=torch.float32)
                scores_tensor = torch.tensor(scores_list, device=self.device, dtype=torch.float32)
                scenario_indices_tensor = torch.tensor(scenario_indices_list, device=self.device, dtype=torch.long)
                
                # 检查并修复 latents_tensor 的形状
                assert latents_tensor.dim() >= 4, f"latents_tensor should be at least 4D, got {latents_tensor.dim()}D, shape={latents_tensor.shape}"
                if latents_tensor.dim() == 5:
                    # 如果是 (B, 1, P, T, D)，需要检查
                    B, B_inner, P, T, D = latents_tensor.shape
                    if B_inner == 1:
                        latents_tensor = latents_tensor.squeeze(1)  # (B, P, T, D)
                elif latents_tensor.dim() == 4:
                    B, P, T, D = latents_tensor.shape
                    # 如果 P=1 且 T=140（7*20），需要 reshape 为 (B, 7, 20, 4)
                    if P == 1 and T == 140 and D == 4:
                        latents_tensor = latents_tensor.reshape(B, 7, 20, D)
                    # 如果 P=7 且 T=20，保持原样（这是正确的形状）
                    elif P == 7 and T == 20 and D == 4:
                        pass  # 保持原样
                    # 如果 P=1 且 T=20，可能需要扩展 P 维度，但这里保持原样
                    elif P == 1 and T == 20 and D == 4:
                        pass  # 保持原样
                
                # 检查并修复 x_noised_tensor 的形状
                if x_noised_tensor.dim() == 5:
                    B, B_inner, P, T, D = x_noised_tensor.shape
                    if B_inner == 1:
                        x_noised_tensor = x_noised_tensor.squeeze(1)  # (B, P, T, D)
                elif x_noised_tensor.dim() == 4:
                    B, P, T, D = x_noised_tensor.shape
                    if P == 1 and T == 140 and D == 4:
                        x_noised_tensor = x_noised_tensor.reshape(B, 7, 20, D)
                    # 如果 P=7 且 T=20，保持原样（这是正确的形状）
                    elif P == 7 and T == 20 and D == 4:
                        pass  # 保持原样
                    # 如果 P=1 且 T=20，保持原样
                    elif P == 1 and T == 20 and D == 4:
                        pass  # 保持原样

                # 检查并修复 x_next_tensor 的形状（应与 x_noised_tensor 对齐）
                if x_next_tensor.dim() == 5:
                    B, B_inner, P, T, D = x_next_tensor.shape
                    if B_inner == 1:
                        x_next_tensor = x_next_tensor.squeeze(1)  # (B, P, T, D)
                elif x_next_tensor.dim() == 4:
                    B, P, T, D = x_next_tensor.shape
                    if P == 1 and T == 140 and D == 4:
                        x_next_tensor = x_next_tensor.reshape(B, 7, 20, D)
                    elif P == 7 and T == 20 and D == 4:
                        pass
                    elif P == 1 and T == 20 and D == 4:
                        pass
                
                # 检查并修复 timesteps_tensor 的形状
                # timesteps_tensor 应该是 (B,) 形状，每个元素是一个 timestep
                if timesteps_tensor.dim() == 2:
                    # 如果是 (B, 1)，squeeze
                    if timesteps_tensor.shape[1] == 1:
                        timesteps_tensor = timesteps_tensor.squeeze(1)  # (B,)
                    elif timesteps_tensor.shape[0] == 1:
                        timesteps_tensor = timesteps_tensor.squeeze(0)  # (B,)
                    else:
                        # 如果是其他形状，尝试 flatten
                        timesteps_tensor = timesteps_tensor.flatten()  # (B,)
                elif timesteps_tensor.dim() > 2:
                    # 如果是更高维度，flatten
                    timesteps_tensor = timesteps_tensor.flatten()  # (B,)
                elif timesteps_tensor.dim() == 0:
                    # 如果是标量，扩展为 (1,)
                    timesteps_tensor = timesteps_tensor.unsqueeze(0)  # (1,)
                
                # 确保 timesteps_tensor 的长度等于 batch size
                if timesteps_tensor.shape[0] != B_batch:
                    # 如果长度不匹配，可能需要截断或扩展
                    if timesteps_tensor.shape[0] > B_batch:
                        # 如果长度大于 batch size，截断
                        timesteps_tensor = timesteps_tensor[:B_batch]
                    elif timesteps_tensor.shape[0] == 1:
                        # 如果只有一个元素，扩展到 batch size
                        timesteps_tensor = timesteps_tensor.expand(B_batch)
                    else:
                        logger.warning(f"timesteps_tensor shape {timesteps_tensor.shape} does not match batch size {B_batch}")
                        # 尝试扩展或截断
                        if timesteps_tensor.shape[0] < B_batch:
                            # 如果长度小于 batch size，重复最后一个元素
                            last_timestep = timesteps_tensor[-1:]
                            timesteps_tensor = torch.cat([timesteps_tensor, last_timestep.repeat(B_batch - timesteps_tensor.shape[0])])
                        else:
                            timesteps_tensor = timesteps_tensor[:B_batch]
                
            except Exception as e:
                logger.error(f"Error stacking tensors: {e}")
                continue
            
            # condition 的 CFG 处理（cond/uncond）在下面的重算阶段统一完成
            
            # 前向传播（带梯度，use_cfg=False）
            # 为了与仿真/采样收集 old_log_prob 对齐（那边是 eval 模式），这里重算 log_prob_new 时使用 eval()。
            # 注意：eval() 不会关闭梯度，只是关闭 dropout / 使用 BN 的 running stats。
            was_training = self.model.training
            self.model.eval()
            
            # 批量计算 log_prob_new
            # 批量处理整个 micro_batch，B = micro_batch_size
            with torch.enable_grad():
                # 批量恢复所有 conditions 的 decoder_model_extra
                # 恢复函数：递归地将对象恢复到设备上并设置 requires_grad（仅对浮点类型）
                def restore_to_device(obj, device):
                    """递归地将对象恢复到设备上并设置 requires_grad（仅对浮点类型 tensor）"""
                    if isinstance(obj, torch.Tensor):
                        if obj.device != device:
                            result = obj.to(device)
                        else:
                            result = obj
                        if result.requires_grad is False and result.dtype.is_floating_point:
                            result = result.requires_grad_(True)
                        return result
                    elif isinstance(obj, (list, tuple)):
                        return type(obj)(restore_to_device(item, device) for item in obj)
                    elif isinstance(obj, dict):
                        return {k: restore_to_device(v, device) for k, v in obj.items()}
                    else:
                        return obj
                
                # 批量恢复所有 conditions（CFG：cond + uncond）
                cond_extras_list = []
                uncond_extras_list = []
                for condition in conditions:
                    if 'cond' not in condition:
                        raise KeyError("Missing key 'cond' in collected condition. 请用开启 CFG 的 RL simulate 重新收集数据。")
                    if 'decoder_model_extra' not in condition['cond']:
                        raise KeyError("Missing key 'condition[\"cond\"][\"decoder_model_extra\"]' in collected data.")
                    if 'uncond' not in condition:
                        raise KeyError("Missing key 'uncond' in collected condition. 请用开启 CFG 的 RL simulate 重新收集数据。")
                    if 'decoder_model_extra' not in condition['uncond']:
                        raise KeyError("Missing key 'condition[\"uncond\"][\"decoder_model_extra\"]' in collected data.")

                    cond_extras_list.append(restore_to_device(condition['cond']['decoder_model_extra'], self.device))
                    uncond_extras_list.append(restore_to_device(condition['uncond']['decoder_model_extra'], self.device))

                # 获取 batch size
                B_batch = latents_tensor.shape[0]

                # 获取模型的 flow_ode / 速度变换 / CFG 权重
                flow_ode = self.base_model.flow_ode
                velocity_func = flow_ode.translation_funcs[(self.base_model._model_type, 'velocity')]
                cfg_weight = float(flow_ode.cfg_weight)

                # 白名单制：只允许 decoder 需要的字段进入 merged_extra
                allowed_keys = {"encodings", "masks", "routes_cond", "token_dist", "cfg_flags"}

                def _ensure_tensor_batch_first(t: torch.Tensor, expect_dim: int) -> torch.Tensor:
                    """把可能带多余 1 维的 tensor 规整成 (1, ...) 且维度为 expect_dim"""
                    assert isinstance(t, torch.Tensor), f"expected tensor, got {type(t)}"
                    t = t.to(self.device)
                    while t.dim() > expect_dim and t.shape[0] == 1:
                        t = t.squeeze(0)
                    if t.dim() == expect_dim - 1:
                        t = t.unsqueeze(0)
                    assert t.dim() == expect_dim, f"tensor dim should be {expect_dim}, got {t.dim()} with shape {t.shape}"
                    assert t.shape[0] == 1, f"tensor batch dim should be 1, got shape {t.shape}"
                    return t

                def _as_batch_1d_int(cfg_flag: torch.Tensor) -> torch.Tensor:
                    """确保 cfg_flags 为 (1,) int tensor"""
                    assert isinstance(cfg_flag, torch.Tensor), f"cfg_flags elem should be tensor, got {type(cfg_flag)}"
                    cfg_flag = cfg_flag.to(self.device)
                    if cfg_flag.dim() == 0:
                        cfg_flag = cfg_flag.unsqueeze(0)
                    elif cfg_flag.dim() > 1:
                        cfg_flag = cfg_flag.flatten()[:1]
                    assert cfg_flag.shape == (1,), f"cfg_flags elem should be (1,), got {cfg_flag.shape}"
                    return cfg_flag.to(dtype=torch.int32)

                def _ensure_device(obj, device):
                    """递归地确保所有 tensor 都在正确的设备上"""
                    if isinstance(obj, torch.Tensor):
                        return obj.to(device) if obj.device != device else obj
                    if isinstance(obj, (list, tuple)):
                        return type(obj)(_ensure_device(item, device) for item in obj)
                    if isinstance(obj, dict):
                        return {k: _ensure_device(v, device) for k, v in obj.items()}
                    return obj

                def _build_merged_extra(extras_list, batch_size: int, extra: Dict[str, Any]) -> Dict[str, Any]:
                    base0 = extras_list[0]
                    assert allowed_keys.issubset(set(base0.keys())), (
                        f"decoder_model_extra missing keys: {allowed_keys - set(base0.keys())}"
                    )
                    merged_extra: Dict[str, Any] = {}

                    encodings_list = []
                    for de in extras_list[:batch_size]:
                        enc = de["encodings"]
                        assert isinstance(enc, (list, tuple)) and len(enc) == 2, "encodings should be len=2 list/tuple"
                        enc0 = _ensure_tensor_batch_first(enc[0], expect_dim=3)
                        enc1 = _ensure_tensor_batch_first(enc[1], expect_dim=3)
                        encodings_list.append((enc0, enc1))
                    merged_extra["encodings"] = [
                        torch.cat([e0 for (e0, _) in encodings_list], dim=0),
                        torch.cat([e1 for (_, e1) in encodings_list], dim=0),
                    ]

                    masks_list = []
                    for de in extras_list[:batch_size]:
                        m = de["masks"]
                        assert isinstance(m, (list, tuple)) and len(m) == 2, "masks should be len=2 list/tuple"
                        m0 = _ensure_tensor_batch_first(m[0], expect_dim=2)
                        m1 = _ensure_tensor_batch_first(m[1], expect_dim=2)
                        masks_list.append((m0, m1))
                    merged_extra["masks"] = [
                        torch.cat([m0 for (m0, _) in masks_list], dim=0),
                        torch.cat([m1 for (_, m1) in masks_list], dim=0),
                    ]

                    routes_list = [_ensure_tensor_batch_first(de["routes_cond"], expect_dim=2) for de in extras_list[:batch_size]]
                    merged_extra["routes_cond"] = torch.cat(routes_list, dim=0)

                    td_list = [_ensure_tensor_batch_first(de["token_dist"], expect_dim=3) for de in extras_list[:batch_size]]
                    merged_extra["token_dist"] = torch.cat(td_list, dim=0)

                    cfg_list = [_as_batch_1d_int(de["cfg_flags"]) for de in extras_list[:batch_size]]
                    merged_extra["cfg_flags"] = torch.cat(cfg_list, dim=0)

                    for k, v in extra.items():
                        if k in allowed_keys:
                            merged_extra[k] = v
                    merged_extra = _ensure_device(merged_extra, self.device)

                    return merged_extra

                def _make_model_fn(extras_list):
                    def _fn(x, t, **extra):
                        merged_extra = _build_merged_extra(extras_list, x.shape[0], extra)
                        return self.base_model.decoder(x, t, **merged_extra)
                    return _fn

                from flow_planner.model.flow_planner_model.flow_utils.velocity_model import VelocityModel
                velocity_model_cond = VelocityModel(_make_model_fn(cond_extras_list), flow_ode.path, velocity_func, use_cfg=False, cfg_weight=None)
                velocity_model_uncond = VelocityModel(_make_model_fn(uncond_extras_list), flow_ode.path, velocity_func, use_cfg=False, cfg_weight=None)

                assert flow_ode.sample_params['sample_steps'] == 4
                assert flow_ode.sample_params['sample_method'] == 'midpoint', f"sample_method should be midpoint, got {flow_ode.sample_params['sample_method']}"
                step_size = 1.0 / flow_ode.sample_params['sample_steps']
                
                # 批量处理 latents_tensor：确保形状正确
                latents_processed = latents_tensor
                if latents_tensor.dim() == 5:
                    B_latent, P_latent, action_num, action_len, state_dim = latents_tensor.shape
                    latents_processed = latents_tensor.reshape(B_latent, P_latent, action_num * action_len, state_dim)
                elif latents_tensor.dim() == 4:
                    B_latent, P_latent, T_latent, D_latent = latents_tensor.shape
                    if P_latent == 1 and T_latent == 140 and D_latent == 4:
                        latents_processed = latents_tensor.reshape(B_latent, 7, 20, D_latent)
                
                # 断言：latents_processed 应该是 4D (B, P, T, D)
                assert latents_processed.dim() == 4, f"latents_processed should be 4D (B, P, T, D), got {latents_processed.dim()}D, shape={latents_processed.shape}"
                assert latents_processed.shape[0] == B_batch, f"latents_processed batch size {latents_processed.shape[0]} != B_batch {B_batch}"
                
                # 批量处理 timesteps_tensor：确保形状正确
                # timesteps_tensor 应该已经是 (B,) 形状（在上面的代码中已经处理）
                timesteps_processed = timesteps_tensor
                # 确保形状是 (B,)
                if timesteps_processed.dim() != 1:
                    if timesteps_processed.dim() == 0:
                        timesteps_processed = timesteps_processed.unsqueeze(0).expand(B_batch)
                    else:
                        timesteps_processed = timesteps_processed.flatten()
                
                # 确保长度等于 batch size
                if timesteps_processed.shape[0] != B_batch:
                    if timesteps_processed.shape[0] == 1:
                        timesteps_processed = timesteps_processed.expand(B_batch)
                    elif timesteps_processed.shape[0] > B_batch:
                        timesteps_processed = timesteps_processed[:B_batch]
                    else:
                        raise ValueError(f"timesteps_processed batch size {timesteps_processed.shape[0]} != B_batch {B_batch}")
                
                # 断言：timesteps_processed 应该是 1D (B,)
                assert timesteps_processed.dim() == 1, f"timesteps_processed should be 1D (B,), got {timesteps_processed.dim()}D, shape={timesteps_processed.shape}"
                assert timesteps_processed.shape[0] == B_batch, f"timesteps_processed batch size {timesteps_processed.shape[0]} != B_batch {B_batch}"

                t_tensorize_and_prep += time.perf_counter() - t_tensor0
                
                # CFG：先得到带梯度的 u_cond / u_uncond，再按 solver 计算 x_{t-1}（不加噪）
                x0 = latents_processed.to(self.device)
                t0 = timesteps_processed.to(self.device)

                def _combine(u_c: torch.Tensor, u_u: torch.Tensor) -> torch.Tensor:
                    return (1.0 - cfg_weight) * u_u + cfg_weight * u_c

                t_fwd0 = time.perf_counter()
                if flow_ode.sample_params['sample_method'] == 'midpoint':
                    u1_c = velocity_model_cond(x0, t0)
                    u1_u = velocity_model_uncond(x0, t0)
                    u1 = _combine(u1_c, u1_u)

                    # 与 ManualODESolver._midpoint_step 对齐：在 t + h/2 处评估斜率
                    t_mid = t0 + step_size / 2
                    x_mid = x0 + step_size / 2 * u1

                    u2_c = velocity_model_cond(x_mid, t_mid)
                    u2_u = velocity_model_uncond(x_mid, t_mid)
                    u2 = _combine(u2_c, u2_u)

                    # 用于更新的速度（带梯度）
                    u_cond = u2_c
                    u_uncond = u2_u

                    x_next_new = x0 + step_size * u2
                else:
                    # Euler
                    u_c = velocity_model_cond(x0, t0)
                    u_u = velocity_model_uncond(x0, t0)
                    u = _combine(u_c, u_u)
                    u_cond = u_c
                    u_uncond = u_u
                    x_next_new = x0 + step_size * u
            
                
                # 断言：x_next_new 的形状应该与 latents_processed 一致
                assert x_next_new.dim() == 4, f"x_next_new should be 4D (B, P, T, D), got {x_next_new.dim()}D, shape={x_next_new.shape}"
                assert x_next_new.shape[0] == B_batch, f"x_next_new batch size {x_next_new.shape[0]} != B_batch {B_batch}"
                assert x_next_new.shape[1:] == latents_processed.shape[1:], f"x_next_new shape {x_next_new.shape[1:]} != latents_processed shape {latents_processed.shape[1:]}"
                
                # 批量处理 x_noised_tensor：确保形状正确
                x_noised_processed = x_noised_tensor
                if x_noised_tensor.dim() == 5:
                    B_noised, P_noised, action_num, action_len, state_dim = x_noised_tensor.shape
                    x_noised_processed = x_noised_tensor.reshape(B_noised, P_noised, action_num * action_len, state_dim)
                elif x_noised_tensor.dim() == 4:
                    B_noised, P_noised, T_noised, D_noised = x_noised_tensor.shape
                    if P_noised == 1 and T_noised == 140 and D_noised == 4:
                        x_noised_processed = x_noised_tensor.reshape(B_noised, 7, 20, D_noised)
                
                # 断言：x_noised_processed 应该是 4D (B, P, T, D)
                assert x_noised_processed.dim() == 4, f"x_noised_processed should be 4D (B, P, T, D), got {x_noised_processed.dim()}D, shape={x_noised_processed.shape}"
                assert x_noised_processed.shape[0] == B_batch, f"x_noised_processed batch size {x_noised_processed.shape[0]} != B_batch {B_batch}"

                # 批量处理 x_next_tensor：确保形状正确
                x_next_saved_processed = x_next_tensor
                if x_next_tensor.dim() == 5:
                    B_n, P_n, action_num, action_len, state_dim = x_next_tensor.shape
                    x_next_saved_processed = x_next_tensor.reshape(B_n, P_n, action_num * action_len, state_dim)
                elif x_next_tensor.dim() == 4:
                    B_n, P_n, T_n, D_n = x_next_tensor.shape
                    if P_n == 1 and T_n == 140 and D_n == 4:
                        x_next_saved_processed = x_next_tensor.reshape(B_n, 7, 20, D_n)
                assert x_next_saved_processed.shape == x_noised_processed.shape, (
                    f"x_next_saved_processed shape {x_next_saved_processed.shape} != x_noised_processed shape {x_noised_processed.shape}"
                )
                
                # 确保 x_next_new 的形状与 x_noised_processed 一致
                if x_next_new.dim() == 5:
                    B_next, P_next, action_num, action_len, state_dim = x_next_new.shape
                    x_next_new = x_next_new.reshape(B_next, P_next, action_num * action_len, state_dim)
                elif x_next_new.dim() == 4:
                    B_next, P_next, T_next, D_next = x_next_new.shape
                    if P_next != x_noised_processed.shape[1] or T_next != x_noised_processed.shape[2]:
                        if P_next == 7 and T_next == 140 and D_next == 4:
                            x_next_new = x_next_new.reshape(B_next, 7, 20, D_next)
                
                # 断言：x_next_new 的形状应该与 x_noised_processed 一致
                assert x_next_new.shape == x_noised_processed.shape, f"x_next_new shape {x_next_new.shape} != x_noised_processed shape {x_noised_processed.shape}"
                
                # 断言：x_noised_processed 的形状应该与 x_next_new 一致
                assert x_noised_processed.dim() == 4, f"x_noised_processed should be 4D (B, P, T, D), got {x_noised_processed.dim()}D, shape={x_noised_processed.shape}"
                assert x_noised_processed.shape == x_next_new.shape, f"x_noised_processed shape {x_noised_processed.shape} != x_next_new shape {x_next_new.shape}"
                
                # 批量计算 log_prob_new
                # 输入一致性检查：
                # 仿真阶段生成 x_noised 用的是 flow_ode.rl_noise_std（可学习参数）；
                # 训练阶段重算必须使用同一个 noise_std（从flow_ode获取当前值）
                # 注意：noise_std现在是可学习的，所以每次获取当前值
                # 关键：noise_std 必须保持为 tensor 才能对 rl_noise_std_logit 反向传播
                flow_noise_std = flow_ode.rl_noise_std
                if isinstance(flow_noise_std, torch.Tensor):
                    flow_noise_std = flow_noise_std.to(self.device, dtype=torch.float32)
                else:
                    flow_noise_std = torch.tensor(float(flow_noise_std), device=self.device, dtype=torch.float32)
                # 不再需要与固定值比较，因为noise_std是可学习的

                # 仅打印第一批次：验证输入一致性 + 定位 x_next_new 偏差来源
                # if micro_batch_idx == 0:
                #     with torch.no_grad():
                #         mse_noise = ((x_noised_processed - x_next_saved_processed) ** 2).mean(dim=(-3, -2, -1)).mean()
                #         mse_xnext = ((x_next_new - x_next_saved_processed) ** 2).mean(dim=(-3, -2, -1)).mean()
                #         logp_old_recalc = self.compute_log_prob_new(x_noised_processed, x_next_saved_processed, flow_noise_std).mean()
                #         print(
                #             "debug_mse_noise",
                #             mse_noise,
                #             "debug_logp_old_recalc",
                #             logp_old_recalc,
                #             "debug_mse_xnext_new_vs_saved",
                #             mse_xnext,
                #             "model_training",
                #             torch.tensor(int(self.model.training), device=self.device),
                #         )
                log_probs_new_tensor = self.compute_log_prob_new(
                    x_noised_processed.to(self.device),
                    x_next_new,
                    flow_noise_std * (1-timesteps_processed)**2
                )  # (B,)
                t_forward_and_logprob += time.perf_counter() - t_fwd0
                
                # 断言：log_probs_new_tensor 应该是 1D (B,)
                assert log_probs_new_tensor.dim() == 1, f"log_probs_new_tensor should be 1D (B,), got {log_probs_new_tensor.dim()}D, shape={log_probs_new_tensor.shape}"
                assert log_probs_new_tensor.shape[0] == B_batch, f"log_probs_new_tensor batch size {log_probs_new_tensor.shape[0]} != B_batch {B_batch}"
            
            # 断言：检查输入 tensor 的形状
            assert log_probs_old_tensor.dim() == 1, f"log_probs_old_tensor should be 1D (B,), got {log_probs_old_tensor.dim()}D, shape={log_probs_old_tensor.shape}"
            assert log_probs_new_tensor.dim() == 1, f"log_probs_new_tensor should be 1D (B,), got {log_probs_new_tensor.dim()}D, shape={log_probs_new_tensor.shape}"
            assert scores_tensor.dim() == 1, f"scores_tensor should be 1D (B,), got {scores_tensor.dim()}D, shape={scores_tensor.shape}"
            assert log_probs_old_tensor.shape[0] == log_probs_new_tensor.shape[0], f"log_probs_old_tensor batch size {log_probs_old_tensor.shape[0]} != log_probs_new_tensor batch size {log_probs_new_tensor.shape[0]}"
            assert log_probs_old_tensor.shape[0] == scores_tensor.shape[0], f"log_probs_old_tensor batch size {log_probs_old_tensor.shape[0]} != scores_tensor batch size {scores_tensor.shape[0]}"
            
            if self.joint_logprob:
                assert steps_per_trajectory is not None
                assert trajectories_per_micro_batch is not None
                assert B_batch == trajectories_per_micro_batch * steps_per_trajectory, (
                    f"joint_logprob expects B_batch == trajectories_per_micro_batch * steps_per_trajectory, "
                    f"got B_batch={B_batch}, traj_per_micro={trajectories_per_micro_batch}, steps={steps_per_trajectory}"
                )

                # 轨迹级 joint logprob：把每条轨迹的多个去噪步 logprob 聚合（为数值稳定使用 mean 而不是 sum）
                logp_old_2d = log_probs_old_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                logp_new_2d = log_probs_new_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                logp_old_joint = logp_old_2d.mean(dim=1)  # (N_traj,)
                logp_new_joint = logp_new_2d.mean(dim=1)  # (N_traj,)

                # score / scenario_index：每条轨迹内应一致，取每条轨迹的第 1 个 step
                scores_2d = scores_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                scores_traj = scores_2d[:, 0]
                max_abs_score_diff = (scores_2d - scores_traj.unsqueeze(1)).abs().max().item()
                if max_abs_score_diff > 1e-6:
                    raise ValueError(
                        f"joint_logprob requires per-trajectory score identical across denoising steps, "
                        f"but max_abs_diff={max_abs_score_diff}"
                    )

                sidx_2d = scenario_indices_tensor.view(trajectories_per_micro_batch, steps_per_trajectory)
                sidx_traj = sidx_2d[:, 0]
                max_sidx_diff = (sidx_2d - sidx_traj.unsqueeze(1)).abs().max().item()
                if max_sidx_diff != 0:
                    raise ValueError("joint_logprob requires per-trajectory scenario_index identical across steps.")

                # 轨迹级 advantage：对齐现有逻辑（按 scenario mean/std 标准化并 clamp）
                t_adv0 = time.perf_counter()
                standardized_scores = (scores_traj - scenario_means_tensor[sidx_traj]) / scenario_stds_tensor[sidx_traj]
                advantages = torch.clamp(standardized_scores, -self.adv_clip, self.adv_clip)
                t_advantage += time.perf_counter() - t_adv0

                # PPO loss：用 joint logprob 算 ratio / clip
                t_loss0 = time.perf_counter()
                loss, stats = self.compute_ppo_loss(logp_old_joint, logp_new_joint, advantages)
                t_loss += time.perf_counter() - t_loss0
            else:
                # 计算 advantage（step-level）
                # scores_tensor 已经是每个数据点对应的 score（从保存的数据中获取）
                t_adv0 = time.perf_counter()
                advantages = self.compute_advantage(
                    scores_per_datapoint=scores_tensor,  # 每个数据点的 score
                    scenario_indices=scenario_indices_tensor,
                    scenario_score_means=scenario_means_tensor,
                    scenario_score_stds=scenario_stds_tensor,
                    log_probs_old=log_probs_old_tensor,
                    log_probs_new=log_probs_new_tensor,
                )
                t_advantage += time.perf_counter() - t_adv0
                
                # 断言：advantages 的形状应该与 log_probs_old_tensor 一致
                assert advantages.shape == log_probs_old_tensor.shape, f"advantages shape {advantages.shape} != log_probs_old_tensor shape {log_probs_old_tensor.shape}"
                
                # 计算 loss（step-level）
                t_loss0 = time.perf_counter()
                loss, stats = self.compute_ppo_loss(log_probs_old_tensor, log_probs_new_tensor, advantages)
                t_loss += time.perf_counter() - t_loss0
            
            if accum_k is None or accum_k <= 0:
                raise RuntimeError("Internal error: accum_k is not set")
            loss_for_backward = loss / float(accum_k)
            last_in_accum = ((local_i % grad_acc) == (accum_k - 1)) or (local_i == int(local_total) - 1)

            # 反向传播（DDP: 非最后一步可 no_sync 加速）
            t_bo0 = time.perf_counter()
            if dist_enabled and hasattr(self.model, "no_sync") and (not last_in_accum):
                with self.model.no_sync():  # type: ignore[attr-defined]
                    loss_for_backward.backward()
            else:
                loss_for_backward.backward()

            total_loss += float(loss.item())
            accum_stats.append(stats)

            if last_in_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)
                self.optimizer.step()
                t_backward_opt += time.perf_counter() - t_bo0
                g = float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                for s in accum_stats:
                    s["grad_norm"] = g
                all_stats.extend(accum_stats)
                accum_stats = []

                # 恢复模型模式（避免影响后续训练逻辑）
                if was_training:
                    self.model.train()
            else:
                t_backward_opt += time.perf_counter() - t_bo0
        
        # 6. 统计信息（DDP 下做 all_reduce 聚合）
        local_mb = len(all_stats)
        if local_mb == 0:
            logger.warning("No micro-batches were processed")
            return

        sum_ratio_mean = float(np.sum([s['ratio_mean'] for s in all_stats]))
        sum_ratio_max = float(np.sum([s['ratio_max'] for s in all_stats]))
        sum_ratio_min = float(np.sum([s['ratio_min'] for s in all_stats]))
        sum_clipped_ratio_mean = float(np.sum([s['clipped_ratio_mean'] for s in all_stats]))
        sum_clipped_ratio_max = float(np.sum([s['clipped_ratio_max'] for s in all_stats]))
        sum_clipped_ratio_min = float(np.sum([s['clipped_ratio_min'] for s in all_stats]))
        sum_clip_fraction = float(np.sum([s['clip_fraction'] for s in all_stats]))
        sum_grad_norm = float(np.sum([s.get('grad_norm', 0.0) for s in all_stats]))
        sum_policy_loss = float(np.sum([s.get('policy_loss', 0.0) for s in all_stats]))
        sum_entropy = float(np.sum([s.get('entropy', 0.0) for s in all_stats]))
        sum_entropy_bonus = float(np.sum([s.get('entropy_bonus', 0.0) for s in all_stats]))
        sum_loss = float(total_loss)

        if dist_enabled and world_size > 1:
            agg = torch.tensor(
                [
                    sum_loss,
                    sum_ratio_mean,
                    sum_ratio_max,
                    sum_ratio_min,
                    sum_clipped_ratio_mean,
                    sum_clipped_ratio_max,
                    sum_clipped_ratio_min,
                    sum_clip_fraction,
                    sum_grad_norm,
                    sum_policy_loss,
                    sum_entropy,
                    sum_entropy_bonus,
                    float(local_mb),
                ],
                device=self.device,
                dtype=torch.float32,
            )
            dist.all_reduce(agg, op=dist.ReduceOp.SUM)
            sum_loss = float(agg[0].item())
            sum_ratio_mean = float(agg[1].item())
            sum_ratio_max = float(agg[2].item())
            sum_ratio_min = float(agg[3].item())
            sum_clipped_ratio_mean = float(agg[4].item())
            sum_clipped_ratio_max = float(agg[5].item())
            sum_clipped_ratio_min = float(agg[6].item())
            sum_clip_fraction = float(agg[7].item())
            sum_grad_norm = float(agg[8].item())
            sum_policy_loss = float(agg[9].item())
            sum_entropy = float(agg[10].item())
            sum_entropy_bonus = float(agg[11].item())
            total_mb = int(agg[12].item())
        else:
            total_mb = local_mb

        avg_loss = sum_loss / float(total_mb)
        avg_ratio_mean = sum_ratio_mean / float(total_mb)
        avg_ratio_max = sum_ratio_max / float(total_mb)
        avg_ratio_min = sum_ratio_min / float(total_mb)
        avg_clipped_ratio_mean = sum_clipped_ratio_mean / float(total_mb)
        avg_clipped_ratio_max = sum_clipped_ratio_max / float(total_mb)
        avg_clipped_ratio_min = sum_clipped_ratio_min / float(total_mb)
        avg_clip_fraction = sum_clip_fraction / float(total_mb)
        avg_grad_norm = sum_grad_norm / float(total_mb)
        avg_policy_loss = sum_policy_loss / float(total_mb)
        avg_entropy = sum_entropy / float(total_mb)
        avg_entropy_bonus = sum_entropy_bonus / float(total_mb)
        avg_score = float(np.mean(np.array(scenario_score_means, dtype=np.float32)))
        success_rate = float(np.mean([1.0 if s > 0 else 0.0 for s in scenario_score_means]))  # 假设 score > 0 表示成功

        # 6.1 TensorBoard 记录（如果启用）
        if self.tb_writer is not None:
            # DDP 下只在 rank0 写 TensorBoard，避免多进程抢写
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                    pass
                else:
                    # step_count 由外部循环传入，作为全局 step
                    self.tb_writer.add_scalar("rl/avg_loss", float(avg_loss), step_count)
                    self.tb_writer.add_scalar("rl/avg_score", float(avg_score), step_count)
                    self.tb_writer.add_scalar("rl/success_rate", float(success_rate), step_count)
                    self.tb_writer.add_scalar("rl/ratio_mean", float(avg_ratio_mean), step_count)
                    self.tb_writer.add_scalar("rl/ratio_max", float(avg_ratio_max), step_count)
                    self.tb_writer.add_scalar("rl/ratio_min", float(avg_ratio_min), step_count)
                    self.tb_writer.add_scalar("rl/clipped_ratio_mean", float(avg_clipped_ratio_mean), step_count)
                    self.tb_writer.add_scalar("rl/clipped_ratio_max", float(avg_clipped_ratio_max), step_count)
                    self.tb_writer.add_scalar("rl/clipped_ratio_min", float(avg_clipped_ratio_min), step_count)
                    self.tb_writer.add_scalar("rl/clip_fraction", float(avg_clip_fraction), step_count)
                    self.tb_writer.add_scalar("rl/grad_norm", float(avg_grad_norm), step_count)
                    self.tb_writer.add_scalar("rl/policy_loss", float(avg_policy_loss), step_count)
                    self.tb_writer.add_scalar("rl/entropy", float(avg_entropy), step_count)
                    self.tb_writer.add_scalar("rl/entropy_bonus", float(avg_entropy_bonus), step_count)
                    # 记录可学习的noise_std
                    if hasattr(self.base_model, 'flow_ode') and hasattr(self.base_model.flow_ode, 'rl_noise_std'):
                        ns = self.base_model.flow_ode.rl_noise_std
                        if isinstance(ns, torch.Tensor):
                            current_noise_std = float(ns.detach().cpu().item())
                        else:
                            current_noise_std = float(ns)
                        self.tb_writer.add_scalar("rl/noise_std", current_noise_std, step_count)
                    self.tb_writer.flush()
            except Exception:
                # 非分布式时直接写
                self.tb_writer.add_scalar("rl/avg_loss", float(avg_loss), step_count)
                self.tb_writer.add_scalar("rl/avg_score", float(avg_score), step_count)
                self.tb_writer.add_scalar("rl/success_rate", float(success_rate), step_count)
                self.tb_writer.add_scalar("rl/ratio_mean", float(avg_ratio_mean), step_count)
                self.tb_writer.add_scalar("rl/ratio_max", float(avg_ratio_max), step_count)
                self.tb_writer.add_scalar("rl/ratio_min", float(avg_ratio_min), step_count)
                self.tb_writer.add_scalar("rl/clipped_ratio_mean", float(avg_clipped_ratio_mean), step_count)
                self.tb_writer.add_scalar("rl/clipped_ratio_max", float(avg_clipped_ratio_max), step_count)
                self.tb_writer.add_scalar("rl/clipped_ratio_min", float(avg_clipped_ratio_min), step_count)
                self.tb_writer.add_scalar("rl/clip_fraction", float(avg_clip_fraction), step_count)
                self.tb_writer.add_scalar("rl/grad_norm", float(avg_grad_norm), step_count)
                self.tb_writer.add_scalar("rl/policy_loss", float(avg_policy_loss), step_count)
                self.tb_writer.add_scalar("rl/entropy", float(avg_entropy), step_count)
                self.tb_writer.add_scalar("rl/entropy_bonus", float(avg_entropy_bonus), step_count)
                # 记录可学习的noise_std
                if hasattr(self.base_model, 'flow_ode') and hasattr(self.base_model.flow_ode, 'rl_noise_std'):
                    ns = self.base_model.flow_ode.rl_noise_std
                    if isinstance(ns, torch.Tensor):
                        current_noise_std = float(ns.detach().cpu().item())
                    else:
                        current_noise_std = float(ns)
                    self.tb_writer.add_scalar("rl/noise_std", current_noise_std, step_count)
                self.tb_writer.flush()
        
        # 获取当前noise_std值
        current_noise_std_str = ""
        if hasattr(self.base_model, 'flow_ode') and hasattr(self.base_model.flow_ode, 'rl_noise_std'):
            ns = self.base_model.flow_ode.rl_noise_std
            if isinstance(ns, torch.Tensor):
                current_noise_std = float(ns.detach().cpu().item())
            else:
                current_noise_std = float(ns)
            current_noise_std_str = f", NoiseStd={current_noise_std:.6f}"
        
        logger.info(
            f"Step {step_count}: "
            f"Loss={avg_loss:.4f} (Policy={avg_policy_loss:.4f}, EntropyBonus={avg_entropy_bonus:.4f}), "
            f"Score={avg_score:.4f}, "
            f"RatioMean={avg_ratio_mean:.4f}, "
            f"RatioMax={avg_ratio_max:.4f}, "
            f"RatioMin={avg_ratio_min:.4f}, "
            f"ClipFraction={avg_clip_fraction:.2%}, "
            f"GradNorm={avg_grad_norm:.4f}, "
            f"Entropy={avg_entropy:.4f}"
            f"{current_noise_std_str}"
        )

        t_total = time.perf_counter() - t_step0
        denom = float(max(num_micro_batches, 1))
        # logger.info(
        #     f"Step {step_count} train_step timing: total={t_total:.2f}s | "
        #     f"flatten={t_flatten:.2f}s | shuffle={t_shuffle:.2f}s | "
        #     f"micro_batches={num_micro_batches} (micro_batch_size={self.micro_batch_size}, data={len(flattened_data)}) | "
        #     f"collect_py={t_collect_py:.2f}s(avg={t_collect_py/denom:.3f}s) | "
        #     f"tensorize+prep={t_tensorize_and_prep:.2f}s(avg={t_tensorize_and_prep/denom:.3f}s) | "
        #     f"forward+logprob={t_forward_and_logprob:.2f}s(avg={t_forward_and_logprob/denom:.3f}s) | "
        #     f"advantage={t_advantage:.2f}s(avg={t_advantage/denom:.3f}s) | "
        #     f"loss={t_loss:.2f}s(avg={t_loss/denom:.3f}s) | "
        #     f"backward+opt={t_backward_opt:.2f}s(avg={t_backward_opt/denom:.3f}s)"
        # )
        
        # 7. 保存检查点
        if step_count % self.save_interval == 0:
            self.save_checkpoint(step_count)
    
    def save_checkpoint(self, step_count: int):
        """保存检查点"""
        if self.output_dir is None:
            return

        # DDP 下只在 rank0 保存，避免多进程写同一路径
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
                return
        except Exception:
            pass
        
        checkpoint_path = self.output_dir / f"checkpoint_step_{step_count}.pth"

        # 注意：
        # 1) 按需求：不要保存 model_state_dict，而是保存到 key='ema_state_dict'（仅改 key 名，不代表真实 EMA）
        # 2) 同时把所有权重名加上 "module." 前缀（便于与其它 ckpt 格式对齐）
        raw_state = (self.base_model.state_dict() if hasattr(self, "base_model") else self.model.state_dict())
        if not isinstance(raw_state, dict):
            raise TypeError(f"state_dict must be dict, got {type(raw_state)}")
        ema_state_dict = {}
        for k, v in raw_state.items():
            if not isinstance(k, str):
                raise TypeError(f"state_dict key must be str, got {type(k)}")
            kk = k if k.startswith("module.") else ("module." + k)
            ema_state_dict[kk] = v

        checkpoint = {
            "step_count": int(step_count),
            "ema_state_dict": ema_state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rl_cfg": self.rl_cfg,
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
