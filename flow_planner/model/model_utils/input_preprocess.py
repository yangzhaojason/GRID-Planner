import torch
import math
from flow_planner.data.dataset.nuplan import NuPlanDataSample

class ModelInputProcessor:
    def __init__(
        self,
        future_len,
        obs_normalizer,
        state_normalizer,
        neighbor_pred_num
    ):
        self.future_len = future_len
        self.obs_normalizer = obs_normalizer
        self.state_normalizer = state_normalizer
        self.neighbor_pred_num = neighbor_pred_num

    def state_preprocess(self, x):
        return self.state_normalizer(x) if self.state_normalizer is not None else x
    
    def state_postprocess(self, x):
        return self.state_normalizer.inverse(x) if self.state_normalizer is not None else x

    def x_differentiate(self, x_future, x_current):
        x_all = torch.cat([x_current, x_future], dim=-2)
        return x_all[..., 1:, :] - x_all[..., :-1, :]

    def x_integral(self, dx_future, x_current):
        v_all = torch.cat([x_current, dx_future], dim=-2)
        return torch.cumsum(v_all, dim=-2)[..., 1:, :]

    def sample_to_model_input(
        self,
        data: NuPlanDataSample,
        device,
        kinematic,
        is_training: bool=False
    ):
                
        if self.obs_normalizer is not None:
            data = self.obs_normalizer(data)

        ego_future = data.ego_future
        if ego_future.numel() != 0:
            ego_future = ego_future[..., -self.future_len:, :3] # (x, y, heading)

        model_inputs = {}
        model_inputs['ego_past'] = data.ego_past.to(device)
        model_inputs['neighbor_past'] = data.neighbor_past.to(device)
        model_inputs['lanes'] = data.lanes.to(device)
        model_inputs['lanes_speedlimit'] = data.lanes_speedlimit.to(device)
        model_inputs['lanes_has_speedlimit'] = data.lanes_has_speedlimit.to(device)
        model_inputs['routes'] = data.routes.to(device)
        model_inputs['map_objects'] = data.map_objects.to(device)
        

        ego_current_state = data.ego_current
        model_inputs['ego_current'] = ego_current_state
        ego_current_xy_cos_sin = ego_current_state[..., :4]
        ego_current = torch.cat([
            ego_current_xy_cos_sin[..., :2],
            torch.atan2(ego_current_xy_cos_sin[..., 3:4], ego_current_xy_cos_sin[..., 2:3])
        ], dim=-1)

        current_states = ego_current[:, None]
        
        if is_training:
            gt_future = ego_future[:, None, :, :]

            gt_with_current = torch.cat([
                    current_states[:, :, None, :],
                    gt_future
                ], dim=2)

            gt_with_current = gt_with_current.to(device)
        else:
            gt_with_current = current_states[:, :, None, :].repeat(1, 1, self.future_len + 1, 1)

        if kinematic == 'waypoints':
            gt_with_current = torch.cat([
                gt_with_current[..., :2],
                torch.cat([
                    gt_with_current[..., 2:3].cos(),
                    gt_with_current[..., 2:3].sin()
                ], dim=-1)
            ], dim=-1)
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :])
        elif kinematic == 'velocity':
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            gt_with_current = torch.cat([gt_with_current[..., :1, :], future_velocity], dim=-2)
        elif kinematic == 'acceleration':
            future_velocity = self.x_differentiate(gt_with_current[..., 1:, :], gt_with_current[..., :1, :])
            current_velocity = torch.cat([ego_current_state[..., 4:6], ego_current_state[..., 9:10]], dim=-1)[:, None, None, :]
            future_acc = self.x_differentiate(future_velocity, current_velocity)
            gt_with_current = torch.cat([current_velocity, future_acc], dim=-2)
        elif kinematic == 'xy_only':
            """
            nuPlan 原始模式
            在 nuPlan ego-planning 中，模型需要输出 (x, y, heading) 三个维度——因为自动驾驶规控需要知道车头朝向。对应的 kinematic 模式：
            waypoints（line 79-87）：输出 (x, y, cos_θ, sin_θ) → 4 维，heading 被展开为 cos/sin
            velocity（line 88-90）：输出 (Δx, Δy, Δθ) → 3 维，差分表示
            acceleration（line 91-95）：输出 (Δ²x, Δ²y, Δ²θ) → 3 维，二阶差分

            WOMD 需要 xy_only
            WOMD motion prediction 只需要预测 (x, y) 位置，不需要 heading。原有的三种模式都无法直接用：
            waypoints 输出 4 维 (x, y, cos, sin)——多了 heading
            velocity / acceleration 输出 3 维——也多了 heading 的差分
            
            所以需要 xy_only 分支做两件事：
            gt_with_current[..., :2] — 只取前两维 (x, y)，丢掉第三维（在 WOMD 中存的是 validity flag，不是 heading）
            state_normalizer(gt_with_current[..., 1:, :]) — 对 future 部分用 StateNormalizer 归一化（和 waypoints 模式一样，只是维度从 4 变成 2）
            配合 config 中 state_dim: 2，整个模型就在 2D (x, y) 空间上做 flow matching，和 waypoints 模式中 4D 的流程完全一致，只是少了 heading 维度。
            """
            gt_with_current = gt_with_current[..., :2]
            gt_with_current[..., 1:, :] = self.state_normalizer(gt_with_current[..., 1:, :])
        
        return model_inputs, gt_with_current