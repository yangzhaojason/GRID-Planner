import torch
import torch.nn as nn
from timm.layers import Mlp
from flow_planner.model.modules.decoder_modules import FinalLayer, PostFusion
from flow_planner.model.model_utils.tool_func import sinusoidal_positional_encoding
from flow_planner.model.modules.decoder_modules import RMSNorm, FeedForward, AdaptiveLayerNorm
from flow_planner.model.flow_planner_model.global_attention import JointAttention


class FlowPlannerDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim,
            depth,
            t_embedder,
            agents_hidden_dim=192,
            lane_hidden_dim=192,
            heads=6,
            preproj_hidden=256,
            enable_attn_dist=False,
            act_pe_type: str = 'learnable',
            device: str = 'cuda',
            **planner_params
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = planner_params['action_len'] * planner_params['state_dim']
        action_num = (planner_params['future_len'] - planner_params['action_overlap']) // (planner_params['action_len'] - planner_params['action_overlap'])
        self.action_num = int(action_num)
        self.token_num = planner_params['neighbor_num'] + planner_params['static_num'] + planner_params['lane_num'] + self.action_num
        self.action_len = planner_params['action_len']
        self.action_overlap = planner_params['action_overlap']
        self.state_dim = planner_params['state_dim']
        
        self.dit = FlowPlannerDiT(
            depth=depth,
            dim_modalities=(
                agents_hidden_dim,
                lane_hidden_dim,
                hidden_dim
            ),
            dim_cond=hidden_dim,
            heads=heads,
            dim_head=int(hidden_dim/heads),
            enable_attn_dist=enable_attn_dist,
            token_num=self.token_num
        )
        
        self.post_fusion = PostFusion(hidden_dim=hidden_dim, heads=heads, action_num=self.action_num)
        self.t_embedder = t_embedder        
        self.preproj = Mlp(in_features=self.output_dim, hidden_features=preproj_hidden, out_features=hidden_dim, act_layer=nn.GELU, drop=0.)
        self.final_layer = FinalLayer(hidden_dim, self.output_dim)

        self.cfg_embedding = nn.Embedding(2, hidden_dim) # an embedding that indicates if the neighbor vehicles are dropped
        self.act_pe_type = act_pe_type
        self.load_action_pe(act_pe_type) # load action positional embedding into the model
        
        self.agents_in_proj = nn.Linear(agents_hidden_dim, hidden_dim) if agents_hidden_dim != hidden_dim \
            else nn.Identity()
        self.lane_in_proj = nn.Linear(lane_hidden_dim, hidden_dim) if lane_hidden_dim != hidden_dim \
            else nn.Identity()
        
        self.planner_params = planner_params
        
        self.device = device
        
        # self.initialize_weights()

    def load_action_pe(self, act_pe_type: str):
        if act_pe_type == 'learnable':
            self.action_pe = nn.Parameter(torch.Tensor(self.action_num, self.hidden_dim))
            nn.init.normal_(self.action_pe, mean=0.0, std=1.0)
        elif act_pe_type == 'fixed_sin':
            action_t = (torch.arange(0, self.action_num) * (self.action_len - self.action_overlap) + self.action_len / 2)
            action_pe = sinusoidal_positional_encoding(action_t, self.hidden_dim)
            self.register_buffer('action_pe', action_pe)
        elif act_pe_type == 'none':
            action_pe = torch.zeros((self.action_num, self.hidden_dim))
            self.register_buffer('action_pe', action_pe)
        else:
            raise ValueError(f'Unexpected action embedding type {act_pe_type}')
    
    def initialize_weights(self):
        def basic_init(module):
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                    nn.init.constant_(module.bias, 0.0)
        self.apply(basic_init)
    
    def forward(self, x, t, **model_extra):
        '''
        Forward pass of the FlowPlannerDecoder.
        x: (B, P, output_dim)   -> Embedded out of DiT
        t: (B,)                  -> Time step
        model_extra: dict containing additional information for the model, such as:
            - 'encodings': list of encodings for different modalities, each with shape (B
            , neighbor, lane, static, hidden_dim)
            - 'token_dist': attention distribution for the tokens, shape (B, neighbor, lane, static, token_num)
            - 'masks': list of masks for different modalities, each with shape (B, neighbor, lane, static, token_num)
            - 'cfg_flags': tensor of shape (B,) indicating whether to use configuration flags
            - 'routes_cond': tensor of shape (B, hidden_dim) for route conditions
        '''

        # 断言：检查输入形状
        assert x.dim() == 4, f"x should be 4D (B, P, T, D), got {x.dim()}D, shape={x.shape}"
        assert t.dim() == 1, f"t should be 1D (B,), got {t.dim()}D, shape={t.shape}"
        # assert t.shape[0] == x.shape[0], f"t batch size {t.shape[0]} != x batch size {x.shape[0]}"
        
        B, P, _, _ = x.shape
        # 检查 x 的形状：如果 T*D 不等于 output_dim，可能需要 reshape
        # output_dim = action_len * state_dim = 20 * 4 = 80
        # 如果 x 是 (B, P, T, D) 且 T*D != 80，可能需要特殊处理
        T_D = x.shape[2] * x.shape[3]
        if T_D != self.output_dim:
            # 如果 T*D = 140*4 = 560，但 output_dim = 80，说明 P 和 T 被合并了
            # 需要 reshape 为 (B, 7, 20, 4) 然后 flatten
            if T_D == 560 and P == 1:
                # reshape 为 (B, 7, 20, 4)
                x = x.reshape(B, 7, 20, 4)
                P = 7
            elif T_D == 140 and P == 1:
                # 如果 T=140, D=4，reshape 为 (B, 7, 20, 4)
                x = x.reshape(B, 7, 20, 4)
                P = 7
        
        x = x.to(torch.float32)
        x = x.reshape(B, P, -1)
        
        # 断言：检查 x reshape 后的形状
        assert x.shape[2] == self.output_dim, f"x reshape后第三维应该是 output_dim={self.output_dim}, got {x.shape[2]}"
        
        encodings = list(model_extra['encodings'])
        
        # 断言：检查 encodings
        # encoder 输出的 encodings 只有 2 个 modality：
        #   0: neighbors + static（已 concat）
        #   1: lanes
        # 第 3 个 modality（ego trajectory tokens）由 decoder 这里的 preproj(x) 追加
        assert isinstance(encodings, (list, tuple)), f"encodings should be list/tuple, got {type(encodings)}"
        expected_pre_modalities = self.dit.blocks[0].num_modalities - 1  # 追加 ego traj 之前应有的 modality 数
        assert len(encodings) == expected_pre_modalities, (
            f"encodings should have {expected_pre_modalities} elements before appending ego traj, got {len(encodings)}"
        )
        for i, enc in enumerate(encodings):
            assert isinstance(enc, torch.Tensor), f"encodings[{i}] should be tensor, got {type(enc)}"
            assert enc.dim() == 3, f"encodings[{i}] should be 3D (B, num_tokens, hidden_dim), got {enc.dim()}D, shape={enc.shape}"
            assert enc.shape[0] == B, f"encodings[{i}] batch size {enc.shape[0]} != B {B}"

        attn_dist = model_extra['token_dist']
        # 断言：检查 attn_dist
        assert isinstance(attn_dist, torch.Tensor), f"attn_dist should be tensor, got {type(attn_dist)}"
        assert attn_dist.shape[0] == B, f"attn_dist batch size {attn_dist.shape[0]} != B {B}"

        masks = list(model_extra['masks'])
        # 断言：检查 masks（与 encodings 对齐，进入 decoder 时应为 2 个 modality）
        assert isinstance(masks, (list, tuple)), f"masks should be list/tuple, got {type(masks)}"
        assert len(masks) == expected_pre_modalities, (
            f"masks should have {expected_pre_modalities} elements before appending ego traj, got {len(masks)}"
        )

        cfg_flags = model_extra['cfg_flags']
        # 确保 cfg_flags 是 tensor 并且在正确的设备上
        # 使用 cfg_embedding 的权重设备，确保与模型参数在同一设备上
        target_device = next(self.cfg_embedding.parameters()).device
        if not isinstance(cfg_flags, torch.Tensor):
            cfg_flags = torch.tensor(cfg_flags, device=target_device, dtype=torch.int32)
        elif cfg_flags.device != target_device:
            cfg_flags = cfg_flags.to(target_device)
        cfg_flags = cfg_flags.reshape(B) # (B,)
        cfg_embedding = self.cfg_embedding(cfg_flags).unsqueeze(-2) # (B, 1, hidden_dim)

        routes_cond = model_extra['routes_cond']

        x = self.preproj(x)
        encodings.append(x)
        masks.append(None)

        # 断言：append ego traj 后应与 DiT 的 modality 数一致
        assert len(encodings) == self.dit.blocks[0].num_modalities, (
            f"after appending ego traj, encodings should have {self.dit.blocks[0].num_modalities} modalities, got {len(encodings)}"
        )
        assert len(masks) == self.dit.blocks[0].num_modalities, (
            f"after appending ego traj, masks should have {self.dit.blocks[0].num_modalities} modalities, got {len(masks)}"
        )
        
        time_cond = self.t_embedder(t).unsqueeze(1)
        routes_cond = routes_cond.unsqueeze(1)        
        action_pe = self.action_pe.unsqueeze(0).repeat(B, 1, 1) # concate an extra dimension for path token

        y = time_cond + routes_cond + action_pe + cfg_embedding

        token_tuple = self.dit(
            modality_tokens=encodings, 
            modality_masks=masks, 
            time_cond=time_cond, 
            routes_cond=routes_cond, 
            action_encoding=action_pe, 
            cfg_embedding=cfg_embedding, 
            attn_dist=attn_dist
        )
        
        agents_token, lanes_token, x_token = token_tuple

        agents_token = self.agents_in_proj(agents_token)
        lanes_token = self.lane_in_proj(lanes_token)

        kv_token = torch.cat([agents_token, lanes_token], dim=1)
        key_mapping_mask = torch.cat([masks[i] for i in range(len(masks)-1)], dim=1) # here the valid mask is represented as True (1)

        x_token = self.post_fusion(x_token, kv_token, ~key_mapping_mask)

        prediction = self.final_layer(x_token, y)

        prediction = prediction.reshape(B, P, -1, self.planner_params['state_dim'])

        return prediction

class FlowPlannerDiTBlock(nn.Module):
    def __init__(
        self,
        *,
        dim_modalities: tuple[int, ...],
        dim_cond = None,
        dim_head = 64,
        heads = 8,
        enable_attn_dist = False,
        ff_kwargs: dict = dict(),
        token_num: int = 118,
    ):
        super().__init__()
        self.num_modalities = len(dim_modalities)
        self.dim_modalities = dim_modalities

        self.modalities_gate_proj = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(dim_cond, dim * 2)
            )
        for dim in dim_modalities])

        for layer in self.modalities_gate_proj:
            nn.init.zeros_(layer[-1].weight)
            nn.init.constant_(layer[-1].bias, 1.)

        self.attn_layernorms = nn.ModuleList([AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities])

        self.joint_attn = JointAttention(
            dim_inputs = dim_modalities,
            dim_head = dim_head,
            heads = heads,
            enable_attn_dist = enable_attn_dist,
            token_num=token_num
        )

        self.ff_layernorms = nn.ModuleList([AdaptiveLayerNorm(dim, dim_cond = dim_cond) for dim in dim_modalities])
        self.feedforwards = nn.ModuleList([FeedForward(dim, **ff_kwargs) for dim in dim_modalities])

    def forward(
        self,
        *,
        modality_tokens,
        modality_masks = None,
        modality_conds = None,
        attn_dist = None
    ):
        assert len(modality_tokens) == self.num_modalities
 
        attn_gammas = []
        ff_gammas = []
        for proj, cond in zip(self.modalities_gate_proj, modality_conds):
            gamma = proj(cond)
            attn_g, ff_g = gamma.chunk(2, dim=-1)
            attn_gammas.append(attn_g)
            ff_gammas.append(ff_g)
        
        modality_tokens_attn_res = [token.clone() for token in modality_tokens]
        modality_tokens = [ln(tokens, cond=ln_cond) for ln, tokens, ln_cond in zip(self.attn_layernorms, modality_tokens, modality_conds)]
        modality_tokens = self.joint_attn(inputs = modality_tokens, masks = modality_masks, attn_dist = attn_dist)
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, attn_gammas)]
        modality_tokens = [token + res for token, res in zip(modality_tokens, modality_tokens_attn_res)]

        modality_tokens_ffn_res = [token.clone() for token in modality_tokens]
        modality_tokens = [ln(tokens, cond=ln_cond) for ln, tokens, ln_cond in zip(self.ff_layernorms, modality_tokens, modality_conds)]
        modality_tokens = [ff(tokens) for tokens, ff in zip(modality_tokens, self.feedforwards)]
        modality_tokens = [tokens * gamma for tokens, gamma in zip(modality_tokens, ff_gammas)]
        modality_tokens = [token + res for token, res in zip(modality_tokens, modality_tokens_ffn_res)]

        return modality_tokens

class FlowPlannerDiT(nn.Module):
    def __init__(
        self,
        *,
        depth,
        dim_modalities,
        enable_attn_dist = False,
        **block_kwargs
    ):
        super().__init__()

        blocks = [FlowPlannerDiTBlock(dim_modalities = dim_modalities, enable_attn_dist = enable_attn_dist, **block_kwargs) for _ in range(depth)]
        self.blocks = nn.ModuleList(blocks)

        norms = [RMSNorm(dim) for dim in dim_modalities]
        self.norms = nn.ModuleList(norms)

    def forward(
        self,
        *,
        modality_tokens,
        modality_masks = None,
        time_cond = None,
        routes_cond = None,
        action_encoding = None,
        cfg_embedding = None,
        attn_dist = None,
    ):
        other_modality_conds = [time_cond + routes_cond] * (len(modality_tokens) - 1)
        ego_traj_conds = [time_cond + routes_cond + action_encoding + cfg_embedding]
        modality_conds = other_modality_conds + ego_traj_conds

        for block in self.blocks:
            modality_tokens = block(
                modality_tokens = modality_tokens,
                modality_masks = modality_masks,
                modality_conds = modality_conds,
                attn_dist = attn_dist
            )

        modality_tokens = [norm(tokens) for tokens, norm in zip(modality_tokens, self.norms)]
        
        return tuple(modality_tokens)