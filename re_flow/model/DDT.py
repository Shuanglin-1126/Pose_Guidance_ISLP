# Copyright (c) 2025 Boyang Zheng
from math import sqrt
from typing import *
import torch
from torch import nn

from re_flow.model.GCN import GCNModel
from timm.models.vision_transformer import PatchEmbed, Mlp
from re_flow.model.model_utils import VisionRotaryEmbeddingFast, RMSNorm, SwiGLUFFN, GaussianFourierEmbedding, LabelEmbedder, NormAttention, get_2d_sincos_pos_embed



def DDTModulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        shift: Tensor of shape (B, L, D)
        scale: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * (1 + scale) + shift,
        with shift and scale repeated to match L_x if necessary.
    """
    x1, x2 = x[:, :prefix_tokens, :], x[:, prefix_tokens:, :]
    B, Lx, D = x2.shape
    _, L, _ = shift.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        shift = shift.repeat_interleave(repeat, dim=1)
        scale = scale.repeat_interleave(repeat, dim=1)
    # apply modulation
    return torch.cat((x1, x2 * (1 + scale) + shift), dim=1)


def DDTGate(x: torch.Tensor, gate: torch.Tensor, prefix_tokens: int) -> torch.Tensor:
    """
    Applies per-segment modulation to x.

    Args:
        x: Tensor of shape (B, L_x, D)
        gate: Tensor of shape (B, L, D)
    Returns:
        Tensor of shape (B, L_x, D): x * gate,
        with gate repeated to match L_x if necessary.
    """
    x1, x2 = x[:, :prefix_tokens, :], x[:, prefix_tokens:, :]
    B, Lx, D = x2.shape
    _, L, _ = gate.shape
    if Lx % L != 0:
        raise ValueError(f"L_x ({Lx}) must be divisible by L ({L})")
    repeat = Lx // L
    if repeat != 1:
        # repeat each of the L segments 'repeat' times along the length dim
        # print(f"gate shape: {gate.shape}, x shape: {x.shape}")
        gate = gate.repeat_interleave(repeat, dim=1)
    # apply modulation
    return torch.cat((x1, x2 * gate), dim=1)


class LightningDDTBlock(nn.Module):
    """
    Lightning DiT Block. We add features including:
    - ROPE
    - QKNorm
    - RMSNorm
    - SwiGLU
    - No shift AdaLN.
    Not all of them are used in the final model, please refer to the paper for more details.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        prefix_tokens=8,
        use_qknorm=False,
        use_swiglu=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_LORA=False,
        LORA_rank=8,
        LORA_alpha=1.,
        **block_kwargs
    ):
        super().__init__()

        # Initialize normalization layers
        self.prefix_tokens = prefix_tokens
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)

        # Initialize attention layer
        self.attn = NormAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            use_LORA=use_LORA,
            LORA_rank=LORA_rank,
            LORA_alpha=LORA_alpha,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        def approx_gelu(): return nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0,
            )

        # Initialize AdaLN modulation
        if wo_shift:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 4 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        self.wo_shift = wo_shift

    def forward(self, x, c, feat_rope=None):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)  # (B, 1, C)
        if self.wo_shift:
            scale_msa, gate_msa, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(4, dim=-1)
            shift_msa = None
            shift_mlp = None
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=-1)
        x = x + DDTGate(self.attn(DDTModulate(self.norm1(x),
                        shift_msa, scale_msa, self.prefix_tokens), rope=feat_rope), gate_msa, self.prefix_tokens)
        x = x + DDTGate(self.mlp(DDTModulate(self.norm2(x),
                        shift_mlp, scale_mlp, self.prefix_tokens)), gate_mlp, self.prefix_tokens)
        return x


class DDTFinalLayer(nn.Module):
    """
    The final layer of DDT.
    """

    def __init__(self, hidden_size, patch_size, out_channels, use_rmsnorm=False, prefix_tokens=8):
        super().__init__()
        if not use_rmsnorm:
            self.norm_final = nn.LayerNorm(
                hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(
            hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
        self.prefix_tokens = prefix_tokens

    def forward(self, x, c):
        if len(c.shape) < len(x.shape):
            c = c.unsqueeze(1)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = DDTModulate(self.norm_final(x), shift, scale, self.prefix_tokens)  # no gate
        x = self.linear(x)
        return x


class DiTwDDTHead(nn.Module):
    def __init__(
            self,
            input_size: int = 1,
            patch_size: Union[list, int] = 1,
            in_channels: int = 768,
            hidden_size=[1152, 2048],
            depth=[28, 2],
            prefix_tokens: int = 8,
            num_heads: Union[list[int], int] = [16, 16],
            mlp_ratio=4.0,
            use_qknorm=False,
            use_swiglu=True,
            use_rope=True,
            use_rmsnorm=True,
            use_LORA=False,
            LORA_rank=8,
            LORA_alpha=1.,
            wo_shift=False,
            use_pos_embed: bool = True,
            GCN_cfg: dict = (),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.encoder_hidden_size = hidden_size[0]
        self.decoder_hidden_size = hidden_size[1]
        self.num_heads = [num_heads, num_heads] if isinstance(
            num_heads, int) else list(num_heads)
        self.num_decoder_blocks = depth[1]
        self.num_encoder_blocks = depth[0]
        self.num_blocks = depth[0] + depth[1]
        self.use_rope = use_rope
        # analyze patch size
        if isinstance(patch_size, int) or isinstance(patch_size, float):
            patch_size = [patch_size, patch_size]  # patch size for s , x embed
        assert len(
            patch_size) == 2, f"patch size should be a list of two numbers, but got {patch_size}"
        self.patch_size = patch_size
        self.s_patch_size = patch_size[0]
        self.x_patch_size = patch_size[1]
        s_channel_per_token = in_channels * self.s_patch_size * self.s_patch_size
        s_input_size = input_size
        s_patch_size = self.s_patch_size
        x_input_size = input_size
        x_patch_size = self.x_patch_size
        x_channel_per_token = in_channels * self.x_patch_size * self.x_patch_size
        self.x_embedder = PatchEmbed(
            x_input_size, x_patch_size, x_channel_per_token, self.decoder_hidden_size, bias=True)
        self.s_embedder = PatchEmbed(
            s_input_size, s_patch_size, s_channel_per_token, self.encoder_hidden_size, bias=True)
        self.s_channel_per_token = s_channel_per_token
        self.x_channel_per_token = x_channel_per_token
        self.s_projector = nn.Linear(
            self.encoder_hidden_size, self.decoder_hidden_size) if self.encoder_hidden_size != self.decoder_hidden_size else nn.Identity()
        self.t_embedder = GaussianFourierEmbedding(self.encoder_hidden_size)
        GCN_cfg['hidden_size'] = self.encoder_hidden_size
        self.y_embedder = GCNModel(**GCN_cfg)

        self.final_layer = DDTFinalLayer(
            self.decoder_hidden_size, 1, x_channel_per_token, use_rmsnorm=use_rmsnorm)
        # Will use fixed sin-cos embedding:
        if use_pos_embed:
            num_patches = self.s_embedder.num_patches
            self.pos_embed = nn.Parameter(torch.zeros(
                1, num_patches, self.encoder_hidden_size), requires_grad=False)
            self.x_pos_embed = None
        self.use_pos_embed = use_pos_embed
        enc_num_heads = self.num_heads[0]
        dec_num_heads = self.num_heads[1]
        # use rotary position encoding, borrow from EVA
        if self.use_rope:
            enc_half_head_dim = self.encoder_hidden_size // enc_num_heads // 2
            hw_seq_len = int(sqrt(self.s_embedder.num_patches))
            # print(f"enc_half_head_dim: {enc_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.enc_feat_rope = VisionRotaryEmbeddingFast(
                dim=enc_half_head_dim,
                pt_seq_len=hw_seq_len,
                prefix_tokens=prefix_tokens,
            )
            dec_half_head_dim = self.decoder_hidden_size // dec_num_heads // 2
            hw_seq_len = int(sqrt(self.x_embedder.num_patches))
            # print(f"dec_half_head_dim: {dec_half_head_dim}, hw_seq_len: {hw_seq_len}")
            self.dec_feat_rope = VisionRotaryEmbeddingFast(
                dim=dec_half_head_dim,
                pt_seq_len=hw_seq_len,
                prefix_tokens=prefix_tokens,
            )
        else:
            self.feat_rope = None
        self.blocks = nn.ModuleList([
            LightningDDTBlock(self.encoder_hidden_size if i < self.num_encoder_blocks else self.decoder_hidden_size,
                              enc_num_heads if i < self.num_encoder_blocks else dec_num_heads,
                              mlp_ratio=mlp_ratio,
                              prefix_tokens=prefix_tokens,
                              use_qknorm=use_qknorm,
                              use_rmsnorm=use_rmsnorm,
                              use_swiglu=use_swiglu,
                              wo_shift=wo_shift,
                              use_LORA=use_LORA,
                              LORA_rank=LORA_rank,
                              LORA_alpha=LORA_alpha
                              ) for i in range(self.num_blocks)
        ])

        self.prefix_tokens = prefix_tokens
        self.reference_pool = nn.AdaptiveAvgPool1d(prefix_tokens)
        self.initialize_weights()

        if use_LORA:
            # for name, param in self.s_embedder.named_parameters():
            #     param.requires_grad = False
            # for name, param in self.x_embedder.named_parameters():
            #     param.requires_grad = False
            # for name, param in self.t_embedder.named_parameters():
            #     param.requires_grad = False
            for name, param in self.blocks.named_parameters():
                if 'mlp' in name:
                    param.requires_grad = False


    def initialize_weights(self, xavier_uniform_init: bool = False):
        if xavier_uniform_init:
            def _basic_init(module):
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)
            self.apply(_basic_init)
        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)

        if self.use_pos_embed:
            # Initialize (and freeze) pos_embed by sin-cos embedding:
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1], int(self.s_embedder.num_patches ** 0.5))
            self.pos_embed.data.copy_(
                torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Zero-out adaLN modulation layers in LightningDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        # c = self.out_channels
        c = self.x_channel_per_token
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, joint, reference_image, s=None, is_sample=False):
        # x = self.x_embedder(x) + self.pos_embed
        t = self.t_embedder(t)
        if is_sample:
            y1 = self.y_embedder(x=joint, is_train=self.training, wo_condition=False)
            y2 = self.y_embedder(x=joint, is_train=self.training, wo_condition=True)
            y = torch.cat((y1, y2), dim=0)
        else:
            y = self.y_embedder(joint, self.training)
        c = nn.functional.silu(t + y)
        reference = self.s_embedder(reference_image).detach()  # B L C
        reference = self.reference_pool(reference.permute(0, 2, 1)).permute(0, 2, 1)  # B prefix_tokens C
        if s is None:
            s = self.s_embedder(x)
            if self.use_pos_embed:
                s = s + self.pos_embed
            # print(f"t shape: {t.shape}, y shape: {y.shape}, c shape: {c.shape}, s shape: {s.shape}, pos_embed shape: {self.pos_embed.shape}")
            s = torch.cat((reference, s), dim=1)  # B (prefix_tokens + L) C
            for i in range(self.num_encoder_blocks):
                s = self.blocks[i](s, c, feat_rope=self.enc_feat_rope)
            # broadcast t to s
            s = s[:, self.prefix_tokens:, :]  # remove prefix tokens before broadcasting
            t = t.unsqueeze(1).repeat(1, s.shape[1], 1)
            s = nn.functional.silu(t + s)

        s = self.s_projector(s)
        x = self.x_embedder(x)
        reference = self.x_embedder(reference_image).detach()  # B L C
        reference = self.reference_pool(reference.permute(0, 2, 1)).permute(0, 2, 1)  # B prefix_tokens C
        if self.use_pos_embed and self.x_pos_embed is not None:
            x = x + self.x_pos_embed
        x = torch.cat((reference, x), dim=1)
        for i in range(self.num_encoder_blocks, self.num_blocks):
            x = self.blocks[i](x, s, feat_rope=self.dec_feat_rope)
        x = self.final_layer(x, s)
        x = self.unpatchify(x[:, self.prefix_tokens:, :])
        return x

    def forward_with_cfg(self, x, t, joint, reference_image, cfg_scale, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, joint, reference_image, None, True)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:,
                              :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        t = t[: len(t) // 2] # get t for the conditional half
        half_eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(cond_eps.shape) - 1)),
            uncond_eps + cfg_scale * (cond_eps - uncond_eps), cond_eps
        )
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def forward_with_autoguidance(self, x, t, y, cfg_scale, additional_model_forward, cfg_interval=(0, 1)):
        """
        Forward pass of LightningDiT, but also contain the forward pass for the additional model
        """
        model_out = self.forward(x, t, y)
        ag_model_out = additional_model_forward(x, t, y)
        eps = model_out[:, :self.in_channels]
        ag_eps = ag_model_out[:, :self.in_channels]

        guid_t_min, guid_t_max = cfg_interval
        assert guid_t_min < guid_t_max, "cfg_interval should be (min, max) with min < max"
        eps = torch.where(
            ((t >= guid_t_min) & (t <= guid_t_max)
             ).view(-1, *[1] * (len(eps.shape) - 1)),
            ag_eps + cfg_scale * (eps - ag_eps), eps
        )

        return eps


if __name__ == '__main__':
    from src.utils.train_utils import parse_configs
    import torch
    from src.utils.model_utils import instantiate_from_config
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


    def get_diffusion_model(args) -> object:
        model = instantiate_from_config(args)
        return model

    (
        rae_config,
        model_config,
        dataset_config,
        transport_config,
        sampler_config,
        guidance_config,
        misc_config,
        training_config,
    ) = parse_configs(r'F:\chexiao\project\RAE\re_flow\config\DiDH_XL_DINOv2_B.yaml')

    device = 'cuda:0'
    model = get_diffusion_model(model_config).to(device)
    x = torch.randn(2, 768, 16, 16).to(device)
    t = torch.randn(2).to(device)
    joint = torch.randn(2, 121, 3).to(device)
    reference_image = torch.randn(2, 768, 16, 16).to(device)

    y = model(x, t, joint, reference_image)

    print(y.size())
