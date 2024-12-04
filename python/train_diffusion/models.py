# ruff: noqa
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from mamba_ssm import Mamba2


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c
        ).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class PolicyHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # カメラ操作（連続値）用の出力層
        self.camera_mean = nn.Linear(hidden_dim, 2)
        self.camera_log_std = nn.Parameter(torch.zeros(2))

        # ボタン操作（離散値）用の出力層
        self.button_logits = nn.Linear(hidden_dim, 22)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """方策の計算と行動のサンプリング"""
        features = self.net(x)

        # カメラ操作の確率分布
        camera_mean = torch.tanh(self.camera_mean(features))
        camera_std = self.camera_log_std.exp()
        camera_dist = Normal(camera_mean, camera_std)

        # ボタン操作の確率分布
        button_logits = self.button_logits(features)
        button_dist = Bernoulli(logits=button_logits)

        # 行動のサンプリング
        camera_action = camera_dist.sample() / 20
        camera_action[:, 0] /= 10  # pitch方向は抑える
        button_action = button_dist.sample()
        button_action[:, 0] = 1.0  # "attack"は常に実行
        button_action[:, 13] = 0.0  # "inventory"は常に実行

        # 行動の対数確率を計算
        camera_log_prob = camera_dist.log_prob(camera_action).sum(-1)
        button_log_prob = button_dist.log_prob(button_action).sum(-1)
        log_prob = camera_log_prob + button_log_prob

        # エントロピー計算（探索の度合いを調整するために使用）
        entropy = camera_dist.entropy().sum(-1) + button_dist.entropy().sum(-1)

        # 行動を結合
        action = torch.cat([camera_action, button_action], dim=-1)

        return action, log_prob, entropy

    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """行動に対する確率の評価"""
        features = self.net(x)

        # カメラ操作の確率分布
        camera_mean = torch.tanh(self.camera_mean(features))
        camera_std = self.camera_log_std.exp()
        camera_dist = Normal(camera_mean, camera_std)

        # ボタン操作の確率分布
        button_logits = self.button_logits(features)
        button_dist = Bernoulli(logits=button_logits)

        # 行動を分割
        camera_action, button_action = action.split([2, 22], dim=-1)

        # 行動の対数確率を計算
        camera_log_prob = camera_dist.log_prob(camera_action).sum(-1)
        button_log_prob = button_dist.log_prob(button_action).sum(-1)
        log_prob = camera_log_prob + button_log_prob

        # エントロピー計算（探索の度合いを調整するために使用）
        entropy = camera_dist.entropy().sum(-1) + button_dist.entropy().sum(-1)

        return log_prob, entropy


class ValueHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.dt_embedder = TimestepEmbedder(hidden_size)
        self.action_embedder = nn.Linear(24, hidden_size)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.mamba = Mamba2(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=hidden_size,  # Model dimension d_model
            d_state=64,  # SSM state expansion factor, typically 64 or 128
            d_conv=4,  # Local convolution width
            expand=4,  # Block expansion factor
        )

        self.blocks = nn.ModuleList(
            [DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        self.policy_head = PolicyHead(hidden_size, hidden_size)
        self.value_head = ValueHead(hidden_size, hidden_size)

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5)
        )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.action_embedder.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.dt_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.dt_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

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
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, cond_image, cond_action):
        """
        Forward pass of DiT.
        x: (N, T_pred, C, H, W) tensor of spatial inputs (latent representations of images)
        t: (N,) tensor of diffusion timesteps
        cond_image: (N, T_cond, C, H, W) tensor of spatial inputs (latent representations of images)
        cond_action: (N, T_cond, 24) tensor of class labels
        """
        N, T_pred, C, H, W = x.shape
        image = x  # (N, T_pred, C, H, W)
        image = image.reshape(N * T_pred, C, H, W)
        image = (
            self.x_embedder(image) + self.pos_embed
        )  # (N * T_pred, L, D), where L = H * W / patch_size ** 2
        L, D = image.shape[1:3]
        image = image.reshape(N, T_pred * L, D)  # (N, T_pred * L, D)
        x = image

        last = self.extract_features(cond_image, cond_action)  # (N, D)

        t = self.t_embedder(t)  # (N, D)
        c = t + last  # (N, D)
        for block in self.blocks:
            x = block(x, c)  # (N, T_pred * (L + 1), D)
        x = x[:, 0 : (T_pred * L)]  # (N, T_pred * L, D)
        x = self.final_layer(x, c)  # (N, T_pred * L, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N * T_pred, out_channels, H, W)
        x = x.reshape(N, T_pred, self.out_channels, H, W)  # (N, T_pred, out_channels, H, W)
        return x

    def extract_features(self, image, action):
        """
        Compress the sequence of images and actions into a single feature vector.
        image: (N, T_cond, C, H, W) tensor of spatial inputs (latent representations of images)
        action: (N, T_cond, 24) tensor of class labels
        """
        # image
        N, T, C, H, W = image.shape
        image = image.reshape(N * T, C, H, W)
        image = (
            self.x_embedder(image) + self.pos_embed
        )  # (N * T, L, D), where L = H * W / patch_size ** 2
        L, D = image.shape[1:3]
        image = image.reshape(N, T, L, D)
        image = image.mean(dim=2)  # (N, T, D)

        # action
        action = self.action_embedder(action)  # (N, T, D)

        # feature
        seq_feature = image + action  # (N, T, D)
        seq_feature = self.mamba(seq_feature)  # (N, T, D)
        last = seq_feature[:, -1]  # (N, D)
        return last

    def allocate_inference_cache(self, batch_size, dtype=None):
        conv_state, ssm_state = self.mamba.allocate_inference_cache(
            batch_size, max_seqlen=0, dtype=dtype
        )
        feature = torch.zeros(batch_size, self.mamba.d_model, dtype=dtype, device=conv_state.device)
        return feature, conv_state, ssm_state

    def step(self, curr_image, curr_action, conv_state, ssm_state):
        """
        Update the model state for a single timestep.
        curr_image: (1, C, H, W) tensor of spatial inputs (latent representations of images)
        curr_action: (1, 24) tensor of class labels
        conv_state, ssm_state: mamaba states
        """
        N, C, H, W = curr_image.shape
        assert N == 1

        curr_image = (
            self.x_embedder(curr_image) + self.pos_embed
        )  # (1, L, D), where L = H * W / patch_size ** 2
        curr_image = curr_image.mean(dim=1)  # (1, D)

        curr_action = self.action_embedder(curr_action)  # (1, D)

        curr_cond = curr_image + curr_action  # (1, D)
        curr_cond = curr_cond.unsqueeze(1)  # (1, 1, D)

        return self.mamba.step(curr_cond, conv_state, ssm_state)

    def predict(self, x, t, dt, feature):
        """
        Predict image by using the current feature.
        x: (1, C, H, W) tensor of spatial inputs (latent representations of images)
        t: (1,) tensor of diffusion timesteps
        feature: (1, D) tensor of feature
        """
        x = self.x_embedder(x) + self.pos_embed  # (1, L, D), where L = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (1, D)
        dt = self.dt_embedder(dt)  # (1, D)
        feature = feature.squeeze(1)  # (1, D)
        c = t + dt + feature  # (1, D)
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        x = self.unpatchify(x)  # (1, out_channels, H, W)
        return x

    def policy(self, feature):
        return self.policy_head(feature)

    def evaluate(self, feature, action):
        return self.policy_head.evaluate(feature, action)

    def value(self, feature):
        return self.value_head(feature)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
