# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Implementation of Cross-Covariance Image Transformer (XCiT)
Based on timm and DeiT code bases
https://github.com/rwightman/pytorch-image-models/tree/master/timm
https://github.com/facebookresearch/deit/
"""
import math
import os
import sys
from functools import partial

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, _cfg

from pe_scale_scheduler import ScaleScheduler

# get current file path
file_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(file_dir, "truly_shift_invariant_cnns"))
from truly_shift_invariant_cnns.models.aps_models.apspool import ApsPool

from xcit import XCA, ClassAttentionBlock, PositionalEncodingFourier


def conv3x3(in_planes, out_planes, stride=1, pad_type="circular"):
    """
    3x3 convolution with padding
    SyncBatchNorm is used instead of BatchNorm2d
    """

    norm_layer = nn.SyncBatchNorm(out_planes)
    return torch.nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            padding_mode=pad_type,
        ),
        norm_layer,
    )


# class ConvPatchEmbedAPS(nn.Module):
#     """ Image to Patch Embedding using multiple convolutional layers
#     """

#     def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
#         super().__init__()
#         img_size = to_2tuple(img_size)
#         patch_size = to_2tuple(patch_size)
#         num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
#         self.img_size = img_size
#         self.patch_size = patch_size
#         self.num_patches = num_patches

#         if patch_size[0] == 16:
#             self.proj = torch.nn.Sequential(
#                 conv3x3(3, embed_dim // 8, stride=1, pad_type='circular'),
#                 nn.GELU(),
#                 ApsPool(channels=embed_dim // 8, pad_type='circular', filt_size=1, stride=2, apspool_criterion='l2', return_poly_indices = False),
#                 conv3x3(embed_dim // 8, embed_dim // 4, stride=1, pad_type='circular'),
#                 nn.GELU(),
#                 ApsPool(channels=embed_dim // 4, pad_type='circular', filt_size=1, stride=2, apspool_criterion='l2', return_poly_indices = False),
#                 conv3x3(embed_dim // 4, embed_dim // 2, stride=1, pad_type='circular'),
#                 nn.GELU(),
#                 ApsPool(channels=embed_dim // 2, pad_type='circular', filt_size=1, stride=2, apspool_criterion='l2', return_poly_indices = False),
#                 conv3x3(embed_dim // 2, embed_dim, stride=1, pad_type='circular'),
#             )

#         elif patch_size[0] == 8:
#             self.proj = torch.nn.Sequential(
#                 conv3x3(3, embed_dim // 4, stride=1, pad_type='circular'),
#                 nn.GELU(),
#                 ApsPool(channels=embed_dim // 4, pad_type='circular', filt_size=1, stride=2, apspool_criterion='l2', return_poly_indices = False),
#                 conv3x3(embed_dim // 4, embed_dim // 2, stride=1, pad_type='circular'),
#                 nn.GELU(),
#                 ApsPool(channels=embed_dim // 2, pad_type='circular', filt_size=1, stride=2, apspool_criterion='l2', return_poly_indices = False),
#                 conv3x3(embed_dim // 2, embed_dim, stride=1, pad_type='circular'),
#             )
#         else:
#             raise("For convolutional projection, patch size has to be in [8, 16]")

#     def forward(self, x, padding_size=None):
#         B, C, H, W = x.shape
#         x = self.proj(x)
#         Hp, Wp = x.shape[2], x.shape[3]
#         x = x.flatten(2).transpose(1, 2)

#         return x, (Hp, Wp)


class ConvPatchEmbedAPS(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        if patch_size[0] == 16:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 8, stride=1, pad_type="circular"),
                nn.GELU(),
                ApsPool(
                    channels=embed_dim // 8,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
                conv3x3(embed_dim // 8, embed_dim // 4, stride=1, pad_type="circular"),
                nn.GELU(),
                ApsPool(
                    channels=embed_dim // 4,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
                conv3x3(embed_dim // 4, embed_dim // 2, stride=1, pad_type="circular"),
                nn.GELU(),
                ApsPool(
                    channels=embed_dim // 2,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
                conv3x3(embed_dim // 2, embed_dim, stride=1, pad_type="circular"),
                ApsPool(
                    channels=embed_dim,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
            )

        elif patch_size[0] == 8:
            self.proj = torch.nn.Sequential(
                conv3x3(3, embed_dim // 4, stride=1, pad_type="circular"),
                nn.GELU(),
                ApsPool(
                    channels=embed_dim // 4,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
                conv3x3(embed_dim // 4, embed_dim // 2, stride=1, pad_type="circular"),
                nn.GELU(),
                ApsPool(
                    channels=embed_dim // 2,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
                conv3x3(embed_dim // 2, embed_dim, stride=1, pad_type="circular"),
                ApsPool(
                    channels=embed_dim,
                    pad_type="circular",
                    filt_size=1,
                    stride=2,
                    apspool_criterion="l2",
                    return_poly_indices=False,
                ),
            )
        else:
            raise ("For convolutional projection, patch size has to be in [8, 16]")

    def forward(self, x, padding_size=None):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)

        return x, (Hp, Wp)


class LPIAps(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        kernel_size=3,
    ):
        super().__init__()
        out_features = out_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
            padding_mode="circular",
        )
        self.act = act_layer()
        self.bn = nn.SyncBatchNorm(in_features)
        self.conv2 = torch.nn.Conv2d(
            in_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
            padding_mode="circular",
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class XCABlockAps(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        num_tokens=196,
        eta=None,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        self.norm3 = norm_layer(dim)
        self.local_mp = LPIAps(in_features=dim, act_layer=act_layer)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class XCiTAps(nn.Module):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=None,
        cls_attn_layers=2,
        use_pos=True,
        patch_proj="conv",
        eta=None,
        tokens_norm=False,
        learnable_pos_scale=False,
        **kwargs
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
            cls_attn_layers: (int) Depth of Class attention layers
            use_pos: (bool) whether to use positional encoding
            eta: (float) layerscale initialization value
            tokens_norm: (bool) Whether to normalize all tokens or just the cls_token in the CA
        """
        super().__init__()
        print("[XCiT] unsed kwargs: ", kwargs)
        self.learnable_pos_scale = learnable_pos_scale
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = ConvPatchEmbedAPS(
            img_size=img_size, embed_dim=embed_dim, patch_size=patch_size
        )

        num_patches = self.patch_embed.num_patches

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                XCABlockAps(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    num_tokens=num_patches,
                    eta=eta,
                )
                for i in range(depth)
            ]
        )

        self.features_type = kwargs.pop("features_type", "cls_attn")

        if self.features_type == "cls_attn":
            self.cls_attn_blocks = nn.ModuleList(
                [
                    ClassAttentionBlock(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        norm_layer=norm_layer,
                        eta=eta,
                        tokens_norm=tokens_norm,
                    )
                    for i in range(cls_attn_layers)
                ]
            )
            self.norm = norm_layer(embed_dim)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)

        elif self.features_type == "avgpool":
            self.norm = nn.Identity()

        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )

        self.use_pos = use_pos
        if self.use_pos:
            self.pos_embeder = PositionalEncodingFourier(dim=embed_dim)
            # Optional learnable scale for positional encoding
            if self.learnable_pos_scale:
                self.pos_scale = nn.Parameter(torch.tensor(1.0))
            else:
                self.pos_scale = torch.tensor(1.0)
            
            if kwargs.get("pos_decay", False):
                # Wrap the positional encoding module with a scale scheduler
                self.pos_scale_scheduler = ScaleScheduler(
                    total_steps=kwargs["pos_decay_steps"],
                    schedule_type=kwargs.get("pos_decay_type", "triangular"),
                )
            else:
                self.pos_scale_scheduler = None
            

        # Classifier head

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token"}

    def forward_features(self, x):
        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)

        pe_norm = torch.tensor(0.0, device=x.device)

        if self.use_pos:
            pos_encoding = (
                self.pos_embeder(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            )
            pe = self.pos_scale * pos_encoding
            pe_norm = torch.norm(pe.reshape(-1))

            if self.pos_scale_scheduler is not None:
                pe_schaed_scale = self.pos_scale_scheduler()
                x = x + pe_schaed_scale * pe
            else:
                x = x + pe
        
        else:
            self.pos_scale = None
        
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        if self.features_type == "cls_attn":
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for blk in self.cls_attn_blocks:
                x = blk(x, Hp, Wp)
            x = self.norm(x)[:, 0]

        elif self.features_type == "avgpool":
            x = self.norm(x)
            x = x.mean(dim=1)
        return x, pe_norm

    def forward(self, x):
        x, pe_norm = self.forward_features(x)
        x = self.head(x)

        if self.training:
            return x, pe_norm
        else:
            return x


# Patch size 16x16 models
@register_model
def xcit_aps_nano_12_p16(pretrained=False, **kwargs):
    model = XCiTAps(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_aps_nano_12_p16_avgpool(pretrained=False, **kwargs):
    model = XCiTAps(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        features_type="avgpool",
        **kwargs
    )
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_aps_small_12_p16_avgpool(pretrained=False, **kwargs):
    model = XCiTAps(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        features_type="avgpool",
        **kwargs
    )
    model.default_cfg = _cfg()
    return model

@register_model
def xcit_aps_small_12_p16(pretrained=False, **kwargs):
    model = XCiTAps(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        features_type="cls_attn",
        **kwargs
    )
    model.default_cfg = _cfg()
    return model
