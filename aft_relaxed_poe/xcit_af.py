# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# Implementation of Cross-Covariance Image Transformer (XCiT)
# Based on timm and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
###############################################################################
###############################################################################

import json
import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import Mlp, _cfg

from af_ops import (
    DownSampleAF,
    LayerNormAF,
    LPFPolyActPerChannel,
    PolyActPerChannel,
    UpAct,
    UpActConv,
)
from pe_scale_scheduler import ScaleScheduler
from xcit import XCA, ClassAttentionBlock, ConvPatchEmbed, PositionalEncodingFourier


class XCiTAFConfig:
    def __init__(self, args):
        """
        Initialize the configuration from an argparse.Namespace.
        Args:
            args (argparse.Namespace): Command line arguments containing the configuration.
        """
        # Copy all the attributes from the Namespace to this config object.
        self.__dict__.update(vars(args))

        # Ensure norm_layer is set, if it was not provided on the command line.
        if not hasattr(self, "norm_layer") or self.norm_layer is None:
            self.norm_layer = partial(nn.LayerNorm, eps=1e-6)

        # If any additional adjustments or default fallbacks are needed, add them here.
        # For example, if the patch_embed type is not specified, you might want to set a default:
        if not hasattr(self, "patch_embed") or self.patch_embed is None:
            self.patch_embed = "conv"

        # parse kwargs strings
        if hasattr(self, "pe_act_kwargs") and type(self.pe_act_kwargs) == str:
            self.pe_act_kwargs = json.loads(self.pe_act_kwargs)
        if hasattr(self, "mlp_act_kwargs") and type(self.mlp_act_kwargs) == str:
            self.mlp_act_kwargs = json.loads(self.mlp_act_kwargs)
        if hasattr(self, "lpi_act_kwargs") and type(self.lpi_act_kwargs) == str:
            self.lpi_act_kwargs = json.loads(self.lpi_act_kwargs)

        if (
            hasattr(self, "pe_first_act_kwargs")
            and type(self.pe_first_act_kwargs) == str
        ):
            self.pe_first_act_kwargs = json.loads(self.pe_first_act_kwargs)

        # Set default value for positional constant bias
        if not hasattr(self, "pos_const_bias") or self.pos_const_bias is None:
            self.pos_const_bias = False

        # Set default bias value if not provided
        if not hasattr(self, "pos_bias_value") or self.pos_bias_value is None:
            self.pos_bias_value = 0.05

        # Set default values for positional encoding decay
        if not hasattr(self, "pos_decay") or self.pos_decay is None:
            self.pos_decay = False

        # Create decay configuration if decay is enabled
        if self.pos_decay:
            self.pos_decay_config = {
                "initial_scale": 1.0,
                "final_scale": getattr(self, "pos_decay_final_scale", 0.0),
                "decay_type": getattr(self, "pos_decay_type", "linear"),
                "total_steps": getattr(self, "pos_decay_steps", 10000),
                "warmup_steps": getattr(self, "pos_decay_warmup", 0),
            }

            # Add optional parameters if provided
            if hasattr(self, "pos_decay_gamma") and self.pos_decay_gamma is not None:
                self.pos_decay_config["gamma"] = self.pos_decay_gamma

            if (
                hasattr(self, "pos_decay_step_size")
                and self.pos_decay_step_size is not None
            ):
                self.pos_decay_config["step_size"] = self.pos_decay_step_size
        else:
            self.pos_decay_config = None

        print("[XCiTAFConfig] Configuration: ", self.__dict__)


def conv3x3(in_planes, out_planes, stride=1, padding_mode="circular"):
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
            padding_mode=padding_mode,
        ),
        norm_layer,
    )


def get_activation(
    layer: str,
    channels: Optional[int] = None,
    data_format: Optional[str] = None,
    up: Optional[int] = 2,
    down: Optional[int] = 2,
    **kwargs,
):
    if layer == "gelu":
        act = nn.GELU()
    elif layer == "up_gelu":
        act = UpAct(act_layer=nn.GELU(), up=up, down=down, data_format=data_format)
    elif layer == "poly":
        act = PolyActPerChannel(channels, data_format=data_format)
    elif layer == "up_poly":
        assert channels is not None, "Channels must be provided for PolyAct"
        act = UpAct(
            act_layer=PolyActPerChannel(channels, **kwargs),
            up=up,
            down=down,
            data_format=data_format,
        )
    elif layer == "lpf_poly":
        assert channels is not None, "Channels must be provided for LPF PolyAct"
        act = LPFPolyActPerChannel(channels, **kwargs)
    elif layer == "up_c_gelu":
        act = UpActConv(act_layer=nn.GELU(), up=up, down=down, data_format=data_format)
    else:
        raise ValueError(f"Activation layer {layer} not supported")
    return act


def get_norm_layer(norm_layer, dim):
    if norm_layer == "layer":
        return nn.LayerNorm(dim, eps=1e-6)
    elif norm_layer == "layer_af":
        return LayerNormAF(dim, affine=True)

    else:
        assert False, f"Unknown norm_layer {norm_layer}"


class ConvPatchEmbedAF(nn.Module):
    """Image to Patch Embedding using multiple convolutional layers
    This class transforms images into patch embeddings using a series of convolutional layers.
    The architecture dynamically adjusts based on the patch size and configuration parameters.
    Args:
        img_size (int): Size of the input image (will be converted to a square if a single int is provided)
        patch_size (int): Size of the patches to embed (supported values: 8, 16)
        in_chans (int): Number of input channels
        embed_dim (int): Dimension of the output embeddings
        cfg (XCiTAFConfig): Configuration object containing parameters for:
            - pe_down_af: Controls stride and downsampling
            - pe_act: Activation function type
            - pe_act_kwargs: Additional arguments for the activation function
            - pe_fuse_down: Whether to fuse downsampling with activation
            - conv_padding_mode: Padding mode for convolutions
    Attributes:
        img_size (tuple): Input image dimensions (height, width)
        patch_size (tuple): Patch dimensions (height, width)
        num_patches (int): Total number of patches
        proj (nn.Sequential): Sequential container of convolutional layers and activations
    Returns:
        In forward pass:
            - x (torch.Tensor): Patch embeddings with shape (batch_size, num_patches, embed_dim)
            - (Hp, Wp) (tuple): Height and width of the patch grid

    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_chans: int,
        embed_dim: int,
        cfg: XCiTAFConfig,
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        assert patch_size[0] in [
            8,
            16,
        ], "af patch embedding only supports patch sizes 8, 16"

        if patch_size[0] == 16:
            dims = [embed_dim // 8, embed_dim // 4, embed_dim // 2, embed_dim]
        elif patch_size[0] == 8:
            dims = [embed_dim // 4, embed_dim // 2, embed_dim]

        else:
            raise ("For convolutional projection, patch size has to be in [8, 16]")

        # Set stride based on downsampling configuration
        stride = 1 if cfg.pe_down_af else 2

        # Configure activation upsampling
        act_up = 2 if cfg.pe_act.startswith("up_") else None

        # Configure activation downsampling
        act_down = None
        if act_up is not None and cfg.pe_down_af and cfg.pe_fuse_down:
            act_down = act_up * 2
        layers = []
        in_dim = in_chans
        for i, dim in enumerate(dims):
            layers.append(
                conv3x3(in_dim, dim, stride, padding_mode=cfg.conv_padding_mode)
            )
            if i == 0 and cfg.pe_first_act is not None:
                # First activation layer may be different
                layers.append(
                    get_activation(
                        layer=cfg.pe_first_act,
                        data_format="channels_first",
                        channels=dim,
                        up=act_up,
                        down=act_down,
                        **cfg.pe_first_act_kwargs,
                    )
                )
                layers.append(DownSampleAF(down=2))
            elif i == len(dims) - 1 and not cfg.pe_last_act:
                # Last activation layer is not used
                layers.append(DownSampleAF(down=2))
            else:
                layers.append(
                    get_activation(
                        layer=cfg.pe_act,
                        data_format="channels_first",
                        channels=dim,
                        up=act_up,
                        down=act_down,
                        **cfg.pe_act_kwargs,
                    )
                )
                if cfg.pe_down_af and not cfg.pe_fuse_down:
                    layers.append(DownSampleAF(down=2))
            in_dim = dim

        self.proj = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


# Modified from timm/models/layers/mlp.py
class MlpAF(nn.Module):
    """
    MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(
        self,
        in_features: int,
        cfg: XCiTAFConfig,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop=0.0,
        act_layer=None,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        if act_layer is None:
            act_layer = cfg.mlp_act
        self.act = get_activation(
            layer=act_layer,
            channels=hidden_features,
            data_format="tokens",
            **cfg.mlp_act_kwargs,
        )
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H=None, W=None):
        x = self.fc1(x)
        if getattr(self.act, "requires_input_dimension", False):
            x = self.act(x, H, W)
        else:
            x = self.act(x)

        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LPIAF(nn.Module):
    """
    Local Patch Interaction module that allows explicit communication between tokens in 3x3 windows
    to augment the implicit communcation performed by the block diagonal scatter attention.
    Implemented using 2 layers of separable 3x3 convolutions with GeLU and BatchNorm2d
    """

    def __init__(
        self,
        in_features,
        cfg: XCiTAFConfig,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
        kernel_size: int = 3,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        padding = kernel_size // 2

        self.conv1 = torch.nn.Conv2d(
            in_features,
            hidden_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=hidden_features,
            padding_mode=cfg.conv_padding_mode,
        )
        self.act = get_activation(
            layer=cfg.lpi_act,
            channels=hidden_features,
            data_format="channels_first",
            **cfg.lpi_act_kwargs,
        )
        self.norm = nn.SyncBatchNorm(hidden_features)
        self.conv2 = torch.nn.Conv2d(
            hidden_features,
            out_features,
            kernel_size=kernel_size,
            padding=padding,
            groups=out_features,
            padding_mode=cfg.conv_padding_mode,
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = x.reshape(B, C, N).permute(0, 2, 1)

        return x


class XCABlockAF(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias,
        qk_scale,
        drop,
        attn_drop,
        drop_path,
        eta,
        cfg: XCiTAFConfig,
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm1 = get_norm_layer(norm_layer=cfg.xca_norm_layer, dim=dim)
        self.attn = XCA(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        # self.norm2 = norm_layer(dim)
        self.norm2 = get_norm_layer(norm_layer=cfg.xca_norm_layer, dim=dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpAF(
            in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, cfg=cfg
        )

        # self.norm3 = norm_layer(dim)
        self.norm3 = get_norm_layer(norm_layer=cfg.xca_norm_layer, dim=dim)
        self.local_mp = LPIAF(in_features=dim, cfg=cfg)

        self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        self.gamma3 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma3 * self.local_mp(self.norm3(x), H, W))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x), H, W))
        return x


class ClassAttentionBlockAF(nn.Module):
    """
    Class Attention Block with XCA attention layer and AF activation function
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: int,
        qkv_bias: bool,
        qk_scale: bool,
        drop: float,
        attn_drop: float,
        eta: float,
        tokens_norm,
        cfg: XCiTAFConfig,
    ):
        super().__init__()
        # self.norm1 = norm_layer(dim)
        self.norm1 = get_norm_layer(norm_layer=cfg.xca_norm_layer, dim=dim)

        self.attn = XCA(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # self.norm2 = norm_layer(dim)
        self.norm2 = get_norm_layer(norm_layer=cfg.xca_norm_layer, dim=dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # force the activation layer to be gelu since only affects the cls token
        act_layer = nn.GELU
        # self.mlp = MlpAF(in_features=dim, hidden_features=mlp_hidden_dim,
        #                drop=drop, cfg=cfg, act_layer=act_layer)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if eta is not None:  # LayerScale Initialization (no layerscale when None)
            self.gamma1 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(eta * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1, self.gamma2 = 1.0, 1.0

        # FIXME: A hack for models pre-trained with layernorm over all the tokens not just the CLS
        self.tokens_norm = tokens_norm

    def forward(self, x, H, W, mask=None):
        # x = x + self.drop_path(self.gamma1 * self.attn(self.norm1(x)))
        x = x + self.gamma1 * self.attn(self.norm1(x))
        if self.tokens_norm:
            x = self.norm2(x)
        else:
            x[:, 0:1] = self.norm2(x[:, 0:1])

        x_res = x
        cls_token = x[:, 0:1]
        cls_token = self.gamma2 * self.mlp(cls_token)
        x = torch.cat([cls_token, x[:, 1:]], dim=1)
        # x = x_res + self.drop_path(x)
        x = x_res + x
        return x


class XCiTAF(nn.Module):
    """
    Based on timm and DeiT code bases
    https://github.com/rwightman/pytorch-image-models/tree/master/timm
    https://github.com/facebookresearch/deit/
    """

    def __init__(
        self,
        cfg: Optional[XCiTAFConfig] = None,
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
        patch_proj="linear",
        eta=None,
        tokens_norm=False,
        learnable_pos_scale: bool = False,
        **kwargs,
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
        print("[XCiTAF] unsed kwargs: ", kwargs)
        if cfg is not None:
            print("[XCiTAF] Configuration: ", cfg.__dict__)
        self.cfg = cfg
        # Prefer explicit flag, fallback to cfg if provided
        self.learnable_pos_scale = learnable_pos_scale
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = ConvPatchEmbedAF(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            cfg=cfg,
        )

        num_patches = self.patch_embed.num_patches  # noqa: F841 - kept for reference

        self.use_pos = use_pos
        if self.use_pos:
            # Use standard Fourier positional encoding
            pos_module = PositionalEncodingFourier(dim=embed_dim)
            self.pos_module = pos_module
            # Optional learnable scale for positional encoding
            if self.learnable_pos_scale:
                self.pos_scale = nn.Parameter(torch.tensor(1.0))
            else:
                self.pos_scale = torch.tensor(1.0)

            if kwargs.get("pos_const_bias", False):
                # Add a constant bias to the positional encoding
                self.register_buffer(
                    "pos_const_bias",
                    torch.full(
                        (1, 1, embed_dim), float(kwargs.get("pos_bias_value", 0.05))
                    ),
                )

            if kwargs["pos_decay"]:
                # Wrap the positional encoding module with a scale scheduler
                self.pos_scale_scheduler = ScaleScheduler(
                    total_steps=kwargs["pos_decay_steps"],
                    schedule_type=kwargs.get("pos_decay_type", "triangular"),
                )
            else:
                self.pos_scale_scheduler = None

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [drop_path_rate for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                XCABlockAF(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    eta=eta,
                    cfg=cfg,
                )
                for i in range(depth)
            ]
        )

        if cfg.features_type == "cls_attn":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)
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

        elif cfg.features_type == "cls_attn_af":
            # raise NotImplementedError("ClassAttentionBlockAF not implemented yet")
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=0.02)
            self.cls_attn_blocks = nn.ModuleList(
                [
                    ClassAttentionBlockAF(
                        dim=embed_dim,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop=drop_rate,
                        attn_drop=attn_drop_rate,
                        eta=eta,
                        tokens_norm=tokens_norm,
                        cfg=cfg,
                    )
                    for _ in range(cls_attn_layers)
                ]
            )
            self.norm = norm_layer(embed_dim)

        elif cfg.features_type == "avgpool":
            self.norm = nn.Identity()

        # Classifier head
        self.head = (
            nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        # Initialize positional encoding scale with ones
        if hasattr(self, "pos_scale") and isinstance(self.pos_scale, nn.Parameter):
            nn.init.constant_(self.pos_scale, 1.0)
        # ConstBiasLayer is already initialized in its constructor

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "pos_scale", "bias"}

    def forward_features(self, x):
        B, C, H, W = x.shape

        x, (Hp, Wp) = self.patch_embed(x)
        pe_norm = torch.tensor(0.0, device=x.device)

        if self.use_pos:
            pos_encoding = (
                self.pos_module(B, Hp, Wp).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            )
            pe = self.pos_scale * pos_encoding
            pe_norm = torch.norm(pe.reshape(-1))

            if self.pos_scale_scheduler is not None:
                pe_schaed_scale = self.pos_scale_scheduler()
                x = x + pe_schaed_scale * pe
            else:
                x = x + pe

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, Hp, Wp)

        if (
            self.cfg.features_type == "cls_attn"
            or self.cfg.features_type == "cls_attn_af"
        ):
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

            for blk in self.cls_attn_blocks:
                x = blk(x, Hp, Wp)

            x = self.norm(x)[:, 0]

        elif self.cfg.features_type == "avgpool":
            x = self.norm(x)
            x = x.mean(dim=1)
        return x, pe_norm

    def forward(self, x):
        x, pe_norm = self.forward_features(x)
        x = self.head(x)

        if self.training:
            return (x, x), pe_norm
        else:
            return x


# Patch size 16x16 models
# Patch size 16x16 models


@register_model
def xcit_af_nano_12_p16(pretrained=False, **kwargs):
    model = XCiTAF(
        patch_size=16,
        embed_dim=128,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


# Patch size 16x16 models


@register_model
def xcit_af_tiny_12_p16(pretrained=False, **kwargs):
    model = XCiTAF(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


@register_model
def xcit_af_small_12_p16(pretrained=False, **kwargs):
    model = XCiTAF(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=8,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model


# Patch size 8x8 models


@register_model
def xcit_af_nano_12_p8(pretrained=False, **kwargs):
    model = XCiTAF(
        patch_size=8,
        embed_dim=128,
        depth=12,
        num_heads=4,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        eta=1.0,
        tokens_norm=True,
        **kwargs,
    )
    model.default_cfg = _cfg()
    return model
