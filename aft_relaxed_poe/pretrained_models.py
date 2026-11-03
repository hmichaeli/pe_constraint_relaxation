import argparse
import torch
import torchvision
from timm.models.registry import register_model

import xcit
import xcit_aps
import xcit_af
from xcit_af import XCiTAFConfig


@register_model
def swin_tiny_pretrained(**kwargs):
    model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
    return model


@register_model
def vit_base_pretrained(**kwargs):
    model = torchvision.models.vit_b_16(
        weights=torchvision.models.ViT_B_16_Weights.DEFAULT
    )
    return model


@register_model
def cvt_13(**kwargs):
    try:
        import config as config_lib

        import CvT.tools._init_paths
        from CvT.lib.models.build import build_model
    except ImportError:
        print("CvT not available. Use git submodule update --init --recursive")
        raise ImportError
    cfg_file = "./CvT/experiments/imagenet/cvt/cvt-13-224x224.yaml"
    config_lib._update_config_from_file(config_lib.config, cfg_file)
    model = build_model(config_lib.config)
    return model


# ------------------------------------------------------------------
# Our trained models from huggingface
# ------------------------------------------------------------------
xcit_af_config = dict(
        conv_padding_mode="circular",
        pe_down_af=True,
        pe_fuse_down=True,
        pe_first_act=None,
        pe_first_act_kwargs={},
        pe_last_act=True,
        pe_act="up_gelu",
        pe_act_kwargs={},
        xca_norm_layer="layer_af",
        mlp_act="up_gelu",
        mlp_act_kwargs={},
        lpi_act="up_gelu",
        lpi_act_kwargs={},
        features_type="cls_attn_af",
    )

@register_model
def xcit_af_nano_12_p16_pretrained(**kwargs):
    """
    Loads the official pretrained Alias-Free XCiT Nano model from HuggingFace.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub to load pretrained weights.")

    # Arguments for XCiTAFConfig (positional, norm, activation, etc.)
    # Fixed architecture arguments (no user override to preserve pretrained weights)
    # Note: patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, eta, tokens_norm
    # are hardcoded in xcit_af_nano_12_p16 function, so we don't pass them here
    model_args = dict(
        num_classes=kwargs.get('num_classes', 1000),
        use_pos=False
    )
    cfg = XCiTAFConfig(argparse.Namespace(**xcit_af_config))
    model = xcit_af.xcit_af_nano_12_p16(cfg=cfg, **model_args)
    checkpoint_path = hf_hub_download(
        "hmichaeli/alias_free_vit", "xcit_af_nano_12_p16.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model


@register_model
def xcit_af_small_12_p16_pretrained(**kwargs):
    """
    Loads the official pretrained Alias-Free XCiT Small model from HuggingFace.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub to load pretrained weights.")

    model_args = dict(
        num_classes=kwargs.get('num_classes', 1000),
        use_pos=False,
        drop_path_rate=0.05,
    )
    cfg = XCiTAFConfig(argparse.Namespace(**xcit_af_config))
    model = xcit_af.xcit_af_small_12_p16(cfg=cfg, **model_args)
    checkpoint_path = hf_hub_download(
        "hmichaeli/alias_free_vit", "xcit_af_small_12_p16.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def xcit_aps_nano_12_p16_pretrained(**kwargs):
    """
    Loads the official pretrained Alias-Free XCiT Nano model with APS from HuggingFace.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub to load pretrained weights.")

    # Arguments for XCiTAFConfig (positional, norm, activation, etc.)
    # Fixed architecture arguments (no user override to preserve pretrained weights)
    # Note: patch_size, embed_dim, depth, num_heads, mlp_ratio, qkv_bias, eta, tokens_norm
    # are hardcoded in xcit_af_nano_12_p16 function, so we don't pass them here
    model_args = dict(
        num_classes=kwargs.get('num_classes', 1000),
        use_pos=False
    )
    model = xcit_aps.xcit_aps_nano_12_p16(**model_args)
    checkpoint_path = hf_hub_download(
        "hmichaeli/alias_free_vit", "xcit_aps_nano_12_p16.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def xcit_aps_small_12_p16_pretrained(**kwargs):
    """
    Loads the official pretrained Alias-Free XCiT Small model with APS from HuggingFace.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub to load pretrained weights.")

    model_args = dict(
        num_classes=kwargs.get('num_classes', 1000),
        use_pos=False,
        drop_path_rate=0.05,
    )
    model = xcit_aps.xcit_aps_small_12_p16(**model_args)
    checkpoint_path = hf_hub_download(
        "hmichaeli/alias_free_vit", "xcit_aps_small_12_p16.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def xcit_nano_12_p16_pretrained(**kwargs):
    """
    Loads the official pretrained XCiT Nano model from HuggingFace.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub to load pretrained weights.")

    model_args = dict(
        num_classes=kwargs.get('num_classes', 1000),
        use_pos=True
    )
    model = xcit.xcit_nano_12_p16(**model_args)
    checkpoint_path = hf_hub_download(
        "hmichaeli/alias_free_vit", "xcit_nano_12_p16.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model

@register_model
def xcit_small_12_p16_pretrained(**kwargs):
    """
    Loads the official pretrained XCiT Small model from HuggingFace.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError("Please install huggingface_hub to load pretrained weights.")

    model_args = dict(
        num_classes=kwargs.get('num_classes', 1000),
        use_pos=True,
        drop_path_rate=0.05,
    )
    model = xcit.xcit_small_12_p16(**model_args)
    checkpoint_path = hf_hub_download(
        "hmichaeli/alias_free_vit", "xcit_small_12_p16.pth"
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    return model