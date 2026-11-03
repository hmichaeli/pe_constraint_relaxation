# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# Modifications:
# - Added Alias-Free Transformer (AFT) configuration flags and positional encoding options.
# - Integrated robustness evaluation tasks (adversarial crop, cyclic, fractional shifts).
# - Updated training loop for alias-free activations, mixed precision, and wandb logging.
"""Training and evaluation script for Alias-Free Transformer models.

This module extends the original XCiT training pipeline with Alias-Free components,
additional positional encoding controls, and robustness evaluation entry points.
"""
import argparse
import datetime
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, NativeScaler, get_state_dict

import utils

# import available models for registration
import xcit
import xcit_af
from adversarial_shift_eval import (
    evaluate_adversarial_bilinear_fractional_shift,
    evaluate_adversarial_crop_shift,
    evaluate_adversarial_cyclic_shift,
)
from datasets import build_dataset
from engine import evaluate, train_benchmark, train_one_epoch
from losses import DistillationLoss
from samplers import RASampler
from shift_eval import evaluate_consistency_metrics

try:
    import pretrained_models
    import xcit_aps

except ImportError:
    print(
        "xcit_aps and / or other pretrained models not available. Use git submodule update --init --recursive"
    )

try:
    import wandb
except ImportError:
    wandb = None
    print(
        "To use the Weights & Biases logger please install wandb. "
        "Run `pip install wandb` to install it (or remove this line)."
    )


def get_args_parser():
    """Create the command-line argument parser used across training tasks."""

    parser = argparse.ArgumentParser(
        "XCiT training and evaluation script", add_help=False
    )
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--epochs", default=400, type=int)
    parser.add_argument(
        "--accumulation-steps",
        default=1,
        type=int,
        help="Gradient accumulation steps (default: 1)",
    )

    # Model parameters
    parser.add_argument(
        "--model",
        default="xcit_s_12",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input-size", default=224, type=int, help="images input size")

    parser.add_argument(
        "--drop",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Dropout rate (default: 0.)",
    )
    parser.add_argument(
        "--drop-path",
        type=float,
        default=0.0,
        metavar="PCT",
        help="Drop path rate (default: 0.0)",
    )

    parser.add_argument("--model-ema", action="store_true")
    parser.add_argument("--no-model-ema", action="store_false", dest="model_ema")
    parser.set_defaults(model_ema=False)
    parser.add_argument("--model-ema-decay", type=float, default=0.99996, help="")
    parser.add_argument(
        "--model-ema-force-cpu", action="store_true", default=False, help=""
    )

    # Optimizer parameters
    parser.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw"',
    )
    parser.add_argument(
        "--opt-eps",
        default=1e-8,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: 1e-8)",
    )
    parser.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=None,
        metavar="NORM",
        help="Clip gradient norm (default: None, no clipping)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="SGD momentum (default: 0.9)",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    # Learning rate schedule parameters
    parser.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine"',
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        metavar="LR",
        help="learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--lr-noise",
        type=float,
        nargs="+",
        default=None,
        metavar="pct, pct",
        help="learning rate noise on/off epoch percentages",
    )
    parser.add_argument(
        "--lr-noise-pct",
        type=float,
        default=0.67,
        metavar="PERCENT",
        help="learning rate noise limit percent (default: 0.67)",
    )
    parser.add_argument(
        "--lr-noise-std",
        type=float,
        default=1.0,
        metavar="STDDEV",
        help="learning rate noise std-dev (default: 1.0)",
    )
    parser.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        metavar="LR",
        help="lower lr bound for cyclic schedulers that hit 0 (1e-5)",
    )

    parser.add_argument(
        "--decay-epochs",
        type=float,
        default=30,
        metavar="N",
        help="epoch interval to decay LR",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    parser.add_argument(
        "--cooldown-epochs",
        type=int,
        default=10,
        metavar="N",
        help="epochs to cooldown LR at min_lr, after cyclic schedule ends",
    )
    parser.add_argument(
        "--patience-epochs",
        type=int,
        default=10,
        metavar="N",
        help="patience epochs for Plateau LR scheduler (default: 10",
    )
    parser.add_argument(
        "--decay-rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )

    # Augmentation parameters
    parser.add_argument(
        "--color-jitter",
        type=float,
        default=0.4,
        metavar="PCT",
        help="Color jitter factor (default: 0.4)",
    )
    parser.add_argument(
        "--aa",
        type=str,
        default="rand-m9-mstd0.5-inc1",
        metavar="NAME",
        help=(
            'AutoAugment policy to apply. Options: "v0" or "original". '
            "(default: rand-m9-mstd0.5-inc1)"
        ),
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.1, help="Label smoothing (default: 0.1)"
    )
    parser.add_argument(
        "--train-interpolation",
        type=str,
        default="bicubic",
        help='Training interpolation (random, bilinear, bicubic default: "bicubic")',
    )

    parser.add_argument("--repeated-aug", action="store_true")
    parser.add_argument("--no-repeated-aug", action="store_false", dest="repeated_aug")
    parser.set_defaults(repeated_aug=True)

    # * Random Erase params
    parser.add_argument(
        "--reprob",
        type=float,
        default=0.25,
        metavar="PCT",
        help="Random erase prob (default: 0.25)",
    )
    parser.add_argument(
        "--remode",
        type=str,
        default="pixel",
        help='Random erase mode (default: "pixel")',
    )
    parser.add_argument(
        "--recount", type=int, default=1, help="Random erase count (default: 1)"
    )
    parser.add_argument(
        "--resplit",
        action="store_true",
        default=False,
        help="Do not random erase first (clean) augmentation split",
    )

    # * Mixup params
    parser.add_argument(
        "--mixup",
        type=float,
        default=0.8,
        help="mixup alpha, mixup enabled if > 0. (default: 0.8)",
    )
    parser.add_argument(
        "--cutmix",
        type=float,
        default=1.0,
        help="cutmix alpha, cutmix enabled if > 0. (default: 1.0)",
    )
    parser.add_argument(
        "--cutmix-minmax",
        type=float,
        nargs="+",
        default=None,
        help="cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)",
    )
    parser.add_argument(
        "--mixup-prob",
        type=float,
        default=1.0,
        help="Probability of performing mixup or cutmix when either/both is enabled",
    )
    parser.add_argument(
        "--mixup-switch-prob",
        type=float,
        default=0.5,
        help="Probability of switching to cutmix when both mixup and cutmix enabled",
    )
    parser.add_argument(
        "--mixup-mode",
        type=str,
        default="batch",
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"',
    )

    # Distillation parameters
    parser.add_argument(
        "--teacher-model",
        default="regnety_160",
        type=str,
        metavar="MODEL",
        help='Name of teacher model to train (default: "regnety_160"',
    )
    parser.add_argument("--teacher-path", type=str, default="")
    parser.add_argument(
        "--distillation-type",
        default="none",
        choices=["none", "soft", "hard"],
        type=str,
        help="",
    )
    parser.add_argument("--distillation-alpha", default=0.5, type=float, help="")
    parser.add_argument("--distillation-tau", default=1.0, type=float, help="")

    # Dataset parameters
    parser.add_argument(
        "--data-path", default="/Datasets/imagenet/", type=str, help="dataset path"
    )
    parser.add_argument(
        "--data-set",
        default="IMNET",
        choices=[
            "CIFAR10",
            "CIFAR100",
            "CIFAR100_SUBSET",
            "IMNET",
            "INAT",
            "INAT19",
            "CARS",
            "FLOWERS",
            "IMNET22k",
            "TINY_IMNET",
            "StanfordCars",
            "Flowers102",
        ],
        type=str,
        help="Image Net dataset path",
    )
    parser.add_argument(
        "--dataset-seed",
        default=0,
        type=int,
        help="Seed for dataset sampling (for CIFAR100_SUBSET only)",
    )
    parser.add_argument(
        "--samples-per-class",
        default=None,
        type=int,
        help="Number of samples per class (for CIFAR100_SUBSET only)",
    )
    parser.add_argument(
        "--inat-category",
        default="name",
        choices=[
            "kingdom",
            "phylum",
            "class",
            "order",
            "supercategory",
            "family",
            "genus",
            "name",
        ],
        type=str,
        help="semantic granularity",
    )

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--dist-eval",
        action="store_true",
        default=False,
        help="Enabling distributed evaluation",
    )
    parser.add_argument("--num_workers", default=10, type=int)
    parser.add_argument(
        "--pin-mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no-pin-mem", action="store_false", dest="pin_mem", help="")
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--test-freq",
        default=1,
        type=int,
        help="Number of epochs between validation runs.",
    )

    parser.add_argument(
        "--full_crop",
        action="store_true",
        help="Use crop_ratio=1.0 instead of the default 0.875 (as in CaiT).",
    )
    parser.add_argument("--no_full_crop", action="store_false", dest="full_crop")
    parser.set_defaults(full_crop=True)

    parser.add_argument(
        "--pretrained", default=None, type=str, help="Path to pre-trained checkpoint"
    )
    parser.add_argument(
        "--surgery",
        default=None,
        type=str,
        help=(
            "Path to checkpoint to copy the patch projection from. "
            "Can improve stability for very large models."
        ),
    )

    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=False,
        help="Use Automatic Mixed Precision",
    )
    parser.add_argument(
        "--save-every-epoch",
        type=int,
        default=None,
        help="Save checkpoint every n epochs",
    )

    # Wandb arguments
    parser.add_argument(
        "--wandb_project",
        default=None,
        type=str,
        help="Name of the W&B project for logging (optional).",
    )
    parser.add_argument(
        "--wandb_group", default=None, type=str, help="Group name for W&B"
    )
    parser.add_argument("--wandb_name", default=None, type=str, help="Run name for W&B")
    # XCiT AF config
    parser.add_argument(
        "--use-pos",
        type=utils.str2bool,
        default=True,
        help="Enable positional encoding.",
    )

    parser.add_argument(
        "--pe-down-af",
        type=utils.str2bool,
        default=True,
        help="Enable alias-free downsampling in the convolutional positional embedding.",
    )
    parser.add_argument(
        "--pe-fuse-down",
        type=utils.str2bool,
        default=True,
        help="Fuse alias-free downsampling with the UpACT downsample.",
    )
    parser.add_argument(
        "--pe-act",
        type=str,
        default="gelu",
        help="Activation type for the patch embedding (ConvPatchEmbedAF).",
    )
    parser.add_argument("--pe-act-kwargs", type=str, default="{}")

    parser.add_argument(
        "--pe-first-act",
        type=str,
        default=None,
        help="Override the first activation layer in ConvPatchEmbedAF (defaults to --pe-act).",
    )
    parser.add_argument("--pe-first-act-kwargs", type=str, default="{}")

    parser.add_argument(
        "--pe-last-act",
        action="store_true",
        help="Use a final activation layer in ConvPatchEmbedAF (defaults to True).",
    )
    parser.add_argument("--no-pe-last-act", action="store_false", dest="pe_last_act")
    parser.set_defaults(pe_last_act=True)

    parser.add_argument(
        "--pe-scale-mu",
        type=float,
        default=None,
        help="Scale mu for ConvPatchEmbedAF.",
    )
    parser.add_argument(
        "--pe-scale-sigma",
        type=float,
        default=None,
        help="Scale sigma for ConvPatchEmbedAF.",
    )
    parser.add_argument(
        "--learnable-pos-scale",
        type=utils.str2bool,
        default=False,
        help="Use learnable scaling weights for positional encoding.",
    )
    parser.add_argument(
        "--pos-const-bias",
        type=utils.str2bool,
        default=False,
        help="Use a constant bias instead of positional encoding.",
    )
    parser.add_argument(
        "--pos-bias-value",
        type=float,
        default=0.05,
        help="Value of the constant bias when positional bias is enabled.",
    )

    # Gradual decay of positional encoding
    parser.add_argument(
        "--pos-decay",
        action="store_true",
        help="Enable triangular schedule for positional encodings (0->1->0)",
    )
    parser.add_argument(
        "--pos-decay-steps",
        type=int,
        default=0,
        help="Total steps S for the triangle (peak at S/2)",
    )
    parser.add_argument(
        "--pos-decay-type",
        type=str,
        default="triangular",
        choices=["triangular", "linear"],
        help="Type of decay schedule for positional encoding",
    )
    parser.add_argument(
        "--pos-reg-lambda",
        type=float,
        default=0.0,
        help="Coefficient for L2 norm of learnable-scaled positional encoding",
    )
    # parser.add_argument(
    #     "--pos-decay",
    #     type=utils.str2bool,
    #     default=False,
    #     help="Gradually decay positional encoding during training.",
    # )
    # parser.add_argument(
    #     "--pos-decay-type",
    #     type=str,
    #     default="linear",
    #     choices=["linear", "exponential", "cosine", "step"],
    #     help="Decay schedule type for positional encoding.",
    # )
    # parser.add_argument(
    #     "--pos-decay-steps",
    #     type=int,
    #     default=10000,
    #     help="Total steps for positional encoding decay.",
    # )
    # parser.add_argument(
    #     "--pos-decay-final-scale",
    #     type=float,
    #     default=0.0,
    #     help="Final positional encoding scale.",
    # )
    # parser.add_argument(
    #     "--pos-decay-warmup",
    #     type=int,
    #     default=0,
    #     help="Warmup steps before decay starts.",
    # )
    # parser.add_argument(
    #     "--pos-decay-gamma",
    #     type=float,
    #     default=None,
    #     help="Decay factor for exponential or step schedules.",
    # )
    # parser.add_argument(
    #     "--pos-decay-step-size",
    #     type=int,
    #     default=None,
    #     help="Steps between decay events for step schedule.",
    # )

    parser.add_argument(
        "--conv-padding-mode",
        default="circular",
        help="Padding mode to use in all XCiT convolutions.",
    )

    parser.add_argument(
        "--xca-norm-layer", default="layer", help="Norm layer to use in XCiT XCABlocks"
    )
    parser.add_argument(
        "--mlp-act",
        default="gelu",
        type=str,
        choices=["gelu", "poly", "up_poly", "up_gelu", "up_c_gelu"],
        help="Activation used in the MLP blocks.",
    )
    parser.add_argument("--mlp-act-kwargs", type=str, default="{}")

    parser.add_argument(
        "--lpi-act",
        default="gelu",
        type=str,
        choices=["gelu", "poly", "up_poly", "up_gelu", "up_c_gelu"],
        help="Activation used in the local positional interaction (LPI).",
    )
    parser.add_argument("--lpi-act-kwargs", type=str, default="{}")

    parser.add_argument(
        "--features-type",
        default="cls_token",
        type=str,
        choices=["cls_token", "avgpool", "cls_attn", "cls_attn_af"],
        help="Which features to return for downstream evaluation.",
    )

    parser.add_argument(
        "--distilled-pe",
        default=None,
        type=str,
        help="Path to pre-trained patch embedding checkpoint",
    )
    parser.add_argument(
        "--distilled-pe-freeze",
        action="store_true",
        help="Freeze the patch embedding weights",
    )

    # Adversarial shift eval
    parser.add_argument(
        "--task",
        default="train",
        type=str,
        choices=[
            "train",
            "eval",
            "adversarial_crop_shift",
            "adversarial_cyclic_shift",
            "adversarial_bilinear_fractional_shift",
            "train_benchmark",
        ],
        help="Primary task to run.",
    )
    parser.add_argument(
        "--max-shift", default=32, type=int, help="Maximal shift to evaluate"
    )
    parser.add_argument(
        "--crop-input-size",
        default=224,
        type=int,
        help="Cropped input size in crop-shift eval",
    )
    parser.add_argument(
        "--upsample",
        default=1,
        type=int,
        help=(
            "Image upsample factor used when evaluating fractional shifts. "
            "For example, with upsample=2 and max_shift=2 the function will "
            "evaluate shifts of [0.5, 1] pixels."
        ),
    )

    return parser


def main(args):
    """Run training or evaluation according to parsed command-line arguments."""

    utils.init_distributed_mode(args)
    global_rank = utils.get_rank()

    use_wandb = (
        wandb is not None and args.wandb_project is not None and global_rank == 0
    )
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            group=args.wandb_group,
            name=args.wandb_name,
            config=args,
        )

    else:
        wandb_logger = None

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    print("dataset_train:", dataset_train)
    print("dataset_val:", dataset_val)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print(
                    "Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.nb_classes,
        )

    print(f"Creating model: {args.model}")

    model_af_cfg = None
    if args.model.startswith("xcit_af"):
        model_af_cfg = xcit_af.XCiTAFConfig(args)

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.nb_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        use_pos=args.use_pos,
        learnable_pos_scale=args.learnable_pos_scale,
        cfg=model_af_cfg,
        pos_decay=args.pos_decay,
        pos_decay_steps=args.pos_decay_steps,
        pos_decay_type=args.pos_decay_type,
    )
    print("model:\n", model)

    if args.pretrained:
        if args.pretrained.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.pretrained, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(
                args.pretrained, map_location="cpu", weights_only=False
            )

        checkpoint_model = checkpoint["model"]
        state_dict = model.state_dict()
        for k in ["head.weight", "head.bias"]:
            if (
                k in checkpoint_model
                and checkpoint_model[k].shape != state_dict[k].shape
            ):
                # Remove head weights if has different shape
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # model.load_state_dict(checkpoint_model, strict=True)
        mismatch = model.load_state_dict(checkpoint_model, strict=False)
        if mismatch.missing_keys:
            print("Warning: Missing keys in model state_dict:", mismatch.missing_keys)

    model.to(device)

    if args.surgery:
        print("Loading pretrained Patch-embedding from ", args.surgery)
        checkpoint = torch.load(args.surgery, map_location="cpu")
        checkpoint_model = checkpoint["model"]
        patch_embed_weights = {
            key.replace("patch_embed.", ""): value
            for key, value in checkpoint["model"].items()
            if "patch_embed" in key
        }

        model.patch_embed.load_state_dict(patch_embed_weights)
        for p in model.patch_embed.parameters():
            p.requires_grad = False

    if args.distilled_pe is not None:
        print("Loading pretrained Patch-embedding from ", args.distilled_pe)
        pe_sd = torch.load(args.distilled_pe, map_location="cpu", weights_only=False)
        # remove 'module.' prefix
        pe_sd = {k.replace("module.", ""): v for k, v in pe_sd.items()}
        mismatch = model.patch_embed.load_state_dict(pe_sd, strict=False)
        print("Mismatched keys:", mismatch)

        if args.distilled_pe_freeze:
            print("Freezing patch embedding weights")
            for p in model.patch_embed.parameters():
                p.requires_grad = False

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device="cpu" if args.model_ema_force_cpu else "",
            resume="",
        )

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_scheduler, _ = create_scheduler(args, optimizer)

    criterion = LabelSmoothingCrossEntropy()

    if args.mixup > 0.0:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    print("Optimizer: ", optimizer)
    print("Scheduler: ", lr_scheduler)
    print("Criterion: ", criterion)

    teacher_model = None
    if args.distillation_type != "none":
        assert args.teacher_path, "need to specify teacher-path when using distillation"
        print(f"Creating teacher model: {args.teacher_model}")
        teacher_model = create_model(
            args.teacher_model,
            pretrained=False,
            num_classes=args.nb_classes,
            global_pool="avg",
        )
        if args.teacher_path.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.teacher_path, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.teacher_path, map_location="cpu")

        teacher_model.load_state_dict(checkpoint["model"])

        teacher_model.to(device)
        teacher_model.eval()

    # wrap the criterion in our custom DistillationLoss, which
    # just dispatches to the original criterion if args.distillation_type is 'none'
    criterion = DistillationLoss(
        criterion,
        teacher_model,
        args.distillation_type,
        args.distillation_alpha,
        args.distillation_tau,
    )

    output_dir = Path(args.output_dir)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    resume_path = os.path.join(output_dir, "checkpoint.pth")
    if args.resume and os.path.exists(resume_path):
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            print("Loading from checkpoint ...")
            checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1
            if args.model_ema:
                utils._load_checkpoint_for_ema(model_ema, checkpoint["model_ema"])
            if "scaler" in checkpoint:
                loss_scaler.load_state_dict(checkpoint["scaler"])

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)

        shift_stats = evaluate_consistency_metrics(
            data_loader_val,
            model,
            device,
            use_amp=args.use_amp,
            max_shift=32,
            shift=None,
        )

        log_stats = {
            **{f"test_{k}": v for k, v in test_stats.items()},
            **{f"shift_consistency/{k}": v for k, v in shift_stats.items()},
        }
        if use_wandb:
            wandb.log(log_stats)
        return

    if args.task == "train_benchmark":
        print("Running training benchmark...")
        train_benchmark(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            args.start_epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            surgery=args.surgery,
            use_amp=args.use_amp,
            num_iterations=100,
        )
        return

    if args.task == "adversarial_crop_shift":
        test_stats = evaluate_adversarial_crop_shift(
            data_loader_val,
            model,
            device,
            inp_size=args.crop_input_size,
            max_shift=args.max_shift,
            use_amp=False,
        )
        print("adversarial_crop_shift results:\n", test_stats)
        if use_wandb:
            wandb.log(test_stats)
        return

    if args.task == "adversarial_cyclic_shift":
        test_stats = evaluate_adversarial_cyclic_shift(
            data_loader_val, model, device, args.max_shift, args.upsample, use_amp=False
        )
        print("adversarial_cyclic_shift results:\n", test_stats)
        if use_wandb:
            wandb.log(test_stats)
        return

    if args.task == "adversarial_bilinear_fractional_shift":
        test_stats = evaluate_adversarial_bilinear_fractional_shift(
            data_loader=data_loader_val,
            model=model,
            device=device,
            inp_size=args.crop_input_size,
            max_shift=args.max_shift,
            use_amp=False,
        )
        print("adversarial_bilinear_fractional_shift results:\n", test_stats)
        if use_wandb:
            wandb.log(test_stats)
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            model_ema,
            mixup_fn,
            surgery=args.surgery,
            use_amp=args.use_amp,
            pos_reg_lambda=args.pos_reg_lambda,
            accumulation_steps=args.accumulation_steps,
        )

        lr_scheduler.step(epoch)
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            if args.save_every_epoch is not None and (
                epoch % args.save_every_epoch == 0
            ):
                checkpoint_paths.append(output_dir / f"checkpoint_{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": (
                            get_state_dict(model_ema) if model_ema is not None else None
                        ),
                        "scaler": loss_scaler.state_dict(),
                        "args": args,
                    },
                    checkpoint_path,
                )

        if (epoch % args.test_freq == 0) or (epoch == args.epochs - 1):
            test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)

            if test_stats["acc1"] >= max_accuracy:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "model_ema": (
                            get_state_dict(model_ema) if model_ema is not None else None
                        ),
                        "args": args,
                    },
                    os.path.join(output_dir, "best_model.pth"),
                )

            print(
                f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%"
            )
            max_accuracy = max(max_accuracy, test_stats["acc1"])
            print(f"Max accuracy: {max_accuracy:.2f}%")

            log_stats = {
                **{f"train/{k}": v for k, v in train_stats.items()},
                **{f"test/{k}": v for k, v in test_stats.items()},
                "max_acc1": max_accuracy,
                "epoch": epoch,
                "n_parameters": n_parameters,
            }

            if (epoch % 10 == 0) or epoch == args.epochs - 1:
                # evaluate shift consistency
                shift_stats = evaluate_consistency_metrics(
                    data_loader_val,
                    model,
                    device,
                    use_amp=args.use_amp,
                    max_shift=32,
                    shift=None,
                )
                log_stats.update(
                    {f"shift_consistency/{k}": v for k, v in shift_stats.items()}
                )

            # Log pos decay status
            if args.pos_decay:
                log_stats.update(
                    {
                        "pos_learnable_scale": model_without_ddp.pos_scale.item(),
                        "pos_mix_coef_scheduler": model_without_ddp.pos_scale_scheduler.current_scale.item(),
                    }
                )

            print(log_stats)
            if use_wandb:
                wandb.log(log_stats)

            if args.output_dir and utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "XCiT training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
