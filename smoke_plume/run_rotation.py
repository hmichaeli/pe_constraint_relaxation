#!/usr/bin/env python3
"""Train and evaluate E2CNN, E2CNN + PE and RSteer on the PhiFlow rotation smoke plume.

Follows the data splits and training protocol of Wang et al. (ICML 2022).
`--model Equiv` builds E2CNN_PE: with `--strict_equiv` the PE is never used and the
network is the plain E2CNN; otherwise the PE is added to the input, scaled by the
`--mix_coef_schedule` coefficient, and regularized with `--pe_regularization_lambda`.
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data

from models import E2CNN_PE, Relaxed_Rot_SteerConvNet
from utils import Dataset, get_lr, train_epoch


TRAIN_TASKS = [
    (27, 2), (33, 0), (3, 2), (28, 3), (9, 0),
    (12, 3), (22, 1), (8, 3), (30, 1), (25, 0),
    (16, 3), (11, 2), (23, 2), (29, 0), (36, 3),
    (26, 1), (1, 0), (35, 2), (19, 2), (34, 1),
    (4, 3), (2, 1), (7, 2), (31, 2), (17, 0),
]
DOMAIN_TASKS = [
    (6, 1), (14, 1), (15, 2), (10, 1), (18, 1),
    (20, 3), (24, 3), (13, 0), (21, 0), (5, 0),
]


def parse_args():
    repo_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Rotation smoke-plume training with optional W&B logging"
    )
    parser.add_argument(
        "--model",
        choices=("RSteer", "Equiv"),
        required=True,
        help="RSteer (Wang et al.) or Equiv (E2CNN with optional PE).",
    )
    parser.add_argument(
        "--data_dir",
        default=str(repo_dir / "PhiFlow" / "Rotation"),
    )
    parser.add_argument(
        "--output_dir",
        default=str(repo_dir / "results" / "rotation"),
    )
    parser.add_argument("--hidden_dim", type=int, required=True)
    parser.add_argument("--num_layers", type=int, default=5)
    parser.add_argument("--input_length", type=int, default=10)
    parser.add_argument("--out_length", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epoch", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--decay_rate", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=4e-4)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--group_order", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--schedule_max_epoch",
        type=int,
        default=None,
        help=(
            "Epoch index at which the PE schedule reaches 0 and stays there; "
            "defaults to --num_epoch when unset."
        ),
    )
    parser.add_argument(
        "--strict_equiv",
        action="store_true",
        help="Disable PE injection and train/evaluate the exact equivariant path.",
    )
    parser.add_argument(
        "--mix_coef_schedule",
        choices=("none", "linear", "triangular"),
        default="triangular",
        help="Annealing schedule for PE mixing.",
    )
    parser.add_argument(
        "--pe_regularization_lambda",
        type=float,
        default=0.01,
        help="PE magnitude regularization strength (beta).",
    )
    parser.add_argument(
        "--pe_scale_init",
        type=float,
        default=1.0,
        help="Initial scalar applied to the Fourier PE.",
    )
    parser.add_argument(
        "--restore_best_validation",
        action="store_true",
        help=(
            "RSteer only: evaluate the lowest-validation checkpoint instead of the "
            "final early-stopped model (the behavior of the official run_model.py). "
            "Equiv runs always evaluate their best-validation checkpoints."
        ),
    )
    parser.add_argument(
        "--wandb_mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument(
        "--wandb_project",
        default="pe-smoke-plume-rotation",
    )
    parser.add_argument("--wandb_entity", default="")
    parser.add_argument("--wandb_group", default="")
    parser.add_argument("--wandb_run_name", default="")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def schedule_value(name, epoch, schedule_max_epoch):
    if name == "none":
        return 1.0
    if schedule_max_epoch <= 1:
        progress = 1.0 if epoch > 0 else 0.0
    else:
        progress = epoch / float(schedule_max_epoch - 1)
    progress = min(max(progress, 0.0), 1.0)
    if name == "linear":
        return max(1.0 - progress, 0.0)
    if name == "triangular":
        return max(1.0 - abs(2.0 * progress - 1.0), 0.0)
    raise ValueError(f"Unknown PE schedule: {name}")


def evaluate_rmse(loader, model, loss_function, device, forward_kwargs=None):
    """Match official eval_epoch's batch-mean RMSE without retaining predictions."""
    forward_kwargs = forward_kwargs or {}
    batch_mse = []
    with torch.no_grad():
        for xx, yy in loader:
            xx = xx.to(device)
            yy = yy.to(device)
            loss = 0.0
            for target in yy.transpose(0, 1):
                prediction = model(xx, **forward_kwargs) if forward_kwargs else model(xx)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                xx = torch.cat(
                    [xx[:, prediction.shape[1]:], prediction],
                    dim=1,
                )
                loss += loss_function(prediction, target)
            batch_mse.append(loss.item() / yy.shape[1])
    return round(float(np.sqrt(np.mean(batch_mse))), 5)


def train_epoch_pe(
    train_loader,
    model,
    optimizer,
    loss_function,
    device,
    mix_coef,
    equiv,
    pe_regularization_lambda,
):
    train_mse = []
    reg_ld_sqrt_sum = 0.0
    reg_w_sqrt_sum = 0.0
    reg_loss_sum = 0.0
    batch_count = 0
    eps = 1e-8

    for xx, yy in train_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        optimizer.zero_grad()

        loss = 0.0
        task_mse = xx.new_zeros(())
        reg_loss = xx.new_zeros(())
        reg_ld_sqrt = xx.new_zeros(())
        reg_w_sqrt = xx.new_zeros(())
        for target in yy.transpose(0, 1):
            prediction, n_ld, n = model(xx, equiv, mix_coef)
            reg_ld_sqrt = torch.sqrt(n_ld + eps)
            reg_w_sqrt = torch.sqrt(n + eps)
            reg_loss = pe_regularization_lambda * mix_coef * (reg_ld_sqrt + reg_w_sqrt)
            task_loss = loss_function(prediction, target)
            task_mse += task_loss.detach()
            loss += task_loss + reg_loss
            xx = torch.cat([xx[:, prediction.shape[1]:], prediction], dim=1)

        loss.backward()
        optimizer.step()

        train_mse.append(task_mse.item() / yy.shape[1])
        reg_ld_sqrt_sum += float(reg_ld_sqrt.item())
        reg_w_sqrt_sum += float(reg_w_sqrt.item())
        reg_loss_sum += float(reg_loss.item())
        batch_count += 1

    train_rmse = round(float(np.sqrt(np.mean(train_mse))), 5)
    return train_rmse, {
        "reg_ld_sqrt_avg": reg_ld_sqrt_sum / max(batch_count, 1),
        "reg_w_sqrt_avg": reg_w_sqrt_sum / max(batch_count, 1),
        "reg_loss_avg": reg_loss_sum / max(batch_count, 1),
    }


def copy_state_to_cpu(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    if args.schedule_max_epoch is None:
        args.schedule_max_epoch = args.num_epoch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("These published configurations require a CUDA GPU.")

    data_dir = Path(args.data_dir).resolve()
    expected_files = [
        data_dir / f"raw_data_{task}_{orientation}.pt"
        for task, orientation in TRAIN_TASKS + DOMAIN_TASKS
    ]
    missing_files = [str(path) for path in expected_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing rotation data files:\n" + "\n".join(missing_files)
        )

    mid = args.input_length + 2
    train_set = Dataset(
        input_length=args.input_length,
        mid=mid,
        output_length=args.out_length,
        direc=str(data_dir),
        task_list=TRAIN_TASKS,
        sample_list=list(range(0, 160)),
        stack=True,
    )
    valid_set = Dataset(
        input_length=args.input_length,
        mid=mid,
        output_length=args.out_length,
        direc=str(data_dir),
        task_list=TRAIN_TASKS,
        sample_list=list(range(160, 200)),
        stack=True,
    )
    future_set = Dataset(
        input_length=args.input_length,
        mid=mid,
        output_length=20,
        direc=str(data_dir),
        task_list=TRAIN_TASKS,
        sample_list=list(range(200, 250)),
        stack=True,
    )
    domain_set = Dataset(
        input_length=args.input_length,
        mid=mid,
        output_length=20,
        direc=str(data_dir),
        task_list=DOMAIN_TASKS,
        sample_list=list(range(0, 100)),
        stack=True,
    )

    loader_generator = torch.Generator().manual_seed(args.seed)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "generator": loader_generator,
    }
    train_loader = data.DataLoader(train_set, shuffle=True, **loader_args)
    valid_loader = data.DataLoader(valid_set, shuffle=True, **loader_args)
    future_loader = data.DataLoader(future_set, shuffle=False, **loader_args)
    domain_loader = data.DataLoader(domain_set, shuffle=False, **loader_args)

    if args.model == "RSteer":
        base_model = Relaxed_Rot_SteerConvNet(
            in_frames=args.input_length,
            out_frames=1,
            hidden_dim=args.hidden_dim,
            kernel_size=3,
            num_layers=args.num_layers,
            N=args.group_order,
            alpha=args.alpha,
        )
    else:
        if args.alpha != 0.0:
            raise ValueError("Equiv/E2CNN does not use --alpha; pass --alpha 0.")
        base_model = E2CNN_PE(
            in_frames=args.input_length,
            out_frames=1,
            hidden_dim=args.hidden_dim,
            kernel_size=3,
            num_layers=args.num_layers,
            N=args.group_order,
            pe_scale_init=args.pe_scale_init,
        )

    # e2cnn 0.2.3 hangs while replicating these modules across multiple GPUs
    # under PyTorch 2.7. Keep the wrapper expected by utils.train_epoch (which
    # accesses model.module), but restrict it to a single device.
    model = nn.DataParallel(
        base_model.to(device),
        device_ids=[0],
    )
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if parameter_count >= 10_000_000:
        raise ValueError(
            f"Model has {parameter_count:,} parameters, "
            "violating the paper's <10^7 constraint."
        )

    run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as exc:
            raise ImportError(
                "Install wandb in the selected Python environment."
            ) from exc

        config = vars(args).copy()
        config.update(
            {
                "dataset": "PhiFlow/Rotation",
                "train_tasks": len(TRAIN_TASKS),
                "domain_test_tasks": len(DOMAIN_TASKS),
                "train_time": "0:160",
                "validation_time": "160:200",
                "future_test_time": "200:250",
                "domain_test_time": "0:100",
                "test_rollout_steps": 20,
                "optimizer": "Adam",
                "adam_betas": [0.9, 0.999],
                "parameter_count": parameter_count,
                "schedule_max_epoch": args.schedule_max_epoch,
                "mix_coef_schedule": args.mix_coef_schedule,
                "pe_regularization_lambda": args.pe_regularization_lambda,
                "pe_scale_init": args.pe_scale_init,
                "strict_equiv": args.strict_equiv,
                "checkpoint_selection": (
                    "best_validation"
                    if args.model == "Equiv" or args.restore_best_validation
                    else "final_early_stopped"
                ),
            }
        )
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=(
                args.wandb_run_name
                or f"rotation-{args.model.lower()}-seed{args.seed}"
            ),
            mode=args.wandb_mode,
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        run.define_metric("test/*", step_metric="epoch")

    print(f"Model: {args.model}")
    print(f"Device: {device}; CUDA devices: {torch.cuda.device_count()}")
    print(f"Trainable parameters: {parameter_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=1,
        gamma=args.decay_rate,
    )
    loss_function = torch.nn.MSELoss()

    train_history = []
    validation_history = []
    min_validation_rmse = float("inf")
    best_epoch = 0
    best_state = None

    pe_forward_kwargs_on = None
    pe_forward_kwargs_off = None
    min_validation_rmse_pe_on = float("inf")
    min_validation_rmse_pe_off = float("inf")
    best_epoch_pe_on = 0
    best_epoch_pe_off = 0
    best_state_pe_on = None
    best_state_pe_off = None
    best_forward_kwargs_pe_on = None
    best_forward_kwargs_pe_off = None

    for epoch in range(1, args.num_epoch + 1):
        start = time.time()
        model.train()
        if args.model == "Equiv":
            mix_coef = schedule_value(
                args.mix_coef_schedule,
                epoch - 1,
                args.schedule_max_epoch,
            )
            train_rmse, pe_train_stats = train_epoch_pe(
                train_loader,
                model,
                optimizer,
                loss_function,
                device,
                mix_coef=mix_coef,
                equiv=args.strict_equiv,
                pe_regularization_lambda=args.pe_regularization_lambda,
            )
            pe_forward_kwargs_on = {
                "equiv": args.strict_equiv,
                "mix_coef": 0.0 if args.strict_equiv else mix_coef,
            }
            pe_forward_kwargs_off = {
                "equiv": True,
                "mix_coef": 0.0,
            }
        else:
            mix_coef = 0.0
            pe_train_stats = None
            pe_forward_kwargs_on = None
            pe_forward_kwargs_off = None
            train_rmse = train_epoch(
                train_loader,
                model,
                optimizer,
                loss_function,
            )
        model.eval()

        if args.model == "Equiv":
            validation_rmse_pe_on = evaluate_rmse(
                valid_loader,
                model,
                loss_function,
                device,
                forward_kwargs=pe_forward_kwargs_on,
            )
            validation_rmse_pe_off = evaluate_rmse(
                valid_loader,
                model,
                loss_function,
                device,
                forward_kwargs=pe_forward_kwargs_off,
            )

            # Early stopping follows the strict (PE-off) validation RMSE.
            validation_rmse = validation_rmse_pe_off

            if validation_rmse_pe_on < min_validation_rmse_pe_on:
                min_validation_rmse_pe_on = validation_rmse_pe_on
                best_epoch_pe_on = epoch
                best_state_pe_on = copy_state_to_cpu(model)
                best_forward_kwargs_pe_on = dict(pe_forward_kwargs_on)

            if validation_rmse_pe_off < min_validation_rmse_pe_off:
                min_validation_rmse_pe_off = validation_rmse_pe_off
                best_epoch_pe_off = epoch
                best_state_pe_off = copy_state_to_cpu(model)
                best_forward_kwargs_pe_off = dict(pe_forward_kwargs_off)
        else:
            validation_rmse = evaluate_rmse(
                valid_loader,
                model,
                loss_function,
                device,
                forward_kwargs=None,
            )

        train_history.append(train_rmse)
        validation_history.append(validation_rmse)

        if validation_rmse < min_validation_rmse:
            min_validation_rmse = validation_rmse
            best_epoch = epoch
            if args.restore_best_validation:
                best_state = copy_state_to_cpu(model)

        elapsed_minutes = (time.time() - start) / 60.0
        current_lr = get_lr(optimizer)
        if args.model == "Equiv":
            print(
                f"Epoch {epoch} | T: {elapsed_minutes:.2f} | "
                f"Train RMSE: {train_rmse:.3f} | "
                f"Valid RMSE (PE on): {validation_rmse_pe_on:.3f} | "
                f"Valid RMSE (PE off): {validation_rmse_pe_off:.3f}"
            )
        else:
            print(
                f"Epoch {epoch} | T: {elapsed_minutes:.2f} | "
                f"Train RMSE: {train_rmse:.3f} | "
                f"Valid RMSE: {validation_rmse:.3f}"
            )
        if run is not None:
            log_payload = {
                "epoch": epoch,
                "train/rmse": train_rmse,
                "train/learning_rate": current_lr,
                "train/epoch_minutes": elapsed_minutes,
                "train/mix_coef": mix_coef,
            }
            if args.model == "Equiv":
                log_payload.update(
                    {
                        "validation/rmse": validation_rmse_pe_off,
                        "validation/rmse_pe_on": validation_rmse_pe_on,
                        "validation/rmse_pe_off": validation_rmse_pe_off,
                        "validation/best_rmse": min_validation_rmse_pe_off,
                        "validation/best_rmse_pe_on": min_validation_rmse_pe_on,
                        "validation/best_rmse_pe_off": min_validation_rmse_pe_off,
                        "validation/best_epoch_pe_on": best_epoch_pe_on,
                        "validation/best_epoch_pe_off": best_epoch_pe_off,
                    }
                )
            else:
                log_payload.update(
                    {
                        "validation/rmse": validation_rmse,
                        "validation/best_rmse": min_validation_rmse,
                    }
                )
            if pe_train_stats is not None:
                log_payload.update(
                    {
                        "train/pe_reg_loss": pe_train_stats["reg_loss_avg"],
                        "train/pe_reg_ld_sqrt": pe_train_stats["reg_ld_sqrt_avg"],
                        "train/pe_reg_w_sqrt": pe_train_stats["reg_w_sqrt_avg"],
                    }
                )
                if hasattr(model.module, "pe_statistics"):
                    pe_stats = model.module.pe_statistics()
                    log_payload.update(
                        {
                            "train/pe_gamma": pe_stats["gamma"],
                            "train/pe_gamma_abs": pe_stats["gamma_abs"],
                            "train/pe_raw_norm": pe_stats["raw_norm"],
                            "train/pe_raw_rms": pe_stats["raw_rms"],
                        }
                    )
            run.log(log_payload, step=epoch)

        should_stop = (
            len(train_history) > 100
            and np.mean(validation_history[-5:])
            >= np.mean(validation_history[-10:-5])
        )
        if should_stop:
            break
        scheduler.step()

    if args.model == "Equiv":
        if best_state_pe_on is None or best_state_pe_off is None:
            raise RuntimeError("Best PE checkpoints were not captured during training.")

        model.load_state_dict(best_state_pe_on)
        model.eval()
        future_rmse_pe_on = evaluate_rmse(
            future_loader,
            model,
            loss_function,
            device,
            forward_kwargs=best_forward_kwargs_pe_on,
        )
        domain_rmse_pe_on = evaluate_rmse(
            domain_loader,
            model,
            loss_function,
            device,
            forward_kwargs=best_forward_kwargs_pe_on,
        )

        model.load_state_dict(best_state_pe_off)
        model.eval()
        future_rmse_pe_off = evaluate_rmse(
            future_loader,
            model,
            loss_function,
            device,
            forward_kwargs=best_forward_kwargs_pe_off,
        )
        domain_rmse_pe_off = evaluate_rmse(
            domain_loader,
            model,
            loss_function,
            device,
            forward_kwargs=best_forward_kwargs_pe_off,
        )

        # Keep compatibility with existing output fields.
        future_rmse = future_rmse_pe_off
        domain_rmse = domain_rmse_pe_off
        best_epoch = best_epoch_pe_off
        min_validation_rmse = min_validation_rmse_pe_off

        print(
            f"Best PE-on checkpoint (epoch {best_epoch_pe_on}) | "
            f"Future RMSE: {future_rmse_pe_on}; Domain RMSE: {domain_rmse_pe_on}"
        )
        print(
            f"Best PE-off checkpoint (epoch {best_epoch_pe_off}) | "
            f"Future RMSE: {future_rmse_pe_off}; Domain RMSE: {domain_rmse_pe_off}"
        )
    else:
        if args.restore_best_validation:
            model.load_state_dict(best_state)

        model.eval()
        future_rmse = evaluate_rmse(
            future_loader,
            model,
            loss_function,
            device,
            forward_kwargs=None,
        )
        domain_rmse = evaluate_rmse(
            domain_loader,
            model,
            loss_function,
            device,
            forward_kwargs=None,
        )
        print(
            f"Future Test RMSE: {future_rmse}; "
            f"Domain Test RMSE: {domain_rmse}"
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        output_dir
        / f"PhiFlow_Rotation_{args.model}_seed{args.seed}.pt"
    )
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "args": vars(args),
            "epoch": len(train_history),
            "best_epoch": best_epoch,
            "best_validation_rmse": min_validation_rmse,
            "future_rmse": future_rmse,
            "domain_rmse": domain_rmse,
            "parameter_count": parameter_count,
            "best_pe_on": {
                "epoch": best_epoch_pe_on,
                "validation_rmse": min_validation_rmse_pe_on,
                "forward_kwargs": best_forward_kwargs_pe_on,
                "future_rmse": future_rmse_pe_on if args.model == "Equiv" else None,
                "domain_rmse": domain_rmse_pe_on if args.model == "Equiv" else None,
                "state_dict": best_state_pe_on if args.model == "Equiv" else None,
            },
            "best_pe_off": {
                "epoch": best_epoch_pe_off,
                "validation_rmse": min_validation_rmse_pe_off,
                "forward_kwargs": best_forward_kwargs_pe_off,
                "future_rmse": future_rmse_pe_off if args.model == "Equiv" else None,
                "domain_rmse": domain_rmse_pe_off if args.model == "Equiv" else None,
                "state_dict": best_state_pe_off if args.model == "Equiv" else None,
            },
        },
        checkpoint_path,
    )

    if run is not None:
        test_log_payload = {
            "epoch": len(train_history),
            "test/future_rmse": future_rmse,
            "test/domain_rmse": domain_rmse,
        }
        if args.model == "Equiv":
            test_log_payload.update(
                {
                    "test/future_rmse_pe_on": future_rmse_pe_on,
                    "test/domain_rmse_pe_on": domain_rmse_pe_on,
                    "test/future_rmse_pe_off": future_rmse_pe_off,
                    "test/domain_rmse_pe_off": domain_rmse_pe_off,
                }
            )
        run.log(test_log_payload, step=len(train_history))

        run.summary["test/future_rmse"] = future_rmse
        run.summary["test/domain_rmse"] = domain_rmse
        run.summary["validation/best_rmse"] = min_validation_rmse
        run.summary["validation/best_epoch"] = best_epoch
        if args.model == "Equiv":
            run.summary["validation/best_rmse_pe_on"] = min_validation_rmse_pe_on
            run.summary["validation/best_epoch_pe_on"] = best_epoch_pe_on
            run.summary["validation/best_rmse_pe_off"] = min_validation_rmse_pe_off
            run.summary["validation/best_epoch_pe_off"] = best_epoch_pe_off
            run.summary["test/future_rmse_pe_on"] = future_rmse_pe_on
            run.summary["test/domain_rmse_pe_on"] = domain_rmse_pe_on
            run.summary["test/future_rmse_pe_off"] = future_rmse_pe_off
            run.summary["test/domain_rmse_pe_off"] = domain_rmse_pe_off
        run.summary["train/epochs"] = len(train_history)
        run.summary["checkpoint/path"] = str(checkpoint_path)
        run.finish()


if __name__ == "__main__":
    main()
