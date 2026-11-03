# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# Modifications:
# - Added alias-free positional decay handling and benchmarking helper.
# - Integrated mixed-precision support and GradualDecay wrapper updates.
# - Enhanced logging utilities for distributed training throughput.
"""Training and evaluation utilities for Alias-Free Transformer models.

This module extends the XCiT training routines with AFT-specific functionality,
including positional decay updates and benchmarking helpers.
"""
import math
import sys
import time
from typing import Iterable, Optional

import torch
from timm.data import Mixup
from timm.utils import ModelEma, accuracy
from torch import distributed as dist

import utils
from losses import DistillationLoss
from pe_scale_scheduler import step_all_wrappers


def train_one_epoch(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode: bool = True,
    surgery=None,
    print_freq: int = 500,
    use_amp: bool = False,
    pos_reg_lambda: float = 0.0,
    accumulation_steps: int = 1,
):
    """Run one training epoch with optional mixup, EMA, and positional decay updates."""

    model.train(set_training_mode)

    if surgery:
        model.module.patch_embed.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    optimizer.zero_grad()

    for idx, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets = batch[0], batch[1]

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs, pe_norm = model(samples)
                task_loss = criterion(samples, outputs, targets)
                reg_loss = pos_reg_lambda * pe_norm
        else:
            outputs, pe_norm = model(samples)
            task_loss = criterion(samples, outputs, targets)
            reg_loss = pos_reg_lambda * pe_norm
        loss = task_loss + reg_loss

        loss_value = loss.item()
        reg_loss_value = reg_loss.item()
        task_loss_value = task_loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accumulation_steps

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        
        update_grad = (idx + 1) % accumulation_steps == 0
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
            need_update=update_grad,
        )

        torch.cuda.synchronize()
        if update_grad:
            optimizer.zero_grad()
            if model_ema is not None:
                model_ema.update(model)

            # Update positional encoding scale schedulers (if present) each step
            model_for_sync = model.module if hasattr(model, "module") else model
            step_all_wrappers(model_for_sync)

        metric_logger.update(task_loss=task_loss_value)

        metric_logger.update(reg_loss=reg_loss_value)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_benchmark(
    model: torch.nn.Module,
    criterion: DistillationLoss,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0.0,
    model_ema: Optional[ModelEma] = None,
    mixup_fn: Optional[Mixup] = None,
    set_training_mode: bool = True,
    surgery=None,
    print_freq: int = 500,
    use_amp: bool = False,
    num_iterations: int = 100,
):
    """Run a short training loop purely for throughput benchmarking."""

    model.train(set_training_mode)

    if surgery:
        # keep patch-embed frozen while benchmarking, if requested
        model.module.patch_embed.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    # ------------------------------------------------------------------
    # Timing starts here ― make sure all previous CUDA work is finished.
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    processed_batches = 0

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        if processed_batches >= num_iterations:
            break

        samples = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if use_amp:
            with torch.cuda.amp.autocast():
                outputs = model(samples)
                loss = criterion(samples, outputs, targets)
        else:
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad(set_to_none=True)

        # timm’s loss-scaler handles fp32 / fp16 transparently
        is_second_order = getattr(optimizer, "is_second_order", False)
        loss_scaler(
            loss,
            optimizer,
            clip_grad=max_norm,
            parameters=model.parameters(),
            create_graph=is_second_order,
        )

        if model_ema is not None:
            model_ema.update(model)

        # Update positional encoding scale schedulers (if present) each step
        model_for_sync = model.module if hasattr(model, "module") else model
        step_all_wrappers(model_for_sync, log_freq=500)

        metric_logger.update(loss=loss_value, lr=optimizer.param_groups[0]["lr"])
        processed_batches += 1

    # ------------------------------------------------------------------
    # All CUDA kernels launched above must complete before we stop timing.
    # ------------------------------------------------------------------
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start_time
    avg_time_per_batch = total_time / max(processed_batches, 1)

    # gather stats from all ranks before printing (optional for single-GPU)
    metric_logger.synchronize_between_processes()

    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        # Print only from rank-0 to avoid clutter in multi-GPU jobs
        print("Averaged stats:", metric_logger)
        print(
            f"=> Benchmark finished: {processed_batches} batches in "
            f"{total_time:.2f} s | {avg_time_per_batch:.4f} s / batch"
        )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, print_freq: int = 500, use_amp=False):
    """Evaluate the model on the validation set and report accuracy metrics."""

    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, print_freq, header):
        images, target = batch[0], batch[1]

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(images)
        else:
            output = model(images)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        "* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}".format(
            top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss
        )
    )

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
