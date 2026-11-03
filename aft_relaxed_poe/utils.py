# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""Utility helpers for training and distributed execution.

This module started from the torchvision references and was polished for the
Alias-Free Transformer release.
"""

from __future__ import annotations

import argparse
import datetime
import io
import os
import socket
import time
from collections import defaultdict, deque
from typing import Any, Deque, Iterable, Iterator, MutableMapping, Sequence

import torch  # type: ignore[import]
import torch.distributed as dist  # type: ignore[import]


class SmoothedValue:
    """Track a history of scalar values and expose aggregated statistics."""

    def __init__(self, window_size: int = 20, fmt: str | None = None) -> None:
        fmt = fmt or "{median:.4f} ({global_avg:.4f})"
        self.deque: Deque[float] = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1) -> None:
        """Add a new observation to the tracker."""

        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self) -> None:
        """Synchronize statistics across distributed processes using CUDA."""

        if not is_dist_avail_and_initialized():
            return
        tensor = torch.tensor(
            [self.count, self.total], dtype=torch.float64, device="cuda"
        )
        dist.barrier()
        dist.all_reduce(tensor)
        sync_count, sync_total = tensor.tolist()
        self.count = int(sync_count)
        self.total = sync_total

    @property
    def median(self) -> float:
        values = torch.tensor(list(self.deque))
        return values.median().item()

    @property
    def avg(self) -> float:
        values = torch.tensor(list(self.deque), dtype=torch.float32)
        return values.mean().item()

    @property
    def global_avg(self) -> float:
        return self.total / self.count

    @property
    def max(self) -> float:
        return max(self.deque)

    @property
    def value(self) -> float:
        return self.deque[-1]

    def __str__(self) -> str:
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """Log a dictionary of smoothing metrics during iteration."""

    def __init__(self, delimiter: str = "\t") -> None:
        self.meters: MutableMapping[str, SmoothedValue] = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs: float | torch.Tensor) -> None:
        """Update tracked metrics with keyword arguments."""

        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            if not isinstance(value, (float, int)):
                raise TypeError(f"Expected float or int, received {type(value)!r}")
            self.meters[name].update(float(value))

    def __getattr__(self, attr: str) -> SmoothedValue:
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {attr!r}"
        )

    def __str__(self) -> str:
        loss_str = [f"{name}: {meter}" for name, meter in self.meters.items()]
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self) -> None:
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name: str, meter: SmoothedValue) -> None:
        """Register a new metric under ``name``."""

        self.meters[name] = meter

    def log_every(
        self,
        iterable: Sequence[Any] | Iterable[Any],
        print_freq: int,
        header: str | None = None,
    ) -> Iterator[Any]:
        """Iterate while printing running statistics every ``print_freq`` steps."""

        iterator = iter(iterable)
        size = len(iterable) if hasattr(iterable, "__len__") else None
        i = 0
        resolved_header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        length_string = str(len(str(size))) if size is not None else "4"
        space_fmt = ":" + length_string + "d"
        log_msg = [
            resolved_header,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_msg.append("max mem: {memory:.0f}")
        log_msg = self.delimiter.join(log_msg)
        megabyte = 1024.0 * 1024.0
        while True:
            try:
                obj = next(iterator)
            except StopIteration:
                break
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            size_or_default = size or (i + 1)
            if i % print_freq == 0 or (size is not None and i == size - 1):
                remaining = size_or_default - i
                eta_seconds = iter_time.global_avg * remaining
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            size_or_default,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / megabyte,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            size_or_default,
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        denominator = size or i or 1
        print(
            f"{resolved_header} Total time: {total_time_str} ({total_time / denominator:.4f} s / it)"
        )


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def _find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )

    torch.distributed.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


def str2bool(v):
    """
    Converts string to bool type; enables command line
    arguments in the format of '--arg1 true --arg2 false'
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
