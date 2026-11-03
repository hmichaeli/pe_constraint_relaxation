# scalar_scheduler.py
"""
Scalar Scheduler (0D) with High-Level Updaters
==============================================

What this is:
-------------
A lightweight scalar scheduler (nn.Module) that returns a single tensor scale you can
multiply by anything you want to anneal. It supports per-step or per-epoch driving,
and includes a "triangle" schedule that goes 0 -> 1 -> 0 across the horizon.

What's new vs wrapper:
----------------------
• Not a wrapper. You explicitly multiply using the returned scalar.
• 'triangle' schedule added (0→1→0 over horizon, then sticks at 0).
• High-level API to update ALL schedulers in your model in one call.

Quick start:
------------
from scalar_scheduler import (
    ScalarScheduler,
    update_all_schedulers_step,
    update_all_schedulers_epoch,
)

# Example: triangle over 400 epochs (per-epoch)
sched = ScalarScheduler(
    schedule='triangle',
    update_on='epoch',
    total_epochs=400,
)

# Put the scheduler(s) inside your model so they're discoverable:
# self.pe_scale = ScalarScheduler(...)

# In your training loop:
for epoch in range(num_epochs):
    scale = model.pe_scale()  # or model.pe_scale.forward(), returns 0-D tensor
    # use it: x = x * scale
    update_all_schedulers_epoch(model, log_freq=1)  # advances only epoch-driven schedulers

# Or per-step cosine with warmup:
sched = ScalarScheduler(
    schedule='cosine',
    update_on='step',
    total_steps=100_000,
    warmup_steps=5_000,
)
for step, batch in enumerate(loader):
    scale = model.sched()  # multiply something by this
    # ...
    update_all_schedulers_step(model, log_freq=500)  # advances only step-driven schedulers

Tips:
-----
• Create multiple schedulers (e.g., for different modules) and register as submodules of your model.
• The updaters scan the model for ScalarScheduler instances and tick the right counter.
• Logging is configurable via log_freq and print_fn.
"""

import math
from typing import Optional, Literal, List, Tuple, Dict, Callable
import torch
import torch.nn as nn


class ScalarScheduler(nn.Module):
    """
    A scalar scheduler (no wrapping) that returns a single scale factor as a tensor.
    Useful for gradually turning on/off parts of a model (e.g., PE, regularizers).

    Args:
        schedule (str): One of {'linear','exponential','cosine','step','triangle'}.
        update_on (str): 'step' or 'epoch'. Determines which counter drives the schedule.
        total_steps (int, optional): Horizon in steps if update_on='step'.
        total_epochs (int, optional): Horizon in epochs if update_on='epoch'.
        initial_scale (float): Start value for non-triangle schedules (default 1.0).
        final_scale (float): End value for non-triangle schedules (default 0.0).
        warmup_steps (int): Warmup in steps (only used if update_on='step').
        warmup_epochs (int): Warmup in epochs (only used if update_on='epoch').
        step_size (int, optional): For 'step' schedule (piecewise constant).
        gamma (float, optional): Decay factor for 'exponential' and 'step'. Default chosen if None.
        device (torch.device or str, optional): Device for internal buffers.

    Notes:
        • 'triangle' ignores initial_scale/final_scale (requested behavior):
          0 → 1 → 0 over horizon, then stays 0.
        • After the horizon, non-triangle schedules clamp to final_scale; triangle clamps to 0.
        • forward() returns a 0-D tensor; use it directly in multiplication.
        • Call update_step()/update_epoch() to advance the internal counters.
    """

    def __init__(
        self,
        schedule: Literal['linear','exponential','cosine','step','triangle'] = 'linear',
        update_on: Literal['step','epoch'] = 'step',
        *,
        total_steps: Optional[int] = None,
        total_epochs: Optional[int] = None,
        initial_scale: float = 1.0,
        final_scale:   float = 0.0,
        warmup_steps:  int = 0,
        warmup_epochs: int = 0,
        step_size: Optional[int] = None,
        gamma: Optional[float] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.schedule = schedule
        self.update_on = update_on

        # Horizon validation
        if update_on == 'step' and (total_steps is None or total_steps <= 0):
            raise ValueError("total_steps must be a positive int when update_on='step'.")
        if update_on == 'epoch' and (total_epochs is None or total_epochs <= 0):
            raise ValueError("total_epochs must be a positive int when update_on='epoch'.")

        self.total_steps  = int(total_steps)  if total_steps  is not None else None
        self.total_epochs = int(total_epochs) if total_epochs is not None else None

        # Warmup
        self.warmup_steps  = int(warmup_steps)
        self.warmup_epochs = int(warmup_epochs)

        # Params for non-triangle schedules
        self.initial_scale = float(initial_scale)
        self.final_scale   = float(final_scale)
        self.step_size     = int(step_size) if step_size is not None else None

        # Defaults for gamma if needed
        if gamma is None:
            gamma = 0.99 if schedule == 'exponential' else 0.5 if schedule == 'step' else 1.0
        self.gamma = float(gamma)

        # State
        self.register_buffer('current_step',  torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long, device=device))
        self.register_buffer('current_scale', torch.tensor(0.0, dtype=torch.float, device=device))

        self._validate()
        self._recompute(clamp=True)

    # -----------------------
    # Public API
    # -----------------------
    def forward(self) -> torch.Tensor:
        """Return the current scalar (0-dim tensor) for multiplication."""
        return self.current_scale

    def get_value(self) -> float:
        """Return current scale as Python float."""
        return float(self.current_scale.item())

    @torch.no_grad()
    def update_step(self, n: int = 1) -> float:
        """Increment step counter; if update_on='step', advances schedule."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.current_step += n
        self._recompute(clamp=True)
        return self.get_value()

    @torch.no_grad()
    def update_epoch(self, n: int = 1) -> float:
        """Increment epoch counter; if update_on='epoch', advances schedule."""
        if n < 0:
            raise ValueError("n must be non-negative")
        self.current_epoch += n
        self._recompute(clamp=True)
        return self.get_value()

    @torch.no_grad()
    def reset(self) -> None:
        """Reset counters and scalar to start-of-schedule."""
        self.current_step.zero_()
        self.current_epoch.zero_()
        self._recompute(clamp=True)

    # -----------------------
    # Internals
    # -----------------------
    def _validate(self):
        if self.schedule not in {'linear','exponential','cosine','step','triangle'}:
            raise ValueError(f"Unknown schedule: {self.schedule}")
        if self.update_on not in {'step','epoch'}:
            raise ValueError(f"update_on must be 'step' or 'epoch', got {self.update_on}")
        if self.schedule == 'step' and (self.step_size is None or self.step_size <= 0):
            raise ValueError("For schedule='step', step_size must be a positive integer.")

    def _drive_unit(self) -> int:
        return int(self.current_step.item()) if self.update_on == 'step' else int(self.current_epoch.item())

    def _warmup_units(self) -> int:
        return self.warmup_steps if self.update_on == 'step' else self.warmup_epochs

    def _total_units(self) -> int:
        return self.total_steps if self.update_on == 'step' else self.total_epochs

    def _progress(self, unit: int) -> float:
        """Progress ∈ [0,1] within the active (post-warmup) window (unclamped)."""
        wu = self._warmup_units()
        T  = self._total_units()
        if unit < wu:
            return -1.0  # warmup
        adj = unit - wu
        active = max(T - wu, 1)
        return adj / active

    def _calculate_scale(self, unit: int, clamp: bool) -> float:
        p = self._progress(unit)

        # Warmup
        if p < 0:
            return 0.0 if self.schedule == 'triangle' else self.initial_scale

        # Past horizon
        if p >= 1.0:
            return 0.0 if self.schedule == 'triangle' else self.final_scale

        # Active window
        if self.schedule == 'linear':
            return self.initial_scale + p * (self.final_scale - self.initial_scale)

        if self.schedule == 'exponential':
            wu = self._warmup_units()
            adj_units = max(unit - wu, 0)
            val = self.initial_scale * (self.gamma ** adj_units)
            if self.final_scale != 0.0:
                val = (1 - p) * val + p * self.final_scale
            return val

        if self.schedule == 'cosine':
            cosv = 0.5 * (1.0 + math.cos(math.pi * p))
            return self.final_scale + (self.initial_scale - self.final_scale) * cosv

        if self.schedule == 'step':
            wu = self._warmup_units()
            adj_units = max(unit - wu, 0)
            k = adj_units // self.step_size
            val = self.initial_scale * (self.gamma ** k)
            # clamp toward final if provided
            if self.final_scale < self.initial_scale:
                val = max(val, self.final_scale)
            elif self.final_scale > self.initial_scale:
                val = min(val, self.final_scale)
            return val

        if self.schedule == 'triangle':
            # 0 -> 1 (first half), then 1 -> 0 (second half)
            return 2.0 * p if p <= 0.5 else 2.0 * (1.0 - p)

        return self.initial_scale

    @torch.no_grad()
    def _recompute(self, clamp: bool):
        unit = self._drive_unit()
        val = self._calculate_scale(unit, clamp=clamp)
        self.current_scale.fill_(val)

    def extra_repr(self) -> str:
        horizon = self.total_steps if self.update_on == 'step' else self.total_epochs
        warmup  = self.warmup_steps if self.update_on == 'step' else self.warmup_epochs
        return (f"schedule={self.schedule}, update_on={self.update_on}, "
                f"horizon={horizon}, warmup={warmup}, "
                f"initial_scale={self.initial_scale:.4f}, final_scale={self.final_scale:.4f}, "
                f"gamma={self.gamma:.4f}, step_size={self.step_size}, "
                f"current_step={self.current_step.item()}, current_epoch={self.current_epoch.item()}, "
                f"current_scale={self.current_scale.item():.6f}")


# ======================================================================
# High-level API: find and update ALL ScalarSchedulers inside a model
# ======================================================================

def find_scalar_schedulers(model: nn.Module) -> List[Tuple[str, ScalarScheduler]]:
    """
    Return [(name, module)] for all ScalarScheduler instances registered in `model`.
    """
    scheds: List[Tuple[str, ScalarScheduler]] = []
    for name, module in model.named_modules():
        if isinstance(module, ScalarScheduler):
            scheds.append((name, module))
    return scheds


@torch.no_grad()
def update_all_schedulers_step(
    model: nn.Module,
    *,
    n: int = 1,
    log_freq: int = 500,
    print_fn: Callable[[str], None] = print,
) -> int:
    """
    Increment step by `n` for ALL ScalarSchedulers in `model`.
    Only schedulers with update_on='step' will actually progress their schedule,
    but counters are kept consistent for all.

    Returns:
        int: number of ScalarSchedulers found.
    """
    scheds = find_scalar_schedulers(model)
    for name, sched in scheds:
        prev = sched.get_value()
        sched.update_step(n)
        # Log only for step-driven schedulers, using their own step counter
        if sched.update_on == 'step' and log_freq > 0:
            step = int(sched.current_step.item())
            if step % log_freq == 0:
                print_fn(f"[step] {name}: scale={sched.get_value():.6f} (step={step}, was {prev:.6f})")
    return len(scheds)


@torch.no_grad()
def update_all_schedulers_epoch(
    model: nn.Module,
    *,
    n: int = 1,
    log_freq: int = 1,
    print_fn: Callable[[str], None] = print,
) -> int:
    """
    Increment epoch by `n` for ALL ScalarSchedulers in `model`.
    Only schedulers with update_on='epoch' will actually progress their schedule,
    but counters are kept consistent for all.

    Returns:
        int: number of ScalarSchedulers found.
    """
    scheds = find_scalar_schedulers(model)
    for name, sched in scheds:
        prev = sched.get_value()
        sched.update_epoch(n)
        if sched.update_on == 'epoch' and log_freq > 0:
            ep = int(sched.current_epoch.item())
            if ep % log_freq == 0:
                print_fn(f"[epoch] {name}: scale={sched.get_value():.6f} (epoch={ep}, was {prev:.6f})")
    return len(scheds)


def get_scheduler_states(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Inspect all ScalarSchedulers in `model` and return a dict of their states.
    Returns:
        {
          "module.path": {
              "scale": 0.1234, "step": 1000, "epoch": 5,
              "schedule": "cosine", "update_on": "step"
          },
          ...
        }
    """
    info: Dict[str, Dict[str, float]] = {}
    for name, sched in find_scalar_schedulers(model):
        info[name] = {
            "scale":  sched.get_value(),
            "step":   int(sched.current_step.item()),
            "epoch":  int(sched.current_epoch.item()),
            "schedule": sched.schedule,
            "update_on": sched.update_on,
        }
    return info
