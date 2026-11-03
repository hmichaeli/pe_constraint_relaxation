from typing import Any, Optional

import torch
import torch.nn as nn

# ---------------------------- Triangular PE Wrapper ----------------------------


class ScaleScheduler(nn.Module):
    """
    Triangular (0 -> 1 -> 0) schedule for positional encodings over S steps.

    Schedule (S = total_steps):
      step = 0     -> scale = 0
      step = S/2   -> scale = 1
      step = S     -> scale = 0
      beyond S     -> scale = 0

    Also includes a learnable scalar that multiplies the PE before the schedule,
    and caches the batch-mean L2 norm of (learnable-scaled PE) for the criterion.
    """

    def __init__(self, total_steps: int, schedule_type: str = "triangular") -> None:
        super().__init__()
        self.total_steps = int(total_steps)
        self.schedule_type = schedule_type

        # Cache for the latest norm (kept on-graph for gradients)
        self._last_pos_norm: Optional[torch.Tensor] = None

        # Track training progress & current schedule scale
        self.register_buffer("current_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("current_scale", torch.tensor(0.0, dtype=torch.float))

    # ----- schedule helpers -----
    def _triangular_scale(self, step: int) -> float:
        S = max(1, int(self.total_steps))  # avoid div-by-zero
        if step <= 0:
            return 0.0
        if step >= S:
            return 0.0
        half = S / 2.0
        if step <= half:
            return float(2.0 * step / S)
        # decreasing branch
        return float(max(0.0, 2.0 - 2.0 * step / S))

    def _linear_scale(self, step: int) -> float:
        S = max(1, int(self.total_steps))  # avoid div-by-zero
        if step <= 0:
            return 1.0
        if step >= S:
            return 0.0
        return float(1 - step / S)

    def step(self) -> None:
        with torch.no_grad():
            self.current_step += 1            
            if self.schedule_type == "triangular":
                s = self._triangular_scale(int(self.current_step.item()))
            elif self.schedule_type == "linear":
                s = self._linear_scale(int(self.current_step.item()))
            else:
                raise ValueError(f"Unknown schedule_type: {self.schedule_type}")

            self.current_scale = torch.tensor(
                s, dtype=self.current_scale.dtype, device=self.current_scale.device
            )

    def update_num_steps(self, num_steps: int) -> None:
        with torch.no_grad():
            self.current_step = torch.tensor(
                int(num_steps), dtype=torch.long, device=self.current_step.device
            )
            s = self._triangular_scale(int(self.current_step.item()))
            self.current_scale = torch.tensor(
                s, dtype=self.current_scale.dtype, device=self.current_scale.device
            )

    def get_current_scale(self) -> float:
        return float(self.current_scale.item())

    def get_last_pos_norm(self) -> Optional[torch.Tensor]:
        return self._last_pos_norm

    # ----- attribute pass-throughs (Fairseq expects these on embed_positions) -----
    @property
    def max_positions(self):
        mp = getattr(self.wrapped_module, "max_positions", None)
        if callable(mp):
            return mp()
        return mp

    @property
    def padding_idx(self):
        return getattr(self.wrapped_module, "padding_idx", None)

    # ----- main forward -----
    def forward(self, *args, **kwargs):
        return self.current_scale

    def extra_repr(self) -> str:
        return (
            f"triangular(total_steps={self.total_steps}), "
            f"current_step={int(self.current_step.item())}, "
            f"current_scale={self.current_scale.item():.4f}"
        )


def _sync_wrappers_to_num_updates(
    model: nn.Module, num_updates: int, log_freq: int = 500
) -> int:
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, ScaleScheduler):
            module.update_num_steps(num_updates)
            count += 1
            if log_freq > 0 and (num_updates % log_freq == 0):
                print(
                    f"[PosTri] {name}: scale={module.get_current_scale():.4f} at step {num_updates}"
                )
    return count


def step_all_wrappers(model: nn.Module, log_freq: int = 500) -> int:
    """Increment step on all ScaleScheduler modules and optionally log.

    Returns the number of wrappers updated.
    """
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, ScaleScheduler):
            module.step()
            count += 1
            if log_freq > 0:
                step = int(module.current_step.item())
                if step % log_freq == 0:
                    print(
                        f"[PosTri] {name}: scale={module.get_current_scale():.4f} at step {step}"
                    )
    return count
