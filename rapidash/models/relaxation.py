import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn

LEARNED_PE_INIT_STD = 0.02


class AnnealScheduler:
    def __init__(self, schedule_type: str = "linear"):
        self.schedule_type = schedule_type

    def __call__(self, current_step: int, total_steps: int) -> float:
        total_steps = max(int(total_steps), 1)
        current_step = max(int(current_step), 0)
        t = min(float(current_step) / float(total_steps), 1.0)

        if self.schedule_type == "linear":
            return max(0.0, 1.0 - t)
        elif self.schedule_type == "triangular":
            if t <= 0.5:
                return 2.0 * t
            return max(0.0, 2.0 * (1.0 - t))
        elif self.schedule_type == "constant":
            return 1.0
        raise ValueError(
            f"Unsupported schedule_type={self.schedule_type}. "
            "Supported types: linear, triangular, constant"
        )


class PERelaxation(nn.Module):
    def __init__(
        self,
        pe_enabled: bool = False,
        pe_mode: str = "add",
        pe_type: str = "fixed",
        pe_schedule: str = "linear",
        pe_gamma_init: float = 1.0,
        pe_keep_at_test: bool = False,
        pe_reg_beta: float = 0.0,
    ):
        super().__init__()
        self.pe_enabled = pe_enabled
        self.pe_mode = pe_mode
        self.pe_type = pe_type
        self.pe_keep_at_test = pe_keep_at_test
        self.pe_reg_beta = pe_reg_beta
        self.scheduler = AnnealScheduler(pe_schedule)

        self.pe_scale_gamma = nn.Parameter(torch.tensor(float(pe_gamma_init)))
        self.fixed_pe_param: Optional[nn.Parameter] = None
        self.learned_pe_param: Optional[nn.Parameter] = None

        self.last_lambda = 0.0
        self.last_pe_norm = 0.0
        self.last_reg_loss = None

    def get_lambda(self, step: int, total_steps: int, training: bool) -> float:
        if not self.pe_enabled:
            return 0.0
        if training:
            return self.scheduler(step, total_steps)
        return 1.0 if self.pe_keep_at_test else 0.0

    def _param_shape(self, x_shape: Sequence[int]) -> Tuple[int, ...]:
        if len(x_shape) == 3:
            return (1, 1, int(x_shape[-1]))
        if len(x_shape) == 2:
            return (1, int(x_shape[-1]))
        raise ValueError(f"Unsupported x shape for PE: {x_shape}")

    def _ensure_param(self, name: str, shape: Tuple[int, ...], device, dtype, init: str):
        param = getattr(self, name)
        if param is None or tuple(param.shape) != shape:
            data = torch.zeros(shape, device=device, dtype=dtype)
            if init == "normal":
                data.normal_(0.0, LEARNED_PE_INIT_STD)
            param = nn.Parameter(data)
            setattr(self, name, param)
        return getattr(self, name)

    def build_pe(self, x_shape: Sequence[int], device, dtype):
        if self.pe_type == "sinusoidal":
            if len(x_shape) == 3:
                n_pts, n_ori, channels = int(x_shape[0]), int(x_shape[1]), int(x_shape[2])
            elif len(x_shape) == 2:
                n_pts, n_ori, channels = int(x_shape[0]), 1, int(x_shape[1])
            else:
                raise ValueError(f"Unsupported x shape for PE: {x_shape}")

            idx = torch.arange(n_pts, device=device, dtype=dtype).unsqueeze(-1)
            idx = idx / max(n_pts - 1, 1)
            div = torch.exp(
                torch.arange(0, channels, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(channels, 1))
            )
            pe = torch.zeros((n_pts, channels), device=device, dtype=dtype)
            pe[:, 0::2] = torch.sin(idx * div)
            if channels > 1:
                pe[:, 1::2] = torch.cos(idx * div[: pe[:, 1::2].shape[1]])
            if n_ori > 1:
                pe = pe.unsqueeze(1).expand(-1, n_ori, -1)
            return pe

        param_shape = self._param_shape(x_shape)
        if self.pe_type == "fixed":
            return self._ensure_param("fixed_pe_param", param_shape, device, dtype, init="zero")
        elif self.pe_type == "learned":
            return self._ensure_param("learned_pe_param", param_shape, device, dtype, init="normal")
        raise ValueError(
            f"Unsupported pe_type={self.pe_type}. Supported types: fixed, sinusoidal, learned"
        )

    def apply_pe(
        self,
        x_processed: torch.Tensor,
        step: int,
        total_steps: int,
        training: bool,
        lambda_override: Optional[float] = None,
    ):
        if not self.pe_enabled:
            self.last_lambda = 0.0
            self.last_pe_norm = 0.0
            self.last_reg_loss = x_processed.new_tensor(0.0)
            return x_processed

        if lambda_override is not None:
            lam = float(lambda_override)
        else:
            lam = self.get_lambda(step, total_steps, training)
        pe = self.build_pe(x_processed.shape, x_processed.device, x_processed.dtype)
        scaled_pe = float(lam) * self.pe_scale_gamma * pe

        if self.pe_mode == "add":
            out = x_processed + scaled_pe
        elif self.pe_mode == "concat":
            # torch.cat does not broadcast, so expand scaled_pe to match x_processed
            out = torch.cat([x_processed, scaled_pe.expand_as(x_processed)], dim=-1)
        else:
            raise ValueError(
                f"Unsupported pe_mode={self.pe_mode}. Supported modes: add, concat"
            )

        self.last_lambda = float(lam)
        self.last_pe_norm = float(scaled_pe.detach().pow(2).mean().sqrt().item())
        self.last_reg_loss = float(self.pe_reg_beta) * scaled_pe.pow(2).mean()
        return out

    def regularization_loss(self, step: Optional[int] = None, total_steps: Optional[int] = None):
        if self.last_reg_loss is None:
            return self.pe_scale_gamma.new_tensor(0.0)
        return self.last_reg_loss


class PoseRelaxation(nn.Module):
    def __init__(
        self,
        enabled: bool = False,
        schedule: str = "linear",
        gamma_init: float = 1.0,
        keep_at_test: bool = False,
        reg_beta: float = 0.0,
    ):
        super().__init__()
        self.enabled = enabled
        self.keep_at_test = keep_at_test
        self.reg_beta = reg_beta
        self.scheduler = AnnealScheduler(schedule)
        self.gamma_pose = nn.Parameter(torch.tensor(float(gamma_init)))

        self.last_lambda = 0.0
        self.last_pose_norm = 0.0
        self.last_reg_loss = None

    def get_lambda(self, step: int, total_steps: int, training: bool) -> float:
        if not self.enabled:
            return 1.0
        if training:
            return self.scheduler(step, total_steps)
        return 1.0 if self.keep_at_test else 0.0

    def apply_pose(
        self,
        pose_feature: torch.Tensor,
        step: int,
        total_steps: int,
        training: bool,
        lambda_override: Optional[float] = None,
    ):
        if lambda_override is not None:
            lam = float(lambda_override)
        else:
            lam = self.get_lambda(step, total_steps, training)
        scaled_pose = float(lam) * self.gamma_pose * pose_feature
        self.last_lambda = float(lam)
        self.last_pose_norm = float(scaled_pose.detach().pow(2).mean().sqrt().item())
        self.last_reg_loss = float(self.reg_beta) * scaled_pose.pow(2).mean()
        return scaled_pose

    def regularization_loss(self):
        if self.last_reg_loss is None:
            return self.gamma_pose.new_tensor(0.0)
        return self.last_reg_loss
