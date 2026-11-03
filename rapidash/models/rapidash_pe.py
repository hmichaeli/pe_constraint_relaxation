import torch
import torch.nn as nn

from .rapidash import Rapidash
from .relaxation import PERelaxation


class RapidashPE(nn.Module):
    def __init__(
        self,
        backbone: Rapidash,
        processed_input_dim: int,
        pe_mode: str = "add",
        pe_type: str = "fixed",
        pe_schedule: str = "linear",
        pe_gamma_init: float = 1.0,
        pe_keep_at_test: bool = False,
        pe_reg_beta: float = 0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.pe_relaxation = PERelaxation(
            pe_enabled=True,
            pe_mode=pe_mode,
            pe_type=pe_type,
            pe_schedule=pe_schedule,
            pe_gamma_init=pe_gamma_init,
            pe_keep_at_test=pe_keep_at_test,
            pe_reg_beta=pe_reg_beta,
        )

        # Pre-initialize the learnable PE parameter so it is included in the
        # optimizer when configure_optimizers() is called (before the first
        # forward pass).  Without this, _ensure_param creates the parameter
        # lazily inside apply_pe(), i.e. *after* the optimizer has already
        # snapshotted self.parameters() — so the parameter would receive
        # gradients but never be stepped, keeping pe_norm fixed at 0.
        if pe_type in ("fixed", "learned"):
            num_ori = backbone.num_ori
            if num_ori > 0:
                param_shape = (1, 1, processed_input_dim)
            else:
                param_shape = (1, processed_input_dim)
            param_name = f"{pe_type}_pe_param"
            init = "zero" if pe_type == "fixed" else "normal"
            self.pe_relaxation._ensure_param(
                param_name, param_shape,
                device=torch.device("cpu"),
                dtype=torch.get_default_dtype(),
                init=init,
            )

        proj_in_dim = processed_input_dim * 2 if pe_mode == "concat" else processed_input_dim
        self.input_proj = nn.Linear(proj_in_dim, self.backbone.hidden_dims[0], bias=False)

    def forward(self, x, pos, edge_index, batch=None, vec=None, step: int = None, total_steps: int = None, lambda_override=None):
        if self.training and (step is None or total_steps is None):
            raise ValueError(
                "RapidashPE requires step and total_steps during training for annealing. "
                "Pass self.global_step and self.trainer.estimated_stepping_batches from the Lightning module."
            )
        step = 0 if step is None else int(step)
        total_steps = 1 if total_steps is None else max(int(total_steps), 1)

        spatial_cond = x[..., -1:] if (self.backbone.last_feature_conditioning and x is not None) else None
        layer_data = self.backbone._precompute_layer_data(pos, batch, spatial_cond)

        x = self.backbone._process_inputs(x, vec)
        x = self.pe_relaxation.apply_pe(
            x_processed=x,
            step=step,
            total_steps=total_steps,
            training=self.training,
            lambda_override=lambda_override,
        )
        x = self.input_proj(x)

        residuals = []
        for i in range(self.backbone.num_total_layers):
            residual = x
            spatial_kernel, spherical_kernel, edge_index, batch = layer_data[i]
            x = self.backbone.layers[i](x, spatial_kernel, spherical_kernel, edge_index)

            if self.backbone.skip_connections:
                if residual.shape[0] > x.shape[0]:
                    residuals.append(residual)
                elif residual.shape[0] < x.shape[0]:
                    x = x + residuals.pop()

        output = self.backbone.readout(x)
        return self.backbone._process_output(output, batch)

    def regularization_loss(self):
        return self.pe_relaxation.regularization_loss()
