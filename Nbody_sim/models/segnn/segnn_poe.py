import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps
import e3nn.o3 as o3
import math
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm


class SEGNNPOE(nn.Module):
    """Steerable E(3) equivariant message passing network"""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        num_layers,
        norm=None,
        pool="avg",
        task="graph",
        additional_message_irreps=None,
        poe_type: str = "fixed",
        poe_l1_mask: bool = True,
        poe_dropout: float = 0.0,
        poe_scale_init: float = 1.0,
    ):
        super().__init__()
        self.task = task
        self.poe_type = poe_type
        self.poe_l1_mask = poe_l1_mask
        self.poe_dropout = poe_dropout
        # Create network, embedding first
        # self.embedding_layer_1 = O3TensorProductSwishGate(
        #     input_irreps, hidden_irreps, node_attr_irreps
        # )
        # self.embedding_layer_2 = O3TensorProduct(
        #     hidden_irreps, hidden_irreps, node_attr_irreps
        # )

        self.embedding_layer = O3TensorProduct(
            input_irreps, hidden_irreps, node_attr_irreps
        )

        # Message passing layers.
        layers = []
        for i in range(num_layers):
            layers.append(
                SEGNNLayerEquiv(
                    hidden_irreps,
                    hidden_irreps,
                    hidden_irreps,
                    edge_attr_irreps,
                    node_attr_irreps,
                    norm=norm,
                    additional_message_irreps=additional_message_irreps,
                )
            )
        self.layers = nn.ModuleList(layers)
        self.input_irreps=input_irreps
        self.node_attr_irreps=node_attr_irreps
        self.edge_attr_irreps=edge_attr_irreps
        self.pos_irreps=o3.Irreps("1x1o")

        # Determine total channels corresponding to rotation-equivariant (l=1) parts
        l1_dims = 0
        for mul, ir in self.input_irreps:
            if ir.l == 1:
                l1_dims += mul * ir.dim
        self._l1_dims = l1_dims

        # Positional encoding operator (applied only to l=1 channels)
        if self.poe_type in ("fixed", "sin") and self._l1_dims > 0:
            poe_dim = self._l1_dims if self.poe_l1_mask else self.input_irreps.dim
            self.poe = POEAppr(
                poe_type=self.poe_type,
                dims=poe_dim,  # Option to apply PE to all channels for ablation
                dropout=self.poe_dropout,
                poe_scale_init=poe_scale_init,
            )
        else:
            self.poe = None
        # Prepare for output irreps, since the attrs will disappear after pooling
        if task == "graph":
            pooled_irreps = (
                (output_irreps * hidden_irreps.num_irreps).simplify().sort().irreps
            )
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, pooled_irreps, node_attr_irreps
            )
            self.post_pool1 = O3TensorProductSwishGate(pooled_irreps, pooled_irreps)
            self.post_pool2 = O3TensorProduct(pooled_irreps, output_irreps)
            self.init_pooler(pool)
        elif task == "node":
            self.pre_pool1 = O3TensorProductSwishGate(
                hidden_irreps, hidden_irreps, node_attr_irreps
            )
            self.pre_pool2 = O3TensorProduct(
                hidden_irreps, output_irreps, node_attr_irreps
            )

    def init_pooler(self, pool):
        """Initialise pooling mechanism"""
        if pool == "avg":
            self.pooler = global_mean_pool
        elif pool == "sum":
            self.pooler = global_add_pool

    def catch_isolated_nodes(self, graph):
        """Isolated nodes should also obtain attributes"""
        if (
            graph.contains_isolated_nodes()
            and graph.edge_index.max().item() + 1 != graph.num_nodes
        ):
            nr_add_attr = graph.num_nodes - (graph.edge_index.max().item() + 1)
            add_attr = graph.node_attr.new_tensor(
                np.zeros((nr_add_attr, node_attr.shape[-1]))
            )
            graph.node_attr = torch.cat((graph.node_attr, add_attr), -2)
        # Trivial irrep value should always be 1 (is automatically so for connected nodes, but isolated nodes are now 0)
        graph.node_attr[:, 0] = 1.0

    def forward(self, graph):
        """SEGNN forward pass"""
        x, pos, edge_index, edge_attr, node_attr, batch = (
            graph.x,
            graph.pos,
            graph.edge_index,
            graph.edge_attr,
            graph.node_attr,
            graph.batch,
        )
        try:
            additional_message_features = graph.additional_message_features
        except AttributeError:
            additional_message_features = None

        self.catch_isolated_nodes(graph)

        # POE
        

        # Optionally add positional encoding to rotation-equivariant input channels only
        if self.poe is not None:
            # Build mask over input channels corresponding to l=1 irreps
            poe_add, ld_norm, nrm = self.poe(x, batch)
            x = x.clone()
            if self.poe_l1_mask:
                equiv_mask = []
                for mul, ir in self.input_irreps:
                    ch = mul * ir.dim
                    if ir.l == 1:
                        equiv_mask.extend([True] * ch)
                    else:
                        equiv_mask.extend([False] * ch)
                equiv_mask = torch.tensor(equiv_mask, dtype=torch.bool, device=x.device)

                # Get additive PE and its norms; hook can collect these
                
                x[:, equiv_mask] = x[:, equiv_mask] + poe_add
            else:
                x = x + poe_add
        # Embed
        x = self.embedding_layer(x, node_attr)

        # Pass messages
        for layer in self.layers:
            x = layer(
                x, edge_index, edge_attr, node_attr, batch, additional_message_features
            )

        # Pre pool
        x = self.pre_pool1(x, node_attr)
        x = self.pre_pool2(x, node_attr)

        if self.task == "graph":
            # Pool over nodes
            x = self.pooler(x, batch)

            # Predict
            x = self.post_pool1(x)
            x = self.post_pool2(x)
        return x



class SEGNNLayerEquiv(MessagePassing):
    """E(3) equivariant message passing layer."""

    def __init__(
        self,
        input_irreps,
        hidden_irreps,
        output_irreps,
        edge_attr_irreps,
        node_attr_irreps,
        norm=None,
        additional_message_irreps=None,
    ):
        super().__init__(node_dim=-2, aggr="add")
        self.hidden_irreps = hidden_irreps

        message_input_irreps = (2 * input_irreps + additional_message_irreps).simplify()
        update_input_irreps = (input_irreps + hidden_irreps).simplify()

        self.message_layer_1 = O3TensorProductSwishGate(
            message_input_irreps, hidden_irreps, edge_attr_irreps
        )
        self.message_layer_2 = O3TensorProductSwishGate(
            hidden_irreps, hidden_irreps, edge_attr_irreps
        )
        self.update_layer_1 = O3TensorProductSwishGate(
            update_input_irreps, hidden_irreps, node_attr_irreps
        )
        self.update_layer_2 = O3TensorProduct(
            hidden_irreps, hidden_irreps, node_attr_irreps
        )
        
        self.setup_normalisation(norm)

    def setup_normalisation(self, norm):
        """Set up normalisation, either batch or instance norm"""
        self.norm = norm
        self.feature_norm = None
        self.message_norm = None

        if norm == "batch":
            self.feature_norm = BatchNorm(self.hidden_irreps)
            self.message_norm = BatchNorm(self.hidden_irreps)
        elif norm == "instance":
            self.feature_norm = InstanceNorm(self.hidden_irreps)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        node_attr,
        batch,
        additional_message_features=None,
    ):
        """Propagate messages along edges"""

        x = self.propagate(
            edge_index,
            x=x,
            node_attr=node_attr,
            edge_attr=edge_attr,
            additional_message_features=additional_message_features,
        )
        # Normalise features
        if self.feature_norm:
            if self.norm == "batch":
                x = self.feature_norm(x)
            elif self.norm == "instance":
                x = self.feature_norm(x, batch)
        return x

    def message(self, x_i, x_j, edge_attr, additional_message_features):
        """Create messages"""
        if additional_message_features is None:
            input = torch.cat((x_i, x_j), dim=-1)
        else:
            input = torch.cat((x_i, x_j, additional_message_features), dim=-1)

        message = self.message_layer_1(input, edge_attr)
        message = self.message_layer_2(message, edge_attr)

        if self.message_norm:
            message = self.message_norm(message)
        return message

    def update(self, message, x, node_attr):
        """Update note features"""
        input = torch.cat((x, message), dim=-1)
        update = self.update_layer_1(input, node_attr)
        update = self.update_layer_2(update, node_attr)
        x += update  # Residual connection
        return x



class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding adapted from the Annotated Transformer.

    Source:
        https://github.com/harvardnlp/annotated-transformer/blob/master/the_annotated_transformer.py

    License:
        MIT License (c) 2018 Alexander Rush

    Modification in this repository:
        - Support odd d_model values (e.g., 3). The original implementation assumes
            an even feature dimension when assigning sin/cos to alternating channels.
            We keep sin on even-indexed channels and cos on odd-indexed channels, and
            slice the frequency vector (div_term) for the odd indices to exactly match
            the number of odd columns. This avoids shape mismatches while preserving
            the standard sinusoidal pattern.

    Args:
        d_model: feature dimension (number of channels)
        dropout: dropout probability applied after adding PE
        max_len: maximum sequence length used to precompute the encodings
    """

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        # frequencies for even indices (0,2,4,...)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10000.0) / d_model))
        )
        # assign sin to even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        # assign cos to odd positions; slice div_term to match number of odd columns when d_model is odd
        odd_cols = pe[:, 1::2]
        if odd_cols.shape[1] > 0:
            pe[:, 1::2] = torch.cos(position * div_term[: odd_cols.shape[1]])
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class FixedPOE(nn.Module):
    '''
    Add a fixed learned vector for each point in the point cloud.
    '''
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = torch.nn.Parameter(torch.randn(1, d_model))


    def forward(self, x):
        return torch.repeat_interleave(self.pe, x.size(1), dim=0).unsqueeze(0)


class POEAppr(nn.Module):
    """
    Positional encoding applicator with learnable scale and regularization reporting.
    - Applies only to rotation-equivariant channels (the caller provides l1_dims mask-slice).
    - Returns (poe_addition, ld_norm, norm) to integrate with existing hook/regularization logic.
    """
    def __init__(self, poe_type: str, dims: int, dropout: float = 0.0, poe_scale_init: float = 1.0):
        super().__init__()
        self.poe_type = poe_type
        self.dims = dims
        self.poe_scale = nn.Parameter(torch.tensor(poe_scale_init))
        self.equiv = True
        self.mix_coef = 1.0
        if self.poe_type == "fixed":
            self.fixed_vec = nn.Parameter(torch.randn(self.dims))
            self.pos_enc = None
        elif self.poe_type == "sin":
            self.fixed_vec = None
            self.pos_enc = PositionalEncoding(d_model=self.dims, dropout=dropout, max_len=5000)
        else:
            self.fixed_vec = None
            self.pos_enc = None

    def forward(self, x, batch):
        # If set to equivariant behavior, no PE is applied; also report zero norms.
        if self.equiv or (self.poe_type is None):
            zero = torch.tensor(0.0, device=x.device)
            return zero.expand(x.size(0), self.dims), zero, zero

        if self.poe_type == "fixed":
            poe = self.fixed_vec.unsqueeze(0).expand(x.size(0), -1)
        elif self.poe_type == "sin":
            counts = torch.bincount(batch)
            poe_chunks = []
            for c in counts.tolist():
                if c == 0:
                    continue
                dummy = torch.zeros(1, c, self.dims, device=x.device)
                enc = self.pos_enc(dummy)  # [1, c, d_model]
                poe_chunks.append(enc.squeeze(0))  # [c, d_model]
            poe = torch.cat(poe_chunks, dim=0)
        else:
            poe = torch.zeros(x.size(0), self.dims, device=x.device)

        poe_scaled = self.mix_coef * self.poe_scale * poe
        nrm = torch.norm(poe_scaled.flatten())**2
        ld_n = torch.tensor(0.0, device=x.device)
        return poe_scaled, ld_n, nrm
