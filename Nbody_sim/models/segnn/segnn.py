import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool
from e3nn.nn import BatchNorm
import numpy as np
from e3nn.o3 import Irreps
import e3nn.o3 as o3
from .o3_building_blocks import O3TensorProduct, O3TensorProductSwishGate
from .instance_norm import InstanceNorm


class SEGNN(nn.Module):
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
    ):
        super().__init__()
        self.task = task
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
                SEGNNLayer(
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

        # Embed
        # x = self.embedding_layer_1(x, node_attr)
        # x = self.embedding_layer_2(x, node_attr)
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

def create_gen(irreps):
    gen_x=[]
    gen_y=[]
    gen_z=[]
    for irrep in irreps:
        for (mul,ir) in irrep:
            gen=o3.so3_generators(ir.l)
            for _ in range(mul):
                gen_x.append(gen[0])
                gen_y.append(gen[1])
                gen_z.append(gen[2])
    X_in_bl=torch.block_diag(*gen_x)
    Y_in_bl=torch.block_diag(*gen_y)
    Z_in_bl=torch.block_diag(*gen_z)
    in_bl=torch.cat([X_in_bl,Y_in_bl,Z_in_bl],dim=0)
    return in_bl.T

def LieBracketNorm(in_irreps,out_irreps):

    genIn=create_gen(in_irreps).float().to('cuda')
    genOut=create_gen(out_irreps).float().to('cuda')
    return genIn,genOut

class LinearAppr(nn.Module):
    def __init__(self,in_irrep,out_irrep):
        super().__init__()
        self.lin=nn.Linear(in_irrep.dim,out_irrep.dim,bias=False)
        self.lieBracket=LieBracketNorm([in_irrep],[out_irrep])
        torch.nn.init.normal_(self.lin.weight,mean=0,std=0.0001)
        #torch.nn.init.normal_(self.lin.bias,0)
        self.equiv=True
        self.mix_coef=1
        self.unfl=nn.Unflatten(-1,(-1,3))

    def forward(self,x):
        if self.equiv:
            return torch.tensor(0).to(x.device), torch.tensor(0).to(x.device), torch.tensor(0).to(x.device)
        else:
            wx=self.lin(x)
            Wx=self.lin(x.detach())
            AWx=self.unfl(Wx@self.lieBracket[1]).transpose(-1,-2)
            WAx=self.lin(self.unfl(x.detach()@self.lieBracket[0]).transpose(-1,-2))
            n=torch.norm(Wx.flatten())**2
            ld_n=torch.norm((WAx-AWx).flatten())**2
            return self.mix_coef*wx,ld_n,n


class SEGNNLayer(MessagePassing):
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
        self.linear_pass1=LinearAppr(update_input_irreps,hidden_irreps)
        self.linear_pass2=LinearAppr(hidden_irreps,hidden_irreps)

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
        update = self.update_layer_1(input, node_attr)+self.linear_pass1(input)[0]
        update = self.update_layer_2(update, node_attr)+self.linear_pass2(update)[0]
        x += update  # Residual connection
        return x
