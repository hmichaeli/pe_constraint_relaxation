import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.pointnet_equi.layers import *
from models.pointnet_equi.pointnet import PointNetEncoder_Dual
import math

class get_model(nn.Module):
    def __init__(self, args, k=40, normal_channel=True):
        super(get_model, self).__init__()
        self.args = args
        width = getattr(args, 'width', 1024)
        if normal_channel:
            channel = 6
        else:
            channel = 3
        
        if args.poe == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(channel, dropout=0, max_len=args.num_point)
            self.pos_encoder_scale = torch.nn.Parameter(torch.ones(1))
        elif args.poe == 'fixed':
            self.pos_encoder = FixedPOE(channel, max_len=args.num_point)
            self.pos_encoder_scale = torch.nn.Parameter(torch.ones(1))
        else:
            self.pos_encoder = None

        # Additive / concatenative POE handled in PointNetEncoder_Dual. Default is additive.
        self.poe_concat = getattr(args, 'poe_concat', False)
        if self.poe_concat:
            channel *= 2

        self.feat = PointNetEncoder_Dual(args, global_feat=True, feature_transform=True, channel=channel, in_dim=channel, strict_equiv=True, width=width)
        self.fc1 = nn.Linear(width//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x,equiv,mix):
        if self.pos_encoder is not None and not equiv:
            # use positional encoding when not equivariant
            pe = self.pos_encoder_scale * self.pos_encoder(torch.zeros_like(x).transpose(1,2)).transpose(1,2)
            n_ld = torch.tensor(0.0)
            n = torch.norm(pe.reshape(-1))**2
        else:
            pe = torch.zeros_like(x)
            n_ld = torch.tensor(0.0)
            n = torch.tensor(0.0)
        
        if self.poe_concat:
                if pe.shape[0] != x.shape[0]:
                    pe = pe.repeat(x.shape[0], 1, 1)
                x = torch.cat([x, mix * pe], dim=1)
        else:
            x = x + mix * pe

        x, trans, trans_feat,_,_ = self.feat(x,True,mix)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat,(n_ld,n)

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        #self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        #mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss #+ mat_diff_loss * self.mat_diff_loss_scale
        return total_loss


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
