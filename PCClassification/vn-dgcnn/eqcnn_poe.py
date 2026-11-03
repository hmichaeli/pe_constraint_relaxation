import copy
import torch
import torch.nn as nn
from model_equi import EQCNN_cls_normDual
import math


class EQCNN_POE_cls_normDual(nn.Module):
    def __init__(self, args, output_channels=40):
        super().__init__()
        self.args = args
        self.k = args.k
        # The wrapped EQCNN is always built strictly equivariant (no dense layers).
        # Use a copy of args so the caller's strict_equiv flag, which controls
        # whether the PE is applied during training, is left unchanged.
        backbone_args = copy.copy(args)
        backbone_args.strict_equiv = True
        pe_dim = 3
        if args.poe == 'sinusoidal':
            self.pos_encoder = PositionalEncoding(3, dropout=0, max_len=args.num_points)
        elif args.poe == 'fixed':
            self.pos_encoder = FixedPOE(3, max_len=args.num_points)
        elif args.poe == 'fixed_so':
            self.pos_encoder = FixedSOPOE()
            pe_dim = 9
        else:
            self.pos_encoder = None
        # PE scale gamma: learnable (default) or kept fixed at its initialization.
        gamma = torch.tensor([getattr(args, 'poe_scale_init', 1.0)], dtype=torch.float32)
        if getattr(args, 'poe_learnable_scale', True):
            self.pos_encoder_scale = torch.nn.Parameter(gamma)
        else:
            self.register_buffer('pos_encoder_scale', gamma)

        # Additive / concatenative POE
        self.poe_concat = getattr(args, 'poe_concat', False)
        if pe_dim != 3 and not self.poe_concat:
            raise ValueError("fixed_so PE has 3 vector channels and requires --poe_concat")
        self.pe_dim = pe_dim
        in_channels = 3
        if self.poe_concat:
            in_channels = 3 + pe_dim

        self.eqcnn = EQCNN_cls_normDual(backbone_args, output_channels, in_channels=in_channels)


    def forward(self, x,equiv,mix):
        if self.pos_encoder is not None and not equiv:
            # use positional encoding when not equivariant
            pe = self.pos_encoder_scale * self.pos_encoder(torch.zeros_like(x).transpose(1,2)).transpose(1,2)
            n_ld = torch.tensor(0.0)
            n = torch.norm(pe.reshape(-1))**2
        else:
            pe = x.new_zeros(x.shape[0], self.pe_dim, x.shape[2])
            n_ld = torch.tensor(0.0)
            n = torch.tensor(0.0)
        
        if self.poe_concat:
            if pe.shape[0] != x.shape[0]:
                pe = pe.repeat(x.shape[0], 1, 1)
            x = torch.cat([x, mix * pe], dim=1)
        else:
            x = x + mix * pe
        
        # force equivariant eqcnn
        x, _ = self.eqcnn(x, equiv=True, mix=0)
        
        return x,(n_ld,n)


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


class FixedSOPOE(nn.Module):
    '''
    Fixed frame with a trivial SO(3) stabilizer: three learned axis-aligned vector
    channels (a1*e1, a2*e2, a3*e3), broadcast to every point. Flattened per point,
    this is (a1, 0, 0, 0, a2, 0, 0, 0, a3).
    '''
    def __init__(self):
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(3))

    def forward(self, x):
        frame = torch.diag(self.alpha).reshape(1, 9)
        return torch.repeat_interleave(frame, x.size(1), dim=0).unsqueeze(0)