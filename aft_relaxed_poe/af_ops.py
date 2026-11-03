###############################################################################
###############################################################################

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

ENABLE_FFT_AMP = True

class TruncLPF(nn.Module):
    '''
    Low pass filter implementation by truncating high frequencies
    '''
    def __init__(self, cutoff=0.5, fixed_size=None):
        super(TruncLPF, self).__init__()
        self.cutoff = cutoff
        self.fixed_size = fixed_size
    
    @autocast(enabled=ENABLE_FFT_AMP)
    def forward(self, x):
        if not ENABLE_FFT_AMP: 
            x = x.float()                 # make sure the data is FP32
    
        N = x.shape[-1]

        # Compute cutoff indices
        cutoff_low = int((N * self.cutoff) // 2) + 1
        # cutoff_high = int(N - cutoff_low)
        if N % 4 == 0:
            cutoff_low = cutoff_low - 1


        # Perform 2D RFFT
        x_fft = torch.fft.rfft2(x)

        # Zero out high frequencies in both dimensions
        # Vertical direction (full dimension)
        if cutoff_low == 1:
            # edge case where keeping only DC frequnecy - end index is "0"
            x_fft[..., cutoff_low: , :] = 0
        else:
            x_fft[..., cutoff_low: -cutoff_low + 1, :] = 0

        x_fft[...,:, cutoff_low:] = 0
        # horz_cutoff = cutoff_low + 1
        # Inverse transform with original dimensions
        out = torch.fft.irfft2(x_fft, s=(x.shape[-2], x.shape[-1]))

        return out

    def extra_repr(self) -> str:
        return f"cutoff={self.cutoff}, fixed_size={self.fixed_size}"


class DownSampleAF(nn.Module):
    '''
    Downsample by cropping low frequencies
    '''

    def __init__(self, down=2):
        super(DownSampleAF, self).__init__()
        self.down = down

    @autocast(enabled=ENABLE_FFT_AMP)
    def forward(self, x):
        if not ENABLE_FFT_AMP: 
            x = x.float()                 # make sure the data is FP32

        N = x.shape[-1]

        # Compute cutoff indices
        cutoff_low = int((N / self.down) // 2) + 1
        cutoff_high = int(N - cutoff_low)
        if N % 4 == 0:
            cutoff_low = cutoff_low - 1

        # Perform 2D RFFT
        x_fft = torch.fft.rfft2(x)


        if cutoff_low == 1:
            # edge case where keeping only DC frequnecy - end index is "0"
            x_fft = x_fft[..., :, : 1]
            x_fft = x_fft[..., :1 , :]

        elif N % 4 == 0:
            x_fft = x_fft[..., :, : cutoff_low+1]
            x_fft = torch.cat(
                (x_fft[..., :cutoff_low, :],
                x_fft[..., -cutoff_low:, :]),
                dim=-2
            )

            # zero out nyquist frequency
            x_fft[..., cutoff_low, :] = 0
            x_fft[..., :, cutoff_low] = 0

        else:
            # not divisble by 4
            x_fft = x_fft[..., :, : cutoff_low]
            x_fft = torch.cat(
                (x_fft[..., :cutoff_low, :],
                x_fft[..., -cutoff_low + 1:, :]),
                dim=-2
            )

        # Fix magnitude
        x_fft = x_fft / (self.down ** 2)

        # Inverse transform with original dimensions
        out = torch.fft.irfft2(x_fft, s=(x.shape[-2] // self.down, x.shape[-1] // self.down))

        return out

    def extra_repr(self) -> str:
        return f"down={self.down}"


class UpSampleAF(nn.Module):
    '''
    Upsample by padding in frequency domain
    '''
    def __init__(self, up: int=2):
        super(UpSampleAF, self).__init__()
        self.up = up

    @autocast(enabled=False)          #  ← disables AMP just for this forward
    def forward(self, x: torch.Tensor):
        x = x.float()                 # make sure the data is FP32
        
        B, C, H, W = x.shape
        x_rfft = torch.fft.rfft2(x)

        first_rows = (H // 2) + 1
        last_rows = (H // 2)
        out_rows = H * self.up
        pad_rows = out_rows - (first_rows + last_rows)

        out_cols = (W * self.up // 2) + 1
        inp_cols = (W // 2) + 1
        pad_cols = out_cols - inp_cols

        # pad rows
        x_rfft = torch.cat((
            x_rfft[..., : first_rows , :],
            torch.zeros((B, C, pad_rows, inp_cols), device=x_rfft.device),
            x_rfft[..., -last_rows:, :]
        ), dim=-2)

        # pad columns
        x_rfft = torch.cat((
            x_rfft,
            torch.zeros((B, C, out_rows, pad_cols), device=x_rfft.device),
        ), dim=-1)

        # Fix magnitude
        x_rfft = x_rfft * (self.up ** 2)

        # Fix nyquist frequencies
        if out_rows % 4 == 0:
            x_rfft[..., first_rows - 1, :] *= 0.5
            x_rfft[..., -last_rows, :] *= 0.5
            x_rfft[..., :, inp_cols - 1] *= 0.5

        # print(torch.abs(x_rfft))
        x = torch.fft.irfft2(x_rfft, s=(H * self.up, W * self.up))

        return x

    def extra_repr(self) -> str:
        return f"up={self.up}"

def subpixel_shift(images: torch.Tensor, up: int=2, shift_x: int=1, shift_y: int=1):
    '''
    Effective fractional shift is (shift_x / up, shift_y / up)
    '''

    up_layer = UpSampleAF(up=up).to(images.device)
    up_img_batch = up_layer(images)
    img_batch_1 = torch.roll(up_img_batch, shifts=(-shift_x, -shift_y), dims=(2, 3))[:, :, ::up, ::up]
    return img_batch_1


class LayerNormAF(nn.Module):
    r""" Alias Free version of LayerNorm.
        Supports token format (B, N, D)
    """

    def __init__(self, dim, eps=1e-6, affine=False):
        super().__init__()
        self.dim = dim
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

        self.u_dims = (1,)
        self.s_dims = (1,2)

    def forward(self, x):

        u = x.mean(self.u_dims, keepdim=True)
        s = (x - u).pow(2).mean(self.s_dims, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        if self.affine:
            x = self.weight * x + self.bias
        return x

    def extra_repr(self) -> str:
        return super().extra_repr() + f"dim={self.dim}, eps={self.eps}, u_dims={self.u_dims}, s_dims={self.s_dims}, affine={self.affine}"


class UpAct(nn.Module):
    '''
    A module that upsamples the input, applies an activation function, and then downsamples back to original resolution.
    This module is designed to enable activation functions to operate at higher resolution, 
    which can capture more detailed features, before returning to the original resolution.
    Args:
        act_layer (nn.Module): The activation function to apply at the higher resolution.
        up (int, optional): The upsampling factor. Defaults to 2.
        data_format (str, optional): The format of the input tensor. One of:
            - 'channels_first': Input in (N, C, H, W) format
            - 'channels_last': Input in (N, H, W, C) format
            - 'tokens': Input in (B, N, D) format (batch, sequence length, embedding dim)
            Defaults to 'channels_first'.
        down (Optional[int], optional): The downsampling factor. If None, uses the same value as 'up'.
            Defaults to None.
    Attributes:
        requires_input_dimension (bool): Flag indicating that this module may need explicit H,W dimensions
            when operating on token sequences.
    Input:
        x (Tensor): The input tensor in the format specified by data_format.
        H (Optional[int]): Height of the spatial dimensions, required when data_format='tokens' 
                           and the dimensions cannot be inferred.
        W (Optional[int]): Width of the spatial dimensions, required when data_format='tokens'
                           and the dimensions cannot be inferred.
    Returns:
        Tensor: The transformed tensor in the same format as the input.
    
    '''

    def __init__(self, act_layer: nn.Module, up: int = 2, data_format: str="channels_first", down: Optional[int] = None):
        super(UpAct, self).__init__()
        self.up = up
        self.down = down if down is not None else up
        self.upsample = UpSampleAF(self.up)
        self.act_layer = act_layer
        self.downsmaple = DownSampleAF(down=self.down)
        self.data_format = data_format

        self.requires_input_dimension = True

    def forward(self, x, H: Optional[int] = None,
                         W: Optional[int] = None,):

        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        elif self.data_format == 'tokens':
            B, N, D = x.shape
            if H is None or W is None:
                H = int(N ** 0.5)
                W = H
            x = torch.reshape(x.transpose(1,2), (B, D, H, W)) # B, N, D -> B, D, H, W


        out = self.upsample(x)
        out = self.act_layer(out)
        out = self.downsmaple(out)


        if self.data_format == 'channels_last':
            out = out.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        elif self.data_format == 'tokens':
            out = out.flatten(2).transpose(1, 2) # B, D, H, W -> B, N, D

        return out

    def extra_repr(self) -> str:
        return f"data_format={self.data_format}"


class PolyActPerChannel(nn.Module):
    def __init__(self, channels, init_coef=None, data_format="channels_first", in_scale=1,
                 out_scale=1,
                 train_scale=True,
                 degree=2,
                 per_channel=True,
                 **kwargs):
        super(PolyActPerChannel, self).__init__()
        if kwargs:
            print(f"Warning: PolyActPerChannel got unexpected kwargs: {kwargs}")
        self.channels = channels
        self.per_channel = per_channel
        if init_coef is None:
            # Default initialization - fit to gelu in range (-2, 2)
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
            if degree > 2:
                init_coef += [0.0] * (degree - 2)
        self.deg = degree
        coef = torch.Tensor(init_coef)
        if per_channel:
            coef = coef.repeat([channels, 1])
        else:
            coef = coef.repeat([1, 1])

        coef = torch.unsqueeze(coef, -1)
        if data_format != "tokens":
            coef = torch.unsqueeze(coef, -1)

        self.coef = nn.Parameter(coef, requires_grad=True)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None

        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        elif self.data_format == 'tokens':
            x = x.permute(0, 2, 1) # (N, D, C) -> (N, C, D)

        if self.in_scale is not None:
            x = self.in_scale * x

        x = self.calc_polynomial(x)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        elif self.data_format == 'tokens':
            x = x.permute(0, 2, 1) # (N, C, D) -> (N, D, C)

        return x

    def __repr__(self):
        # print_coef = self.coef.cpu().detach().numpy()
        print_in_scale = self.in_scale.cpu().detach().numpy() if self.in_scale else None
        print_out_scale = self.out_scale.cpu().detach().numpy() if self.out_scale else None

        rep = "PolyActPerChannel(channels={}, in_scale={}, out_scale={}, degree={}, per_channel={}".format(
            self.channels, print_in_scale, print_out_scale, self.deg, self.per_channel)
        if not self.per_channel:
            rep += ", coef={}".format(self.coef.cpu().detach().flatten().numpy())
        rep += ")"
        return rep
        
    def calc_polynomial(self, x):

        if self.deg == 2:
            # maybe this is faster?
            res = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x ** 2)
        else:
            res = self.coef[:, 0]
            for i in range(1, self.deg+1):
                res = res + self.coef[:, i] * (x ** i)

        return res


class LPFPolyActPerChannel(nn.Module):
    def __init__(self, channels, init_coef=None, data_format="channels_first", in_scale=1,
                 out_scale=1,
                 train_scale=True,
                 cutoff=0.5):
        super(LPFPolyActPerChannel, self).__init__()
        self.lpf = TruncLPF(cutoff=cutoff)

        self.channels = channels
        if init_coef is None:
            init_coef = [0.0169394634313126, 0.5, 0.3078363963999393]
        self.deg = len(init_coef) - 1
        coef = torch.Tensor(init_coef)
        coef = coef.repeat([channels, 1])
        coef = torch.unsqueeze(torch.unsqueeze(coef, -1), -1)
        self.coef = nn.Parameter(coef, requires_grad=True)

        if train_scale:
            self.in_scale = nn.Parameter(torch.tensor([in_scale * 1.0]), requires_grad=True)
            self.out_scale = nn.Parameter(torch.tensor([out_scale * 1.0]), requires_grad=True)

        else:
            if in_scale != 1:
                self.register_buffer('in_scale', torch.tensor([in_scale * 1.0]))
            else:
                self.in_scale = None

            if out_scale != 1:
                self.register_buffer('out_scale', torch.tensor([out_scale * 1.0]))
            else:
                self.out_scale = None
        assert data_format in ['channels_first', 'channels_last'], "[LPFPolyActPerChannel] data_format must be 'channels_first' or 'channels_last'"
        self.data_format = data_format

    def forward(self, x):
        if self.data_format == 'channels_last':
            x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        if self.in_scale is not None:
            x = self.in_scale * x

        x_lpf = self.lpf(x)

        x = self.coef[:, 0] + self.coef[:, 1] * x + self.coef[:, 2] * (x * x_lpf)

        if self.out_scale is not None:
            x = self.out_scale * x

        if self.data_format == 'channels_last':
            x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        return x


class ScaleShift(nn.Module):
    """
    Module that computes sigma * (x - mu) with learnable parameters.
    
    Args:
        alpha_init (float): Initial value for alpha parameter
        beta_init (float): Initial value for beta parameter
    """

    def __init__(self, mu_init=None, sigma_init=None):
        super().__init__()
        if mu_init is None:
            self.mu = 0.0
        else:
            self.mu = nn.Parameter(torch.tensor(float(mu_init)))
        if sigma_init is None:
            self.sigma = 1.0
        else:
            self.sigma = nn.Parameter(torch.tensor(float(sigma_init)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: sigma * (x - mu)
        """
        return self.sigma * (x - self.mu)
    
    def extra_repr(self) -> str:
        mu = self.mu.item() if isinstance(self.mu, torch.Tensor) else self.mu
        sigma = self.sigma.item() if isinstance(self.sigma, torch.Tensor) else self.sigma
        return f"mu={mu}, sigma={sigma}"




# ----------------------------------------------------------------------
# helpers: strided convolutions with *shared* learnable kernel
# ----------------------------------------------------------------------

class UpSampleConv(nn.Module):
    """
    Depth-wise transposed convolution used as an up-sampler.

    Args
    ----
    up           : stride / scaling factor
    kernel_size  : convolution kernel size (defaults to `up`)
    init         : "ones" (nearest-neighbour) or a float, used to fill weights
    """

    def __init__(self, up: int = 2, kernel_size: Optional[int] = None, init="ones"):
        super().__init__()
        self.up = up
        self.k = kernel_size if kernel_size is not None else up
        print("[UpSampleConv] kernel size:", self.k)
        # one (1×1×k×k) learnable kernel shared by all channels
        w_init = 1.0 if init == "ones" else float(init)
        # use register_buffer to avoid making it a trainable parameter
        self.weight = nn.Parameter(torch.full((1, 1, self.k, self.k), w_init))
        # self.register_buffer('weight', torch.full((1, 1, self.k, self.k), w_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]                                    # channels at runtime
        # repeat the 1×1 kernel C times → shape (C,1,k,k), groups=C
        w = self.weight.repeat(C, 1, 1, 1)
        return F.conv_transpose2d(x, w, stride=self.up, groups=C)


class DownSampleConv(nn.Module):
    """
    Depth-wise strided convolution used as an anti-aliased down-sampler.

    Args
    ----
    down         : stride / reduction factor
    kernel_size  : convolution kernel size (defaults to `down`)
    init         : "avg" → 1/(k²) (average pooling), or a float
    """

    def __init__(self, down: int = 2, kernel_size: Optional[int] = None, init="avg"):
        super().__init__()
        self.down = down
        self.k = kernel_size if kernel_size is not None else down

        if init == "avg":
            w_init = 1.0 / (self.k * self.k)
        else:
            w_init = float(init)

        self.weight = nn.Parameter(torch.full((1, 1, self.k, self.k), w_init))
        # self.register_buffer('weight', torch.full((1, 1, self.k, self.k), w_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        C = x.shape[1]
        w = self.weight.repeat(C, 1, 1, 1)
        return F.conv2d(x, w, stride=self.down, groups=C)

# ----------------------------------------------------------------------
# the new UpAct module (conv-based up / down)
# ----------------------------------------------------------------------

class UpActConv(nn.Module):
    """
    Upsample → activation → downsample implemented with depth-wise
    strided convolutions.

    Args
    ----
    channels      : number of feature-map channels
    act_layer     : any point-wise non-linearity (e.g. GELU, ReLU, PolyAct…)
    up            : upsampling factor (default: 2)
    down          : downsampling factor (default: up)
    data_format   : 'channels_first', 'channels_last', or 'tokens'
    """

    def __init__(
        self,
        act_layer: nn.Module,
        up: int = 2,
        down: Optional[int] = None,
        data_format: str = "channels_first",
    ):
        super().__init__()
        assert data_format in ("channels_first", "channels_last", "tokens")
        self.up = up
        self.down = down if down is not None else up
        self.data_format = data_format
        print(f"[UpActConv] up={self.up}, down={self.down}, data_format='{self.data_format}'")
        self.upsample = UpSampleConv(self.up)
        self.downsample = DownSampleConv(down=self.down)
        self.act = act_layer

        # Token layout needs height / width hints
        self.requires_input_dimension = data_format == "tokens"

    # --------------------------------------------------------------

    def forward(self, x: torch.Tensor, H: Optional[int] = None, W: Optional[int] = None):
        # ─── reshape helpers ───────────────────────────────────────
        if self.data_format == "channels_last":          # (N, H, W, C) → (N,C,H,W)
            x = x.permute(0, 3, 1, 2)
        elif self.data_format == "tokens":               # (B,N,D) → (B,D,H,W)
            B, N, D = x.shape
            if H is None or W is None:
                H = int(N**0.5)
                W = H
            x = x.transpose(1, 2).reshape(B, D, H, W)

        # ─── core ──────────────────────────────────────────────────
        x = self.upsample(x)        # ↑
        x = self.act(x)             # σ
        x = self.downsample(x)      # ↓

        # ─── back to original layout ───────────────────────────────
        if self.data_format == "channels_last":          # (N,C,H,W) → (N,H,W,C)
            x = x.permute(0, 2, 3, 1)
        elif self.data_format == "tokens":               # (B,D,H,W) → (B,N,D)
            x = x.flatten(2).transpose(1, 2)

        return x

    def extra_repr(self) -> str:
        return (f"up={self.up}, down={self.down}, "
                f"data_format='{self.data_format}'")
