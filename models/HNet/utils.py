import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

"""
# --------------------------------------------
# LayerNorm for Vision Normalization
# --------------------------------------------
"""
class LayerNorm(nn.Module):
    '''
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    '''

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


def sequential(*args):
    """Advanced nn.Sequential.

    Args:
        nn.Sequential, nn.Module

    Returns:
        nn.Sequential
    """
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]  # No sequential is needed.
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

"""
# --------------------------------------------
# implementation of Converse2d operator ^_^
# --------------------------------------------
"""
class Converse2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale=1, padding=2, padding_mode='circular', eps=1e-5):
        super(Converse2D, self).__init__()
        """
        Converse2D Operator for Image Restoration Tasks.

        Args:
            x (Tensor): Input tensor of shape (N, in_channels, H, W), where
                        N is the batch size, H and W are spatial dimensions.
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the operation.
            kernel_size (int): Size of the kernel.
            scale (int): Upsampling factor. For example, `scale=2` doubles the resolution.
            padding (int): Padding size. Recommended value is `kernel_size - 1`.
            padding_mode (str, optional): Padding method. One of {'reflect', 'replicate', 'circular', 'constant'}.
                                        Default is `circular`.
            eps (float, optional): Small value added to denominators for numerical stability.
                                Default is a small value like 1e-5.

        Returns:
            Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale), where spatial dimensions
                    are upsampled by the given scale factor.
        """
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size =  kernel_size
        self.scale = scale
        self.padding = padding
        self.padding_mode = padding_mode
        self.eps = eps


        # ensure depthwise
        assert self.out_channels == self.in_channels
        self.weight = nn.Parameter(torch.randn(1, self.in_channels, self.kernel_size, self.kernel_size))
        self.bias = nn.Parameter(torch.zeros(1, self.in_channels, 1, 1))
        self.weight.data = nn.functional.softmax(self.weight.data.view(1,self.in_channels,-1), dim=-1).view(1, self.in_channels, self.kernel_size, self.kernel_size)

        
    def forward(self, x):

        if self.padding > 0:
            x = nn.functional.pad(x, pad=[self.padding, self.padding, self.padding, self.padding], mode=self.padding_mode, value=0)

        self.biaseps = torch.sigmoid(self.bias-9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, scale=self.scale)
        if self.scale != 1:
            x = nn.functional.interpolate(x, scale_factor=self.scale, mode='nearest')
            # x = nn.functional.interpolate(x, scale_factor=self.scale, mode='bilinear',align_corners=False)
        # x = torch.zeros_like(x)

        FB = self.p2o(self.weight, (h*self.scale, w*self.scale))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)
        FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
        
        FR = FBFy + torch.fft.fftn(self.biaseps*x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, self.scale), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, self.scale), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + self.biaseps)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, self.scale, self.scale)
        FX = (FR-FCBinvWBR)/self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if self.padding > 0:
            out = out[..., self.padding*self.scale:-self.padding*self.scale, self.padding*self.scale:-self.padding*self.scale]

        return out

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b


    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2]/2), -int(psf.shape[-1]/2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2,-1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z


"""
# --------------------------------------------
# implementation of Converse Block
# --------------------------------------------
"""
class ConverseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=1, padding=2, padding_mode='replicate', eps=1e-5):
        super(ConverseBlock, self).__init__()
        """
        ConverseBlock: A Convolutional Block for Image Restoration using Converse2D Operations.

        This block consists of two main sub-blocks, each incorporating normalization, pointwise convolution,
        non-linearity, and (optionally) a custom reverse convolution (`Converse2D`) for learnable upsampling.
        It also includes residual connections to preserve information and improve gradient flow.

        Args:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels to be produced by the block.
            kernel_size (int, optional): Kernel size used in the `Converse2D` operation. Default: 3.
            scale (int, optional): Upsampling scale factor. Default: 1 (no upsampling).
            padding (int, optional): Padding size for `Converse2D`. Default: 2.
            padding_mode (str, optional): Padding mode to use in `Converse2D`. One of {'reflect', 'replicate', 'circular', 'constant'}. Default: 'circular'.
            eps (float, optional): A small epsilon value for numerical stability in normalization layers. Default: 1e-6.

        Forward:
            x (Tensor): Input tensor of shape (N, in_channels, H, W)
            Returns:
                Tensor: Output tensor of shape (N, out_channels, H * scale, W * scale)
        """

        self.conv1 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(in_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   Converse2D(2*out_channels, 2*out_channels, kernel_size, scale=scale, padding=padding, padding_mode=padding_mode, eps=eps), 
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
        self.conv2 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(out_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
    def forward(self, x):
        x = self.conv1(x) + x
        x = self.conv2(x) + x
        return x


class ConverseBlockAlphaVariant(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, scale=1, padding=2, padding_mode='replicate', eps=1e-5):
        super(ConverseBlockAlphaVariant, self).__init__()
        """
        ConverseBlockAlphaVariant: Only difference is the addition of alpha parameters to the convolution blocks.
        """
        
        self.alpha1 = nn.Parameter(torch.zeros((1, out_channels, 1, 1)), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.zeros((1, out_channels, 1, 1)), requires_grad=True)


        self.conv1 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(in_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   Converse2D(2*out_channels, 2*out_channels, kernel_size, scale=scale, padding=padding, padding_mode=padding_mode, eps=eps), 
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
                                  
        self.conv2 = nn.Sequential(LayerNorm(in_channels, eps=1e-5, data_format="channels_first"),
                                   nn.Conv2d(out_channels, 2*out_channels, 1, 1, 0),
                                   nn.GELU(),
                                   nn.Conv2d(2*out_channels, out_channels, 1, 1, 0))
        
                                  
    def forward(self, x):
        x = self.alpha1 * self.conv1(x) + x
        x = self.alpha2 * self.conv2(x) + x
        return x

class ResidualBlock(nn.Module):
    """Residual block
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, num_features=64):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return x + out

class KernelNet(nn.Module):
    def __init__(self, kernel_size=7):
        super(KernelNet, self).__init__()
        self.kernel_size = kernel_size
        self.fc1 = nn.Linear(kernel_size ** 2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 16 * (kernel_size ** 2))
        self.gelu = nn.GELU()

    def forward(self, k):
        k = k.to(torch.float)
        b, _, _, _ = k.shape
        k = k.view(b, -1)  # flatten
        # fully connected
        k = self.gelu(self.fc1(k))
        k = self.gelu(self.fc2(k))
        k = self.fc3(k)
        k = k.view(b, 16, self.kernel_size, self.kernel_size)
        return k

class Converse2D_kernel(nn.Module):
    def __init__(self, features, eps=1e-3):
        super(Converse2D_kernel, self).__init__()
        '''
        Converse2d operator for condition the kernel
        '''
        self.alpha = nn.Parameter(torch.zeros(1, features, 1, 1))
        self.eps = eps

    def forward(self, x, k, sf, padding=0, padding_mode='circular'):

        if padding > 0:
            x = nn.functional.pad(x, pad=[padding, padding, padding, padding], mode=padding_mode, value=0)
        self.biaseps = torch.sigmoid(self.alpha - 9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, sf)
        if sf != 1:
            x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')

        FB = self.p2o(k, (h * sf, w * sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)

        FBFy = FBC * torch.fft.fftn(STy, dim=(-2, -1))
        FR = FBFy + torch.fft.fftn(self.biaseps * x, dim=(-2, -1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + self.biaseps)
        FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
        FX = (FR - FCBinvWBR) / self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if padding > 0:
            out = out[..., padding * sf:-padding * sf, padding * sf:-padding * sf]

        return out

    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3,
                                                          len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b

    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[..., :psf.shape[-2], :psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2] / 2), -int(psf.shape[-1] / 2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2, -1))

        return otf

    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2] * scale, x.shape[3] * scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z

