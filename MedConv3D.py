import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _triple
from typing import Optional
import random
import numpy as np
import os

# set spacing env variable for testing
os.environ['SPACING'] = '0.3,0.7,0.9'

os.environ['SPACING_TYPE'] = 'during'



class NaiveConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x, spacing = None):

        if spacing is not None:
            patch_spacing = self.get_spacing_kernel(spacing)

        B, C, D, H, W = x.shape
        k = self.kernel_size
        out_D = D - k + 1
        out_H = H - k + 1
        out_W = W - k + 1

        out = torch.zeros(
            B, self.weight.shape[0], out_D, out_H, out_W,
            device=x.device
        )

        for b in range(B):
            for oc in range(self.weight.shape[0]):
                for ic in range(C):
                    for d in range(out_D):
                        for h in range(out_H):
                            for w in range(out_W):
                                patch = x[b, ic, d:d+k, h:h+k, w:w+k]
                                if spacing is not None:
                                    patch = patch * patch_spacing

                                out[b, oc, d, h, w] += torch.sum(
                                    patch * self.weight[oc, ic]
                                )
                out[b, oc] += self.bias[oc]

        return out
    
    def get_spacing_kernel(self, spacings):

        sx, sy, sz = spacings

        # Offsets relative to center voxel
        offsets = torch.tensor([-1, 0, 1], dtype=torch.float32)

        # Create coordinate grid (z, y, x)
        dz, dy, dx = torch.meshgrid(offsets, offsets, offsets, indexing="ij")

        # Physical distances
        dist = torch.sqrt(
            (dx * sx) ** 2 +
            (dy * sy) ** 2 +
            (dz * sz) ** 2
        )

        # Enforce center value = 1.0
        dist[1, 1, 1] = 1.0

        return dist
    
class MedConv3D_old(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

    def forward(self, x, spacing=None):
        weight = self.weight

        if spacing is not None:
            spacing_kernel = self.get_spacing_kernel(spacing).to(
                device=x.device, dtype=x.dtype
            )

            # Broadcast spacing kernel over (out_channels, in_channels)
            weight = weight * spacing_kernel[None, None, ...]

        return F.conv3d(
            x,
            weight,
            bias=self.bias,
            stride=1,
            padding=0
        )

    @staticmethod
    def get_spacing_kernel(spacings):
        sx, sy, sz = spacings

        offsets = torch.tensor([-1, 0, 1], dtype=torch.float32)
        dz, dy, dx = torch.meshgrid(offsets, offsets, offsets, indexing="ij")

        dist = torch.sqrt(
            (dx * sx) ** 2 +
            (dy * sy) ** 2 +
            (dz * sz) ** 2
        )

        dist[1, 1, 1] = 1.0
        return dist

class MedConv3D(torch.nn.Conv3d):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1,
                 padding=0, 
                 dilation=1, 
                 groups=1, 
                 bias=True, 
                 padding_mode='zeros',
                 device=None,
                 dtype=None) -> None:
        super(MedConv3D, self).__init__(in_channels, out_channels, kernel_size,
                                        stride, padding, dilation, groups, bias, padding_mode,
                                        device=device, dtype=dtype)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        
        spacing_env = os.getenv('SPACING', None)
        if spacing_env is not None:
            spacing = tuple(map(float, spacing_env.split(',')))

        spacing_type = os.getenv('SPACING_TYPE')

        if spacing is not None and spacing_type == 'during':
            spacing_kernel = self.get_spacing_kernel(spacing).to(
                device=input.device, dtype=input.dtype
            )

            # Broadcast spacing kernel over (out_channels, in_channels)
            weight = weight * spacing_kernel[None, None, ...]
        
        if self.padding_mode != "zeros":
            assert "weshouldnt pad"
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        x = F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

        if spacing is not None and spacing_type == 'after':
            spacing_kernel = self.get_spacing_kernel(spacing).to(
                device=input.device, dtype=input.dtype
            )

            spacing_kernel.unsqueeze_(0).unsqueeze_(0)

            x = F.conv3d(
                x, spacing_kernel, bias, self.stride, "same", self.dilation, self.groups
            )

        return x

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)

    @staticmethod
    def get_spacing_kernel(spacings):
        
        sx, sy, sz = spacings

        offsets = torch.tensor([-1, 0, 1], dtype=torch.float32)
        dz, dy, dx = torch.meshgrid(offsets, offsets, offsets, indexing="ij")

        dist = torch.sqrt(
            (dx * sx) ** 2 +
            (dy * sy) ** 2 +
            (dz * sz) ** 2
        )

        dist[1, 1, 1] = 1.0
        dist = 1/dist
        dist[1, 1, 1] = torch.sqrt(torch.sum(dist))

        return dist


torch.nn.Conv3d = MedConv3D

if __name__ == "__main__":

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    input_tensor = torch.randn(1, 1, 256, 256, 256)
    spacings = (0.3, 0.7, 0.7)

    # # Example usage
    model_original = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3)
    # model_naive = NaiveConv3D(in_channels=1, out_channels=1, kernel_size=3)
    model_med = MedConv3D(in_channels=1, out_channels=1, kernel_size=3)    
    model_med.load_state_dict(model_original.state_dict())
    
    t1 = time.time()
    output_tensor_original = model_original(input_tensor)
    t2 = time.time()
    print("Original Conv3D output shape:", output_tensor_original.shape)
    print("Original Conv3D time taken:", t2 - t1)

    # t1 = time.time()
    # output_tensor_naive = model_naive(input_tensor)
    # t2 = time.time()
    # # print("NaiveConv3D output shape:", output_tensor_naive.shape)
    # print("NaiveConv3D time taken:", t2 - t1)

    t1 = time.time()
    output_tensor_med = model_med(input_tensor, spacing=spacings)
    t2 = time.time()
    print("MedConv3D output shape:", output_tensor_med.shape)
    print("MedConv3D time taken:", t2 - t1)    

    assert torch.allclose(output_tensor_original, output_tensor_med, atol=1e-6), "Outputs are not close!"
    print("Outputs are close!")

    # spacings = (1.0, 1.0, 1.0)
    # spacing_kernel = model_med.get_spacing_kernel(spacings)
    # print("Spacing kernel:\n", spacing_kernel)



