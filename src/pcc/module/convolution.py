import torch
import torch.nn as nn


class ComplexConv2d(nn.Module):
    """Complex convolution using two real Conv2d ops."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 3,
        padding: int = 1,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.wr = nn.Conv2d(in_ch, out_ch, k, padding=padding, stride=stride, bias=bias)
        self.wi = nn.Conv2d(in_ch, out_ch, k, padding=padding, stride=stride, bias=bias)

    def forward(self, xr, xi):
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return yr, yi


class RealBlockComplexConv2d(nn.Module):
    """
    Real-valued layer equivalent to ComplexConv2d but operating on concatenated channels.
    Input:  x = cat([xr, xi], dim=1) with shape [B, 2*Cin, H, W]
    Output: y = cat([yr, yi], dim=1) with shape [B, 2*Cout, H', W']
    """

    def __init__(self, cin: int, cout: int, k=3, padding=1, stride=1, bias=True):
        super().__init__()
        self.cin = cin
        self.cout = cout
        self.wr = nn.Conv2d(cin, cout, k, padding=padding, stride=stride, bias=bias)
        self.wi = nn.Conv2d(cin, cout, k, padding=padding, stride=stride, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.split(self.cin, dim=1)
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return torch.cat([yr, yi], dim=1)
