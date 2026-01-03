import torch
import torch.nn as nn
import torch.nn.functional as F


class ModReLU(nn.Module):
    """modReLU: relu(|z| + b) * z/|z|"""

    def __init__(self, channels: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, xr, xi):
        mag = torch.sqrt(xr * xr + xi * xi + 1e-8)
        scale = F.relu(mag + self.b) / (mag + 1e-8)
        return xr * scale, xi * scale


class RealBlockModReLU(nn.Module):
    """
    modReLU on concatenated real tensor:
    x = [xr, xi] -> [xr * s, xi * s], where s depends on magnitude.
    """

    def __init__(self, channels_complex: int):
        super().__init__()
        self.channels = channels_complex
        self.b = nn.Parameter(torch.zeros(1, channels_complex, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = x.split(self.channels, dim=1)
        mag = torch.sqrt(xr * xr + xi * xi + 1e-8)
        scale = F.relu(mag + self.b) / (mag + 1e-8)
        xr = xr * scale
        xi = xi * scale
        return torch.cat([xr, xi], dim=1)
