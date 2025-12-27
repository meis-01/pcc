import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Real CNN backbone ----------
class RealCNN(nn.Module):
    def __init__(self, in_ch: int, width: int = 32, num_classes: int = 2):
        super().__init__()
        w = width
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(w, 2*w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(2*w, 2*w, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(2*w, 4*w, 3, padding=1), nn.ReLU(),
            nn.Conv2d(4*w, 4*w, 3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4*w, num_classes),
        )

    def forward(self, x):
        return self.head(self.net(x))

# ---------- Complex layers implemented via real tensors ----------
class ComplexConv2d(nn.Module):
    """Complex convolution using two real Conv2d ops."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, padding: int = 1, bias: bool = True):
        super().__init__()
        self.wr = nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=bias)
        self.wi = nn.Conv2d(in_ch, out_ch, k, padding=padding, bias=bias)

    def forward(self, xr, xi):
        # (wr + i wi) * (xr + i xi) = (wr*xr - wi*xi) + i(wr*xi + wi*xr)
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return yr, yi

class ModReLU(nn.Module):
    """modReLU: relu(|z| + b) * z/|z|"""
    def __init__(self, channels: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, xr, xi):
        mag = torch.sqrt(xr*xr + xi*xi + 1e-8)
        scale = F.relu(mag + self.b) / (mag + 1e-8)
        return xr * scale, xi * scale

class ComplexMaxPool2d(nn.Module):
    def __init__(self, k=2):
        super().__init__()
        self.pool = nn.MaxPool2d(k)

    def forward(self, xr, xi):
        return self.pool(xr), self.pool(xi)

class ComplexCNN(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 24, num_classes: int = 2):
        """
        in_ch refers to number of complex channels.
        We represent complex activations as (real, imag) tensors with shape [B, C, H, W].
        """
        super().__init__()
        w = width
        self.c1 = ComplexConv2d(in_ch, w, 3, padding=1)
        self.a1 = ModReLU(w)
        self.c2 = ComplexConv2d(w, w, 3, padding=1)
        self.a2 = ModReLU(w)
        self.p1 = ComplexMaxPool2d(2)

        self.c3 = ComplexConv2d(w, 2*w, 3, padding=1)
        self.a3 = ModReLU(2*w)
        self.c4 = ComplexConv2d(2*w, 2*w, 3, padding=1)
        self.a4 = ModReLU(2*w)
        self.p2 = ComplexMaxPool2d(2)

        self.c5 = ComplexConv2d(2*w, 4*w, 3, padding=1)
        self.a5 = ModReLU(4*w)
        self.c6 = ComplexConv2d(4*w, 4*w, 3, padding=1)
        self.a6 = ModReLU(4*w)

        self.avg = nn.AdaptiveAvgPool2d(1)
        # Convert complex -> real for classification by concatenating pooled real+imag
        self.fc = nn.Linear(8*w, num_classes)

    def forward(self, xr, xi):
        xr, xi = self.c1(xr, xi); xr, xi = self.a1(xr, xi)
        xr, xi = self.c2(xr, xi); xr, xi = self.a2(xr, xi)
        xr, xi = self.p1(xr, xi)

        xr, xi = self.c3(xr, xi); xr, xi = self.a3(xr, xi)
        xr, xi = self.c4(xr, xi); xr, xi = self.a4(xr, xi)
        xr, xi = self.p2(xr, xi)

        xr, xi = self.c5(xr, xi); xr, xi = self.a5(xr, xi)
        xr, xi = self.c6(xr, xi); xr, xi = self.a6(xr, xi)

        pr = self.avg(xr).flatten(1)
        pi = self.avg(xi).flatten(1)
        feat = torch.cat([pr, pi], dim=1)
        return self.fc(feat)
