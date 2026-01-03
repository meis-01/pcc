import torch
import torch.nn as nn
import torch.nn.functional as F
from pcc.module.convolution import ComplexConv2d, RealBlockComplexConv2d
from pcc.module.normalization import ComplexBatchNorm2d, ComplexSplitBatchNorm2d
from pcc.module.activation import ModReLU, RealBlockModReLU


class RealConstrainedComplexCNN(nn.Module):
    """
    Constrained-real baseline: same algebra as complex convs, but represented as
    a real network that processes 2-channel input [Re, Im].

    Uses ordinary BatchNorm2d on 2C channels (not full complex whitening BN).
    This is intended as a *real* control for the conv-structure constraint.
    """

    def __init__(
        self,
        in_ch_complex: int = 1,
        width: int = 24,
        num_classes: int = 2,
        normalize: bool = True,
    ):
        super().__init__()
        w = width
        Cin = in_ch_complex  # complex channels
        # Note: real channel count is 2*Cin etc.
        self.normalize = normalize

        self.net = nn.Sequential(
            # Stage 1
            RealBlockComplexConv2d(Cin, w, k=3, padding=1, stride=1),
            nn.BatchNorm2d(2 * w) if self.normalize else nn.Identity(),
            RealBlockModReLU(w),
            # Stage 2 (downsample)
            RealBlockComplexConv2d(w, 2 * w, k=3, padding=1, stride=2),
            nn.BatchNorm2d(4 * w) if self.normalize else nn.Identity(),
            RealBlockModReLU(2 * w),
            # Stage 3 (downsample)
            RealBlockComplexConv2d(2 * w, 4 * w, k=3, padding=1, stride=2),
            nn.BatchNorm2d(8 * w) if self.normalize else nn.Identity(),
            RealBlockModReLU(4 * w),
        )

        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8 * w, num_classes)  # 2*(4w) pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.avg(x).flatten(1)
        return self.fc(x)


class RealCNN(nn.Module):
    def __init__(
        self, in_ch: int, width: int = 32, num_classes: int = 2, normalize: bool = True
    ):
        super().__init__()
        w = width

        self.net = nn.Sequential(
            # Stage 1 (H,W)
            nn.Conv2d(in_ch, w, 3, padding=1, stride=1),
            nn.BatchNorm2d(w) if normalize else nn.Identity(),
            nn.ReLU(),
            # Stage 2 (H/2,W/2) via stride-2 conv
            nn.Conv2d(w, 2 * w, 3, padding=1, stride=2),
            nn.BatchNorm2d(2 * w) if normalize else nn.Identity(),
            nn.ReLU(),
            # Stage 3 (H/4,W/4) via stride-2 conv
            nn.Conv2d(2 * w, 4 * w, 3, padding=1, stride=2),
            nn.BatchNorm2d(4 * w) if normalize else nn.Identity(),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * w, num_classes),
        )

    def forward(self, x):
        return self.head(self.net(x))


class ComplexCNN(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        width: int = 24,
        num_classes: int = 2,
        normalize: bool = True,
    ):
        """
        in_ch refers to number of complex channels.
        We represent complex activations as (real, imag) tensors with shape [B, C, H, W].
        """
        super().__init__()
        w = width
        self.c1 = ComplexConv2d(in_ch, w, 3, padding=1)
        self.b1 = ComplexBatchNorm2d(w) if normalize else nn.Identity()
        self.a1 = ModReLU(w)

        self.c3 = ComplexConv2d(w, 2 * w, 3, padding=1, stride=2)
        self.b3 = ComplexBatchNorm2d(2 * w) if normalize else nn.Identity()
        self.a3 = ModReLU(2 * w)

        self.c5 = ComplexConv2d(2 * w, 4 * w, 3, padding=1, stride=2)
        self.b5 = ComplexBatchNorm2d(4 * w) if normalize else nn.Identity()
        self.a5 = ModReLU(4 * w)

        self.avg = nn.AdaptiveAvgPool2d(1)
        # Convert complex -> real for classification by concatenating pooled real+imag
        self.fc = nn.Linear(8 * w, num_classes)

    def forward(self, xr, xi):
        xr, xi = self.c1(xr, xi)
        xr, xi = self.b1(xr, xi)
        xr, xi = self.a1(xr, xi)

        xr, xi = self.c3(xr, xi)
        xr, xi = self.b3(xr, xi)
        xr, xi = self.a3(xr, xi)

        xr, xi = self.c5(xr, xi)
        xr, xi = self.b5(xr, xi)
        xr, xi = self.a5(xr, xi)

        pr = self.avg(xr).flatten(1)
        pi = self.avg(xi).flatten(1)
        feat = torch.cat([pr, pi], dim=1)
        return self.fc(feat)


# -----------------------------
def make_model(model_kind: str, normalize: bool = True) -> nn.Module:
    if model_kind == "mag":
        return RealCNN(in_ch=1, width=32, normalize=normalize)
    if model_kind == "R2":
        return RealCNN(in_ch=2, width=32, normalize=normalize)
    if model_kind == "R2c":
        return RealConstrainedComplexCNN(in_ch_complex=1, width=24, normalize=normalize)
    if model_kind == "cossin":
        return RealCNN(in_ch=3, width=32, normalize=normalize)
    if model_kind == "complex":
        return ComplexCNN(in_ch=1, width=24, normalize=normalize)
    raise ValueError(model_kind)
