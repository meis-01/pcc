import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Real CNN backbone ----------
class RealCNN(nn.Module):
    def __init__(self, in_ch: int, width: int = 32, num_classes: int = 2):
        super().__init__()
        w = width

        self.net = nn.Sequential(
            # Stage 1 (H,W)
            nn.Conv2d(in_ch, w, 3, padding=1, stride=1),
            nn.BatchNorm2d(w),
            nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1, stride=1),
            nn.BatchNorm2d(w),
            nn.ReLU(),
            # Stage 2 (H/2,W/2) via stride-2 conv
            nn.Conv2d(w, 2 * w, 3, padding=1, stride=2),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(),
            nn.Conv2d(2 * w, 2 * w, 3, padding=1, stride=1),
            nn.BatchNorm2d(2 * w),
            nn.ReLU(),
            # Stage 3 (H/4,W/4) via stride-2 conv
            nn.Conv2d(2 * w, 4 * w, 3, padding=1, stride=2),
            nn.BatchNorm2d(4 * w),
            nn.ReLU(),
            nn.Conv2d(4 * w, 4 * w, 3, padding=1, stride=1),
            nn.BatchNorm2d(4 * w),
            nn.ReLU(),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(4 * w, num_classes),
        )

    def forward(self, x):
        return self.head(self.net(x))


# ---------- Complex layers implemented via real tensors ----------
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


class ModReLU(nn.Module):
    """modReLU: relu(|z| + b) * z/|z|"""

    def __init__(self, channels: int):
        super().__init__()
        self.b = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, xr, xi):
        mag = torch.sqrt(xr * xr + xi * xi + 1e-8)
        scale = F.relu(mag + self.b) / (mag + 1e-8)
        return xr * scale, xi * scale


class ComplexBatchNorm2d(nn.Module):
    """
    Proper Complex BatchNorm (2D) with *joint* whitening of (real, imag) per channel.

    This treats each complex channel as a 2D random vector [xr, xi]^T and normalizes it by:
      - subtracting the 2D mean
      - whitening with Σ^{-1/2} where Σ is the 2x2 covariance of (xr, xi) over (B,H,W)
      - (optional) applying a learnable 2x2 affine transform + learnable 2D bias

    Input/Output:
      xr, xi: tensors of shape [B, C, H, W]
      returns: (yr, yi) same shape

    Notes:
      - This matches the "full" complex BN spirit (e.g., Deep Complex Networks style).
      - Uses torch.linalg.eigh for stable inverse square-root of 2x2 cov matrices.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if affine:
            # Learnable 2x2 transform per channel (initialized to identity)
            self.weight = nn.Parameter(
                torch.eye(2, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(num_features, 1, 1)
            )
            # Learnable 2D bias per channel
            self.bias = nn.Parameter(torch.zeros(num_features, 2, dtype=torch.float32))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if track_running_stats:
            self.register_buffer(
                "running_mean", torch.zeros(num_features, 2, dtype=torch.float32)
            )
            self.register_buffer(
                "running_cov",
                torch.eye(2, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(num_features, 1, 1),
            )
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_cov", None)
            self.register_buffer("num_batches_tracked", None)

    @staticmethod
    def _inv_sqrt_2x2(cov: torch.Tensor, eps: float) -> torch.Tensor:
        """
        cov: [C, 2, 2] symmetric positive definite
        returns cov^{-1/2}: [C, 2, 2]
        """
        # Ensure symmetry numerically
        cov = 0.5 * (cov + cov.transpose(-1, -2))
        # Eigen-decomp for symmetric matrices: cov = Q diag(l) Q^T
        l, Q = torch.linalg.eigh(cov)  # l: [C,2], Q: [C,2,2]
        l = torch.clamp(l, min=eps)
        inv_sqrt_l = torch.rsqrt(l)  # 1/sqrt(l)
        # Q diag(inv_sqrt_l) Q^T
        return Q @ torch.diag_embed(inv_sqrt_l) @ Q.transpose(-1, -2)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        if xr.shape != xi.shape:
            raise ValueError(
                f"xr and xi must have same shape, got {xr.shape} vs {xi.shape}"
            )
        if xr.dim() != 4:
            raise ValueError(f"Expected [B,C,H,W], got {xr.shape}")

        B, C, H, W = xr.shape
        if C != self.num_features:
            raise ValueError(f"Expected C={self.num_features}, got C={C}")

        # Stack into 2-vector: x: [B,C,H,W,2]
        x = torch.stack([xr, xi], dim=-1)

        if self.training:
            # Compute batch mean over N=B*H*W for each channel and each component
            # mean: [C,2]
            mean = x.mean(dim=(0, 2, 3))

            # Center: xc: [B,C,H,W,2]
            xc = x - mean.view(1, C, 1, 1, 2)

            # Reshape for covariance: [C, N, 2]
            N = B * H * W
            xc_flat = xc.permute(1, 0, 2, 3, 4).contiguous().view(C, N, 2)

            # Covariance per channel: Σ = (1/N) sum_n v_n v_n^T
            # cov: [C,2,2]
            cov = (xc_flat.transpose(1, 2) @ xc_flat) / float(N)

            # Add epsilon * I for numerical stability
            eye = torch.eye(2, device=cov.device, dtype=cov.dtype).view(1, 2, 2)
            cov = cov + self.eps * eye

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_mean.mul_(1.0 - m).add_(m * mean)
                    self.running_cov.mul_(1.0 - m).add_(m * cov)
        else:
            if not self.track_running_stats:
                raise RuntimeError(
                    "Eval mode requires track_running_stats=True for this module."
                )
            mean = self.running_mean
            cov = self.running_cov
            xc = x - mean.view(1, C, 1, 1, 2)

        # Whitening: y = Σ^{-1/2} (x - mean)
        inv_sqrt = self._inv_sqrt_2x2(cov, self.eps)  # [C,2,2]
        # Apply inv_sqrt to last-dim vector:
        # xc: [B,C,H,W,2], inv_sqrt: [C,2,2]
        y = torch.einsum("bchwk,ckl->bchwl", xc, inv_sqrt)

        # Optional learnable affine: y = W y + b
        if self.affine:
            y = torch.einsum("bchwk,ckl->bchwl", y, self.weight) + self.bias.view(
                1, C, 1, 1, 2
            )

        yr, yi = y[..., 0], y[..., 1]
        return yr, yi


class ComplexCNN(nn.Module):
    def __init__(self, in_ch: int = 1, width: int = 24, num_classes: int = 2):
        """
        in_ch refers to number of complex channels.
        We represent complex activations as (real, imag) tensors with shape [B, C, H, W].
        """
        super().__init__()
        w = width
        self.c1 = ComplexConv2d(in_ch, w, 3, padding=1)
        self.b1 = ComplexBatchNorm2d(w)
        self.a1 = ModReLU(w)

        self.c2 = ComplexConv2d(w, w, 3, padding=1)
        self.b2 = ComplexBatchNorm2d(w)
        self.a2 = ModReLU(w)

        self.c3 = ComplexConv2d(w, 2 * w, 3, padding=1, stride=2)
        self.b3 = ComplexBatchNorm2d(2 * w)
        self.a3 = ModReLU(2 * w)
        self.c4 = ComplexConv2d(2 * w, 2 * w, 3, padding=1)
        self.b4 = ComplexBatchNorm2d(2 * w)
        self.a4 = ModReLU(2 * w)

        self.c5 = ComplexConv2d(2 * w, 4 * w, 3, padding=1, stride=2)
        self.b5 = ComplexBatchNorm2d(4 * w)
        self.a5 = ModReLU(4 * w)
        self.c6 = ComplexConv2d(4 * w, 4 * w, 3, padding=1)
        self.b6 = ComplexBatchNorm2d(4 * w)
        self.a6 = ModReLU(4 * w)

        self.avg = nn.AdaptiveAvgPool2d(1)
        # Convert complex -> real for classification by concatenating pooled real+imag
        self.fc = nn.Linear(8 * w, num_classes)

    def forward(self, xr, xi):
        xr, xi = self.c1(xr, xi)
        xr, xi = self.b1(xr, xi)
        xr, xi = self.a1(xr, xi)
        xr, xi = self.c2(xr, xi)
        xr, xi = self.b2(xr, xi)
        xr, xi = self.a2(xr, xi)

        xr, xi = self.c3(xr, xi)
        xr, xi = self.b3(xr, xi)
        xr, xi = self.a3(xr, xi)
        xr, xi = self.c4(xr, xi)
        xr, xi = self.b4(xr, xi)
        xr, xi = self.a4(xr, xi)

        xr, xi = self.c5(xr, xi)
        xr, xi = self.b5(xr, xi)
        xr, xi = self.a5(xr, xi)
        xr, xi = self.c6(xr, xi)
        xr, xi = self.b6(xr, xi)
        xr, xi = self.a6(xr, xi)

        pr = self.avg(xr).flatten(1)
        pi = self.avg(xi).flatten(1)
        feat = torch.cat([pr, pi], dim=1)
        return self.fc(feat)
