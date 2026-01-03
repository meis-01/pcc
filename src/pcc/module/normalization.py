import torch
import torch.nn as nn


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


class ComplexSplitBatchNorm2d(ComplexBatchNorm2d):
    """
    Complex BatchNorm2d with split normalization: real and imag parts are normalized independently.
    """

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

        # Compute mean and var for real and imag separately
        if self.training:
            mean_r = xr.mean(dim=(0, 2, 3))
            mean_i = xi.mean(dim=(0, 2, 3))
            var_r = xr.var(dim=(0, 2, 3), unbiased=False)
            var_i = xi.var(dim=(0, 2, 3), unbiased=False)

            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked += 1
                    m = self.momentum
                    self.running_mean[..., 0].mul_(1.0 - m).add_(m * mean_r)
                    self.running_mean[..., 1].mul_(1.0 - m).add_(m * mean_i)
                    self.running_cov[:, 0, 0].mul_(1.0 - m).add_(m * var_r)
                    self.running_cov[:, 1, 1].mul_(1.0 - m).add_(m * var_i)
        else:
            if not self.track_running_stats:
                raise RuntimeError(
                    "Eval mode requires track_running_stats=True for this module."
                )
            mean_r = self.running_mean[..., 0]
            mean_i = self.running_mean[..., 1]
            var_r = self.running_cov[:, 0, 0]
            var_i = self.running_cov[:, 1, 1]

        xr_norm = (xr - mean_r.view(1, C, 1, 1)) / (
            var_r.view(1, C, 1, 1) + self.eps
        ).sqrt()
        xi_norm = (xi - mean_i.view(1, C, 1, 1)) / (
            var_i.view(1, C, 1, 1) + self.eps
        ).sqrt()

        if self.affine:
            # Only use diagonal and bias
            weight = self.weight  # [C,2,2]
            bias = self.bias  # [C,2]
            xr_norm = xr_norm * weight[:, 0, 0].view(1, C, 1, 1) + bias[:, 0].view(
                1, C, 1, 1
            )
            xi_norm = xi_norm * weight[:, 1, 1].view(1, C, 1, 1) + bias[:, 1].view(
                1, C, 1, 1
            )

        return xr_norm, xi_norm
