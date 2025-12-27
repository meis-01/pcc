from dataclasses import dataclass
import math
import numpy as np
import torch
from torch.utils.data import Dataset


def _gaussian_kernel1d(sigma: float, radius: int):
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """img: (H,W) float32"""
    if sigma <= 0:
        return img
    radius = max(1, int(3 * sigma))
    k = _gaussian_kernel1d(sigma, radius).to(img.device)
    # separable conv
    img2 = img[None, None, :, :]  # 1x1xHxW
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    img2 = torch.nn.functional.conv2d(img2, kx, padding=(0, radius))
    img2 = torch.nn.functional.conv2d(img2, ky, padding=(radius, 0))
    return img2[0, 0]


def highpass_noise(shape, sigma_low: float, scale: float, device):
    """White noise -> remove low-freq component by subtracting blur -> scale."""
    n = torch.randn(shape, device=device)
    low = gaussian_blur(n, sigma_low)
    hp = n - low
    hp = hp / (hp.std() + 1e-6) * scale
    return hp


def quantile_uniformize_phase(theta: torch.Tensor) -> torch.Tensor:
    """
    Make the phase histogram exactly uniform on (-pi, pi] by rank mapping.
    Preserves spatial ordering only through rank, removing marginal histogram cues.
    """
    flat = theta.flatten()
    idx = torch.argsort(flat)
    # Uniform targets in (-pi, pi]
    n = flat.numel()
    targets = torch.linspace(
        -math.pi, math.pi, steps=n, device=theta.device, dtype=theta.dtype
    )
    out = torch.empty_like(flat)
    out[idx] = targets
    return out.view_as(theta)


def wrap_pi(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2 * math.pi) - math.pi


@dataclass
class PCCConfig:
    N: int = 128
    # A3 amplitude params
    amp_radial_decay: float = 2.2  # larger -> stronger decay
    amp_smooth_sigma: float = 6.0  # tissue variation smoothness
    amp_range: tuple = (0.7, 1.4)  # multiplicative tissue range
    # phase params
    phase_smooth_sigma: float = 2.0  # coherence scale (bigger -> smoother)
    incoh_highpass_sigma: float = 16.0  # low-sigma blur for high-pass extraction
    incoh_scale: float = 0.2  # scramble strength (radians)
    global_phase: bool = True
    uniformize_phase_hist: bool = True
    # nuisances
    translate_px: int = 16
    rotate_deg: float = 30.0
    noise_std: float = 0.1  # complex noise std (optional)
    renorm_amp: bool = False  # if True, rescale to keep amplitude distribution tight


def _make_coords(N, device):
    xs = torch.linspace(-1, 1, steps=N, device=device)
    ys = torch.linspace(-1, 1, steps=N, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    return xx, yy, r


def _rand_rotate(img: torch.Tensor, max_deg: float):
    if max_deg <= 0:
        return img
    deg = (torch.rand((), device=img.device) * 2 - 1) * max_deg
    rad = deg * math.pi / 180.0
    c, s = torch.cos(rad), torch.sin(rad)
    # affine grid for single-channel (H,W) -> add batch/channel
    H, W = img.shape
    theta = torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0]], device=img.device, dtype=img.dtype
    ).unsqueeze(0)
    grid = torch.nn.functional.affine_grid(
        theta, size=(1, 1, H, W), align_corners=False
    )
    out = torch.nn.functional.grid_sample(
        img.view(1, 1, H, W),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    return out[0, 0]


def _rand_translate(img: torch.Tensor, max_px: int):
    if max_px <= 0:
        return img
    tx = int(torch.randint(-max_px, max_px + 1, (1,)).item())
    ty = int(torch.randint(-max_px, max_px + 1, (1,)).item())
    return torch.roll(img, shifts=(ty, tx), dims=(0, 1))


def make_sample(cfg: PCCConfig, y: int, device="cpu"):
    """
    Returns:
      z: complex tensor (N,N) complex64
      A: amplitude (N,N) float32
      theta: phase (N,N) float32
    """
    N = cfg.N
    xx, yy, r = _make_coords(N, device=device)

    # ---- A3 amplitude: radial envelope * smooth tissue variation ----
    radial = torch.exp(-cfg.amp_radial_decay * (r**2))
    tissue = torch.randn((N, N), device=device)
    tissue = gaussian_blur(tissue, cfg.amp_smooth_sigma)
    tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min() + 1e-6)
    lo, hi = cfg.amp_range
    tissue = lo + (hi - lo) * tissue
    A = radial * tissue
    # normalize amplitude to stable scale
    A = A / (A.mean() + 1e-6)

    # ---- coherent base phase ----
    psi = torch.randn((N, N), device=device)
    psi = gaussian_blur(psi, cfg.phase_smooth_sigma)
    psi = psi / (psi.std() + 1e-6)
    # scale to cover roughly [-pi, pi]
    psi = psi * (math.pi / 1.5)

    if cfg.global_phase:
        psi = psi + (torch.rand((), device=device) * 2 - 1) * math.pi

    if y == 0:
        theta = psi
    else:
        eta = highpass_noise(
            (N, N),
            sigma_low=cfg.incoh_highpass_sigma,
            scale=cfg.incoh_scale,
            device=device,
        )
        theta = psi + eta

    theta = wrap_pi(theta)

    # Random nuisances applied consistently to A and theta
    # (Rotation + translation preserve the "physics" feel)
    if cfg.rotate_deg > 0:
        A = _rand_rotate(A, cfg.rotate_deg)
        theta = _rand_rotate(theta, cfg.rotate_deg)
    if cfg.translate_px > 0:
        A = _rand_translate(A, cfg.translate_px)
        theta = _rand_translate(theta, cfg.translate_px)

    if cfg.uniformize_phase_hist:
        theta = quantile_uniformize_phase(theta)

    # complex field
    z = A * torch.exp(1j * theta)

    if cfg.noise_std > 0:
        noise = (
            torch.randn((N, N), device=device) + 1j * torch.randn((N, N), device=device)
        ) * cfg.noise_std
        z = z + noise
        if cfg.renorm_amp:
            # preserve mean amplitude roughly
            amp = torch.abs(z)
            z = z / (amp.mean() + 1e-6) * (A.mean() + 1e-6)

    return z.to(torch.complex64), A.to(torch.float32), theta.to(torch.float32)


class PCCDataset(Dataset):
    def __init__(self, size: int, cfg: PCCConfig, seed: int = 0):
        self.size = size
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        y = int(self.rng.integers(0, 2))
        z, A, theta = make_sample(self.cfg, y=y, device="cpu")
        return z, A, theta, torch.tensor(y, dtype=torch.long)
