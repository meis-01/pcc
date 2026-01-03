from dataclasses import dataclass
import math
import torch
from torch.utils.data import Dataset
from pcc.params import PCCConfig

# ---------------- RNG helpers ----------------


def _make_generator(base_seed: int, idx: int, device="cpu") -> torch.Generator:
    g = torch.Generator(device=device)
    # A simple bijection of (base_seed, idx). Works fine in practice.
    # 1_000_003 is a large prime. Avoids collisions for small idx over datasets
    g.manual_seed(int(base_seed) * 1_000_003 + int(idx))
    return g


# ---------------- Core ops with generator ----------------


def _gaussian_kernel1d(sigma: float, radius: int, device, dtype):
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    k = k / k.sum()
    return k


def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return img
    radius = max(1, int(3 * sigma))
    k = _gaussian_kernel1d(sigma, radius, device=img.device, dtype=img.dtype)
    img2 = img[None, None, :, :]  # 1x1xHxW
    kx = k.view(1, 1, 1, -1)
    ky = k.view(1, 1, -1, 1)
    img2 = torch.nn.functional.conv2d(img2, kx, padding=(0, radius))
    img2 = torch.nn.functional.conv2d(img2, ky, padding=(radius, 0))
    return img2[0, 0]


def highpass_noise(shape, sigma_low: float, scale: float, device, g: torch.Generator):
    n = torch.randn(shape, device=device, generator=g)
    low = gaussian_blur(n, sigma_low)
    hp = n - low
    hp = hp / (hp.std() + 1e-6) * scale
    return hp


def quantile_uniformize_phase(theta: torch.Tensor) -> torch.Tensor:
    flat = theta.flatten()
    idx = torch.argsort(flat)
    n = flat.numel()
    targets = torch.linspace(
        -math.pi, math.pi, steps=n, device=theta.device, dtype=theta.dtype
    )
    out = torch.empty_like(flat)
    out[idx] = targets
    return out.view_as(theta)


def wrap_pi(theta: torch.Tensor) -> torch.Tensor:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def _make_coords(N, device, dtype):
    xs = torch.linspace(-1, 1, steps=N, device=device, dtype=dtype)
    ys = torch.linspace(-1, 1, steps=N, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2)
    return xx, yy, r


def _apply_rotate(img: torch.Tensor, rad: torch.Tensor):
    # rad is a scalar tensor on same device
    c, s = torch.cos(rad), torch.sin(rad)
    H, W = img.shape
    theta = (
        torch.stack(
            [
                torch.stack(
                    [c, -s, torch.tensor(0.0, device=img.device, dtype=img.dtype)]
                ),
                torch.stack(
                    [s, c, torch.tensor(0.0, device=img.device, dtype=img.dtype)]
                ),
            ]
        )
        .unsqueeze(0)
        .to(dtype=img.dtype)
    )

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


def _apply_translate(img: torch.Tensor, tx: int, ty: int):
    return torch.roll(img, shifts=(ty, tx), dims=(0, 1))


# ---------------- Deterministic sample ----------------


def make_sample(cfg: PCCConfig, y: int, device="cpu", g: torch.Generator | None = None):
    assert g is not None, "Pass a torch.Generator for determinism."

    N = cfg.N
    dtype = torch.float32
    xx, yy, r = _make_coords(N, device=device, dtype=dtype)

    # amplitude
    radial = torch.exp(-cfg.amp_radial_decay * (r**2))
    tissue = torch.randn((N, N), device=device, dtype=dtype, generator=g)
    tissue = gaussian_blur(tissue, cfg.amp_smooth_sigma)
    tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min() + 1e-6)
    lo, hi = cfg.amp_range
    tissue = lo + (hi - lo) * tissue
    A = radial * tissue
    A = A / (A.mean() + 1e-6)

    # coherent phase
    psi = torch.randn((N, N), device=device, dtype=dtype, generator=g)
    psi = gaussian_blur(psi, cfg.phase_smooth_sigma)
    psi = psi / (psi.std() + 1e-6)
    psi = psi * (math.pi / 1.5)

    if cfg.global_phase:
        psi = (
            psi
            + (torch.rand((), device=device, dtype=dtype, generator=g) * 2 - 1)
            * math.pi
        )

    if y == 0:
        theta = psi
    else:
        eta = highpass_noise(
            (N, N),
            sigma_low=cfg.incoh_highpass_sigma,
            scale=cfg.incoh_scale,
            device=device,
            g=g,
        )
        theta = psi + eta

    theta = wrap_pi(theta)

    # sample nuisances ONCE and apply to both A and theta
    if cfg.rotate_deg > 0:
        deg = (
            torch.rand((), device=device, dtype=dtype, generator=g) * 2 - 1
        ) * cfg.rotate_deg
        rad = deg * math.pi / 180.0
        A = _apply_rotate(A, rad)
        theta = _apply_rotate(theta, rad)

    if cfg.translate_px > 0:
        tx = int(
            torch.randint(
                -cfg.translate_px, cfg.translate_px + 1, (1,), generator=g
            ).item()
        )
        ty = int(
            torch.randint(
                -cfg.translate_px, cfg.translate_px + 1, (1,), generator=g
            ).item()
        )
        A = _apply_translate(A, tx, ty)
        theta = _apply_translate(theta, tx, ty)

    if cfg.uniformize_phase_hist:
        theta = quantile_uniformize_phase(theta)

    z = A * torch.exp(1j * theta)

    if cfg.noise_std > 0:
        noise = (
            torch.randn((N, N), device=device, generator=g)
            + 1j * torch.randn((N, N), device=device, generator=g)
        ) * cfg.noise_std
        z = z + noise
        if cfg.renorm_amp:
            amp = torch.abs(z)
            z = z / (amp.mean() + 1e-6) * (A.mean() + 1e-6)

    return z.to(torch.complex64), A.to(torch.float32), theta.to(torch.float32)


# ---------------- Dataset ----------------


class PCCDataset(Dataset):
    def __init__(self, size: int, cfg: PCCConfig, seed: int = 0, device="cpu"):
        self.size = size
        self.cfg = cfg
        self.seed = int(seed)
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g = _make_generator(self.seed, idx, device="cpu")
        y = int(torch.randint(0, 2, (1,), generator=g).item())
        z, A, theta = make_sample(self.cfg, y=y, device="cpu", g=g)
        return z, A, theta, torch.tensor(y, dtype=torch.long)
