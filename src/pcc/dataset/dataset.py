import math
from typing import Optional

import numpy as np

from pcc.params import PCCConfig

# ---------------- RNG helpers ----------------


def make_generator(base_seed: int, idx: int) -> np.random.Generator:
    # A simple bijection of (base_seed, idx). Works fine in practice.
    # 1_000_003 is a large prime. Avoids collisions for small idx over datasets.
    seed = (int(base_seed) * 1_000_003 + int(idx)) % (2**64)
    return np.random.default_rng(seed)


# ---------------- Core ops with generator ----------------


def _gaussian_kernel1d(sigma: float, radius: int, dtype):
    x = np.arange(-radius, radius + 1, dtype=dtype)
    k = np.exp(-0.5 * (x / sigma) ** 2).astype(dtype, copy=False)
    return k / k.sum()


def _convolve_axis(img: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = kernel.size // 2
    pad_width = [(0, 0)] * img.ndim
    pad_width[axis] = (radius, radius)
    padded = np.pad(img, pad_width, mode="constant")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=kernel.size, axis=axis
    )
    return np.tensordot(windows, kernel, axes=([-1], [0])).astype(
        img.dtype, copy=False
    )


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    radius = max(1, int(3 * sigma))
    k = _gaussian_kernel1d(sigma, radius, dtype=img.dtype)
    return _convolve_axis(_convolve_axis(img, k, axis=1), k, axis=0)


def highpass_noise(
    shape,
    sigma_low: float,
    scale: float,
    g: np.random.Generator,
) -> np.ndarray:
    n = g.standard_normal(shape).astype(np.float32)
    low = gaussian_blur(n, sigma_low)
    hp = n - low
    return hp / (hp.std() + 1e-6) * scale


def quantile_uniformize_phase(theta: np.ndarray) -> np.ndarray:
    flat = theta.reshape(-1)
    idx = np.argsort(flat)
    targets = np.linspace(-math.pi, math.pi, num=flat.size, dtype=theta.dtype)
    out = np.empty_like(flat)
    out[idx] = targets
    return out.reshape(theta.shape)


def wrap_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + math.pi) % (2 * math.pi) - math.pi


def _make_coords(N, dtype):
    xs = np.linspace(-1, 1, num=N, dtype=dtype)
    ys = np.linspace(-1, 1, num=N, dtype=dtype)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r = np.sqrt(xx**2 + yy**2).astype(dtype, copy=False)
    return xx, yy, r


def _bilinear_sample_zeros(img: np.ndarray, src_x: np.ndarray, src_y: np.ndarray):
    H, W = img.shape
    x0 = np.floor(src_x).astype(np.int64)
    y0 = np.floor(src_y).astype(np.int64)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = src_x - x0
    wy = src_y - y0

    out = np.zeros_like(img)
    for x, y, weight in (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        valid = (0 <= x) & (x < W) & (0 <= y) & (y < H)
        out[valid] += img[y[valid], x[valid]] * weight[valid]

    return out.astype(img.dtype, copy=False)


def _apply_rotate(img: np.ndarray, rad: float):
    c, s = math.cos(float(rad)), math.sin(float(rad))
    H, W = img.shape

    yy, xx = np.meshgrid(
        np.arange(H, dtype=img.dtype),
        np.arange(W, dtype=img.dtype),
        indexing="ij",
    )
    x_norm = 2 * (xx + 0.5) / W - 1
    y_norm = 2 * (yy + 0.5) / H - 1

    src_x_norm = c * x_norm - s * y_norm
    src_y_norm = s * x_norm + c * y_norm
    src_x = ((src_x_norm + 1) * W - 1) / 2
    src_y = ((src_y_norm + 1) * H - 1) / 2

    return _bilinear_sample_zeros(img, src_x, src_y)


def _apply_translate(img: np.ndarray, tx: int, ty: int):
    return np.roll(img, shift=(ty, tx), axis=(0, 1))


# ---------------- Deterministic sample ----------------


def make_sample(cfg: PCCConfig, y: int, g: Optional[np.random.Generator] = None):
    if g is None:
        raise ValueError("Pass a numpy random Generator for deterministic sampling.")

    N = cfg.N
    dtype = np.float32
    _, _, r = _make_coords(N, dtype=dtype)

    # amplitude
    radial = np.exp(-cfg.amp_radial_decay * (r**2)).astype(dtype, copy=False)
    tissue = g.standard_normal((N, N)).astype(dtype)
    tissue = gaussian_blur(tissue, cfg.amp_smooth_sigma)
    tissue = (tissue - tissue.min()) / (tissue.max() - tissue.min() + 1e-6)
    lo, hi = cfg.amp_range
    tissue = lo + (hi - lo) * tissue
    A = radial * tissue
    A = A / (A.mean() + 1e-6)

    # coherent phase
    psi = g.standard_normal((N, N)).astype(dtype)
    psi = gaussian_blur(psi, cfg.phase_smooth_sigma)
    psi = psi / (psi.std() + 1e-6)
    psi = psi * (math.pi / 1.5)

    if cfg.global_phase:
        psi = psi + (g.random() * 2 - 1) * math.pi

    if y == 0:
        theta = psi
    else:
        eta = highpass_noise(
            (N, N),
            sigma_low=cfg.incoh_highpass_sigma,
            scale=cfg.incoh_scale,
            g=g,
        )
        theta = psi + eta

    theta = wrap_pi(theta)

    # sample nuisances once and apply to both A and theta
    if cfg.rotate_deg > 0:
        deg = (g.random() * 2 - 1) * cfg.rotate_deg
        rad = deg * math.pi / 180.0
        A = _apply_rotate(A.astype(dtype, copy=False), rad)
        theta = _apply_rotate(theta.astype(dtype, copy=False), rad)

    if cfg.translate_px > 0:
        tx = int(g.integers(-cfg.translate_px, cfg.translate_px + 1))
        ty = int(g.integers(-cfg.translate_px, cfg.translate_px + 1))
        A = _apply_translate(A, tx, ty)
        theta = _apply_translate(theta, tx, ty)

    if cfg.uniformize_phase_hist:
        theta = quantile_uniformize_phase(theta.astype(dtype, copy=False))

    z = A * np.exp(1j * theta)

    if cfg.noise_std > 0:
        noise = (
            g.standard_normal((N, N)) + 1j * g.standard_normal((N, N))
        ) * cfg.noise_std
        z = z + noise
        if cfg.renorm_amp:
            amp = np.abs(z)
            z = z / (amp.mean() + 1e-6) * (A.mean() + 1e-6)

    return (
        z.astype(np.complex64, copy=False),
        A.astype(np.float32, copy=False),
        theta.astype(np.float32, copy=False),
    )


def make_deterministic_sample(cfg: PCCConfig, y: int, seed: int = 0, idx: int = 0):
    g = make_generator(seed, idx)
    return make_sample(cfg, y=y, g=g)


# ---------------- Dataset ----------------


class PCCDataset:
    def __init__(self, size: int, cfg: PCCConfig, seed: int = 0):
        self.size = size
        self.cfg = cfg
        self.seed = int(seed)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        g = make_generator(self.seed, idx)
        y = np.int64(g.integers(0, 2))
        z, A, theta = make_sample(self.cfg, y=int(y), g=g)
        return z, A, theta, y
