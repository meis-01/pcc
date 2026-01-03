import argparse
import torch
from pcc.dataset.dataset import PCCConfig, make_sample
import math


def phase_coherence_index(theta: torch.Tensor) -> float:
    """
    Mean local alignment of phase differences (4-neighborhood).
    High for coherent fields; low for incoherent.
    """
    # wrap differences via complex exponentials
    z = torch.exp(1j * theta)
    # neighbor products measure alignment
    prod_x = z[:, 1:] * torch.conj(z[:, :-1])
    prod_y = z[1:, :] * torch.conj(z[:-1, :])
    # average magnitude of mean phasor
    m = torch.abs(torch.mean(torch.cat([prod_x.flatten(), prod_y.flatten()])))
    return float(m.item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--samples", type=int, default=50)
    args = ap.parse_args()

    cfg = PCCConfig(N=args.N)
    c0, c1 = [], []
    for _ in range(args.samples):
        _, _, th0 = make_sample(cfg, y=0, device="cpu")
        _, _, th1 = make_sample(cfg, y=1, device="cpu")
        c0.append(phase_coherence_index(th0))
        c1.append(phase_coherence_index(th1))

    print("Coherence index (mean±std)")
    print(
        f"Class 0 coherent:   {sum(c0)/len(c0):.4f} ± {torch.tensor(c0).std().item():.4f}"
    )
    print(
        f"Class 1 incoherent: {sum(c1)/len(c1):.4f} ± {torch.tensor(c1).std().item():.4f}"
    )


if __name__ == "__main__":
    main()
