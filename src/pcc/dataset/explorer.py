# holoviews+panel UI to explore PCCConfig and plot generated samples
# pip install panel holoviews bokeh torch numpy
import numpy as np
import holoviews as hv
import panel as pn
import param
import torch
from pcc.dataset.dataset import make_sample, _make_generator
from pcc.evaluation.metric import phase_coherence_index
from pcc.params import PCCConfig

hv.extension("bokeh")
pn.extension()


def _to_numpy(t: torch.Tensor):
    return t.detach().cpu().numpy()


def _angle(z: torch.Tensor):
    return torch.atan2(z.imag, z.real)


class PCCExplorer(param.Parameterized):
    pcc = PCCConfig()
    # --- sample controls ---
    y = param.ObjectSelector(default=0, objects=[0, 1], doc="Class label")
    seed = param.Integer(default=0, bounds=(0, 999999), doc="Reproducibility seed")
    device = param.ObjectSelector(
        default="cpu", objects=["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    )

    # --- config controls (mirrors PCCConfig) ---
    N = param.Integer(default=pcc.N, bounds=(32, 512), step=16)

    amp_radial_decay = param.Number(default=pcc.amp_radial_decay, bounds=(0.1, 10.0))
    amp_smooth_sigma = param.Number(default=pcc.amp_smooth_sigma, bounds=(0.0, 30.0))
    amp_range_lo = param.Number(default=pcc.amp_range[0], bounds=(0.0, 3.0))
    amp_range_hi = param.Number(default=pcc.amp_range[1], bounds=(0.0, 3.0))

    phase_smooth_sigma = param.Number(
        default=pcc.phase_smooth_sigma, bounds=(0.0, 60.0)
    )
    incoh_highpass_sigma = param.Number(
        default=pcc.incoh_highpass_sigma, bounds=(0.0, 30.0)
    )
    incoh_scale = param.Number(default=pcc.incoh_scale, bounds=(0.0, 6.0))

    global_phase = param.Boolean(default=pcc.global_phase)
    uniformize_phase_hist = param.Boolean(default=pcc.uniformize_phase_hist)

    translate_px = param.Integer(default=pcc.translate_px, bounds=(0, 64))
    rotate_deg = param.Number(default=pcc.rotate_deg, bounds=(0.0, 90.0))

    noise_std = param.Number(default=pcc.noise_std, bounds=(0.0, 1.0))
    renorm_amp = param.Boolean(default=pcc.renorm_amp)

    # --- display controls ---
    show_complex_plane = param.Boolean(default=True)
    plane_points = param.Integer(default=6000, bounds=(500, 40000), step=500)

    def _cfg(self):
        lo = float(self.amp_range_lo)
        hi = float(self.amp_range_hi)
        if hi < lo:
            lo, hi = hi, lo  # keep it sane
        return PCCConfig(
            N=int(self.N),
            amp_radial_decay=float(self.amp_radial_decay),
            amp_smooth_sigma=float(self.amp_smooth_sigma),
            amp_range=(lo, hi),
            phase_smooth_sigma=float(self.phase_smooth_sigma),
            incoh_highpass_sigma=float(self.incoh_highpass_sigma),
            incoh_scale=float(self.incoh_scale),
            global_phase=bool(self.global_phase),
            uniformize_phase_hist=bool(self.uniformize_phase_hist),
            translate_px=int(self.translate_px),
            rotate_deg=float(self.rotate_deg),
            noise_std=float(self.noise_std),
            renorm_amp=bool(self.renorm_amp),
        )

    def _make(self):
        # make all randomness deterministic w.r.t. seed
        g = _make_generator(int(self.seed), idx=0, device=self.device)
        cfg = self._cfg()
        z, A, theta = make_sample(cfg, y=int(self.y), device=self.device, g=g)

        # return numpy versions for plotting
        return cfg, _to_numpy(z), _to_numpy(A), _to_numpy(theta)

    @param.depends(
        "y",
        "seed",
        "device",
        "N",
        "amp_radial_decay",
        "amp_smooth_sigma",
        "amp_range_lo",
        "amp_range_hi",
        "phase_smooth_sigma",
        "incoh_highpass_sigma",
        "incoh_scale",
        "global_phase",
        "uniformize_phase_hist",
        "translate_px",
        "rotate_deg",
        "noise_std",
        "renorm_amp",
        "show_complex_plane",
        "plane_points",
    )
    def view(self):
        cfg, z_np, A_np, th_np = self._make()

        # Amplitude image
        amp = hv.Image(A_np, kdims=["x", "y"], vdims=["A"]).opts(
            title=f"Amplitude A (y={self.y})",
            colorbar=True,
            tools=["hover"],
            width=360,
            height=360,
        )
        # Amplitude histogram
        amp_hist = hv.Histogram(np.histogram(A_np.flatten(), bins=100)).opts(
            title="Amplitude Histogram",
            xlabel="A",
            ylabel="Count",
            width=360,
            height=360,
            tools=["hover"],
        )

        # Phase histogram
        phase_hist = hv.Histogram(
            np.histogram(th_np.flatten(), bins=100, range=[-np.pi, np.pi])
        ).opts(
            title="Phase Histogram",
            xlabel="θ",
            ylabel="Count",
            width=360,
            height=360,
            tools=["hover"],
        )
        # Phase image (keep in [-pi, pi])
        phase = hv.Image(th_np, kdims=["x", "y"], vdims=["theta"]).opts(
            title="Phase θ",
            colorbar=True,
            tools=["hover"],
            width=360,
            height=360,
        )

        # Optional complex plane scatter
        if self.show_complex_plane:
            z_flat = z_np.reshape(-1)
            n = z_flat.size
            m = min(int(self.plane_points), n)
            # cheap deterministic subsample (stride) so it doesn't jitter
            stride = max(1, n // m)
            pts = z_flat[::stride][:m]
            xs = pts.real.astype(np.float32)
            ys = pts.imag.astype(np.float32)
            plane = hv.Points(np.column_stack([xs, ys]), kdims=["Re(z)", "Im(z)"]).opts(
                title="Complex plane scatter (subsampled)",
                size=2,
                alpha=0.25,
                tools=["hover"],
                width=360,
                height=360,
                aspect="equal",
            )
            layout = (
                (amp + phase + plane + amp_hist + phase_hist)
                .cols(3)
                .opts(shared_axes=False)
            )
        else:
            layout = (
                (amp + phase + amp_hist + phase_hist).cols(2).opts(shared_axes=False)
            )

        # quick numeric sanity panel
        z_abs = np.abs(z_np)
        z_ang = np.angle(z_np)
        stats = pn.pane.Markdown(
            f"""
**Quick stats**
- `N`: {cfg.N}
- `A.mean`: {A_np.mean():.4f} | `A.std`: {A_np.std():.4f}
- `|z|.mean`: {z_abs.mean():.4f} | `|z|.std`: {z_abs.std():.4f}
- `angle(z)` mean: {z_ang.mean():.4f} | std: {z_ang.std():.4f}
- `uniformize_phase_hist`: {cfg.uniformize_phase_hist}
- `phase coherence index`: {phase_coherence_index(torch.from_numpy(th_np)):.4f}
""".strip()
        )

        return pn.Row(layout, stats)


# -----------------------------
# Build the app
# -----------------------------
explorer = PCCExplorer()

controls = pn.Param(
    explorer,
    parameters=[
        "y",
        "seed",
        "device",
        "N",
        "amp_radial_decay",
        "amp_smooth_sigma",
        "amp_range_lo",
        "amp_range_hi",
        "phase_smooth_sigma",
        "incoh_highpass_sigma",
        "incoh_scale",
        "global_phase",
        "uniformize_phase_hist",
        "translate_px",
        "rotate_deg",
        "noise_std",
        "renorm_amp",
        "show_complex_plane",
        "plane_points",
    ],
    show_name=False,
    widgets={
        "y": pn.widgets.RadioButtonGroup,
        "device": pn.widgets.RadioButtonGroup,
        "global_phase": pn.widgets.Checkbox,
        "uniformize_phase_hist": pn.widgets.Checkbox,
        "renorm_amp": pn.widgets.Checkbox,
        "show_complex_plane": pn.widgets.Checkbox,
    },
)

app = pn.template.FastListTemplate(
    title="PCC Sample Explorer (Holoviews + Panel)",
    sidebar=[pn.pane.Markdown("### Controls"), controls],
    main=[explorer.view],
)

# In a notebook: just display `app`
# In a script: run `panel serve this_file.py --show`
if __name__ == "__main__":
    app.servable()
    pn.serve(app)
