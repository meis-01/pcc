# PCC Sampler

`pcc-sampler` (Phase Coherence Complex - Sampler) generates synthetic complex-valued image samples where the label is
defined by spatial phase coherence rather than amplitude cues.

The package is intentionally small: config in, deterministic complex samples out.
It is meant to be imported by experiments that study complex-valued neural networks.

## Install

The examples below use the `uv` command. If `uv` is installed but not on PATH,
run the same commands as `python -m uv ...`.

Install the project environment with uv:

```powershell
uv sync
```

Run a quick import check:

```powershell
uv run python -c "from pcc import load_default_config; print(load_default_config())"
```

From another local uv project, add this repo in editable mode:

```powershell
uv add --editable C:\Users\meisa\Projects\pcc
```

Or add it from a Git URL after publishing the repo:

```bash
uv add "pcc-sampler @ git+https://github.com/<owner>/<repo>.git"
```

## Python Usage

```python
from pcc import PCCConfig, PCCDataset

cfg = PCCConfig(
    N=128,
    phase_smooth_sigma=2.0,
    incoh_highpass_sigma=16.0,
    incoh_scale=0.2,
    uniformize_phase_hist=True,
)

dataset = PCCDataset(size=10_000, cfg=cfg, seed=42)
z, amplitude, phase, y = dataset[0]
```

Returned values:

- `z`: complex-valued sample, shape `(N, N)`, dtype `numpy.complex64`
- `amplitude`: amplitude field, shape `(N, N)`, dtype `numpy.float32`
- `phase`: phase field, shape `(N, N)`, dtype `numpy.float32`
- `y`: label scalar, where `0` is coherent and `1` is incoherent

## YAML Config Usage

Create a YAML file:

```yaml
N: 128
amp_radial_decay: 2.2
amp_smooth_sigma: 6.0
amp_range: [0.7, 1.4]

phase_smooth_sigma: 2.0
incoh_highpass_sigma: 16.0
incoh_scale: 0.2
global_phase: true
uniformize_phase_hist: true

translate_px: 16
rotate_deg: 30.0
noise_std: 0.1
renorm_amp: false
```

Load it directly:

```python
from pcc import dataset_from_yaml

dataset = dataset_from_yaml("configs/pcc_phase.yaml", size=50_000, seed=7)
```

You can also load the built-in default:

```python
from pcc import PCCDataset, load_default_config

cfg = load_default_config()
dataset = PCCDataset(size=1_000, cfg=cfg, seed=0)
```

## Single Sample

```python
from pcc import PCCConfig, make_deterministic_sample

cfg = PCCConfig()
z, amplitude, phase = make_deterministic_sample(cfg, y=0, seed=123, idx=0)
```

The same `(seed, idx, y, config)` combination produces the same sample.

## API Parameters

### `PCCConfig`

`PCCConfig` controls the image size, amplitude field, phase field, nuisance
transforms, and complex noise added to each sample.

| Parameter | Default | Description |
| --- | --- | --- |
| `N` | `128` | Height and width of each square sample. Returned arrays have shape `(N, N)`. |
| `amp_radial_decay` | `2.2` | Strength of the radial amplitude falloff from the center toward the image edges. Larger values make the amplitude fade faster. |
| `amp_smooth_sigma` | `6.0` | Gaussian smoothing scale for the random amplitude texture. Larger values produce smoother amplitude fields. |
| `amp_range` | `(0.7, 1.4)` | Minimum and maximum range used to rescale the smoothed amplitude texture before radial weighting. |
| `phase_smooth_sigma` | `2.0` | Gaussian smoothing scale for the coherent phase field. Larger values produce more slowly varying phase. |
| `incoh_highpass_sigma` | `16.0` | Low-pass scale removed from incoherent phase noise. Larger values leave broader high-frequency phase variations. |
| `incoh_scale` | `0.2` | Standard-deviation scale of the high-pass phase perturbation added to incoherent samples. |
| `global_phase` | `True` | Adds one random global phase offset to each sample when enabled. |
| `uniformize_phase_hist` | `True` | Remaps phase values by rank so each sample has an approximately uniform phase histogram. This reduces label leakage through phase histogram differences. |
| `translate_px` | `0` | Maximum integer translation, in pixels, sampled independently for x and y. A value of `16` samples shifts from `-16` to `16`. |
| `rotate_deg` | `0.0` | Maximum absolute rotation angle in degrees. A value of `30.0` samples rotations from `-30` to `30` degrees. |
| `noise_std` | `0.1` | Standard deviation of complex Gaussian noise added to `z`. Set to `0` for noiseless samples. |
| `renorm_amp` | `False` | If enabled after noise is added, rescales `z` so its mean amplitude matches the pre-noise amplitude field. |

### Dataset And Sampling Arguments

| Function or class | Parameter | Description |
| --- | --- | --- |
| `PCCDataset(size, cfg, seed=0)` | `size` | Number of deterministic items exposed by the dataset. |
| `PCCDataset(size, cfg, seed=0)` | `cfg` | A `PCCConfig` instance that defines how samples are generated. |
| `PCCDataset(size, cfg, seed=0)` | `seed` | Base seed. The dataset combines this with the item index so `dataset[i]` is repeatable. |
| `make_deterministic_sample(cfg, y, seed=0, idx=0)` | `cfg` | A `PCCConfig` instance. |
| `make_deterministic_sample(cfg, y, seed=0, idx=0)` | `y` | Label to generate: `0` for coherent phase, `1` for incoherent phase. |
| `make_deterministic_sample(cfg, y, seed=0, idx=0)` | `seed` | Base seed for deterministic generation. |
| `make_deterministic_sample(cfg, y, seed=0, idx=0)` | `idx` | Index mixed with `seed` to produce a repeatable per-sample random generator. |
| `dataset_from_yaml(path, size, seed=0)` | `path` | YAML file containing `PCCConfig` fields, either at the top level or under a `pcc:` key. |
| `dataset_from_yaml(path, size, seed=0)` | `size` | Number of deterministic dataset items to expose. |
| `dataset_from_yaml(path, size, seed=0)` | `seed` | Base seed used by the returned dataset. |
