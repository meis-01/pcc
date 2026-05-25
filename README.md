# PCC Sampler

`pcc-sampler` generates synthetic complex-valued image samples where the label is
defined by spatial phase coherence rather than amplitude cues.

The package is intentionally small: config in, deterministic complex samples out.
It is meant to be imported by experiments that compare real-valued and
complex-valued neural networks.

## Install

From another local project, install this repo in editable mode:

```powershell
pip install -e C:\Users\meisa\Projects\pcc
```

Or install from a Git URL after publishing the branch:

```bash
pip install git+https://github.com/<owner>/<repo>.git
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

- `z`: complex-valued sample, shape `(N, N)`, dtype `torch.complex64`
- `amplitude`: amplitude field, shape `(N, N)`, dtype `torch.float32`
- `phase`: phase field, shape `(N, N)`, dtype `torch.float32`
- `y`: label tensor, where `0` is coherent and `1` is incoherent

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
