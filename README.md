# Phase-Coherence Classification (PCC) Benchmark — MRI-ish Complex Fields

This project generates a synthetic complex-valued image benchmark where **amplitude cues are controlled** and the label depends on **spatial phase coherence**.
It is designed to test whether **complex-valued neural networks** learn phase-relational structure more naturally than real-valued baselines.

## Task

Binary classification:

- **Class 0 (coherent):** smooth phase field
- **Class 1 (incoherent):** same base phase + high-frequency phase scramble

Key properties:

- **MRI-ish amplitude envelope (A3):** radial decay + smooth tissue-like variation
- **Global phase shift** applied (absolute phase irrelevant)
- **Per-sample phase histogram equalization** (marginal phase distribution matched across classes)
- Label signal is **only in spatial phase relations**

## Models included

- **mag:** real CNN on magnitude only |z|
- **R2:** real CNN on real and imaginary components as input channels
- **R2c:** real CNN on real and imaginary components as input channels constraind on convolution similar to complex (preserves rotation)
- **cossin (real-fair):** real CNN on (cos θ, sin θ) (+ optional amplitude channel)
- **complex:** complex CNN using complex convolutions + modReLU, trained on complex input z

## Quickstart

### 1) Create env

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Train runs

The experiments are stored in experiments forlder and the train.train module should be called with the patht to the experiment yaml file. 

To run the default training:

```bash
python -m pcc.train.main
```

## Notes

- This benchmark is intentionally synthetic but MRI-inspired: complex fields with controlled amplitude and phase behavior.
- If you want to mimic k-space effects later, add a Fourier step + sampling mask before reconstruction; the generator is modular.
