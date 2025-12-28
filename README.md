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
- **mag (doomed):** real CNN on magnitude only |z|
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

### 2) Train a single run
```bash
python -m src.train --model complex --N 128 --train_size 20000 --val_size 2000 --test_size 5000
```

Try baselines:
```bash
python -m src.train --model mag
python -m src.train --model cossin
```

### 3) Sample-efficiency sweep
```bash
python -m src.sweep --model complex --N 128 --train_sizes 500,1000,2000,5000,10000,20000
```

Outputs are written to `runs/` with a JSON summary and the best checkpoint.

## Notes
- This benchmark is intentionally synthetic but MRI-inspired: complex fields with controlled amplitude and phase behavior.
- If you want to mimic k-space effects later, add a Fourier step + sampling mask before reconstruction; the generator is modular.
