from dataclasses import dataclass, field


@dataclass
class PCCConfig:
    N: int = 128
    amp_radial_decay: float = 2.2
    amp_smooth_sigma: float = 6.0
    amp_range: tuple = (0.7, 1.4)

    phase_smooth_sigma: float = 2.0
    incoh_highpass_sigma: float = 16.0
    incoh_scale: float = 0.2
    global_phase: bool = True
    uniformize_phase_hist: bool = True

    translate_px: int = 0
    rotate_deg: float = 0.0
    noise_std: float = 0.1
    renorm_amp: bool = False


@dataclass
class HYPERPARAMS:
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 200
    batch_size: int = 32
    epochs: int = 2
    lr: float = 2e-3
    seed: int = 0
    run_dir: str = "runs"


@dataclass
class TrainConfig:
    pcc: PCCConfig = field(default_factory=PCCConfig)
    hyperparams: HYPERPARAMS = field(default_factory=HYPERPARAMS)
    model: str = "complex"
    normalize: bool = True
