from dataclasses import asdict, dataclass


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

    def __post_init__(self):
        if len(self.amp_range) != 2:
            raise ValueError("amp_range must contain exactly two values.")
        self.amp_range = tuple(float(v) for v in self.amp_range)

    def as_dict(self) -> dict:
        return asdict(self)
