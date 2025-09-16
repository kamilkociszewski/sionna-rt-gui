from __future__ import annotations

from dataclasses import dataclass, field

import yaml

import mitsuba as mi
from omegaconf import OmegaConf


@dataclass(kw_only=True)
class RadioMapConfig:
    auto_update: bool = True
    accumulate_max_samples_per_tx: int = int(1e9)

    # -- Computation
    center: tuple[float, float, float] | None = None
    orientation: tuple[float, float, float] | None = None
    size: tuple[float, float] | None = None
    cell_size: tuple[float, float] = (1.0, 1.0)
    measurement_surface: str | None = None
    # precoding_vec: tuple[mi.TensorXf, mi.TensorXf] | None = None
    log_samples_per_it: float = 8.0
    max_depth: int = 5
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = True
    refraction: bool = True
    # seed: int = 42
    # rr_depth: int = -1
    # rr_prob: float = 0.95
    # stop_threshold: float | None = None

    # -- Display
    color_map: str = "magma"
    vmin: float = -150
    vmax: float = -50

    @property
    def samples_per_it(self) -> int:
        return int(10**self.log_samples_per_it)


@dataclass(kw_only=True)
class PathsConfig:
    auto_update: bool = True
    accumulate_max_samples_per_src: int = int(1e9)

    max_depth: int = 5
    max_num_paths_per_src: int = 1000000
    samples_per_src: int = 1000000
    synthetic_array: bool = True
    los: bool = True
    specular_reflection: bool = True
    diffuse_reflection: bool = False
    refraction: bool = True


@dataclass(kw_only=True)
class GuiConfig:
    title: str = "Sionna RT"
    config_path: str

    use_live_reload: bool = False
    use_vsync: bool = True
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_resolution: tuple[int, int] = (1920, 1080)

    scene_filename: str | None = None
    loaded_from_snapshot: bool = False

    # Radio map
    radio_map: RadioMapConfig = field(default_factory=lambda: RadioMapConfig())

    # Paths
    paths: PathsConfig = field(default_factory=lambda: PathsConfig())


def load_config(config_path: str, data_path: str | None) -> GuiConfig:
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(config_path, "r") as f:
        loaded = yaml.load(f, Loader=Loader)

    loaded = OmegaConf.create(loaded)
    # Make sure interpolations are resolved, if any
    OmegaConf.resolve(loaded)
    merged = OmegaConf.merge(OmegaConf.structured(GuiConfig), loaded)
    return GuiConfig(config_path=config_path, **merged)
