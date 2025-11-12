from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
import os
import yaml

from omegaconf import OmegaConf
from sionna import rt
from sionna.rt.antenna_pattern import (
    antenna_pattern_registry,
    polarization_registry,
)

from . import DATA_DIR

# ------------------------


@dataclass(kw_only=True)
class AntennaArrayConfig:
    """
    Class holding configuration parameters for an antenna array.
    This is needed because once created, antenna array objects don't
    expose those fields.
    """

    num_rows: int = 1
    num_cols: int = 1
    vertical_spacing: float = 0.5
    horizontal_spacing: float = 0.5
    pattern_i: int = antenna_pattern_registry.list().index("iso")
    polarization_i: int = polarization_registry.list().index("V")

    @property
    def pattern(self) -> str:
        return antenna_pattern_registry.list()[self.pattern_i]

    @property
    def polarization(self) -> str:
        return polarization_registry.list()[self.polarization_i]

    def create(self) -> rt.AntennaArray:
        return rt.PlanarArray(
            num_rows=self.num_rows,
            num_cols=self.num_cols,
            vertical_spacing=self.vertical_spacing,
            horizontal_spacing=self.horizontal_spacing,
            pattern=self.pattern,
            polarization=self.polarization,
        )


# ------------------------


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
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = True
    # seed: int = 42
    # rr_depth: int = -1
    # rr_prob: float = 0.95
    # stop_threshold: float | None = None

    # -- Display
    color_map: str = "viridis"
    use_alpha: bool = True
    vmin: float = -150
    vmax: float = -50
    show_colorbar: bool = True
    # When updating the radio map, upload new values directly from the device.
    # This is only supported when using a CUDA variant.
    use_direct_update_from_device: bool = True

    @property
    def samples_per_it(self) -> int:
        return int(10**self.log_samples_per_it)


# ------------------------


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
    refraction: bool = False
    diffraction: bool = False
    edge_diffraction: bool = False
    diffraction_lit_region: bool = False


# ------------------------


class RenderingMode(Enum):
    RASTERIZATION = 0
    RAY_TRACING = 1


RENDERING_MODE_NAMES = ["Rasterization", "Ray tracing"]
assert len(RENDERING_MODE_NAMES) == len(RenderingMode)


@dataclass(kw_only=True)
class RenderingConfig:
    mode: RenderingMode = RenderingMode.RAY_TRACING
    # Full resolution of the window
    default_resolution: tuple[int, int] = (1920, 1080)
    # Relative resolution for ray traced rendering
    # TODO: relative rendering resolution picker (10%, 25%, 50%, 100%)
    relative_resolution: float = 0.5
    spp_per_frame: int = 16
    max_accumulated_spp: int = 256

    # Use the OptiX denoiser (CUDA variant only)
    use_denoiser: bool = True

    envmap: str | None = os.path.join(
        DATA_DIR, "envmaps", "teufelsberg_ground_2_1k.exr"
    )
    # Brightness factor to apply to the environment map.
    envmap_factor: float = 1.0
    # Rotation of the environment map along the vertical axis, in degrees.
    envmap_rotation_deg: float = -60

    @property
    def rendering_resolution(self) -> tuple[int, int]:
        return (
            int(self.default_resolution[0] * self.relative_resolution),
            int(self.default_resolution[1] * self.relative_resolution),
        )


# ------------------------


class GuiMode(Enum):
    HIDDEN = 0
    FULL = 1


GUI_MODE_NAMES = ["Hidden", "Full"]
assert len(GUI_MODE_NAMES) == len(GuiMode)

# ------------------------


@dataclass(kw_only=True)
class GuiConfig:
    title: str = "Sionna RT"
    config_path: str

    gui_mode: GuiMode = GuiMode.FULL
    show_polyscope_gui: bool = False
    use_live_reload: bool = False
    use_vsync: bool = True
    background_color: tuple[float, float, float] = (0.0, 0.0, 0.0)

    scene_filename: str | None = None
    loaded_from_snapshot: bool = False

    # Logging
    log_level: int = logging.INFO

    # Antenna arrays
    tx_array: AntennaArrayConfig = field(default_factory=AntennaArrayConfig)
    rx_array: AntennaArrayConfig = field(default_factory=AntennaArrayConfig)

    # Rendering
    rendering: RenderingConfig = field(default_factory=RenderingConfig)

    # Radio map
    radio_map: RadioMapConfig = field(default_factory=RadioMapConfig)

    # Paths
    paths: PathsConfig = field(default_factory=PathsConfig)


# ------------------------


def load_config(config_path: str, data_path: str | None) -> GuiConfig:
    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    with open(config_path, "r") as f:
        loaded = yaml.load(f, Loader=Loader)
        # The config file might be empty.
        if loaded is None:
            loaded = {}

    loaded = OmegaConf.create(loaded)
    # Resolve interpolations, if any.
    OmegaConf.resolve(loaded)
    loaded["config_path"] = config_path
    merged = OmegaConf.merge(OmegaConf.structured(GuiConfig), loaded)
    return OmegaConf.to_object(merged)
