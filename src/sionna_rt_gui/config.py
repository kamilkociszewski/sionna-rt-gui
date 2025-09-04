from dataclasses import dataclass, field

import yaml

from omegaconf import OmegaConf


@dataclass(kw_only=True)
class RadioMapConfig:
    auto_update: bool = True
    color_map: str = "viridis"
    vmin: float = -150
    vmax: float = -50


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
    auto_update_paths: bool = True


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
