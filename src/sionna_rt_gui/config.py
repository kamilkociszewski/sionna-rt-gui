from dataclasses import dataclass

import omegaconf


@dataclass(kw_only=True)
class GuiConfig:
    use_live_reload: bool = False

    # TODO: remove these if not relevant
    config_path: str
    snapshot_filename: str | None = None
    loaded_from_snapshot: bool = False
