import argparse
import os


from common import add_project_root_to_path

add_project_root_to_path()

from sionna import rt
import sionna_rt_gui as gui
from sionna_rt_gui import AppHolder, GuiConfig, DEFAULT_CONFIG_PATH
import numpy as np
import polyscope as ps


def main():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Interactive Sionna RT GUI"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the GUI configuration file to use.",
    )
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument(
        "--watch", action="store_true", dest="watch", default=False
    )
    watch_group.add_argument("--no-watch", action="store_false", dest="watch")
    # TODO: load a specific Sionna RT scene and / or config
    args = parser.parse_args()

    cfg_overrides = {
        "use_live_reload": args.watch,
    }
    # TODO
    cfg = GuiConfig(config_path=args.config)
    data_path = ""

    # --- Initialization
    app = AppHolder(cfg, data_path=data_path, overrides=cfg_overrides)

    # --- Running loop
    app.show()


if __name__ == "__main__":
    main()
