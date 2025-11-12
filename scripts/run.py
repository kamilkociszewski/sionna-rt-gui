import argparse
import logging
import os

from common import add_project_root_to_path

add_project_root_to_path()

from sionna_rt_gui import AppHolder, DEFAULT_CONFIG_PATH
from sionna_rt_gui.config import load_config


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
    data_path = None
    cfg = load_config(args.config, data_path=data_path)

    # Configure logging
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Initialization
    app = AppHolder(cfg, data_path=data_path, overrides=cfg_overrides)

    # --- Running loop
    app.show()


if __name__ == "__main__":
    main()
