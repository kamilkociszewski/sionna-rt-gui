from common import add_project_root_to_path

add_project_root_to_path()

from sionna import rt
import sionna_rt_gui as gui
import numpy as np
import polyscope as ps


def main():
    gui.hello()

if __name__ == "__main__":
    main()
