import polyscope as ps
from sionna import rt

from sionna_rt_gui.config import GuiConfig


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        was_initialized = ps.is_initialized()
        if not was_initialized:
            ps.set_up_dir("z_up")
            ps.init()

    def tick(self):
        pass
