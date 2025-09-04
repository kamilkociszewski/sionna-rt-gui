import os

import mitsuba as mi
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

from . import PROJECT_DIR
from .config import GuiConfig


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        # --- Inputs state
        self.last_mouse_pos: mi.ScalarVector2f | None = None
        self.reset_accumulation_requested = False

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        # --- Polyscope setup
        self.build_default_ps_gui: bool = False
        # Pre-init settings
        ps.set_program_name(self.cfg.title)
        # Window size and position will be loaded from the last run from `.polyscope.ini`
        ps.set_use_prefs_file(True)
        # ps.set_navigation_style("none")
        ps.set_enable_vsync(cfg.use_vsync)
        ps.set_max_fps(-1)
        ps.set_user_gui_is_on_right_side(False)
        ps.set_verbosity(1)
        ps.set_build_default_gui_panels(self.build_default_ps_gui)
        ps.set_background_color(self.cfg.background_color)
        ps.set_ground_plane_mode("none")
        ps.set_window_resizable(True)
        ps.set_window_size(*self.cfg.default_resolution)
        ps.set_give_focus_on_show(True)
        ps.set_transparency_mode("pretty")

        was_initialized = ps.is_initialized()
        if not was_initialized:
            ps.set_up_dir("z_up")
            ps.init()

        # TODO: add slice plane controls (Polyscope has built-in support)
        # TODO: add scene drag & drop support

        # # Apply default camera pose
        # ps.set_automatically_compute_scene_extents(False)
        # ps.set_bounding_box((0, 0, 0), (1, 1, 1))

        # TODO: get scene path from config
        scene_path = os.path.join(
            PROJECT_DIR,
            "..",
            "sionna-rt/src/sionna/rt/scenes/box_two_screens/box_two_screens.xml",
        )
        self.load_scene(scene_path)

    def load_scene(self, scene_path: str):
        self.scene = rt.load_scene(scene_path)

        # Clear Polyscope state
        ps.remove_all_structures()

        # Add the meshes to Polyscope
        # TODO: apply consistent materials (based on radio material)
        for mesh in self.scene.mi_scene.shapes():
            vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
            faces = mesh.faces_buffer().numpy().reshape(-1, 3)
            ps.register_surface_mesh(mesh.id(), vertices, faces)

    def tick(self):
        self.gui()

    def gui(self):
        if psim.CollapsingHeader("Rendering"):
            changed, self.cfg.use_vsync = psim.Checkbox("VSync", self.cfg.use_vsync)
            if changed:
                ps.set_enable_vsync(self.cfg.use_vsync)

        if psim.CollapsingHeader("Interface", psim.ImGuiTreeNodeFlags_DefaultOpen):
            self.cfg.background_color = ps.get_background_color()
            changed, self.cfg.background_color = psim.ColorEdit4(
                "Background",
                self.cfg.background_color,
                psim.ImGuiColorEditFlags_NoInputs,
            )
            if changed:
                ps.set_background_color(self.cfg.background_color)

            changed, self.build_default_ps_gui = psim.Checkbox(
                "Show Polyscope UI", self.build_default_ps_gui
            )
            if changed:
                ps.set_build_default_gui_panels(self.build_default_ps_gui)
