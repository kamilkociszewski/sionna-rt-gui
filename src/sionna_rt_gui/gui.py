import os

import mitsuba as mi
import numpy as np
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

        # Polyscope structures & groups
        self.setup_ps_structures()

        # TODO: add slice plane controls (Polyscope has built-in support)
        # TODO: add scene drag & drop support

        # # Apply default camera pose
        # ps.set_automatically_compute_scene_extents(False)
        # ps.set_bounding_box((0, 0, 0), (1, 1, 1))

        # TODO: get scene path from config
        self.cfg.scene_filename = os.path.join(
            PROJECT_DIR,
            "..",
            "sionna-rt/src/sionna/rt/scenes/box_two_screens/box_two_screens.xml",
        )
        self.load_scene(self.cfg.scene_filename)

    def setup_ps_structures(self):
        # Clear Polyscope state
        ps.remove_all_structures()
        ps.remove_all_groups()

        self.ps_groups = {
            "scene": ps.create_group("Scene meshes"),
            "rd": ps.create_group("Radio devices"),
        }

    def load_scene(self, scene_path: str):
        self.scene = rt.load_scene(scene_path)

        # Add the meshes to Polyscope
        # TODO: apply consistent materials (based on radio material)
        for mesh in self.scene.mi_scene.shapes():
            vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
            faces = mesh.faces_buffer().numpy().reshape(-1, 3)
            struct = ps.register_surface_mesh(mesh.id(), vertices, faces)
            struct.add_to_group(self.ps_groups["scene"])

    def tick(self):
        self.process_inputs()
        self.gui()

    def add_radio_device(self, position: list[float], is_transmitter: bool):
        # TODO: controllable offset to the clicked surface (along normal?)
        existing_rd = (
            self.scene.transmitters if is_transmitter else self.scene.receivers
        )

        # Create or update point cloud for transmitters or receivers
        position_np = np.array(position)[None, :]
        name = "Transmitters" if is_transmitter else "Receivers"
        if ps.has_point_cloud(name):
            # TODO: is there an easier way to get the existing points?
            # existing_points = ps.get_point_cloud(name).get_position()
            existing_points = [rd.position.numpy().T for rd in existing_rd.values()]
            position_np = np.concatenate(existing_points + [position_np], axis=0)

        struct = ps.register_point_cloud(name, position_np)
        struct.add_to_group(self.ps_groups["rd"])

        # Add actual radio device to Sionna scene
        new_rd = (rt.Transmitter if is_transmitter else rt.Receiver)(
            name=f"{'tx' if is_transmitter else 'rx'}-{len(existing_rd)}",
            position=position,
            orientation=[0, 0, 0],
        )
        self.scene.add(new_rd)

    def clear_radio_devices(self):
        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        for name in self.ps_groups["rd"].get_child_structure_names():
            ps.get_point_cloud(name).remove()

    def process_inputs(self):
        imgui_io = psim.GetIO()

        # Ctrl + left click: add transmitter
        if imgui_io.KeyCtrl and imgui_io.MouseClicked[0]:
            self.add_radio_device(
                ps.screen_coords_to_world_position(imgui_io.MousePos), True
            )
        # Ctrl + right click: add receiver
        if imgui_io.KeyCtrl and imgui_io.MouseClicked[1]:
            self.add_radio_device(
                ps.screen_coords_to_world_position(imgui_io.MousePos), False
            )

        # R: reload code
        if psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("R")), repeat=False):
            self.code_reload_requested = True

        # Exit
        if (
            imgui_io.KeyCtrl
            and psim.IsKeyPressed(psim.ImGuiKey(ps.get_key_code("Q")))
            or psim.IsKeyPressed(psim.ImGuiKey_Escape)
        ):
            ps.unshow()

    def gui(self):

        if psim.CollapsingHeader("Radio devices", psim.ImGuiTreeNodeFlags_DefaultOpen):
            clicked = psim.Button("Clear all radio devices")
            if clicked:
                self.clear_radio_devices()

            # TODO: button to place radio devices: at random; or samples from a radio map
            # TODO: one entry for each transmitter & receiver (delete button, scattering pattern, orientation, etc)
            # TODO: widget to move & rotate existing radio devices
            psim.Spacing()

        if psim.CollapsingHeader("Radio map", psim.ImGuiTreeNodeFlags_DefaultOpen):
            clicked = psim.Button("Compute radio map")
            if clicked:
                print("Should compute radio map")

            # TODO: simulation parameters (diffuse, specular, sample count, height, etc)
            # TODO: plane / mesh picker (changes type of radio map)
            # TODO: display parameters (transparency, colormap, etc)

            psim.Spacing()

        if psim.CollapsingHeader("Paths", psim.ImGuiTreeNodeFlags_DefaultOpen):
            clicked = psim.Button("Compute paths")
            if clicked:
                print("Should compute paths")

            # TODO: simulation parameters (diffuse, specular, sample count, height, etc)

            psim.Spacing()

        if psim.CollapsingHeader("Rendering"):
            changed, self.cfg.use_vsync = psim.Checkbox("VSync", self.cfg.use_vsync)
            if changed:
                ps.set_enable_vsync(self.cfg.use_vsync)

            psim.Spacing()

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

            psim.Spacing()
