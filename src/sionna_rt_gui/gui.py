import os

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

from .config import GuiConfig
from .sionna_utils import (
    add_radio_map_to_polyscope,
    add_radio_device_to_polyscope,
    add_scene_to_polyscope,
    get_built_in_scenes,
    add_paths_to_polyscope,
)


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        # --- Sionna RT
        # Scene
        built_in_scenes = get_built_in_scenes()
        self.built_in_scene_names = ["None"] + list(built_in_scenes.keys())
        self.built_in_scene_paths = [None] + list(built_in_scenes.values())
        self.current_scene_idx: int = 0

        # Radio map results
        self.radio_map: rt.RadioMap | None = None
        self.rm_accumulated_samples: int = 0
        self.rm_color_map_options = [v for v in plt.colormaps() if not v.endswith("_r")]
        self.rm_color_map_index = self.rm_color_map_options.index(
            self.cfg.radio_map.color_map
        )

        # Paths results
        self.paths: rt.Paths | None = None

        # --- Inputs state
        self.last_mouse_pos: mi.ScalarVector2f | None = None
        self.reset_accumulation_requested = False

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        # --- Polyscope setup
        # Can be used to derive e.g. random seeds.
        self.ps_groups: dict[str, ps.Group] = {}
        self.frame_i: int = 0
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
            ps.set_front_dir("y_front")
            ps.init()

        # Polyscope structures & groups
        self.reset_and_setup_structures()

        # TODO: add slice plane controls (Polyscope has built-in support)
        # TODO: add scene drag & drop support

        # # Apply default camera pose
        # ps.set_automatically_compute_scene_extents(False)
        # ps.set_bounding_box((0, 0, 0), (1, 1, 1))

        # Load scene
        # TODO: preserve currently-selected scene across reloads
        self.load_scene(
            self.cfg.scene_filename or built_in_scenes["simple_street_canyon_with_cars"]
        )

        # TODO: remove this
        if False:
            # Add some example transmitters
            for pos in [
                [50, -10, 29 + 2.5],
                [13, 13, 51 + 2.5],
                [-34, -10, 22 + 2.5],
            ]:
                self.add_radio_device(pos, is_transmitter=True, allow_auto_update=False)

                shifted = [pos[0] + 5, pos[1] + 15, pos[2] - 10]
                self.add_radio_device(
                    shifted, is_transmitter=False, allow_auto_update=False
                )

        if False:
            self.set_radio_map(self.compute_radio_map(), show=True)
        if False:
            self.paths = self.compute_paths()
            add_paths_to_polyscope(self.paths, self.ps_groups, self.cfg.paths)

    def reset_and_setup_structures(self):
        # Clear Sionna state
        self.clear_radio_map()
        self.paths = None

        # Clear Polyscope state
        ps.remove_all_structures()
        ps.remove_all_groups()

        self.ps_groups = {
            "scene": ps.create_group("Scene meshes"),
            "rd": ps.create_group("Radio devices"),
            "radio_maps": ps.create_group("Radio maps"),
            "paths": ps.create_group("Paths"),
        }

    def load_scene(self, scene_path: str, recenter_camera: bool = True):
        self.reset_and_setup_structures()
        self.scene = rt.load_scene(scene_path)

        # TODO: make antenna arrays configurable
        self.scene.tx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="cross",
        )
        self.scene.rx_array = rt.PlanarArray(
            num_rows=1,
            num_cols=1,
            vertical_spacing=0.5,
            horizontal_spacing=0.5,
            pattern="iso",
            polarization="cross",
        )

        # TODO: setup configurable radio material diffuse & thickness
        for sh in self.scene.mi_scene.shapes():
            if isinstance(sh.bsdf(), rt.RadioMaterialBase):
                sh.bsdf().scattering_coefficient = 0.2
                sh.bsdf().thickness = 10.0

        self.cfg.scene_filename = scene_path
        try:
            self.current_scene_idx = self.built_in_scene_paths.index(
                self.cfg.scene_filename
            )
        except ValueError:
            pass

        add_scene_to_polyscope(self.scene, self.ps_groups)
        if recenter_camera:
            # TODO: automatically zoom to fill the screen, if scene has changed
            ps.set_view_center(self.scene.mi_scene.bbox().center(), fly_to=False)

    # ------------------------

    def tick(self):
        # TODO: automatic refinement & accumulation of the radio map, if enabled
        self.process_inputs()

        # Automatic refinement of the radio map
        if self.radio_map is not None:
            if (
                self.rm_accumulated_samples
                < self.cfg.radio_map.accumulate_max_samples_per_tx
            ):
                rm_new = self.compute_radio_map()
                self.radio_map._pathgain_map += rm_new.path_gain
                self.rm_accumulated_samples += self.cfg.radio_map.samples_per_it // len(
                    self.scene._transmitters
                )
                add_radio_map_to_polyscope(
                    "radio_map", self.radio_map, self.ps_groups, self.cfg.radio_map
                )

        self.gui()
        self.frame_i += 1

    # ------------------------

    def set_radio_map(self, radio_map: rt.RadioMap, show: bool = False):
        self.radio_map = radio_map
        self.rm_accumulated_samples = 0
        if show:
            add_radio_map_to_polyscope(
                "radio_map", self.radio_map, self.ps_groups, self.cfg.radio_map
            )

    def compute_radio_map(self) -> rt.RadioMap | None:
        if not self.scene._transmitters:
            return None

        solver = rt.RadioMapSolver()
        # TODO: expose and pass down all the relevant parameters
        samples_per_tx = self.cfg.radio_map.samples_per_it // len(
            self.scene._transmitters
        )
        return solver(
            self.scene,
            seed=self.frame_i,
            center=self.cfg.radio_map.center,
            orientation=self.cfg.radio_map.orientation,
            size=self.cfg.radio_map.size,
            cell_size=self.cfg.radio_map.cell_size,
            measurement_surface=self.cfg.radio_map.measurement_surface,
            # precoding_vec=self.cfg.radio_map.precoding_vec,
            samples_per_tx=samples_per_tx,
            max_depth=self.cfg.radio_map.max_depth,
            los=self.cfg.radio_map.los,
            specular_reflection=self.cfg.radio_map.specular_reflection,
            diffuse_reflection=self.cfg.radio_map.diffuse_reflection,
            refraction=self.cfg.radio_map.refraction,
        )

    # ------------------------

    def compute_paths(self) -> rt.Paths | None:
        if not self.scene._transmitters or not self.scene._receivers:
            return None

        solver = rt.PathSolver()
        return solver(
            self.scene,
            max_depth=self.cfg.paths.max_depth,
            max_num_paths_per_src=self.cfg.paths.max_num_paths_per_src,
            samples_per_src=self.cfg.paths.samples_per_src,
            synthetic_array=self.cfg.paths.synthetic_array,
            los=self.cfg.paths.los,
            specular_reflection=self.cfg.paths.specular_reflection,
            diffuse_reflection=self.cfg.paths.diffuse_reflection,
            refraction=self.cfg.paths.refraction,
            seed=self.frame_i,
        )

    # ------------------------

    def add_radio_device(
        self,
        position: list[float],
        is_transmitter: bool,
        allow_auto_update: bool = True,
    ):
        # TODO: controllable offset to the clicked surface (along normal?)
        existing_rd = (
            self.scene.transmitters if is_transmitter else self.scene.receivers
        )

        # Add actual radio device to Sionna scene
        new_rd = (rt.Transmitter if is_transmitter else rt.Receiver)(
            name=f"{'tx' if is_transmitter else 'rx'}-{len(existing_rd)}",
            position=position,
            orientation=[0, 0, 0],
        )
        self.scene.add(new_rd)

        add_radio_device_to_polyscope(
            position, is_transmitter, existing_rd, self.ps_groups
        )

        if allow_auto_update and self.cfg.radio_map.auto_update and is_transmitter:
            self.set_radio_map(self.compute_radio_map(), show=True)
        if allow_auto_update and self.cfg.paths.auto_update:
            self.paths = self.compute_paths()
            add_paths_to_polyscope(self.paths, self.ps_groups, self.cfg.paths)

    def clear_radio_devices(self):
        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        for name in self.ps_groups["rd"].get_child_structure_names():
            ps.get_point_cloud(name).remove()

        if self.cfg.radio_map.auto_update:
            self.clear_radio_map()
        if self.cfg.paths.auto_update:
            self.clear_paths()

    def clear_radio_map(self):
        self.radio_map = None
        self.rm_accumulated_samples = 0
        if ps.has_surface_mesh("radio_map"):
            ps.get_surface_mesh("radio_map").remove()

    def clear_paths(self):
        self.paths = None
        if ps.has_curve_network("paths"):
            ps.get_curve_network("paths").remove()

    # ------------------------

    def process_inputs(self):
        imgui_io = psim.GetIO()

        # Ctrl + left/right click: add transmitter/receiver
        if imgui_io.KeyCtrl and (imgui_io.MouseClicked[0] or imgui_io.MouseClicked[1]):
            is_transmitter = imgui_io.MouseClicked[0]
            # TODO: configurable placement offset along the normal
            rd_position = ps.screen_coords_to_world_position(imgui_io.MousePos)
            rd_position += (0, 0, 2.5)
            self.add_radio_device(rd_position, is_transmitter)

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
        # TODO: set ImGui window title

        if psim.CollapsingHeader("Scene", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            # psim.Text(f"Current scene:\n{os.path.basename(self.cfg.scene_filename)}")

            # Quick pick from built-in scenes
            psim.Text("Scene selection:")
            changed, combo_i = psim.Combo(
                "##scene_picker",
                self.current_scene_idx,
                self.built_in_scene_names,
            )
            if changed:
                new_scene = self.built_in_scene_paths[combo_i]
                if new_scene is not None:
                    self.load_scene(new_scene)

            psim.Spacing()

        if psim.CollapsingHeader("Radio devices", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()

            # TODO: legend (= color picker) for each radio device type

            clicked = psim.Button("Clear all radio devices")
            if clicked:
                self.clear_radio_devices()

            # TODO: scene-wide TX and RX array configuration
            # TODO: button to place radio devices: at random; or samples from a radio map
            # TODO: one entry for each transmitter & receiver (delete button, scattering pattern, orientation, etc)
            # TODO: widget to move & rotate existing radio devices
            psim.Spacing()

        if psim.CollapsingHeader("Radio map", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False
            needs_visual_update = False

            clicked = psim.Button("Compute radio map")
            if clicked:
                self.set_radio_map(self.compute_radio_map(), show=True)

            psim.SameLine()
            _, self.cfg.radio_map.auto_update = psim.Checkbox(
                "Automatic update##rm", self.cfg.radio_map.auto_update
            )

            # -- Radio map computation options
            psim.Spacing()

            changed, self.cfg.radio_map.cell_size = psim.InputFloat2(
                "Cell size",
                self.cfg.radio_map.cell_size,
                # v_min=0.1,
                # v_max=100,
                format="%.2f",
            )
            if changed:
                self.cfg.radio_map.cell_size = tuple(
                    min(max(v, 0.01), 100) for v in self.cfg.radio_map.cell_size
                )
            needs_update |= changed

            changed, self.cfg.radio_map.log_samples_per_it = psim.SliderFloat(
                "Samples / it (log 10)",
                self.cfg.radio_map.log_samples_per_it,
                v_min=0,
                v_max=9,
                format="10^%.1f",
            )
            needs_update |= changed

            changed, self.cfg.radio_map.max_depth = psim.SliderInt(
                "Max depth##rm",
                self.cfg.radio_map.max_depth,
                v_min=1,
                v_max=10,
            )
            needs_update |= changed

            # TODO: why is the width ignored?
            psim.SetNextItemWidth(200)
            changed, self.cfg.radio_map.los = psim.Checkbox(
                "Line of sight##rm", self.cfg.radio_map.los
            )
            needs_update |= changed

            psim.SameLine()
            psim.SetNextItemWidth(200)
            changed, self.cfg.radio_map.specular_reflection = psim.Checkbox(
                "Specular reflection##rm", self.cfg.radio_map.specular_reflection
            )
            needs_update |= changed

            psim.SetNextItemWidth(200)
            changed, self.cfg.radio_map.diffuse_reflection = psim.Checkbox(
                "Diffuse reflection##rm", self.cfg.radio_map.diffuse_reflection
            )
            needs_update |= changed

            psim.SameLine()
            psim.SetNextItemWidth(200)
            changed, self.cfg.radio_map.refraction = psim.Checkbox(
                "Refraction##rm", self.cfg.radio_map.refraction
            )
            needs_update |= changed

            # -- Radio map display options
            if self.radio_map is not None:
                psim.Spacing()

                psim.Text("Accumulating samples:")
                psim.ProgressBar(
                    min(
                        self.rm_accumulated_samples
                        / self.cfg.radio_map.accumulate_max_samples_per_tx,
                        1.0,
                    ),
                )

                struct = ps.get_surface_mesh("radio_map")
                changed, show_rm = psim.Checkbox("Show radio map", struct.is_enabled())
                if changed:
                    struct.set_enabled(show_rm)

                changed_cmap, self.rm_color_map_index = psim.Combo(
                    "Colormap",
                    self.rm_color_map_index,
                    self.rm_color_map_options,
                )
                if changed_cmap:
                    self.cfg.radio_map.color_map = self.rm_color_map_options[
                        self.rm_color_map_index
                    ]

                changed_vmin, self.cfg.radio_map.vmin = psim.SliderFloat(
                    "vmin",
                    self.cfg.radio_map.vmin,
                    v_min=-200,
                    v_max=0,
                )
                changed_vmax, self.cfg.radio_map.vmax = psim.SliderFloat(
                    "vmax",
                    self.cfg.radio_map.vmax,
                    v_min=-200,
                    v_max=0,
                )
                self.cfg.radio_map.vmin = min(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )
                self.cfg.radio_map.vmax = max(
                    self.cfg.radio_map.vmin, self.cfg.radio_map.vmax
                )

                needs_visual_update = changed_cmap or changed_vmin or changed_vmax

            if self.cfg.radio_map.auto_update and needs_update:
                self.set_radio_map(self.compute_radio_map(), show=False)
            if needs_update or needs_visual_update:
                add_radio_map_to_polyscope(
                    "radio_map", self.radio_map, self.ps_groups, self.cfg.radio_map
                )

            # TODO: simulation parameters (diffuse, specular, sample count, height, etc)
            # TODO: plane / mesh picker (changes type of radio map)
            # TODO: display parameters (transparency, colormap, etc)

            psim.Spacing()

        if psim.CollapsingHeader("Paths", psim.ImGuiTreeNodeFlags_DefaultOpen):
            psim.Spacing()
            needs_update = False

            _, self.cfg.paths.auto_update = psim.Checkbox(
                "Automatic update##paths", self.cfg.paths.auto_update
            )

            clicked = psim.Button("Compute paths")
            if clicked:
                self.paths = self.compute_paths()
                add_paths_to_polyscope(self.paths, self.ps_groups, self.cfg.paths)

            changed, self.cfg.paths.max_depth = psim.SliderInt(
                "Max depth##paths",
                self.cfg.paths.max_depth,
                v_min=1,
                v_max=10,
            )
            needs_update |= changed

            # TODO:
            # max_num_paths_per_src
            # samples_per_src

            psim.SetNextItemWidth(200)
            changed, self.cfg.paths.synthetic_array = psim.Checkbox(
                "Synthetic array##paths", self.cfg.paths.synthetic_array
            )
            needs_update |= changed

            psim.SetNextItemWidth(200)
            changed, self.cfg.paths.los = psim.Checkbox(
                "Line of sight##paths", self.cfg.paths.los
            )
            needs_update |= changed

            # TODO: why is the width ignored?
            psim.SameLine()
            psim.SetNextItemWidth(200)
            changed, self.cfg.paths.specular_reflection = psim.Checkbox(
                "Specular reflection##paths", self.cfg.paths.specular_reflection
            )
            needs_update |= changed

            psim.SetNextItemWidth(200)
            changed, self.cfg.paths.diffuse_reflection = psim.Checkbox(
                "Diffuse reflection##paths", self.cfg.paths.diffuse_reflection
            )
            needs_update |= changed

            psim.SameLine()
            psim.SetNextItemWidth(200)
            changed, self.cfg.paths.refraction = psim.Checkbox(
                "Refraction##paths", self.cfg.paths.refraction
            )
            needs_update |= changed

            # TODO: rendering parameters (radius, transparency, etc)
            # TODO: show & configure segment color per type of interaction
            needs_visual_update = False

            if self.cfg.paths.auto_update and needs_update:
                self.paths = self.compute_paths()
            if needs_update or needs_visual_update:
                add_paths_to_polyscope(self.paths, self.ps_groups, self.cfg.paths)

            psim.Spacing()

        if psim.CollapsingHeader("Rendering"):
            psim.Spacing()

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
