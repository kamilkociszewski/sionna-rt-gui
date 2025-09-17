import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

from .animation import AnimationConfig, animation_gui, animation_tick
from .antenna_array import AntennaArrayConfig, antenna_array_gui
from .config import GuiConfig
from .sionna_utils import (
    add_radio_map_to_polyscope,
    set_or_update_radio_devices_polyscope,
    add_scene_to_polyscope,
    get_built_in_scenes,
    add_paths_to_polyscope,
)
from .selection import SelectionType, selection_gui


class SionnaRtGui:
    def __init__(self, cfg: GuiConfig):
        self.cfg = cfg

        # --- Sionna RT
        # Scene
        built_in_scenes = get_built_in_scenes()
        self.built_in_scene_names = ["None"] + list(built_in_scenes.keys())
        self.built_in_scene_paths = [None] + list(built_in_scenes.values())
        self.current_scene_idx: int = 0
        self.scene: rt.Scene | None = None
        self.tx_array_config: AntennaArrayConfig = AntennaArrayConfig()
        self.rx_array_config: AntennaArrayConfig = AntennaArrayConfig()

        # Radio map results
        self.radio_map: rt.RadioMap | None = None
        self.rm_accumulated_samples: int = 0
        self.rm_color_map_options = [v for v in plt.colormaps() if not v.endswith("_r")]
        self.rm_color_map_index = self.rm_color_map_options.index(
            self.cfg.radio_map.color_map
        )

        # Paths results
        self.paths: rt.Paths | None = None

        # --- Animation state
        self.animation_config: AnimationConfig = AnimationConfig()

        # --- Inputs state
        self.last_mouse_pos: mi.ScalarVector2f | None = None
        self.reset_accumulation_requested = False

        self.snapshot_load_requested: bool = False
        self.code_reload_requested: bool = False

        # --- Selections
        self.selected_object: rt.SceneObject | None = None
        self.selected_type: SelectionType | None = None

        # --- Polyscope setup
        # Can be used to derive e.g. random seeds.
        self.ps_groups: dict[str, ps.Group] = {}
        self.frame_i: int = 0
        self.build_default_ps_gui: bool = False
        self.was_mouse_dragging: bool = False
        self.prev_gizmo_to_world: np.ndarray | None = None
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

        # --- Test data (for convenience)
        # TODO: remove this
        if True:
            # Add some example transmitters
            for pos in [
                # [50, -10, 29 + 2.5],
                # [13, 13, 51 + 2.5],
                [-34, -10, 22 + 2.5],
            ]:
                self.add_radio_device(pos, is_transmitter=True, allow_auto_update=False)

                # shifted = [pos[0] + 5, pos[1] + 15, pos[2] - 10]
                # self.add_radio_device(
                #     shifted, is_transmitter=False, allow_auto_update=False
                # )

            if False:
                p = self.scene.get("tx-0").position.numpy().squeeze()
                traj = self.animation_config.trajectories["tx-0"]
                traj.add_point(p - [30, 10, 0])
                traj.add_point(p + [30, 10, 0])
                traj.enabled = True
                self.animation_config.playing = True
            if True:
                self.selected_object = self.scene.get("tx-0")
                self.selected_type = SelectionType.Transmitter

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

        self.scene.tx_array = self.tx_array_config.create()
        self.scene.rx_array = self.rx_array_config.create()

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
                if rm_new is not None:
                    self.radio_map._pathgain_map += rm_new.path_gain
                    self.rm_accumulated_samples += (
                        self.cfg.radio_map.samples_per_it
                        // len(self.scene._transmitters)
                    )
                    add_radio_map_to_polyscope(
                        "radio_map", self.radio_map, self.ps_groups, self.cfg.radio_map
                    )

        animation_tick(self, psim.GetIO().DeltaTime)

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
            self.scene._transmitters if is_transmitter else self.scene._receivers
        )

        # Add actual radio device to Sionna scene
        prefix = "tx" if is_transmitter else "rx"
        free_index = len(existing_rd)
        while f"{prefix}-{free_index}" in existing_rd:
            free_index += 1

        new_rd = (rt.Transmitter if is_transmitter else rt.Receiver)(
            name=f"{prefix}-{free_index}",
            position=position,
            orientation=[0, 0, 0],
        )
        self.scene.add(new_rd)

        set_or_update_radio_devices_polyscope(
            existing_rd,
            is_transmitter,
            self.ps_groups,
        )

        if allow_auto_update and self.cfg.radio_map.auto_update and is_transmitter:
            self.set_radio_map(self.compute_radio_map(), show=True)
        if allow_auto_update and self.cfg.paths.auto_update:
            self.paths = self.compute_paths()
            add_paths_to_polyscope(self.paths, self.ps_groups, self.cfg.paths)

    def remove_object(self, object: rt.SceneObject, selected_type: SelectionType):
        match selected_type:
            case SelectionType.Transmitter:
                del self.scene._transmitters[object.name]
            case SelectionType.Receiver:
                del self.scene._receivers[object.name]
            case _:
                print(f"[!] Unexpected selection type: {selected_type}")
                pass

        if object.name in self.animation_config.trajectories:
            del self.animation_config.trajectories[object.name]

    def clear_radio_devices(self):
        for name in self.scene._transmitters.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]
        for name in self.scene._receivers.keys():
            if name in self.animation_config.trajectories:
                del self.animation_config.trajectories[name]

        self.scene._transmitters.clear()
        self.scene._receivers.clear()
        for name in self.ps_groups["rd"].get_child_structure_names():
            ps.get_point_cloud(name).remove()
        if self.selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
            self.clear_selection()

        if self.cfg.radio_map.auto_update:
            self.clear_radio_map()
        if self.cfg.paths.auto_update:
            self.clear_paths()

    def reset_radio_map(self):
        """
        Attempts to reset the radio map to zero, as well as the accumulation counter.
        However, if the number of transmitters has changed, the radio map will be
        completely removed (and re-computed if auto-updates are enabled).
        """
        self.rm_accumulated_samples = 0
        if self.radio_map is None:
            return

        if len(self.scene._transmitters) != self.radio_map.num_tx:
            self.clear_radio_map()
            if self.cfg.radio_map.auto_update:
                self.set_radio_map(self.compute_radio_map(), show=True)
            return

        self.radio_map._pathgain_map *= 0.0

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
        allow_click = not imgui_io.WantCaptureMouse
        has_left_click = allow_click and psim.IsMouseClicked(psim.ImGuiMouseButton_Left)
        has_mouse_drag = psim.IsMouseDragging(
            psim.ImGuiMouseButton_Left
        ) or psim.IsMouseDragging(psim.ImGuiMouseButton_Right)
        has_left_release = allow_click and psim.IsMouseReleased(
            psim.ImGuiMouseButton_Left
        )
        has_right_click = allow_click and psim.IsMouseClicked(
            psim.ImGuiMouseButton_Right
        )

        # Ctrl + left/right click: add transmitter/receiver
        if imgui_io.KeyCtrl and (has_left_click or has_right_click):
            is_transmitter = imgui_io.MouseClicked[0]
            # TODO: configurable placement offset along the normal
            rd_position = ps.screen_coords_to_world_position(imgui_io.MousePos)
            rd_position += (0, 0, 2.5)
            self.add_radio_device(rd_position, is_transmitter)

        # Plain left click: object selection
        if has_left_release and not (
            imgui_io.KeyCtrl
            or imgui_io.KeyShift
            or imgui_io.KeyAlt
            or has_mouse_drag
            or self.was_mouse_dragging
        ):
            self.process_pick_result(ps.pick(screen_coords=imgui_io.MousePos))

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

        self.was_mouse_dragging = has_mouse_drag

    def process_pick_result(self, pick_result: ps.PickResult) -> bool:
        # TODO: how to ignore clicks that hit the GUI?
        if not pick_result.is_hit:
            self.clear_selection()
            return False

        picked_index = pick_result.structure_data["index"]
        if pick_result.structure_name in "Transmitters":
            self.selected_object = list(self.scene._transmitters.values())[picked_index]
            self.selected_type = SelectionType.Transmitter
            return True
        elif pick_result.structure_name in "Receivers":
            self.selected_object = list(self.scene._receivers.values())[picked_index]
            self.selected_type = SelectionType.Receiver
            return True

        self.clear_selection()
        return False

    def clear_selection(self):
        self.selected_object = None
        self.selected_type = None
        if ps.has_point_cloud("Gizmo"):
            ps.get_point_cloud("Gizmo").remove()
        if ps.has_curve_network("Trajectory"):
            ps.get_curve_network("Trajectory").remove()

    def gui(self):
        # TODO: set ImGui window title
        # TODO: change GUI accent color to a non-default color.

        psim.SetWindowSize((430, 800), psim.ImGuiCond_FirstUseEver)
        psim.SetWindowPos((10, 10), psim.ImGuiCond_FirstUseEver)

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
            # TODO: button to place radio devices: at random; or samples from a radio map
            antenna_array_gui(self)

            clicked = psim.Button("Clear all radio devices")
            if clicked:
                self.clear_radio_devices()

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

            # TODO: maybe this should be done automatically at the next tick
            if self.cfg.radio_map.auto_update and needs_update:
                self.set_radio_map(self.compute_radio_map(), show=False)
            if needs_update or needs_visual_update:
                add_radio_map_to_polyscope(
                    "radio_map", self.radio_map, self.ps_groups, self.cfg.radio_map
                )

            # TODO: plane / mesh picker (changes type of radio map)
            # TODO: some way to move the radio map (at least the plane z offset)

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

        if psim.CollapsingHeader("Animation"):
            animation_gui(self)
            psim.Spacing()

        if psim.CollapsingHeader("Rendering"):
            psim.Spacing()
            # TODO: option to show/hide radio device orientations

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

        if self.selected_object is not None:
            selection_gui(self, self.selected_object, self.selected_type)
