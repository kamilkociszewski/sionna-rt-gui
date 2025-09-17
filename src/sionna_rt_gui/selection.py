from __future__ import annotations

from enum import StrEnum

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt
from sionna.rt.utils.geometry import rotation_matrix

from .sionna_utils import set_or_update_radio_devices_polyscope, add_paths_to_polyscope


class SelectionType(StrEnum):
    Transmitter = "Transmitter"
    Receiver = "Receiver"
    RadioMap = "Radio map"
    Path = "Path"
    Mesh = "Mesh"


GIZMO_SCALE = 0.3


def vec_str(vec: np.ndarray) -> str:
    vec = vec.squeeze()
    return f"({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})"


def selection_gui(
    gui: "SionnaRtGui",
    selected_object: rt.SceneObject | None,
    selected_type: SelectionType | None,
):

    psim.Begin("Selection##sionna", open=True)

    psim.Text(f"Selected: {selected_type.value} '{selected_object.name}'\n")
    psim.Spacing()

    rd_update_needed = False
    is_transmitter = selected_type == SelectionType.Transmitter
    if selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
        rd = selected_object
        array = gui.scene.tx_array if is_transmitter else gui.scene.rx_array
        pattern = array.antenna_pattern

        changed, rd.color = psim.ColorEdit3("Color", rd.color)
        rd_update_needed |= changed

        # TODO: color picker to set RD color
        # TODO: do not trigger constant GPU -> CPU transfers
        psim.Spacing()
        if psim.TreeNodeEx(
            "Characteristics:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Text(
                f"Position [m]: {vec_str(rd.position.numpy())}\n"
                f"Orientation (angles): {vec_str(rd.orientation.numpy())}\n"
                f"Velocity [m/s]: {vec_str(rd.velocity.numpy())}\n"
                + (f"Transmit power [W]: {rd.power[0]:.2f}\n" if is_transmitter else "")
            )
            psim.TreePop()

        psim.Spacing()
        if psim.TreeNodeEx(
            "Antenna array:##selection", psim.ImGuiTreeNodeFlags_DefaultOpen
        ):
            psim.Text(
                f"Type: {type(array).__name__}\n"
                f"Array size: {dr.width(array.normalized_positions)}\n"
                f"Pattern: {type(pattern).__name__}\n"
            )

            psim.TreePop()

        # Transformation gizmo
        # TODO: make RD's orientation visible while the gizmo is shown.
        if not ps.has_point_cloud("Gizmo"):
            struct = ps.register_point_cloud(
                "Gizmo", rd.position.numpy().T, enabled=False
            )
            struct.set_transform_gizmo_enabled(True)
        else:
            struct = ps.get_point_cloud("Gizmo")

        to_world = struct.get_transform()

        # Gizmo moved since the last frame
        if (gui.prev_gizmo_to_world is not None) and not np.allclose(
            gui.prev_gizmo_to_world, to_world
        ):
            # Remove scaling
            to_world[:3, :3] = (
                GIZMO_SCALE
                * to_world[:3, :3]
                / np.linalg.norm(to_world[:3, :3], axis=0)
            )
            struct.set_transform(to_world)

            # Apply transform to the selected object
            rd.position = mi.Point3f(to_world[:3, -1])
            target = to_world[:3, -1] + (
                to_world[:3, 0] / np.linalg.norm(to_world[:3, 0])
            )
            rd.look_at(target)
            # For debugging:
            # ps.register_point_cloud("Gizmo target", target[None, :], color=(1, 0, 1))
            dr.make_opaque(rd.position, rd.orientation)
            rd_update_needed = True

        else:
            # Reset position & orientation of gizmo to match the selected object
            to_world = np.eye(4)
            to_world[:3, :3] = (
                GIZMO_SCALE * rotation_matrix(rd.orientation).numpy()[..., 0]
            )
            to_world[:3, -1] = rd.position.numpy().squeeze()
            struct.set_transform(to_world)

        gui.prev_gizmo_to_world = to_world

    psim.Spacing()
    if psim.Button(f"Remove {selected_type.value.lower()}"):
        match selected_type:
            case SelectionType.Transmitter:
                del gui.scene._transmitters[selected_object.name]
            case SelectionType.Receiver:
                del gui.scene._receivers[selected_object.name]
            case _:
                print(f"[!] Unexpected selection type: {selected_type}")
                pass
        rd_update_needed = True
        gui.clear_selection()

    if rd_update_needed:
        set_or_update_radio_devices_polyscope(
            gui.scene.transmitters if is_transmitter else gui.scene.receivers,
            is_transmitter=is_transmitter,
            ps_groups=gui.ps_groups,
        )
        if is_transmitter:
            # Note: receivers don't affect radio maps.
            gui.reset_radio_map()

        if gui.cfg.paths.auto_update:
            # TODO: probably should move this to a little method
            gui.clear_paths()
            gui.paths = gui.compute_paths()
            add_paths_to_polyscope(gui.paths, gui.ps_groups, gui.cfg.paths)

    psim.End()
