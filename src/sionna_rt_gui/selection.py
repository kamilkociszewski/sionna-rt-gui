from __future__ import annotations

from enum import StrEnum

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from sionna import rt

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
    psim.Begin(f'{selected_type.value} "{selected_object.name}"', open=True)

    psim.Spacing()

    if selected_type in (SelectionType.Transmitter, SelectionType.Receiver):
        rd = selected_object
        is_transmitter = selected_type == SelectionType.Transmitter
        array = gui.scene.tx_array if is_transmitter else gui.scene.rx_array
        pattern = array.antenna_pattern
        rd_update_needed = False

        changed, rd.color = psim.ColorEdit3("Color", rd.color)
        rd_update_needed |= changed

        # TODO: color picker to set RD color
        # TODO: do not trigger constant GPU -> CPU transfers
        psim.Spacing()
        psim.Text(
            "Characteristics:\n"
            f"- Position [m]: {vec_str(rd.position.numpy())}\n"
            f"- Orientation (angles): {vec_str(rd.orientation.numpy())}\n"
            f"- Velocity [m/s]: {vec_str(rd.velocity.numpy())}\n"
            + (f"- Transmit power [W]: {rd.power[0]:.2f}\n" if is_transmitter else "")
        )

        psim.Spacing()
        psim.Text(
            "Antenna array:\n"
            f"- Type: {type(array).__name__}\n"
            f"- Array size: {dr.width(array.normalized_positions)}\n"
            f"- Pattern: {type(pattern).__name__}\n"
        )

        # Transformation gizmo
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
            # TODO: apply rotation too

            dr.make_opaque(rd.position, rd.orientation)

            rd_update_needed = True
            if is_transmitter:
                # Note: receivers don't affect radio maps.
                gui.reset_radio_map()

            if gui.cfg.paths.auto_update:
                # TODO: probably should move this to a little method
                gui.clear_paths()
                gui.paths = gui.compute_paths()
                add_paths_to_polyscope(gui.paths, gui.ps_groups, gui.cfg.paths)

        else:
            # Reset position of gizmo to match the selected object
            to_world = GIZMO_SCALE * np.eye(4)
            to_world[:3, -1] = rd.position.numpy().squeeze()
            to_world[-1, -1] = 1
            struct.set_transform(to_world)

        gui.prev_gizmo_to_world = to_world
        if rd_update_needed:
            set_or_update_radio_devices_polyscope(
                gui.scene.transmitters,
                is_transmitter=is_transmitter,
                ps_groups=gui.ps_groups,
            )

    psim.Spacing()
    if psim.Button(f"Remove {selected_type.value.lower()}"):
        # TODO: remove the object from the scene
        pass

    psim.End()
