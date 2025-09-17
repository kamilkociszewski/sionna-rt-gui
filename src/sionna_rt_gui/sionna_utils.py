import drjit as dr
import numpy as np
import polyscope as ps
from sionna import rt
from sionna.rt.constants import DEFAULT_TRANSMITTER_COLOR, DEFAULT_RECEIVER_COLOR

from .config import RadioMapConfig, PathsConfig


ITU_TO_PS_MATERIAL = {
    "marble": "ceramic",
    "concrete": "clay",
    "wood": "clay",
    "metal": "candy",
    "brick": "clay",
    "glass": "ceramic",
    "floorboard": "clay",
    "ceiling_board": "clay",
    "chipboard": "clay",
    "plasterboard": "clay",
    "plywood": "clay",
    "very_dry_ground": "clay",
    "medium_dry_ground": "clay",
    "wet_ground": "clay",
}


def get_built_in_scenes() -> dict[str, str]:
    result = {}
    for var_name in dir(rt.scene):
        var = getattr(rt.scene, var_name)
        if isinstance(var, str) and var.endswith(".xml"):
            result[var_name] = var

    return result


def add_scene_to_polyscope(scene: rt.Scene, ps_groups: dict[str, ps.Group]):
    # Add the meshes to Polyscope
    # TODO: apply consistent materials (based on radio material)
    for mesh in scene.mi_scene.shapes():
        mat = mesh.bsdf()
        ps_mat = None
        if isinstance(mat, rt.RadioMaterialBase):
            color = mat.color
            # TODO: use fancier materials
            # if isinstance(mat, rt.ITURadioMaterial):
            #     ps_mat = ITU_TO_PS_MATERIAL.get(mat.itu_type)
        else:
            color = (0.65, 0.65, 0.65)

        vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
        faces = mesh.faces_buffer().numpy().reshape(-1, 3)
        struct = ps.register_surface_mesh(
            mesh.id(), vertices, faces, color=color, material=ps_mat
        )
        struct.add_to_group(ps_groups["scene"])


def set_or_update_radio_devices_polyscope(
    radio_devices: dict[str, rt.RadioDevice],
    is_transmitter: bool,
    ps_groups: dict[str, ps.Group],
):
    name = "Transmitters" if is_transmitter else "Receivers"
    if not radio_devices:
        if ps.has_point_cloud(name):
            ps.get_point_cloud(name).remove()
        return

    position_np = np.concatenate(
        [rd.position.numpy().T for rd in radio_devices.values()], axis=0
    )
    struct = None
    if ps.has_point_cloud(name):
        # Update existing point cloud (only possible if it has the same size)
        candidate = ps.get_point_cloud(name)
        if candidate.n_points() == position_np.shape[0]:
            candidate.update_point_positions(position_np)
            struct = candidate

    if struct is None:
        struct = ps.register_point_cloud(
            name,
            position_np,
            color=(
                DEFAULT_TRANSMITTER_COLOR if is_transmitter else DEFAULT_RECEIVER_COLOR
            ),
        )
        struct.add_to_group(ps_groups["rd"])

    # Also update per-point colors
    rd_colors = np.array([rd.color for rd in radio_devices.values()])
    struct.add_color_quantity(
        name + "_colors",
        rd_colors,
        enabled=True,
    )


def add_radio_map_to_polyscope(
    name: str,
    radio_map: rt.RadioMap | None,
    ps_groups: dict[str, ps.Group],
    cfg: RadioMapConfig,
):
    if radio_map is None:
        return

    # TODO: make all of this faster, we shouldn't be bottlenecked on the rendering / transfers
    if isinstance(radio_map, rt.PlanarRadioMap):
        struct = get_or_add_planar_radio_map_mesh(name, radio_map, ps_groups)

        rm_values = np.max(radio_map.path_gain.numpy(), axis=0)
        texture, alpha = rt.radio_map_texture(
            rm_values,
            db_scale=True,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            premultiply_alpha=True,
            rm_cmap=cfg.color_map,
        )
        # TODO: change texture interpolation to nearest neighbor
        struct.add_color_quantity(
            name,
            texture,
            defined_on="texture",
            param_name="uv",
            enabled=True,
            image_origin="lower_left",
        )

        # Note: texture-space alpha is not supported yet.
        # TODO: figure out the correct flip / transpose combination or change vertex ordering
        struct.add_scalar_quantity(
            f"{name}_alpha",
            alpha.ravel(),
            defined_on="vertices",
            enabled=False,
        )
        struct.set_transparency_quantity(f"{name}_alpha")

    elif isinstance(radio_map, rt.MeshRadioMap):
        raise NotImplementedError("Mesh radio maps are not supported yet")
    else:
        raise ValueError(f"Unsupported radio map type: {type(radio_map)}")


def get_or_add_planar_radio_map_mesh(
    name: str,
    radio_map: rt.PlanarRadioMap,
    ps_groups: dict[str, ps.Group],
) -> ps.SurfaceMesh:

    rm_shape = radio_map.path_gain.shape[1:]

    if ps.has_surface_mesh(name):
        struct = ps.get_surface_mesh(name)

        n_entries = dr.prod(rm_shape)
        if n_entries == struct.n_vertices():
            # TODO(!): need to update the vertices if pose changed, too
            return struct

    # Create rectangle mesh to display the planar radio map.
    # We need one vertex per entry in the radio map because spatially-varying
    # transparency cannot be defined in texture space.
    vertices_x, vertices_y = np.meshgrid(
        np.linspace(-1, 1, rm_shape[1]),
        np.linspace(-1, 1, rm_shape[0]),
        indexing="xy",
    )
    vertices = np.stack(
        [
            vertices_x,
            vertices_y,
            np.zeros_like(vertices_x),
            np.ones_like(vertices_x),
        ],
        axis=-1,
    ).reshape(-1, 4)
    # Transform vertices to world coordinates (accounts from plane pose)
    to_world = radio_map.to_world.matrix.numpy().squeeze()
    vertices = (vertices @ to_world.T)[:, :3]

    # Faces: two triangles per cell
    # Adapted from: https://stackoverflow.com/a/44935368
    r = np.arange(vertices.shape[0]).reshape(rm_shape)
    faces = np.empty((rm_shape[0] - 1, rm_shape[1] - 1, 2, 3), dtype=int)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces = faces.reshape(-1, 3)

    # Add plane mesh to Polyscope
    struct = ps.register_surface_mesh(name, vertices=vertices, faces=faces)
    struct.add_to_group(ps_groups["radio_maps"])

    # UV map
    param_vals = np.stack(
        [
            (vertices_x.flatten() + 1) * 0.5,
            (vertices_y.flatten() + 1) * 0.5,
        ],
        axis=-1,
    )
    struct.add_parameterization_quantity(
        "uv", param_vals, defined_on="vertices", enabled=False
    )

    return struct


def add_paths_to_polyscope(
    paths: rt.Paths | None, ps_groups: dict[str, ps.Group], cfg: PathsConfig
):
    if paths is None:
        return

    result = rt.render.paths_to_segments(paths)
    if not result:
        return

    starts, ends, colors = result
    vertices = np.stack([starts, ends], axis=1).reshape(-1, 3)

    # TODO: appropriate colors for each path type
    # TODO: path transparency based on gain at each segment?
    struct = ps.register_curve_network(
        "paths",
        vertices,
        edges="segments",
        enabled=True,
        # color=(0.7, 0.7, 0.7),
        # TODO: make radius adaptive & configurable
        radius=0.001,
    )
    struct.add_color_quantity(
        "path_colors",
        np.array(colors),
        defined_on="edges",
        # param_name="uv",
        enabled=True,
        # image_origin="lower_left",
    )
    struct.set_transparency(0.6)
    struct.add_to_group(ps_groups["paths"])
