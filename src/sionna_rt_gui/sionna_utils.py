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


def add_radio_device_to_polyscope(
    position: list[float],
    is_transmitter: bool,
    existing_rd: list[rt.RadioDevice],
    ps_groups: dict[str, ps.Group],
):
    # Create or update point cloud for transmitters or receivers
    position_np = np.array(position)[None, :]
    name = "Transmitters" if is_transmitter else "Receivers"

    # TODO: is there an easier way to get the existing points?
    # existing_points = ps.get_point_cloud(name).get_position()
    existing_points = [rd.position.numpy().T for rd in existing_rd.values()]
    position_np = np.concatenate(existing_points + [position_np], axis=0)
    struct = ps.register_point_cloud(
        name,
        position_np,
        color=(DEFAULT_TRANSMITTER_COLOR if is_transmitter else DEFAULT_RECEIVER_COLOR),
    )
    struct.add_to_group(ps_groups["rd"])


def add_radio_map_to_polyscope(
    name: str,
    radio_map: rt.RadioMap | None,
    ps_groups: dict[str, ps.Group],
    cfg: RadioMapConfig,
):
    if radio_map is None:
        return

    if isinstance(radio_map, rt.PlanarRadioMap):
        # TODO: do something faster if the radio map is already registered
        # Create rectangle mesh to display the planar radio map
        to_world = radio_map.to_world.matrix.numpy().squeeze()
        p0 = to_world @ np.array([-1, -1, 0, 1])
        p1 = to_world @ np.array([-1, 1, 0, 1])
        p2 = to_world @ np.array([1, -1, 0, 1])
        p3 = to_world @ np.array([1, 1, 0, 1])
        vertices = np.array([p0, p1, p2, p3])[:, :3]
        faces = np.array([[0, 1, 2], [2, 1, 3]])

        # Add plane mesh to Polyscope
        struct = ps.register_surface_mesh(name, vertices=vertices, faces=faces)
        struct.add_to_group(ps_groups["radio_maps"])

        # UV map
        param_vals = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        struct.add_parameterization_quantity(
            "uv", param_vals, defined_on="vertices", enabled=False
        )

        rm_values = np.max(radio_map.path_gain.numpy(), axis=0)
        # TODO: use configurable colormap
        texture, alpha = rt.radio_map_texture(
            rm_values,
            db_scale=True,
            vmin=cfg.vmin,
            vmax=cfg.vmax,
            premultiply_alpha=False,
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
        # rgba = np.concatenate([texture, alpha[..., None]], axis=-1)
        # struct.add_color_alpha_image_quantity(
        #     name, rgba, defined_on="texture", param_name="uv"
        # )

    elif isinstance(radio_map, rt.MeshRadioMap):
        raise NotImplementedError("Mesh radio maps are not supported yet")
    else:
        raise ValueError(f"Unsupported radio map type: {type(radio_map)}")


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
