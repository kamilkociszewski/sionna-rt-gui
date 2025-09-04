import numpy as np
import polyscope as ps
from sionna import rt


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
        if isinstance(mat, rt.RadioMaterialBase):
            color = mat.color
        else:
            color = (0.65, 0.65, 0.65)

        vertices = mesh.vertex_positions_buffer().numpy().reshape(-1, 3)
        faces = mesh.faces_buffer().numpy().reshape(-1, 3)
        struct = ps.register_surface_mesh(mesh.id(), vertices, faces, color=color)
        struct.add_to_group(ps_groups["scene"])


def add_radio_map_to_polyscope(
    name: str, radio_map: rt.RadioMap, ps_groups: dict[str, ps.Group]
):
    if isinstance(radio_map, rt.PlanarRadioMap):
        # TODO: do something faster if the radio map is already registered
        # Create rectangle mesh to display the planar radio map
        to_world = radio_map.to_world.matrix.numpy().squeeze()
        p0 = to_world @ np.array([-1, -1, 0, 0])
        p1 = to_world @ np.array([-1, 1, 0, 0])
        p2 = to_world @ np.array([1, -1, 0, 0])
        p3 = to_world @ np.array([1, 1, 0, 0])
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
        texture, alpha = rt.radio_map_texture(rm_values, db_scale=True)
        struct.add_color_quantity(name, texture, defined_on="texture", param_name="uv")
        # rgba = np.concatenate([texture, alpha[..., None]], axis=-1)
        # struct.add_color_alpha_image_quantity(
        #     name, rgba, defined_on="texture", param_name="uv"
        # )

    elif isinstance(radio_map, rt.MeshRadioMap):
        raise NotImplementedError("Mesh radio maps are not supported yet")
    else:
        raise ValueError(f"Unsupported radio map type: {type(radio_map)}")
