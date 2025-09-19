from __future__ import annotations

from enum import Enum
from typing import Any

import drjit as dr
import mitsuba as mi
import numpy as np
import polyscope as ps
from sionna import rt
from sionna.rt.renderer import visual_scene_from_wireless_scene

from .config import RenderingConfig


class RenderingMode(Enum):
    RASTERIZATION = 0
    RAY_TRACING = 1


RENDERING_MODE_NAMES = ["Rasterization", "Ray tracing"]
assert len(RENDERING_MODE_NAMES) == len(RenderingMode)


def _render_scene(
    cfg: RenderingConfig,
    scene: rt.Scene,
    seed: int,
    camera_changed: bool,
    cache: dict[str, Any],
) -> tuple[mi.TensorXf, mi.TensorXf, dict[str, Any]]:

    sensor = cache.get("sensor")
    visual_scene = cache.get("visual_scene")
    integrator = cache.get("integrator")

    if sensor is None or camera_changed:
        # Camera to world transform
        view_pose = ps.get_camera_view_matrix()

        # TODO: only update camera if it actually moved.
        # TODO: maybe reuse `rt.utils.render.make_render_sensor()`
        ps_to_world = np.linalg.inv(view_pose)
        del view_pose
        view_rotation = np.eye(4)
        view_rotation[:3, :3] = ps_to_world[:3, :3]
        view_translation = np.eye(4)
        view_translation[:3, -1] = ps_to_world[:3, -1]

        # Conversions:
        # - World space stays the same (Z up, Y forward), so the translation is left intact.
        # - However, the local camera space needs to be converted from Polyscope's
        #   left-handed (+y up, -z forward, +x right) to Mitsuba's right-handed
        #   system (y up, -z forward, +x left).
        # TODO: double-check the comment above for correctness.
        to_world = (
            mi.ScalarTransform4f(view_translation)
            @ mi.ScalarTransform4f(
                [
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [0, 0, -1, 0],
                    [0, 0, 0, 1],
                ]
            )
            @ mi.ScalarTransform4f(view_rotation)
        )

    if sensor is None:
        # TODO: adapt if the window is resized
        sensor = mi.load_dict(
            {
                "type": "perspective",
                "fov": ps.get_view_camera_parameters().get_fov_vertical_deg(),
                "fov_axis": "y",
                # TODO: adapt automatically based on scene size
                "near_clip": 0.1,
                "far_clip": 10000,
                "to_world": to_world,
                "film": {
                    "type": "hdrfilm",
                    "pixel_format": "rgba",
                    "width": int(cfg.default_resolution[0] * cfg.relative_resolution),
                    "height": int(cfg.default_resolution[1] * cfg.relative_resolution),
                    "filter": {
                        "type": "box",
                    },
                },
            }
        )
    elif camera_changed:
        params = mi.traverse(sensor)
        params["to_world"] = to_world
        params.update()

    if visual_scene is None:
        # NOTE: this assumes that the scene does *not* change between renders!
        visual_scene_dict = visual_scene_from_wireless_scene(
            scene,
            sensor=sensor,
            max_depth=8,
            # # TODO: ship a nice one by default (low-res should be enough)
            # envmap="/home/merlin/nvidia/mitsuba3-next/tutorials/scenes/textures/envmap.exr",
            #  clip_at: float | None = None,
            #  clip_plane_orientation: tuple[float, float, float] = (0, 0, -1),
            #  envmap: str | None = None,
            #  lighting_scale: float = 1.0,
            #  exclude_mesh_ids: set[str] = None
        )
        visual_scene = mi.load_dict(visual_scene_dict)

        # Hack for better sampling: zero-out the bottom half or so of the envmap,
        # since our scenes always have a ground plane.
        params = mi.traverse(visual_scene)
        k = "emitter.data"
        if k in params:
            # Height, width, channels
            h, *_ = dr.shape(params[k])
            params[k][int(0.6 * h) :, :, :] = 0

    if integrator is None:
        integrator = mi.load_dict(
            {
                "type": "aov",
                "aovs": "dd.y:depth",
                "rgb": {
                    "type": "path",
                    "hide_emitters": True,
                    # "hide_emitters": False,
                    "max_depth": 5,
                },
            }
        )

    # TODO: use kernel freezing for this
    img = mi.render(
        visual_scene,
        sensor=sensor,
        integrator=integrator,
        spp=cfg.spp_per_frame,
        seed=seed,
    )
    img = img[..., :4]
    # TODO: pre-multiply alpha to avoid fringing?
    # img[..., :3] *= img[..., -2]

    # TODO: depth convention mismatch with Polyscope?
    depth = img[..., -1]

    cache["sensor"] = sensor
    cache["visual_scene"] = visual_scene
    cache["integrator"] = integrator
    return img, depth, cache


def render_scene(
    cfg: RenderingConfig,
    scene: rt.Scene,
    seed: int,
    camera_changed: bool,
    cache: dict[str, Any] = None,
) -> tuple[mi.TensorXf, mi.TensorXf, dict[str, Any]]:
    assert cfg.mode == RenderingMode.RAY_TRACING

    if cache is None:
        cache = {}

    with mi.scoped_set_variant("cuda_ad_rgb", "llvm_ad_rgb"):
        return _render_scene(cfg, scene, seed, camera_changed, cache=cache)
