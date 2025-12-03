#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import drjit as dr
import mitsuba as mi
import polyscope as ps

from .dlpack_utils import pointer_from_dlpack


def get_array_ptr(arr: dr.ArrayBase) -> tuple[int, int, int, int]:
    value = arr.array if dr.is_tensor_v(arr) else arr
    if hasattr(value, "data_"):
        ptr = value.data_()
    else:
        ptr = pointer_from_dlpack(value)

    return (
        # arr_ptr
        ptr,
        # arr_shape
        dr.shape(arr),
        # arr_dtype
        None,
        # arr_nbytes
        get_array_size_bytes(arr),
    )


def get_array_size_bytes(arr: dr.ArrayBase) -> int:
    assert dr.is_tensor_v(arr) or dr.depth_v(arr) == 1
    match dr.type_v(arr):
        case dr.VarType.UInt8:
            ts = 1
        case dr.VarType.Float16:
            ts = 2
        case dr.VarType.Float32:
            ts = 4
        case dr.VarType.Float64:
            ts = 8
        case _:
            raise ValueError(f"Unsupported array type: {dr.type_v(arr)}")
    return ts * dr.width(arr)


def memcpy_2d_to_array_async(dst_ptr, src_ptr, width, height):
    # TODO: not sure why this is needed. Polyscope implementation might have a bug.
    src_pitch = (width // 12) * 4 * 4
    return dr.cuda.memcpy_2d_to_array_async(
        dst_ptr, src_ptr, src_pitch=src_pitch, height=height, from_host=False
    )


def set_polyscope_device_interop_funcs():
    ps_device_func_dict = {
        "map_resource_and_get_array": lambda handle: dr.cuda.map_graphics_resource_array(
            handle
        ),
        "map_resource_and_get_pointer": lambda handle: dr.cuda.map_graphics_resource_ptr(
            handle
        ),
        "unmap_resource": lambda handle: dr.cuda.unmap_graphics_resource(handle),
        "register_gl_buffer": lambda native_id: dr.cuda.register_gl_buffer(native_id),
        "register_gl_image_2d": lambda native_id: dr.cuda.register_gl_texture(
            native_id
        ),
        "unregister_resource": lambda handle: dr.cuda.unregister_cuda_resource(handle),
        "get_array_ptr": get_array_ptr,
        "memcpy_2d": memcpy_2d_to_array_async,
    }

    ps.set_device_interop_funcs(ps_device_func_dict)


def supports_direct_update_from_device() -> bool:
    return "cuda" in mi.variant()
