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


def set_custom_imgui_style():
    import polyscope.imgui as psim

    def style_cb():
        style = psim.GetStyle()
        style.WindowRounding = 1
        style.FrameRounding = 1
        style.FramePadding = (style.FramePadding[0], 4)
        style.ScrollbarRounding = 1
        style.ScrollbarSize = 20
        style.ScaleAllSizes(ps.get_ui_scale())

        # # TODO: make a nice custom theme.
        # colors = style.GetColors()
        # colors[psim.ImGuiCol_Text] = (0.90, 0.90, 0.90, 1.00)
        # colors[psim.ImGuiCol_TextDisabled] = (0.60, 0.60, 0.60, 1.00)
        # colors[psim.ImGuiCol_WindowBg] = (0.00, 0.00, 0.00, 0.70)
        # colors[psim.ImGuiCol_ChildBg] = (0.00, 0.00, 0.00, 0.00)
        # colors[psim.ImGuiCol_PopupBg] = (0.11, 0.11, 0.14, 0.92)
        # colors[psim.ImGuiCol_Border] = (0.50, 0.50, 0.50, 0.50)
        # colors[psim.ImGuiCol_BorderShadow] = (0.00, 0.00, 0.00, 0.00)
        # colors[psim.ImGuiCol_FrameBg] = (0.63, 0.63, 0.63, 0.39)
        # colors[psim.ImGuiCol_FrameBgHovered] = (0.47, 0.69, 0.59, 0.40)
        # colors[psim.ImGuiCol_FrameBgActive] = (0.41, 0.64, 0.53, 0.69)
        # colors[psim.ImGuiCol_TitleBg] = (0.27, 0.54, 0.42, 0.83)
        # colors[psim.ImGuiCol_TitleBgActive] = (0.32, 0.63, 0.49, 0.87)
        # colors[psim.ImGuiCol_TitleBgCollapsed] = (0.27, 0.54, 0.42, 0.83)
        # colors[psim.ImGuiCol_MenuBarBg] = (0.40, 0.55, 0.48, 0.80)
        # colors[psim.ImGuiCol_ScrollbarBg] = (0.63, 0.63, 0.63, 0.39)
        # colors[psim.ImGuiCol_ScrollbarGrab] = (0.00, 0.00, 0.00, 0.30)
        # colors[psim.ImGuiCol_ScrollbarGrabHovered] = (0.40, 0.80, 0.62, 0.40)
        # colors[psim.ImGuiCol_ScrollbarGrabActive] = (0.39, 0.80, 0.61, 0.60)
        # colors[psim.ImGuiCol_CheckMark] = (0.90, 0.90, 0.90, 0.50)
        # colors[psim.ImGuiCol_SliderGrab] = (1.00, 1.00, 1.00, 0.30)
        # colors[psim.ImGuiCol_SliderGrabActive] = (0.39, 0.80, 0.61, 0.60)
        # colors[psim.ImGuiCol_Button] = (0.35, 0.61, 0.49, 0.62)
        # colors[psim.ImGuiCol_ButtonHovered] = (0.40, 0.71, 0.57, 0.79)
        # colors[psim.ImGuiCol_ButtonActive] = (0.46, 0.80, 0.64, 1.00)
        # colors[psim.ImGuiCol_Header] = (0.40, 0.90, 0.67, 0.45)
        # colors[psim.ImGuiCol_HeaderHovered] = (0.45, 0.90, 0.69, 0.80)
        # colors[psim.ImGuiCol_HeaderActive] = (0.53, 0.87, 0.71, 0.80)
        # colors[psim.ImGuiCol_Separator] = (0.50, 0.50, 0.50, 1.00)
        # colors[psim.ImGuiCol_SeparatorHovered] = (0.60, 0.70, 0.66, 1.00)
        # colors[psim.ImGuiCol_SeparatorActive] = (0.70, 0.90, 0.81, 1.00)
        # colors[psim.ImGuiCol_ResizeGrip] = (1.00, 1.00, 1.00, 0.16)
        # colors[psim.ImGuiCol_ResizeGripHovered] = (0.78, 1.00, 0.90, 0.60)
        # colors[psim.ImGuiCol_ResizeGripActive] = (0.78, 1.00, 0.90, 0.90)
        # colors[psim.ImGuiCol_PlotLines] = (1.00, 1.00, 1.00, 1.00)
        # colors[psim.ImGuiCol_PlotLinesHovered] = (0.90, 0.70, 0.00, 1.00)
        # colors[psim.ImGuiCol_PlotHistogram] = (0.90, 0.70, 0.00, 1.00)
        # colors[psim.ImGuiCol_PlotHistogramHovered] = (1.00, 0.60, 0.00, 1.00)
        # colors[psim.ImGuiCol_TextSelectedBg] = (0.00, 0.00, 1.00, 0.35)
        # colors[psim.ImGuiCol_ModalWindowDimBg] = (0.20, 0.20, 0.20, 0.35)
        # colors[psim.ImGuiCol_DragDropTarget] = (1.00, 1.00, 0.00, 0.90)
        # colors[psim.ImGuiCol_Tab] = (0.27, 0.54, 0.42, 0.83)
        # colors[psim.ImGuiCol_TabHovered] = (0.34, 0.68, 0.53, 0.83)
        # colors[psim.ImGuiCol_TabSelected] = (0.38, 0.76, 0.58, 0.83)
        # style.SetColors(colors)

    ps.set_configure_imgui_style_callback(style_cb)
