import logging
from typing import List, Optional, Union

from pygfx.renderers.wgpu import (
    Binding,
    BaseShader,
    register_wgpu_render_function,
)

import wgpu
import pygfx as gfx


class CheckerboardPlane(gfx.WorldObject):
    """
    The actual geometry is implicit: the shader draws a full-screen quad and
    computes the intersection with the world XY-plane procedurally, using
    a fragment shader to generate the checkerboard pattern in world space.
    """
    pass


class CheckerboardPlaneMaterial(gfx.Material):
    """
    Material for the checkerboard ground.

    Stores parameters (colours, cell size, fade distances) in a uniform buffer  that the shader reads
    in the fragment stage.
    """

    uniform_type = dict(
        gfx.Material.uniform_type,
        grid_color1="4xf4",  # RGBA for light squares
        grid_color2="4xf4",  # RGBA for dark squares
        grid_size="f4",  # world-space size of each checker cell
        fade_start="f4",  # start distance for fading
        fade_end="f4",  # end distance for fading
    )

    def __init__(self, color1: Union[str, tuple] = (0.92, 0.92, 0.92, 1.0),
                 color2: Union[str, tuple] = (0.75, 0.75, 0.75, 1.0),
                 grid_size: float = 10.0,
                 fade_start: float = 2000.0, fade_end: float = 8000.0,
            **kwargs,
    ) -> None:
        super().__init__()
        self.grid_color1 = color1
        self.grid_color2 = color2
        self.grid_size = float(grid_size)
        self.fade_start = float(fade_start)
        self.fade_end = float(fade_end)

    @property
    def grid_color1(self):
        return gfx.Color(self.uniform_buffer.data["grid_color1"])

    @grid_color1.setter
    def grid_color1(self, color):
        self.uniform_buffer.data["grid_color1"] = gfx.Color(color)
        self.uniform_buffer.update_range(0, 99999)

    @property
    def grid_color2(self):
        return gfx.Color(self.uniform_buffer.data["grid_color2"])

    @grid_color2.setter
    def grid_color2(self, color):
        self.uniform_buffer.data["grid_color2"] = gfx.Color(color)
        self.uniform_buffer.update_range(0, 99999)

    @property
    def grid_size(self) -> float:
        return float(self.uniform_buffer.data["grid_size"])

    @grid_size.setter
    def grid_size(self, value: float) -> None:
        self.uniform_buffer.data["grid_size"] = float(value)
        self.uniform_buffer.update_range(0, 99999)

    @property
    def fade_start(self) -> float:
        return float(self.uniform_buffer.data["fade_start"])

    @fade_start.setter
    def fade_start(self, value: float) -> None:
        self.uniform_buffer.data["fade_start"] = float(value)
        self.uniform_buffer.update_range(0, 99999)

    @property
    def fade_end(self) -> float:
        return float(self.uniform_buffer.data["fade_end"])

    @fade_end.setter
    def fade_end(self, value: float) -> None:
        self.uniform_buffer.data["fade_end"] = float(value)
        self.uniform_buffer.update_range(0, 99999)


@register_wgpu_render_function(CheckerboardPlane, CheckerboardPlaneMaterial)
class CheckerboardPlaneShader(BaseShader):
    """
    WGSL shader that draws an infinite checkerboard on the world XY plane.

    Draws a full-screen triangle and in the fragment shader compute the world-space position of the pixel's ray
    intersection with z=0, then apply a checkerboard pattern in world XY.
    """

    type = "render"

    def get_bindings(self, wobject, shared):  # type: ignore[override]
        # Standard info + world-object + material uniforms
        bindings = {
            0: Binding("u_stdinfo", "buffer/uniform", shared.uniform_buffer),
            1: Binding("u_wobject", "buffer/uniform", wobject.uniform_buffer),
            2: Binding("u_material", "buffer/uniform", wobject.material.uniform_buffer),
        }
        self.define_bindings(0, bindings)
        return {0: bindings}

    def get_pipeline_info(self, wobject, shared):  # type: ignore[override]
        # We draw a single full-screen triangle; no culling
        return {
            "primitive_topology": wgpu.PrimitiveTopology.triangle_list,  # type: ignore[name-defined]
            "cull_mode": wgpu.CullMode.none,  # type: ignore[name-defined]
        }

    def get_render_info(self, wobject, shared):  # type: ignore[override]
        # 3 vertices, 1 instance, driven by vertex_index
        return {"indices": (3, 1)}

    def get_code(self):  # type: ignore[override]
        # Use the std library from pygfx and define our own vertex/fragment
        return """
        {$ include 'pygfx.std.wgsl' $}

        // 4x4 matrix inverse function (needed for camera/projection transforms)
        fn inverse_mat4(m: mat4x4<f32>) -> mat4x4<f32> {
            let a00 = m[0][0]; let a01 = m[0][1]; let a02 = m[0][2]; let a03 = m[0][3];
            let a10 = m[1][0]; let a11 = m[1][1]; let a12 = m[1][2]; let a13 = m[1][3];
            let a20 = m[2][0]; let a21 = m[2][1]; let a22 = m[2][2]; let a23 = m[2][3];
            let a30 = m[3][0]; let a31 = m[3][1]; let a32 = m[3][2]; let a33 = m[3][3];

            let b00 = a00 * a11 - a01 * a10;
            let b01 = a00 * a12 - a02 * a10;
            let b02 = a00 * a13 - a03 * a10;
            let b03 = a01 * a12 - a02 * a11;
            let b04 = a01 * a13 - a03 * a11;
            let b05 = a02 * a13 - a03 * a12;
            let b06 = a20 * a31 - a21 * a30;
            let b07 = a20 * a32 - a22 * a30;
            let b08 = a20 * a33 - a23 * a30;
            let b09 = a21 * a32 - a22 * a31;
            let b10 = a21 * a33 - a23 * a31;
            let b11 = a22 * a33 - a23 * a32;

            let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
            let inv_det = 1.0 / det;

            var result: mat4x4<f32>;
            result[0][0] = (a11 * b11 - a12 * b10 + a13 * b09) * inv_det;
            result[0][1] = (a02 * b10 - a01 * b11 - a03 * b09) * inv_det;
            result[0][2] = (a31 * b05 - a32 * b04 + a33 * b03) * inv_det;
            result[0][3] = (a22 * b04 - a21 * b05 - a23 * b03) * inv_det;
            result[1][0] = (a12 * b08 - a10 * b11 - a13 * b07) * inv_det;
            result[1][1] = (a00 * b11 - a02 * b08 + a03 * b07) * inv_det;
            result[1][2] = (a32 * b02 - a30 * b05 - a33 * b01) * inv_det;
            result[1][3] = (a20 * b05 - a22 * b02 + a23 * b01) * inv_det;
            result[2][0] = (a10 * b10 - a11 * b08 + a13 * b06) * inv_det;
            result[2][1] = (a01 * b08 - a00 * b10 - a03 * b06) * inv_det;
            result[2][2] = (a30 * b04 - a31 * b02 + a33 * b00) * inv_det;
            result[2][3] = (a21 * b02 - a20 * b04 - a23 * b00) * inv_det;
            result[3][0] = (a11 * b07 - a10 * b09 - a12 * b06) * inv_det;
            result[3][1] = (a00 * b09 - a01 * b07 + a02 * b06) * inv_det;
            result[3][2] = (a31 * b01 - a30 * b03 - a32 * b00) * inv_det;
            result[3][3] = (a20 * b03 - a21 * b01 + a22 * b00) * inv_det;

            return result;
        }

        @vertex
        fn vs_main(@builtin(vertex_index) index: u32) -> Varyings {
            // Full-screen triangle in NDC
            var positions = array<vec2<f32>, 3>(
                vec2<f32>(-1.0, -1.0),
                vec2<f32>( 3.0, -1.0),
                vec2<f32>(-1.0,  3.0)
            );

            var varyings: Varyings;
            let ndc_xy = positions[index];
            varyings.position = vec4<f32>(ndc_xy, 0.0, 1.0);

            // Store NDC coordinates in world_pos for fragment shader
            varyings.world_pos = vec3<f32>(ndc_xy, 0.0);
            return varyings;
        }

        @fragment
        fn fs_main(varyings: Varyings) -> FragmentOutput {
            var out: FragmentOutput;

            // Reconstruct view direction from NDC position
            let ndc_xy = varyings.world_pos.xy;
            let ndc = vec4<f32>(ndc_xy, 1.0, 1.0);
            let inv_proj = inverse_mat4(u_stdinfo.projection_transform);
            let view_pos = inv_proj * ndc;
            let view_dir_local = normalize(view_pos.xyz / view_pos.w);

            // Transform to world space
            let inv_cam = inverse_mat4(u_stdinfo.cam_transform);
            let world_dir4 = inv_cam * vec4<f32>(view_dir_local, 0.0);
            let dir = normalize(world_dir4.xyz);

            // Ray origin: camera position in world space
            let cam_pos = (inv_cam * vec4<f32>(0.0, 0.0, 0.0, 1.0)).xyz;

            // Intersect ray with plane z = 0 in world space
            let t = -cam_pos.z / dir.z;

            // Handle edge cases with transparency instead of discarding
            var alpha_multiplier = 1.0;
            if (t <= 0.0 || abs(dir.z) < 1e-6) {
                // Ray doesn't intersect or is parallel - make fully transparent
                alpha_multiplier = 0.0;
            }

            let world_pos = cam_pos + t * dir;

            // Compute proper depth for the intersection point
            let clip_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * vec4<f32>(world_pos, 1.0);
            let ndc_depth = clip_pos.z / clip_pos.w;
            out.depth = ndc_depth;

            let gsize = u_material.grid_size;
            let c1 = u_material.grid_color1;
            let c2 = u_material.grid_color2;
            let fade_start = u_material.fade_start;
            let fade_end = u_material.fade_end;

            let coord = world_pos.xy / gsize;
            let cx = floor(coord.x);
            let cy = floor(coord.y);
            let checker = i32(cx + cy) & 1;

            let light_col = c1.rgb;
            let dark_col = c2.rgb;
            let base_col = select(light_col, dark_col, checker == 1);

            // Distance-based fade
            let dist = length(world_pos.xy);
            let tfade_dist = clamp((dist - fade_start) / (fade_end - fade_start), 0.0, 1.0);

            // Camera height-based fade: fade out when camera is below the plane
            let camera_height = cam_pos.z;
            let height_fade_range = 50.0;  // Fade out over 50 units below the plane
            var tfade_height = 0.0;
            if (camera_height < 0.0) {
                // Camera is below plane - fade based on how far below
                tfade_height = clamp(-camera_height / height_fade_range, 0.0, 1.0);
            }

            // Combine both fades (use maximum fade value)
            let tfade_combined = max(tfade_dist, tfade_height);

            // Fade to transparent by reducing alpha (1.0 = opaque, 0.0 = transparent)
            let alpha = (1.0 - tfade_combined) * alpha_multiplier;

            out.color = vec4<f32>(base_col, alpha);
            return out;
        }
        """