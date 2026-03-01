"""
Overhang mesh masking for ray-casting intersection filtering.

This module provides GPU-accelerated tools for filtering ray intersections against
overhang mesh regions. It uses depth buffer rendering to create a height map of the
overhang surface, then filters intersection points by checking if they are above or
below the surface.

"""

from typing import Optional, Union
import numpy as np
import trimesh
import wgpu


class OverhangMaskRenderer:
    """
    A WGPU accelerated renderer for overhang mesh masking using shader-based filtering.

    The technique uses a two-pass rendering approach, by exploiting existing depth buffers to
    identifying the overall height map for a projected region emulating a ray-tracing approach.
    This is achieved by having a two -stage pipeline:

    1. Render overhang mesh to create a depth/height texture
    2. Render primary mesh with shader that discards fragments above the overhang surface

    """

    def __init__(self, mesh: trimesh.Trimesh,
                 overhangMesh: trimesh.Trimesh,
                 resolution: float = 0.05,
                 flipDir: bool = False,
                 bbox: np.ndarray = None) -> None:
        """
        Initialise the overhang mask renderer.

        :param mesh: Primary mesh to render and get intersections from
        :param overhangMesh: esh defining the overhang mask region (e.g., support structure)
        :param resolution: Ray Projection resolution
        :param flipDir: Viewing direction. `False` = from above, `True` = from plane below
        :param bbox: Bounding box of region. If `None`, uses combined mesh bounds.
        """
        self.flipDir = flipDir
        self._resolution = resolution
        self._mesh = mesh
        self._overhangMesh = overhangMesh

        # Use combined bounds if no bbox provided
        if bbox is not None:
            self._bbox = bbox
        else:
            # Combine bounds of both meshes
            combinedBounds = np.array([
                np.minimum(mesh.bounds[0], overhangMesh.bounds[0]),
                np.maximum(mesh.bounds[1], overhangMesh.bounds[1])
            ])
            self._bbox = combinedBounds

        self.heightMap = None  # Final filtered height map

        # Calculate viewport size
        meshExtents = np.diff(self._bbox, axis=0)
        self._visSize = (int((meshExtents / self.resolution).flatten()[0]),
                         int((meshExtents / self.resolution).flatten()[1]))

        # Initialize GPU resources
        self._init_wgpu()

        # Two-pass rendering with shader-based masking
        self.renderWithOverhangMask()

    def _init_wgpu(self):
        """Initialize WGPU device and rendering pipelines"""
        self.adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.device = self.adapter.request_device_sync()

        # Shader for first pass: render overhang mesh to texture
        overhangShader = """
        struct Uniforms {
            mvp_matrix: mat4x4<f32>,
        };
        
        @group(0) @binding(0)
        var<uniform> uniforms: Uniforms;
        
        struct VertexInput {
            @location(0) position: vec3<f32>,
        };
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) world_z: f32,
        };
        
        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            let pos = uniforms.mvp_matrix * vec4f(input.position, 1.0);
            output.position = pos;
            output.world_z = input.position.z + 1000.0;
            return output;
        }
        
        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            let z = input.world_z;
            return vec4<f32>(z, z, z, 1.0);
        }
        """

        # Shader for second pass: render main mesh with overhang mask
        maskedShader = """
        struct Uniforms {
            mvp_matrix: mat4x4<f32>,
            bbox_min: vec2<f32>,
            resolution: f32,
            _padding: f32,
        };
        
        @group(0) @binding(0)
        var<uniform> uniforms: Uniforms;
        
        @group(0) @binding(1)
        var overhang_texture: texture_2d<f32>;
        
        struct VertexInput {
            @location(0) position: vec3<f32>,
        };
        
        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) world_pos: vec3<f32>,
        };
        
        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            let pos = uniforms.mvp_matrix * vec4f(input.position, 1.0);
            output.position = pos;
            output.world_pos = input.position + vec3<f32>(0.0, 0.0, 1000.0);
            return output;
        }
        
        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            // Convert world position to texture coordinates
            let px_x = (input.world_pos.x - uniforms.bbox_min.x) / uniforms.resolution;
            let px_y = (input.world_pos.y - uniforms.bbox_min.y) / uniforms.resolution;
            
            // Get texture dimensions
            let tex_dims = textureDimensions(overhang_texture);
            
            // Clamp to valid range
            let px_x_clamped = clamp(i32(px_x), 0, i32(tex_dims.x) - 1);
            let px_y_clamped = clamp(i32(px_y), 0, i32(tex_dims.y) - 1);
            
            // Sample overhang height at this pixel
            let overhang_z = textureLoad(overhang_texture, vec2<i32>(px_x_clamped, px_y_clamped), 0).r;

            if (overhang_z <  1e-5) {
                discard;
            }
            
            /*
            // Not very accurate
            if( abs(input.world_pos.z - overhang_z) < 1e-2) {
                discard; // the original part-surface is actually the overhang surface
            }
            */
            
            // Offset the overhang by 0.1 so that it is below the actual surface therefore avoiding Z-fighting.
            if (input.world_pos.z > overhang_z-1e-1) {
                discard;
            }
            
            // Output the fragment's Z value
            let z = input.world_pos.z;
            return vec4<f32>(z, z, z, 1.0);
        }
        """

        self.overhangShader = self.device.create_shader_module(code=overhangShader)
        self.maskedShader = self.device.create_shader_module(code=maskedShader)

        # Create uniform buffer (extended for masked pass)
        self.uniformBuffer = self.device.create_buffer(
            size=80,  # 64 (MVP) + 16 (bbox_min, resolution, padding)
            usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
        )

        # Bind group layouts
        # Layout for overhang pass (just MVP)
        self.overhangBindLayout = self.device.create_bind_group_layout(
            entries=[{
                "binding": 0,
                "visibility": wgpu.ShaderStage.VERTEX,
                "buffer": {"type": wgpu.BufferBindingType.uniform},
            }]
        )

        # Layout for masked pass (MVP + texture)
        self.maskedBindLayout = self.device.create_bind_group_layout(
            entries=[
                {
                    "binding": 0,
                    "visibility": wgpu.ShaderStage.VERTEX | wgpu.ShaderStage.FRAGMENT,
                    "buffer": {"type": wgpu.BufferBindingType.uniform},
                },
                {
                    "binding": 1,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.unfilterable_float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    },
                }
            ]
        )

        # Create render targets
        self._createRenderTargets()

        # Create pipelines
        self._createPipelines()

    def _createRenderTargets(self):
        """Create GPU textures for rendering"""


        # Overhang texture (stores overhang mesh height map)
        self.overhangTexture = self.device.create_texture(
            size=(self._visSize[0], self._visSize[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_SRC,
        )

        self.overhangTextureVew = self.overhangTexture.create_view()

        # Final render texture (masked output)
        self.renderTexture = self.device.create_texture(
            size=(self._visSize[0], self._visSize[1], 1),
            format=wgpu.TextureFormat.rgba32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )

        self.renderTexture_view = self.renderTexture.create_view()

        # Depth textures
        self.depthTexture = self.device.create_texture(
            size=(self._visSize[0], self._visSize[1], 1),
            format=wgpu.TextureFormat.depth32float,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT,
        )
        self.depthTexture_view = self.depthTexture.create_view()

    def _createPipelines(self):
        """Create render pipelines for both passes"""


        # Pipeline for overhang pass
        overhangPipelineLayout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.overhangBindLayout]
        )

        self.overhangPipeline = self.device.create_render_pipeline(
            layout=overhangPipelineLayout,
            vertex={
                "module": self.overhangShader,
                "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 12,
                    "attributes": [{
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 0,
                    }],
                }],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
                "front_face": "ccw",
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
            },
            fragment={
                "module": self.overhangShader,
                "entry_point": "fs_main",
                "targets": [{
                    "format": wgpu.TextureFormat.rgba32float,
                    "blend": None,
                }],
            },
        )

        # Pipeline for masked pass
        maskedPipelineLayout = self.device.create_pipeline_layout(
            bind_group_layouts=[self.maskedBindLayout]
        )

        self.maskedPipeline = self.device.create_render_pipeline(
            layout=maskedPipelineLayout,
            vertex={
                "module": self.maskedShader,
                "entry_point": "vs_main",
                "buffers": [{
                    "array_stride": 12,
                    "attributes": [{
                        "format": wgpu.VertexFormat.float32x3,
                        "offset": 0,
                        "shader_location": 0,
                    }],
                }],
            },
            primitive={
                "topology": wgpu.PrimitiveTopology.triangle_list,
                "cull_mode": wgpu.CullMode.none,
                "front_face": "ccw",
            },
            depth_stencil={
                "format": wgpu.TextureFormat.depth32float,
                "depth_write_enabled": True,
                "depth_compare": wgpu.CompareFunction.less,
            },
            multisample={
                "count": 1,
                "mask": 0xFFFFFFFF,
            },
            fragment={
                "module": self.maskedShader,
                "entry_point": "fs_main",
                "targets": [{
                    "format": wgpu.TextureFormat.rgba32float,
                    "blend": None,
                }],
            },
        )

    def _prepareMeshBuffers(self, mesh: trimesh.Trimesh):
        """
        Upload mesh data to GPU buffers
        """
        vertices = mesh.vertices.astype(np.float32)
        indices = mesh.faces.astype(np.uint32).flatten()

        self.vertexBuffer = self.device.create_buffer_with_data(
            data=vertices,
            usage=wgpu.BufferUsage.VERTEX,
        )

        self.indexBuffer = self.device.create_buffer_with_data(
            data=indices,
            usage=wgpu.BufferUsage.INDEX,
        )

        self.indexCount = len(indices)

    def renderWithOverhangMask(self) -> None:
        """
        Perform two-pass rendering: overhang mask first, then filtered main mesh
        """
        mvp = self._computeMVPMatrix()

        # PASS 1: Render overhang mesh to texture
        self._prepareMeshBuffers(self._overhangMesh)

        # Create bind group for overhang pass
        overhang_bind_group = self.device.create_bind_group(
            layout=self.overhangBindLayout,
            entries=[{
                "binding": 0,
                "resource": {
                    "buffer": self.uniformBuffer,
                    "offset": 0,
                    "size": 64,
                },
            }]
        )

        self.device.queue.write_buffer(self.uniformBuffer, 0, mvp.tobytes())

        command_encoder = self.device.create_command_encoder()

        # Render overhang mesh
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": self.overhangTextureVew,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }],
            depth_stencil_attachment={
                "view": self.depthTexture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.discard,
            },
        )

        render_pass.set_pipeline(self.overhangPipeline)
        render_pass.set_bind_group(0, overhang_bind_group)
        render_pass.set_vertex_buffer(0, self.vertexBuffer)
        render_pass.set_index_buffer(self.indexBuffer, wgpu.IndexFormat.uint32)
        render_pass.draw_indexed(self.indexCount, 1, 0, 0, 0)
        render_pass.end()

        self.device.queue.submit([command_encoder.finish()])

        # Read back overhang mask for CPU-side filtering
        bytes_per_pixel = 16
        bytes_per_row = self._visSize[0] * bytes_per_pixel
        bytes_per_row = (bytes_per_row + 255) & ~255

        buffer_size = bytes_per_row * self._visSize[1]
        overhang_readback = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        command_encoder2 = self.device.create_command_encoder()
        command_encoder2.copy_texture_to_buffer(
            {
                "texture": self.overhangTexture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": overhang_readback,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": self._visSize[1],
            },
            (self._visSize[0], self._visSize[1], 1),
        )
        self.device.queue.submit([command_encoder2.finish()])

        overhang_readback.map_sync(mode=wgpu.MapMode.READ)
        overhang_data = overhang_readback.read_mapped()
        overhang_readback.unmap()

        overhang_array = np.frombuffer(overhang_data, dtype=np.float32).reshape(
            self._visSize[1], bytes_per_row // 4
        )
        overhang_array = overhang_array[:, :self._visSize[0] * 4].reshape(
            self._visSize[1], self._visSize[0], 4
        )
        self.overhang_mask = np.flipud(overhang_array[:, :, 0])

        # PASS 2: Render main mesh with overhang mask texture
        self._prepareMeshBuffers(self._mesh)

        # Prepare extended uniform data (MVP + bbox_min + resolution)
        bbox_min = self._bbox[0, :2].astype(np.float32)
        resolution = np.array([self._resolution], dtype=np.float32)
        padding = np.array([0.0], dtype=np.float32)

        uniform_data = mvp.tobytes() + bbox_min.tobytes() + resolution.tobytes() + padding.tobytes()

        # Create bind group for masked pass
        masked_bind_group = self.device.create_bind_group(
            layout=self.maskedBindLayout,
            entries=[
                {
                    "binding": 0,
                    "resource": {
                        "buffer": self.uniformBuffer,
                        "offset": 0,
                        "size": 80,
                    },
                },
                {
                    "binding": 1,
                    "resource": self.overhangTextureVew,
                }
            ]
        )

        self.device.queue.write_buffer(self.uniformBuffer, 0, uniform_data)

        command_encoder = self.device.create_command_encoder()

        # Render main mesh with masking
        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": self.renderTexture_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }],
            depth_stencil_attachment={
                "view": self.depthTexture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.discard,
            },
        )

        render_pass.set_pipeline(self.maskedPipeline)
        render_pass.set_bind_group(0, masked_bind_group)
        render_pass.set_vertex_buffer(0, self.vertexBuffer)
        render_pass.set_index_buffer(self.indexBuffer, wgpu.IndexFormat.uint32)
        render_pass.draw_indexed(self.indexCount, 1, 0, 0, 0)
        render_pass.end()

        # Read back the final result
        bytes_per_pixel = 16
        bytes_per_row = self._visSize[0] * bytes_per_pixel
        bytes_per_row = (bytes_per_row + 255) & ~255

        buffer_size = bytes_per_row * self._visSize[1]
        readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        command_encoder.copy_texture_to_buffer(
            {
                "texture": self.renderTexture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": readback_buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": self._visSize[1],
            },
            (self._visSize[0], self._visSize[1], 1),
        )

        self.device.queue.submit([command_encoder.finish()])

        # Read back and extract height data
        readback_buffer.map_sync(mode=wgpu.MapMode.READ)
        data = readback_buffer.read_mapped()
        readback_buffer.unmap()

        depthArray = np.frombuffer(data, dtype=np.float32).reshape(
            self._visSize[1], bytes_per_row // 4
        )
        depthArray = depthArray[:, :self._visSize[0] * 4].reshape(
            self._visSize[1], self._visSize[0], 4
        )

        # Extract Z-coordinates and flip vertically
        self.heightMap = np.flipud(depthArray[:, :, 0])

    def renderWithoutOverhangMask(self) -> None:
        """
        Implementation of a single-pass render without using the overhang mask.
        """
        mvp = self._computeMVPMatrix()

        # Upload MVP to the beginning of the uniform buffer (matches overhang pass layout)
        self.device.queue.write_buffer(self.uniformBuffer, 0, mvp.tobytes())

        # Prepare GPU buffers for the primary mesh
        self._prepareMeshBuffers(self._mesh)

        # Bind group for simple pass (only MVP uniform needed)
        simple_bind_group = self.device.create_bind_group(
            layout=self.overhangBindLayout,
            entries=[{
                "binding": 0,
                "resource": {
                    "buffer": self.uniformBuffer,
                    "offset": 0,
                    "size": 64,
                },
            }]
        )

        command_encoder = self.device.create_command_encoder()

        render_pass = command_encoder.begin_render_pass(
            color_attachments=[{
                "view": self.renderTexture_view,
                "resolve_target": None,
                "clear_value": (0, 0, 0, 0),
                "load_op": wgpu.LoadOp.clear,
                "store_op": wgpu.StoreOp.store,
            }],
            depth_stencil_attachment={
                "view": self.depthTexture_view,
                "depth_clear_value": 1.0,
                "depth_load_op": wgpu.LoadOp.clear,
                "depth_store_op": wgpu.StoreOp.discard,
            },
        )

        # Use the overhang pipeline which writes Z into RGBA
        render_pass.set_pipeline(self.overhangPipeline)
        render_pass.set_bind_group(0, simple_bind_group)
        render_pass.set_vertex_buffer(0, self.vertexBuffer)
        render_pass.set_index_buffer(self.indexBuffer, wgpu.IndexFormat.uint32)
        render_pass.draw_indexed(self.indexCount, 1, 0, 0, 0)
        render_pass.end()

        # Submit and read back
        self.device.queue.submit([command_encoder.finish()])

        bytes_per_pixel = 16
        bytes_per_row = self._visSize[0] * bytes_per_pixel
        bytes_per_row = (bytes_per_row + 255) & ~255

        buffer_size = bytes_per_row * self._visSize[1]
        readback_buffer = self.device.create_buffer(
            size=buffer_size,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.MAP_READ,
        )

        command_encoder2 = self.device.create_command_encoder()
        command_encoder2.copy_texture_to_buffer(
            {
                "texture": self.renderTexture,
                "mip_level": 0,
                "origin": (0, 0, 0),
            },
            {
                "buffer": readback_buffer,
                "offset": 0,
                "bytes_per_row": bytes_per_row,
                "rows_per_image": self._visSize[1],
            },
            (self._visSize[0], self._visSize[1], 1),
        )
        self.device.queue.submit([command_encoder2.finish()])

        readback_buffer.map_sync(mode=wgpu.MapMode.READ)
        data = readback_buffer.read_mapped()
        readback_buffer.unmap()

        depth_array = np.frombuffer(data, dtype=np.float32).reshape(
            self._visSize[1], bytes_per_row // 4
        )
        depth_array = depth_array[:, :self._visSize[0] * 4].reshape(
            self._visSize[1], self._visSize[0], 4
        )

        # Extract Z-coordinates and flip vertically to match other APIs
        self.heightMap = np.flipud(depth_array[:, :, 0])

    def _computeMVPMatrix(self) -> np.ndarray:
        """
        Compute Model-View-Projection matrix for orthographic rendering
        """

        avg = np.mean(self._bbox, axis=0)

        left = self._bbox[0, 0]
        right = self._bbox[1, 0]
        bottom = self._bbox[0, 1]
        top = self._bbox[1, 1]
        near = -1e4
        far = 1e4

        # Orthographic projection
        proj = np.array([
            [2.0 / (right - left), 0, 0, 0],
            [0, 2.0 / (top - bottom), 0, 0],
            [0, 0, -1.0 / (far - near), -near / (far - near)],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # View transform
        if self.flipDir:
            # View from below
            view = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)
        else:
            # View from above
            view = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ], dtype=np.float32)

        # Model transform - center mesh
        model = np.array([
            [1, 0, 0, -avg[0]],
            [0, 1, 0, -avg[1]],
            [0, 0, 1, -avg[2]] ,
            [0, 0, 0, 1],
        ], dtype=np.float32)

        mvp = proj @ view @ model
        return mvp.T.astype(np.float32)

    def setMesh(self, mesh: Optional[trimesh.Trimesh] = None,
                overhangMesh: Optional[trimesh.Trimesh] = None,
                bbox: Optional[np.ndarray] = None,
                resolution: Optional[float] = None) -> None:
        """
        Update meshes and re-render the height maps.

        This method allows reusing the same renderer instance with different meshes,
        which improves performance by avoiding GPU resource reinitialisation, including
        the recompilation of shaders.

        :param mesh: New mesh - if `None`, keeps current mesh)
        :param overhangMesh:  The overhang mesh - if `None`, keeps current overhang mesh)
        :param bbox: Bounding box. If `None`, recalculates from mesh bounds.
        :param resolution:  Ray projection resolution. If `None`, keeps current resolution.
        """
        # Update meshes if provided
        if mesh is not None:
            self._mesh = mesh

        if overhangMesh is not None:
            self._overhangMesh = overhangMesh

        # Update resolution if provided
        if resolution is not None:
            self._resolution = resolution

        # Update bounding box
        if bbox is not None:
            self._bbox = bbox
        else:
            # Recalculate combined bounds from current meshes
            combined_bounds = np.array([
                np.minimum(self._mesh.bounds[0], self._overhangMesh.bounds[0]),
                np.maximum(self._mesh.bounds[1], self._overhangMesh.bounds[1])
            ])
            self._bbox = combined_bounds

        # Check if viewport size needs to change
        meshExtents = np.diff(self._bbox, axis=0)
        new_visSize = (int((meshExtents / self._resolution).flatten()[0]),
                       int((meshExtents / self._resolution).flatten()[1]))

        size_changed = new_visSize != self._visSize
        self._visSize = new_visSize

        # Recreate render targets if size changed
        if size_changed:
            self._createRenderTargets()

    @property
    def resolution(self) -> float:
        """Get pixel resolution"""
        return self._resolution

    @property
    def viewortSize(self):
        """Get viewport size (width, height)"""
        return self._visSize

    @property
    def bbox(self) -> np.ndarray:
        """Get bounding box"""
        return self._bbox

    def close(self):
        """Clean up GPU resources"""
        # WGPU handles cleanup automatically
        pass


def projectOverhangHeightMap(overhang_mesh: trimesh.Trimesh,
                             resolution: float = 0.05,
                             flipDir: bool = False,
                             bbox: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Project an overhang mesh to a 2D height map.

    Renders the mesh using GPU acceleration to create a 2D array where each
    pixel contains the Z-coordinate of the mesh surface at that (x, y) location.

    :param overhang_mesh: Trimesh object to project
    :param resolution: The raytracing resolution
    :param flipDir: Viewing direction. `False` = from above, `True` = from below.
    :param bbox: Bounding box of the region to project. If `None`, uses mesh bounds.
    :return: 2D numpy array (height map) with float32 Z-coordinates
    """

    # Use the same mesh for both primary and overhang to just get a height map
    renderer = OverhangMaskRenderer(overhang_mesh, overhang_mesh, resolution, flipDir, bbox)

    try:
        return renderer.heightMap
    finally:
        renderer.close()
