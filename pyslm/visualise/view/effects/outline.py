"""
Outline/Edge Detection Effect Pass for CAD Visualisation

This implementation uses depth and normal-based edge detection to create
clean outlines for technical/CAD rendering. Inspired by: https://roystan.net/articles/outline-shader/

The effect detects edges by:
1. Sampling depth values in a cross pattern around each pixel
2. Computing depth-based edge detection (Roberts Cross operator)
3. Optionally combining with normal-based edge detection
4. Applying configurable edge color and thickness
"""

import wgpu

from pygfx.renderers.wgpu.engine.effectpasses import EffectPass, FullQuadPass
from pygfx.renderers.wgpu.engine.shared import get_shared


class OutlinePass(EffectPass):
    """
    CAD-style outline/edge detection effect pass.

    Creates clean, technical outlines by detecting depth and normal discontinuities
    in the scene. Perfect for technical visualization, CAD models, and stylized
    rendering.

    Parameters
    ----------
    edgeColour : tuple[float, float, float], default (0.0, 0.0, 0.0)
        RGB color for the outline/edges (0-1 range).
    edgeThickness : float, default 1.0
        Thickness of the detected edges (multiplier on sampling offset).
    depthThreshold : float, default 0.1
        Sensitivity for depth-based edge detection. Lower = more sensitive.
    depthNormalThreshold : float, default 0.5
        Sensitivity for normal-based edge detection. Lower = more sensitive.
    depthNormalThresholdScale : float, default 7.0
        Scaling factor for normal threshold based on distance.
    useNormals : bool, default True
        Whether to use normal-based edge detection (requires normal buffer).
    """

    USES_DEPTH = True

    class _OutlinePass(FullQuadPass):
        """Internal full-screen pass for edge detection."""

        uniform_type = dict(
            edgeColor="3xf4",
            edgeThickness="f4",
            depthThreshold="f4",
            depthNormalThreshold="f4",
            depthNormalThresholdScale="f4",
            useNormals="f4",  # Using f4 as bool (0.0 or 1.0)
        )

        # Edge detection shader using Roberts Cross operator
        wgsl = """
            fn sample_depth(coord: vec2<f32>) -> f32 {
                let uv = clamp(coord, vec2<f32>(0.0), vec2<f32>(1.0));
                return textureSample(depthTex, texSampler, uv);
            }
            
            fn normal_from_depth(depth: f32, texcoords: vec2<f32>, offset: f32) -> vec3<f32> {
                let offset1 = vec2<f32>(offset, 0.0);
                let offset2 = vec2<f32>(0.0, offset);
                
                let depth1 = sample_depth(texcoords + offset1);
                let depth2 = sample_depth(texcoords + offset2);
                
                let p1 = vec3<f32>(offset1, depth1 - depth);
                let p2 = vec3<f32>(offset2, depth2 - depth);
                
                var normal = cross(p1, p2);
                normal.z = -normal.z;
                return normalize(normal);
            }
            
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let uv = varyings.texCoord;
                let halfScaleFloor = floor(u_effect.edgeThickness * 0.5);
                let halfScaleCeil = ceil(u_effect.edgeThickness * 0.5);
                
                // Calculate sampling offset based on edge thickness
                let texelSize = vec2<f32>(0.001, 0.001); // Approximate texel size
                let bottomLeftUV = uv - texelSize * halfScaleFloor;
                let topRightUV = uv + texelSize * halfScaleCeil;
                let bottomRightUV = uv + vec2<f32>(texelSize.x * halfScaleCeil, -texelSize.y * halfScaleFloor);
                let topLeftUV = uv + vec2<f32>(-texelSize.x * halfScaleFloor, texelSize.y * halfScaleCeil);
                
                // Sample depth values for Roberts Cross operator
                let depth0 = sample_depth(bottomLeftUV);
                let depth1 = sample_depth(topRightUV);
                let depth2 = sample_depth(bottomRightUV);
                let depth3 = sample_depth(topLeftUV);
                
                // Roberts Cross depth edge detection
                let depthFiniteDifference0 = depth1 - depth0;
                let depthFiniteDifference1 = depth3 - depth2;
                
                var edgeDepth = sqrt(
                    depthFiniteDifference0 * depthFiniteDifference0 + 
                    depthFiniteDifference1 * depthFiniteDifference1
                ) * 100.0;
                
                edgeDepth = select(0.0, 1.0, edgeDepth > u_effect.depthThreshold);
                
                // Optional: Normal-based edge detection
                var edgeNormal = 0.0;
                if (u_effect.useNormals > 0.5) {
                    let normal0 = normal_from_depth(depth0, bottomLeftUV, 0.001);
                    let normal1 = normal_from_depth(depth1, topRightUV, 0.001);
                    let normal2 = normal_from_depth(depth2, bottomRightUV, 0.001);
                    let normal3 = normal_from_depth(depth3, topLeftUV, 0.001);
                    
                    let normalFiniteDifference0 = normal1 - normal0;
                    let normalFiniteDifference1 = normal3 - normal2;
                    
                    let edgeNormalRaw = sqrt(
                        dot(normalFiniteDifference0, normalFiniteDifference0) + 
                        dot(normalFiniteDifference1, normalFiniteDifference1)
                    );
                    
                    // Scale threshold based on depth
                    let depthCenter = sample_depth(uv);
                    let normalThreshold = u_effect.depthNormalThreshold * depthCenter * u_effect.depthNormalThresholdScale;
                    edgeNormal = select(0.0, 1.0, edgeNormalRaw > normalThreshold);
                }
                
                // Combine depth and normal edges
                let edge = max(edgeDepth, edgeNormal);
                
                // Return edge value: 0.0 for edges (black), 1.0 for non-edges (white)
                // This is inverted so the thickening pass can detect black pixels as edges
                return vec4<f32>(1.0 - edge, 1.0 - edge, 1.0 - edge, 1.0);
            }
        """

    class _ThickenPass(FullQuadPass):
        """Pass to thicken the outline edges using disc-based sampling."""

        uniform_type = dict(
            lineThickness="f4",
        )

        wgsl = """
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let uv = varyings.texCoord;
                let base = textureSample(outlineTex, texSampler, uv);
                let black = vec4<f32>(0.0, 0.0, 0.0, 1.0);
                let white = vec4<f32>(1.0, 1.0, 1.0, 1.0);
                
                let thickness = i32(u_effect.lineThickness);
                let thickness2 = thickness * thickness;
                
                // Texel size (approximate)
                let texelSize = vec2<f32>(0.001, 0.001);
                
                // Check if any pixel within a disc of radius "thickness" is black (edge)
                var is_black = false;
                
                let halfThickness = thickness / 2;
                for (var i = -halfThickness; i < halfThickness; i = i + 1) {
                    let i2 = i * i;
                    for (var j = -halfThickness; j < halfThickness; j = j + 1) {
                        let offset = vec2<f32>(f32(i), f32(j)) * texelSize;
                        let sample_uv = clamp(uv + offset, vec2<f32>(0.0), vec2<f32>(1.0));
                        let smpl = textureSample(outlineTex, texSampler, sample_uv);
                        
                        // Check if within disc radius and is black (edge pixel)
                        let dist2 = i2 + j * j;
                        let is_edge = smpl.r < 0.5; // Edge pixels are darker
                        is_black = is_black || (dist2 < thickness2 && is_edge);
                    }
                }
                
                // Return thickened edge
                return select(base, black, is_black);
            }
        """

    class _CompositePass(FullQuadPass):
        """Composite pass that blends edges with the scene."""

        uniform_type = dict(
            edgeColor="3xf4",
        )

        wgsl = """
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let color = textureSample(colorTex, texSampler, varyings.texCoord);
                let edge_value = textureSample(outlineTex, texSampler, varyings.texCoord).r;
                
                // Edge is black (0.0), so invert to get edge mask (1.0 = edge, 0.0 = no edge)
                let edge = 1.0 - edge_value;
                
                // Blend edge color with scene color
                let edgeColorRGB = vec3<f32>(u_effect.edgeColor[0], u_effect.edgeColor[1], u_effect.edgeColor[2]);
                let result = mix(color.rgb, edgeColorRGB, edge);
                
                return vec4<f32>(result, color.a);
            }
        """

    def __init__(self,
                 edgeColour: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 edgeThickness: float = 1.0,
                 depthThreshold: float = 0.1,
                 depthNormalThreshold: float = 0.5,
                 depthNormalThresholdScale: float = 7.0,
                 useNormals: bool = True,
                 lineThickness: int = 2,
                 ) -> None:

        super().__init__()

        self._edgeColour = tuple(float(c) for c in edgeColour)
        self._edgeThickness = float(edgeThickness)
        self._depthThreshold = float(depthThreshold)
        self._depthNormalThreshold = float(depthNormalThreshold)
        self._depthNormalThresholdScale = float(depthNormalThresholdScale)
        self._useNormals = bool(useNormals)
        self._lineThickness = int(lineThickness)

        self._outlinePass = self._OutlinePass()
        self._thickenPass = self._ThickenPass()
        self._compositePass = self._CompositePass()

        # Internal outline textures
        self._outlineTexture = None
        self._thickenedTexture = None
        self._current_size = (0, 0)

    # --- Properties -----------------------------------------------------

    @property
    def edgeColour(self) -> tuple[float, float, float]:
        """RGB color for edges (0-1 range)."""
        return self._edgeColour

    @edgeColour.setter
    def edgeColour(self, value: tuple[float, float, float]) -> None:
        self._edgeColour = tuple(float(c) for c in value)

    @property
    def edgeThickness(self) -> float:
        """Thickness multiplier for edges."""
        return self._edgeThickness

    @edgeThickness.setter
    def edgeThickness(self, value: float) -> None:
        self._edgeThickness = float(value)

    @property
    def depthThreshold(self) -> float:
        """Sensitivity for depth-based edges."""
        return self._depthThreshold

    @depthThreshold.setter
    def depthThreshold(self, value: float) -> None:
        self._depthThreshold = float(value)

    @property
    def depthNormalThreshold(self) -> float:
        """Sensitivity for normal-based edges."""
        return self._depthNormalThreshold

    @depthNormalThreshold.setter
    def depthNormalThreshold(self, value: float) -> None:
        self._depthNormalThreshold = float(value)

    @property
    def depthNormalThreshold_scale(self) -> float:
        """Scaling factor for normal threshold."""
        return self._depthNormalThresholdScale

    @depthNormalThreshold_scale.setter
    def depthNormalThreshold_scale(self, value: float) -> None:
        self._depthNormalThresholdScale = float(value)

    @property
    def useNormals(self) -> bool:
        """Whether to use normal-based edge detection."""
        return self._useNormals

    @useNormals.setter
    def useNormals(self, value: bool) -> None:
        self._useNormals = bool(value)

    @property
    def lineThickness(self) -> int:
        """Thickness of the lines in pixels (for thickening pass)."""
        return self._lineThickness

    @lineThickness.setter
    def lineThickness(self, value: int) -> None:
        self._lineThickness = int(value)

    def _ensure_outline_texture(self, source_texture) -> None:
        """
        Ensure outline texture matches source texture size.
        """
        device = get_shared().device

        size = (source_texture.size[0], source_texture.size[1])
        if self._outlineTexture is not None and size == self._current_size:
            return

        self._current_size = size

        self._outlineTexture = device.create_texture(
            size=(size[0], size[1], 1),
            format=source_texture.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            sample_count=1,
            mip_level_count=1,
            dimension=wgpu.TextureDimension.d2,
        )

        self._thickenedTexture = device.create_texture(
            size=(size[0], size[1], 1),
            format=source_texture.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            sample_count=1,
            mip_level_count=1,
            dimension=wgpu.TextureDimension.d2,
        )

    # --- EffectPass interface ------------------------------------------

    def render(self, command_encoder, colourTex, depthTex, targetTex):
        """
        Run the outline pass and composite with the scene.

        :param command_encoder: Encoder to record commands for WGPU
        :param colourTex: Scene colour buffer
        :param depthTex: Depth buffer
        :param targetTex: Output colour buffer
        """

        source_texture = colourTex.texture
        self._ensure_outline_texture(source_texture)

        # Update uniforms for outline detection
        self._outlinePass._uniform_data["edgeColor"] = list(self._edgeColour)
        self._outlinePass._uniform_data["edgeThickness"] = float(self._edgeThickness)
        self._outlinePass._uniform_data["depthThreshold"] = float(self._depthThreshold)
        self._outlinePass._uniform_data["depthNormalThreshold"] = float(self._depthNormalThreshold)
        self._outlinePass._uniform_data["depthNormalThresholdScale"] = float(self._depthNormalThresholdScale)
        self._outlinePass._uniform_data["useNormals"] = 1.0 if self._useNormals else 0.0

        # 1) Detect edges and render to outline texture
        outlineView = self._outlineTexture.create_view()
        self._outlinePass.render(
            command_encoder,
            colorTex=colourTex,
            depthTex=depthTex,
            targetTex=outlineView,
        )

        # 2) Thicken the edges
        self._thickenPass._uniform_data["lineThickness"] = float(self._lineThickness)
        thickenedView = self._thickenedTexture.create_view()
        
        self._thickenPass.render(
            command_encoder,
            outlineTex=outlineView,
            targetTex=thickenedView,
        )

        # Update composite pass uniforms
        self._compositePass._uniform_data["edgeColor"] = list(self._edgeColour)

        # Composite thickened outlines with the scene
        self._compositePass.render(
            command_encoder,
            colorTex=colourTex,
            outlineTex=thickenedView,
            targetTex=targetTex,
        )

    def __repr__(self) -> str:
        return (
            f"<OutlinePass edge_color={self.edgeColour} edge_thickness={self.edgeThickness} "
            f"depth_threshold={self.depthThreshold} use_normals={self.useNormals} at {hex(id(self))}>"
        )

