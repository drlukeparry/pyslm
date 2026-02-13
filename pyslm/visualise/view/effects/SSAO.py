import wgpu

from pygfx.renderers.wgpu.engine.effectpasses import (
    EffectPass,
    FullQuadPass,
)
from pygfx.renderers.wgpu.engine.shared import get_shared


class SSAOPass(EffectPass):
    """
    A screen-space ambient occlusion (SSAO) effect pass based on pure depth sampling.
    
    This implementation reconstructs normals from the depth buffer and uses a sampling
    sphere with random rotation for more accurate occlusion. 
    
    Inspired by the pure depth: https://theorangeduck.com/page/pure-depth-ssao.
    """

    USES_DEPTH = True

    class _SSAOPass(FullQuadPass):
        """
        Internal fullscreen pass computing the AO term from depth buffer
        """

        uniform_type = dict(
            totalStrength="f4",
            base="f4",
            area="f4",
            falloff="f4",
            radius="f4",
        )

        # NOTE: We rely on the standard FullQuadPass bindings:
        #   - colorTex : the colour buffer
        #   - depthTex : the depth buffer
        #   - texSampler : sampler
        wgsl = """
            fn normal_from_depth(depth: f32, texcoords: vec2<f32>) -> vec3<f32> {
                let offset1 = vec2<f32>(0.0, 0.001);
                let offset2 = vec2<f32>(0.001, 0.0);
                let depth1 = textureSample(depthTex, texSampler, texcoords + offset1);
                let depth2 = textureSample(depthTex, texSampler, texcoords + offset2);
                let p1 = vec3<f32>(offset1, depth1 - depth);
                let p2 = vec3<f32>(offset2, depth2 - depth);
                var normal = cross(p1, p2);
                normal.z = -normal.z;
                return normalize(normal);
            }

            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let uv = varyings.texCoord;
                let depth = textureSample(depthTex, texSampler, uv);

                if (depth >= 0.999) {
                    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
                }

                let position = vec3<f32>(uv, depth);
                let normal = normal_from_depth(depth, uv);

                let radius_depth = u_effect.radius / max(depth, 0.001);
                let random = normalize(vec3<f32>(0.5, 0.5, 0.5));
                var occlusion = 0.0;
                let samples = 16;
                let sample_sphere = array<vec3<f32>, 16>(
                    vec3<f32>( 0.5381, 0.1856,-0.4319), vec3<f32>( 0.1379, 0.2486, 0.4430),
                    vec3<f32>( 0.3371, 0.5679,-0.0057), vec3<f32>(-0.6999,-0.0451,-0.0019),
                    vec3<f32>( 0.0689,-0.1598,-0.8547), vec3<f32>( 0.0560, 0.0069,-0.1843),
                    vec3<f32>(-0.0146, 0.1402, 0.0762), vec3<f32>( 0.0100,-0.1924,-0.0344),
                    vec3<f32>(-0.3577,-0.5301,-0.4358), vec3<f32>(-0.3169, 0.1063, 0.0158),
                    vec3<f32>( 0.0103,-0.5869, 0.0046), vec3<f32>(-0.0897,-0.4940, 0.3287),
                    vec3<f32>( 0.7119,-0.0154,-0.0918), vec3<f32>(-0.0533, 0.0596,-0.5411),
                    vec3<f32>( 0.0352,-0.0631, 0.5460), vec3<f32>(-0.4776, 0.2847,-0.0271)
                );

                for (var i = 0; i < samples; i = i + 1) {
                    let reflected = reflect(sample_sphere[i], random);
                    let ray = radius_depth * reflected;
                    let dot_prod = dot(ray, normal);
                    let hemi_ray = position + sign(dot_prod) * ray;
                    let occ_uv = clamp(hemi_ray.xy, vec2<f32>(0.0), vec2<f32>(1.0));
                    let occ_depth = textureSample(depthTex, texSampler, occ_uv);
                    let difference = depth - occ_depth;
                    
                    // Use step and smoothstep for occlusion calculation
                    if (difference > u_effect.falloff) {
                        let smooth_val = 1.0 - smoothstep(u_effect.falloff, u_effect.area, difference);
                        occlusion = occlusion + smooth_val;
                    }
                }

                let ao = 1.0 - u_effect.totalStrength * occlusion * (1.0 / f32(samples));
                let final_ao = clamp(ao + u_effect.base, 0.0, 1.0);
                return vec4<f32>(final_ao, final_ao, final_ao, 1.0);
            }
        """

    class _CompositePass(FullQuadPass):
        """
        Composite pass that multiplies AO into the scene colour.
        """

        uniform_type = dict(_unused="f4")

        wgsl = """
            @fragment
            fn fs_main(varyings: Varyings) -> @location(0) vec4<f32> {
                let colour = textureSample(colorTex, texSampler, varyings.texCoord);
                let ao = textureSample(aoTex, texSampler, varyings.texCoord).r;
                let result = colour.rgb * ao;
                return vec4<f32>(result, colour.a);
            }
        """

    def __init__(self,
                 radius: float = 0.0002,
                 totalStrength: float = 1.0,
                 base: float = 0.2,
                 area: float = 0.0075,
                 falloff: float = 0.00001) -> None:

        super().__init__()

        self._radius = float(radius)
        self._totalStrength = float(totalStrength)
        self._base = float(base)
        self._area = float(area)
        self._falloff = float(falloff)

        self._ssaoPass = self._SSAOPass()
        self._compositePass = self._CompositePass()

        # Internal AO texture matching the colour buffer
        self._ao_texture = None
        self._current_size = (0, 0)

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = float(value)

    @property
    def totalStrength(self) -> float:
        return self._totalStrength

    @totalStrength.setter
    def totalStrength(self, value: float) -> None:
        self._totalStrength = float(value)

    @property
    def base(self) -> float:
        return self._base

    @base.setter
    def base(self, value: float) -> None:
        self._base = float(value)

    @property
    def area(self) -> float:
        return self._area

    @area.setter
    def area(self, value: float) -> None:
        self._area = float(value)

    @property
    def falloff(self) -> float:
        return self._falloff

    @falloff.setter
    def falloff(self, value: float) -> None:
        self._falloff = float(value)

    def _ensure_ao_texture(self, source_texture) -> None:
        device = get_shared().device

        size = (source_texture.size[0], source_texture.size[1])
        if self._ao_texture is not None and size == self._current_size:
            return

        self._current_size = size

        self._ao_texture = device.create_texture(
            size=(size[0], size[1], 1),
            format=source_texture.format,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.TEXTURE_BINDING,
            sample_count=1,
            mip_level_count=1,
            dimension=wgpu.TextureDimension.d2,
        )

    def render(self, command_encoder, color_tex, depth_tex, target_tex):
        """
        Run the SSAO pass and composite the result into ``target_tex``.

        :param command_encoder:  Encoder to record commands for WGPU
        :param color_tex:  Scene colour buffer.
        :param depth_tex:  Depth buffer
        :param target_tex: Output colour buffer.
        """

        source_texture = color_tex.texture
        self._ensure_ao_texture(source_texture)

        # Update uniforms
        self._ssaoPass._uniform_data["radius"] = float(self._radius)
        self._ssaoPass._uniform_data["totalStrength"] = float(self._totalStrength)
        self._ssaoPass._uniform_data["base"] = float(self._base)
        self._ssaoPass._uniform_data["area"] = float(self._area)
        self._ssaoPass._uniform_data["falloff"] = float(self._falloff)

        # Compute AO term into the AO texture
        ao_view = self._ao_texture.create_view()
        self._ssaoPass.render(
            command_encoder,
            colorTex=color_tex,
            depthTex=depth_tex,
            targetTex=ao_view,
        )

        # Composite AO with the original colour buffer into the target
        self._compositePass.render(
            command_encoder,
            colorTex=color_tex,
            aoTex=ao_view,
            targetTex=target_tex,
        )

    def __repr__(self) -> str:
        return (
            f"<SSAOPass radius={self.radius} totalStrength={self.totalStrength} "
            f"base={self.base} area={self.area} falloff={self.falloff} at {hex(id(self))}>"
        )
