import numpy as np
import trimesh
from vispy import app, gloo
from vispy.util.transforms import translate, rotate, ortho

import matplotlib.pyplot as plt
#app.use_app('pyside6')  # Set backend

vert = """

uniform   mat4 u_model;
uniform   mat4 u_view;
uniform   mat4 u_projection;

attribute vec3 a_position;
attribute vec3 a_color;

varying vec4 v_color;

void main()
{
    v_color = vec4(a_color, 1.0);
    gl_Position = u_projection * u_view * u_model * vec4(a_position,1.0);
}
"""

frag = """
varying vec4 v_color;

void main()
{
    //gl_FragColor = vec4( (gl_FragCoord.z / 1e5)+0.,(gl_FragCoord.z / 1e5)+0.,(gl_FragCoord.z / 1e5)+0.5,1.0);
    gl_FragColor = vec4( v_color.z,v_color.z,v_color.z,1.0);
}
"""


class Canvas(app.Canvas):

    @property
    def resolution(self):
        return self._resolution

    @resolution.setter
    def resolution(self, res: float):
        self._resolution = res

    @property
    def visSize(self):
        return self._visSize

    @property
    def mesh(self) -> trimesh.Trimesh:
        return self._mesh

    @property
    def bbox(self) -> np.ndarray:
        return self._bbox

    def setMesh(self, mesh: trimesh.Trimesh):

        self.vertices = np.copy(mesh.vertices).astype(np.float32)
        self.filled = np.copy(mesh.faces).astype(np.int32)
        self.verticesColor = np.copy(mesh.vertices).astype(np.float32)

        self._bbox = mesh.bounds
        self._mesh = mesh

    def __enter__(self):
        self._backend._vispy_warmup()
        return self

    def __init__(self, mesh: trimesh.Trimesh = None, rasterResolution: float = 0.05, flipDir = False, bbox = None):

        self.flipDir = flipDir
        self.rgb = None
        self.vertices = None
        self.filled = None
        self.verticesColor = None
        self._resolution = rasterResolution
        self._mesh = None
        self._bbox = None

        self.program = None
        self._fbo = None  # Initialize to None, create later
        self._fbo_ready = False  # Track FBO initialization
        self._program_ready = False

        if mesh:
            self.setMesh(mesh)

        if bbox is not None:
            self._bbox = bbox

        meshExtents = np.diff(self.bbox, axis=0)

        self._visSize = (meshExtents / self.resolution).flatten()

        app.Canvas.__init__(self, 'interactive', show=False, resizable=True, autoswap=False, decorate=False,
                            vsync=False,
                            size=(self.visSize[0], self.visSize[1]),
                            )


        self.filled = self.filled.astype(np.uint32).flatten()
        self.filled_buf = gloo.IndexBuffer(self.filled)

        self.vertex_data = np.zeros(self.vertices.shape[0], dtype=[('a_position', np.float32, 3),
                                                              ('a_color', np.float32, 3)])

        self.vertex_data['a_position'] = self.vertices.astype(np.float32)
        self.vertex_data['a_color'] = self.vertices.astype(np.float32)

        avg = np.mean(self.bbox, axis=0)

        if flipDir:
            self.view = rotate(0, [-1, 0, 0])
        else:
            self.view = np.dot(np.dot(translate((-avg[0], -avg[1], -avg[2])),
                              rotate(-180, [1,0,0])),
                              translate((avg[0], avg[1], avg[2])))


        self.model = np.eye(4, dtype=np.float32)

        # Store shape for later FBO creation
        self._fbo_shape = (int(self._visSize[1]), int(self._visSize[0]))

        # Create the render texture
        if False:
            shape = int(self._visSize[1]), int(self._visSize[0])
            self._rendertex = gloo.Texture2D((shape + (4,)), format='rgba', internalformat='rgba32f')
            self._depthRenderBuffer = gloo.RenderBuffer(shape, format='depth')

            # Create FBO, attach the color buffer and depth buffer
            self._fbo = gloo.FrameBuffer(self._rendertex, self._depthRenderBuffer)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        gloo.set_viewport(0, 0, self._visSize[0], self._visSize[1])
        self.projection = ortho(self.bbox[1, 0], self.bbox[0, 0], self.bbox[1, 1], self.bbox[0, 1], -1e4, 1e4)

        if False:

            # Set MVP variables for shaders
            self.program['u_projection'] = self.projection
            self.program['u_model'] = self.model
            self.program['u_view'] = self.view

        gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))
        gloo.set_state('opaque')

        self.update()

    def _ensure_fbo(self):
        """Create FBO when OpenGL context is ready"""
        if self._fbo_ready:
            return True

        try:
            self._rendertex = gloo.Texture2D((self._fbo_shape + (4,)), format='rgba', internalformat='rgba32f')
            self._depthRenderBuffer = gloo.RenderBuffer(self._fbo_shape, format='depth')
            self._fbo = gloo.FrameBuffer(self._rendertex, self._depthRenderBuffer)

            # Explicitly activate and validate
            self._fbo.activate()

            # Check if FBO is complete

            status = gloo.gl.glCheckFramebufferStatus(gloo.gl.GL_FRAMEBUFFER)

            if status != gloo.gl.GL_FRAMEBUFFER_COMPLETE:
                print(f"FBO incomplete: {status}")
                self._fbo = None
                return False

            self._fbo.deactivate()
            self._fbo_ready = True
            return True
        except Exception as e:
            print(f"Failed to create FBO: {e}")
            self._fbo = None
            return False

    def on_resize(self, event):

        # TODO - find a better way to set the bounds for the orthographic projection

        gloo.set_viewport(0, 0, self._visSize[0]*2, self._visSize[1]*2)
        self.finalSize = (event.physical_size[0], event.physical_size[1])

        """        
        self.projection = ortho(self.bbox[1, 0], self.bbox[0, 0],
                                self.bbox[1, 1], self.bbox[0, 1],
                                -self.bbox[1, 2], self.bbox[0, 2])
                                """

        self.projection = ortho(self.bbox[0, 0], self.bbox[1, 0],
                                self.bbox[0, 1], self.bbox[1, 1],
                                -1e4,  1e4)

        if self.program:
            self.program['u_projection'] = self.projection

    def on_draw(self, event):
        # Compile shader program on first draw when context is ready
        if not self._program_ready:
            try:


                self.program = gloo.Program(vert, frag)
                self.program.bind(gloo.VertexBuffer(self.vertex_data))

                self.program['u_projection'] = self.projection
                self.program['u_model'] = self.model
                self.program['u_view'] = self.view

                self._program_ready = True
            except Exception as e:
                print(f"Failed to compile shader: {e}")
                return

        self._ensure_fbo()

        with self._fbo:
            gloo.clear()
            gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))

            gloo.set_viewport(0, 0, *self.finalSize)
            gloo.set_viewport(0, 0, self._visSize[0], self._visSize[1])
            gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False, cull_face=False)

            self.program.draw('triangles', self.filled_buf)
            self.rgb = gloo.read_pixels((0, 0, self._visSize[0], self._visSize[1]), True, out_type='float')


def projectHeightMap(mesh: trimesh.Trimesh,
                     resolution: float = 0.05,
                     flipDir: bool = False,
                     bbox: np.ndarray = None) -> np.ndarray:

    c = Canvas(mesh, resolution, flipDir, bbox)

    #c.show(visible=True) #previous
    c.show(visible=True)

    # Multiple event cycles to ensure initialization
    for _ in range(10):
        c.app.process_events()

    # Manually trigger draw
    c.update()
    c.app.process_events()
    c.close()

    if c.rgb is None:
        pass
        #mesh.show()
        #raise Exception()
    else:
        return c.rgb[:, :, 1]
