import numpy as np
import trimesh
from vispy import app, gloo
from vispy.util.transforms import translate, rotate, ortho

from matplotlib import pyplot as plt
app.use_app('pyqt5')  # Set backend

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

        if mesh:
            self.setMesh(mesh)

        if bbox is not None:
            self._bbox = bbox

        meshExtents = np.diff(self.bbox, axis=0)

        self._visSize = (meshExtents / self.resolution).flatten()

        app.Canvas.__init__(self, 'interactive', show=False, resizable=True, autoswap=False, decorate=False,
                            size=(self.visSize[0], self.visSize[1]))

        print('size', self._visSize)
        print('dpi', self.dpi)

        self.filled = self.filled.astype(np.uint32).flatten()
        self.filled_buf = gloo.IndexBuffer(self.filled)

        vertex_data = np.zeros(self.vertices.shape[0], dtype=[('a_position', np.float32, 3),
                                                              ('a_color', np.float32, 3)])

        vertex_data['a_position'] = self.vertices.astype(np.float32)
        vertex_data['a_color'] = self.vertices.astype(np.float32)

        self.program = gloo.Program(vert, frag)
        self.program.bind(gloo.VertexBuffer(vertex_data))

        avg = np.mean(self.bbox, axis=0)

        if flipDir:
            self.view = rotate(0, [1, 0, 0])
        else:
            self.view = np.dot(np.dot(translate((-avg[0], -avg[1], -avg[2])),
                              rotate(-180, [1,0,0])),
                              translate((avg[0], avg[1], avg[2])))


        self.model = np.eye(4, dtype=np.float32)
        shape = int(self._visSize[1]), int(self._visSize[0])

        # Create the render texture
        self._rendertex = gloo.Texture2D((shape + (4,)), format='rgba', internalformat='rgba32f')
        # self._colorBuffer = gloo.RenderBuffer(self.shape, format='color')
        self._depthRenderBuffer = gloo.RenderBuffer(shape, format='depth')
        #self._depthRenderBuffer.resize(shape, format=gloo.gl.GL_DEPTH_COMPONENT16)

        # Create FBO, attach the color buffer and depth buffer
        self._fbo = gloo.FrameBuffer(self._rendertex)#, self._depthRenderBuffer)

        gloo.set_viewport(0, 0, self.physical_size[0], self.physical_size[1])
        gloo.set_viewport(0, 0, self._visSize[0], self._visSize[1])
        self.projection = ortho(self.bbox[1, 0], self.bbox[0, 0], self.bbox[1, 1], self.bbox[0, 1], 2, 40)

        # Set MVP variables for shaders
        self.program['u_projection'] = self.projection
        self.program['u_model'] = self.model
        self.program['u_view'] = self.view

        gloo.set_clear_color('white')
        gloo.set_state('opaque')

        self.update()

    def on_resize(self, event):

        # TODO - find a better way to set the bounds for the orthographic projection

        gloo.set_viewport(0, 0, self._visSize[0]*2, self._visSize[1]*2)
        self.finalSize = (event.physical_size[0], event.physical_size[1])
        print('event physical size', event.physical_size[0], event.physical_size[1])
        # Zself.projection = ortho(self.box[0, 0], self.box[1, 0], self.box[0, 1], self.box[1, 1], -10, 40)
        self.projection = ortho(self.bbox[1, 0], self.bbox[0, 0],
                                self.bbox[1, 1], self.bbox[0, 1],
                                -self.bbox[1, 2], self.bbox[0, 2])

        self.projection = ortho(self.bbox[0, 0], self.bbox[1, 0],
                                self.bbox[0, 1], self.bbox[1, 1],
                                -1e4,  1e4)

        self.program['u_projection'] = self.projection

    def on_draw(self, event):

        with self._fbo:
            gloo.clear()
            gloo.set_clear_color((0.0, 0.0, 0.0, 0.0))

            gloo.set_viewport(0, 0, *self.finalSize)
            gloo.set_viewport(0, 0, self._visSize[0], self._visSize[1])
            gloo.set_state(blend=True, depth_test=True, polygon_offset_fill=False, cull_face=False)

            self.program.draw('triangles', self.filled_buf)
            # self.rgb = np.copy(self._fbo.read('color')) #_screenshot((0, 0, self.size[0], self.size[1]))  #self._fbo.read('color')
            self.rgb = gloo.read_pixels((0, 0, self._visSize[0], self._visSize[1]), True, out_type='float')
            #self.rgb = _screenshot((0, 0, *self.physical_size))


def projectHeightMap(mesh: trimesh.Trimesh,
                     resolution: float = 0.05,
                     flipDir: bool = False,
                     bbox: np.ndarray = None):

    c = Canvas(mesh, resolution, flipDir, bbox)

    c.show(visible=True)
    c.close()

    return c.rgb[:, :, 1]
