"""
3D Viewer for PySLM using pygfx

This module provides interactive 3D visualization capabilities for Part objects and Trimesh meshes
using the pygfx library with WGPU backend.

"""

import logging
from typing import List, Optional, Union

import numpy as np
import trimesh

from pyslm.core import Part

import pylinalg as la
import pygfx as gfx
import wgpu
from rendercanvas.auto import RenderCanvas, loop

from .view.effects import SSAOPass, OutlinePass
from .view.objects import CheckerboardPlane, CheckerboardPlaneMaterial


class MeshViewer:
    """
    Interactive 3D viewer for visualisation of trimesh meshes and Part objects using pygfx.

    This viewer supports:
    - Multiple meshes in the same scene
    - Face and vertex colours from trimesh visual attributes
    - Interactive camera controls (orbit, pan, zoom)

    .. code-block::
        from pyslm.visualise import MeshViewer
        from pyslm import Part
        viewer = MeshViewer()
        part = Part('MyPart')
        part.setGeometry('model.stl')
        viewer.addPart(part)
        viewer.show()
    """

    def __init__(self, width: int = 1024, height: int = 768, title: str = "PySLM 3D Viewer"):
        """
        Initialise the 3D mesh viewer.

        :param width: Window width in pixels
        :param height: Window height in pixels
        :param title: Window title
        """

        self.width = width
        self.height = height
        self.title = title

        # Create canvas and renderer
        self.canvas = RenderCanvas(size=(width, height), title=title, vsync=False, max_fps=60, update_mode='ondemand')

        self.renderer = gfx.renderers.WgpuRenderer(self.canvas)

        """Register Shader Effect Passes"""

        # Surface-Sampled Ambient Occlusion Pass
        self.ssao = SSAOPass()

        self.ssao.base = 0.1
        self.ssao.area = 0.01
        self.ssao.radius = 0.01
        self.ssao.totalStrength = 1.0
        self.ssao.falloff = 1e-4

        # Outline Pass
        self.outline = OutlinePass(edgeColour=(0.0, 0.0, 0.0), edgeThickness=1.0, lineThickness=2.0,
                                   depthThreshold=0.1, useNormals=False)

        # Add the shader passes to the renderer
        self.renderer.effect_passes = [self.ssao, self.outline]

        # Set the camera options
        self.camera = gfx.PerspectiveCamera(50, width / height)
        self.camera.local.position = (50, 50, 50)
        # Set camera up vector to Z-axis
        self.camera.up = (0, 0, 1)
        # Create scene
        self.scene = gfx.Scene()


        """ Create camera objects """
        # Create camera controller for interaction
        self.controller = gfx.OrbitController(self.camera, register_events=self.renderer)
        self.canvas.add_event_handler(self._onKeyPress, "key_down")

        """ Add ambient and directional lights """
        self.scene.add(gfx.AmbientLight(intensity=0.4))
        directional_light = gfx.DirectionalLight(intensity=0.6)
        directional_light.local.position = (1, 1, 1)
        self.scene.add(directional_light)

        self.setBackgroundColour([255,255,255,2550])  # White background

        # Add axes helper
        self.axes = gfx.AxesHelper(size=10)
        self.axes.local.rotation_matrix = la.mat_from_axis_angle([1, 0, 0], np.pi / 2.0)
        self.scene.add(self.axes)

        """ Flags for the viewer """
        self._wireframe_enabled = False
        self._gridEnabled = False
        self._clippingEnabled = False
        self._SSAOEnabled = True
        self._outlineEnabled = True

        """ Clipping Planes """
        self.clippingPlaneNormal = np.array([0.0, 0.0, 1.0])
        self.clippingPlaneConstant = 10.0
        self.sliceHeight = 10.0  # Current Z-height for slicing
        self.sliceStep = 1.0  # Step size for adjusting slice height

        # Cap meshes for clipping visualization
        self.capMeshes = []
        self.capColor = (0.3, 0.5, 0.9)  # Default blue color for caps
        self.capOpacity = 0.8  # Default opacity for caps

        # Add grid
        self.grid = None

        material = CheckerboardPlaneMaterial(
            color1=(0.92, 0.92, 0.92, 1.0),
            color2=(0.75, 0.75, 0.75, 1.0),
            grid_size=10.0,
            fade_start=300.0,
            fade_end=1000.0,
        )

        # Rotate plane from XZ to XY so it lies in z=0 plane with normal +Z
        self.checkerboardGrid = CheckerboardPlane(material=material)
        self.scene.add(self.checkerboardGrid)

        # Store meshes for reference
        self.meshObjects = []

        # Setup animation callback
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))


    def _onKeyPress(self, event) -> None:
        """Handle keyboard events."""
        if event['event_type'] == 'key_down':
            if event['key'] == 'w':
                # Toggle wireframe mode
                self._wireframe_enabled = not self._wireframe_enabled
                self.setWireframeMode(self._wireframe_enabled)
                logging.info(f"Wireframe mode: {'enabled' if self._wireframe_enabled else 'disabled'}")
            elif event['key'] == 'g':
                self._gridEnabled = not self._gridEnabled
                self.setGridMode(self._gridEnabled)
            elif event['key'] == 'q':
                # close the canvas and window
                self.canvas.close()
            elif event['key'] == 'f':
                self.fitCameraToScene()
            elif event['key'] == 's':
                self._SSAOEnabled = not self._SSAOEnabled
                if self._SSAOEnabled:
                    self.renderer.effect_passes  = [self.ssao] + list(self.renderer.effect_passes)
                else:
                    # remove self.ssao in effect passes
                    self.renderer.effect_passes = [pass_ for pass_ in self.renderer.effect_passes if pass_ != self.ssao]

                self.renderer.request_draw()
            elif event['key'] == 'o':
                self._outlineEnabled = not self._outlineEnabled
                if self._outlineEnabled:
                    self.renderer.effect_passes = list(self.renderer.effect_passes) +  [self.outline]
                else:
                    # remove self.ssao in effect passes
                    self.renderer.effect_passes = [pass_ for pass_ in self.renderer.effect_passes if pass_ != self.outline]

                self.renderer.request_draw()
            elif event['key'] == 'c':
                # Toggle clipping mode
                self._clippingEnabled = not self._clippingEnabled
                self.setClippingMode(self._clippingEnabled)
                logging.info(f"Clipping mode: {'enabled' if self._clippingEnabled else 'disabled'}")
            elif event['key'] == 'ArrowUp':
                # Increase slice height
                if self._clippingEnabled:
                    self.sliceHeight += self.sliceStep
                    self.setClippingPlane(constant=self.sliceHeight)
                    logging.info(f"Slice height: {self.sliceHeight:.2f}")
            elif event['key'] == 'ArrowDown':
                # Decrease slice height
                if self._clippingEnabled:
                    self.sliceHeight -= self.sliceStep
                    self.setClippingPlane(constant=self.sliceHeight)
                    logging.info(f"Slice height: {self.sliceHeight:.2f}")



    def setGridMode(self, enabled: bool = True) -> None:
        """
        Toggle grid helper visibility.

        :param enabled: If `True`, show the grid helper. If `False`, hide the checkerboard grid.
        """

        self.checkerboardGrid.visible = enabled

        logging.info(f"Set grid mode to {enabled}")
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def setClippingMode(self, enabled: bool = True) -> None:
        """
        Toggle clipping plane with caps for all objects in the scene

        When enabled, applies clipping at the specified plane and renders a cap
        at the cross-section using trimesh slicing.

        :param enabled: If `True`, enable clipping with caps. If False, disable clipping.
        """
        self._clippingEnabled = enabled

        if enabled:
            # Enable GPU-based clipping on all meshes
            plane_normal = self.clippingPlaneNormal
            plane_constant = self.clippingPlaneConstant

            for mesh_info in self.meshObjects:
                mesh_obj = mesh_info['mesh']
                # Set clipping plane for pygfx mesh using clipping_planes list
                # Format: [nx, ny, nz, constant] where n is the plane normal
                mesh_obj.material.clipping_planes = [[
                    float(plane_normal[0]),
                    float(plane_normal[1]),
                    float(-plane_normal[2]),
                    float(-plane_constant)
                ]]

            # Generate caps using trimesh slicing
            self._updateSliceCaps()
        else:
            # Disable clipping on all meshes
            for mesh_info in self.meshObjects:
                mesh_obj = mesh_info['mesh']
                mesh_obj.material.clipping_planes = []

            # Remove all cap meshes
            self._clearSliceCaps()

        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def setSliceHeight(self, height: float) -> None:
        """
        Set the Z-height for slicing and update the clipping plane.

        :param height: The Z-height for the slice plane
        """
        self.sliceHeight = height

        # Update clipping plane constant (distance from origin along normal)
        # For a Z-plane at height h: plane equation is n·p + d = 0
        # With normal (0,0,1) and point (0,0,h): d = -h

        # Update clipping if enabled
        if self._clippingEnabled:
            # Update clipping plane on all meshes using clipping_planes list
            for mesh_info in self.meshObjects:
                mesh_obj = mesh_info['mesh']
                mesh_obj.material.clipping_planes = [[
                    float(self.clippingPlaneNormal[0]),
                    float(self.clippingPlaneNormal[1]),
                    float(self.clippingPlaneNormal[2]),
                    float(self.clippingPlaneConstant)
                ]]

            # Update slice caps
            self._updateSliceCaps()
            self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def setClippingPlane(self, normal: np.ndarray = None, constant: float = None) -> None:
        """
        Set the clipping plane position and orientation.

        :param normal: Normal vector of the clipping plane (default: [0, 0, 1])
        :param constant: Distance along the normal (default: 0.0)
        """
        if normal is not None:
            self.clippingPlaneNormal = np.array(normal, dtype=np.float32)
            self.clippingPlaneNormal /= np.linalg.norm(self.clippingPlaneNormal)

        if constant is not None:
            self.clippingPlaneConstant = float(constant)

        # Update existing clipping if enabled
        if self._clippingEnabled:
            # Refresh clipping mode to apply new plane settings
            self.setClippingMode(False)
            self.setClippingMode(True)

        logging.info(f"Set clipping plane: normal={self.clippingPlaneNormal}, constant={self.clippingPlaneConstant}")

    def _updateSliceCaps(self) -> None:
        """
        Generate cap meshes for the current slice plane using trimesh slicing. (CPU Based)
        """

        # Clear existing caps
        self._clearSliceCaps()

        # Slice each mesh and create cap geometry
        plane_origin = self.clippingPlaneNormal * self.sliceHeight
        plane_normal = self.clippingPlaneNormal

        for mesh_info in self.meshObjects:

            trimesh_obj = mesh_info.get('trimesh')

            if trimesh_obj is None:
                continue

            # Use trimesh to slice the mesh and get the cross-section
            slice_2d = trimesh_obj.section(plane_origin=plane_origin,
                                           plane_normal=plane_normal)

            if slice_2d is None:
                continue

            # Get the planar representation with triangulation

            planar, transform = slice_2d.to_planar()
            if len(planar.dangling) > 0:
                continue

            # Triangulate - returns (vertices_2d, faces) as a tuple
            vertices_2d, faces_2d = planar.triangulate()

            if vertices_2d is None or len(vertices_2d) == 0:
                continue

            # Create 3D vertices by adding z=0
            vertices_3d = np.column_stack([vertices_2d, np.zeros(len(vertices_2d))])

            # Flip face winding order to reverse normals (flip in z-direction)
            faces_2d_flipped = faces_2d[:, ::-1]

            # Create a trimesh from 2D triangulation
            cap_mesh_2d = trimesh.Trimesh(vertices=vertices_3d, faces=faces_2d_flipped)

            # Transform back to 3D space at the correct slice plane
            # Use forward transform (not inverse) to go from 2D to 3D
            cap_mesh = cap_mesh_2d.copy()
            cap_mesh.apply_transform(transform)

            # Create pygfx mesh for the cap
            vertices = np.ascontiguousarray(cap_mesh.vertices.astype(np.float32))
            faces = np.ascontiguousarray(cap_mesh.faces.astype(np.int32))

            # Compute normals for the cap
            normals = np.tile(plane_normal, (len(vertices), 1)).astype(np.float32)

            geometry = gfx.Geometry(
                positions=vertices,
                indices=faces,
                normals=normals
            )

            # Use a distinct material for caps
            material = gfx.MeshPhongMaterial(
                color=self.capColor,
                opacity=self.capOpacity,
                side='both'  # Render both sides
            )

            material.alpha_mode = 'blend'

            cap_mesh_obj = gfx.Mesh(geometry, material)

            self.scene.add(cap_mesh_obj)
            self.capMeshes.append(cap_mesh_obj)


    def _clearSliceCaps(self) -> None:
        """
        Remove all cap meshes from the scene.
        """
        for cap_mesh in self.capMeshes:
            self.scene.remove(cap_mesh)
        self.capMeshes.clear()

    def setCapAppearance(self, color: tuple = None, opacity: float = None) -> None:
        """
        Configure the appearance of slice caps.

        :param color: RGB tuple for cap color (0-1 range)
        :param opacity: Opacity value (0-1 range), e.g., 0.8
        """
        if color is not None:
            self.capColor = color
        if opacity is not None:
            self.capOpacity = opacity

        # Refresh caps if clipping is enabled
        if self._clippingEnabled:
            self._updateSliceCaps()
            self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))

    def setWireframeMode(self, enabled: bool = True) -> None:
        """
        Toggle wireframe rendering for all meshes in the scene.

        :param enabled: If True, render all meshes as wireframe. If False, render as solid.
        """
        for mesh_info in self.meshObjects:
            mesh_obj = mesh_info['mesh']
            mesh_obj.material.wireframe = enabled

        logging.info(f"Set wireframe mode to {enabled} for {len(self.meshObjects)} meshes")
        self.canvas.request_draw(lambda: self.renderer.render(self.scene, self.camera))


    def clear(self) -> None:
        """
        Remove all the meshes from the scene
        """

        for mesh_info in self.meshObjects:
            self.scene.remove(mesh_info['mesh'])
        self.meshObjects.clear()


    def addMesh(self, mesh: trimesh.Trimesh,
                name: Optional[str] = None,
                colour: Optional[np.ndarray] = None,
                wireframe: bool = False) -> gfx.Mesh:
        """
        Add a trimesh mesh to the viewer.

        This function extracts vertex and face colour information from the mesh's visual
        attribute if available. If no visual data is present, a default colour is used.

        :param mesh: The trimesh mesh to visualize
        :param name: Optional name for the mesh
        :param colour: Optional override colour as RGBA array (0-1 range) or RGB array
        :param wireframe: If `True`, render as wireframe instead of solid
        :return: The pygfx Mesh object added to the scene
        """

        if mesh is None or not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Invalid trimesh mesh provided")

        mesh.fix_normals()
        # Get vertices and faces
        vertices = np.ascontiguousarray(mesh.vertices.astype(np.float32))
        faces = np.ascontiguousarray(mesh.faces.astype(np.int32))
        normals = np.ascontiguousarray(mesh.vertex_normals.astype(np.float32))

        # Create geometry - indices must be flattened and contiguous
        geometry = gfx.Geometry(
            positions=vertices,
            indices=faces,
            normals=normals
        )

        # Handle colours - priority: override colour > visual face colours > visual vertex colours > default
        if colour is not None:
            # Use provided colour
            if len(colour) == 3:
                colour = np.append(colour, 1.0)  # Add alpha channel

            material = gfx.MeshPhongMaterial(color=colour[:3], opacity=colour[3])
            if colour[3] < 1.0:
                material.alpha_mode = 'blend'
            else:
                material.alpha_mode = 'solid'

        elif hasattr(mesh, 'visual') and hasattr(mesh.visual, 'face_colours'):
            # Try to use face colours from trimesh visual
            try:
                face_colours = mesh.visual.face_colours
                if face_colours is not None and len(face_colours) > 0:
                    # Convert from 0-255 to 0-1 range if needed
                    if face_colours.max() > 1.0:
                        face_colours = face_colours.astype(np.float32) / 255.0
                    else:
                        face_colours = face_colours.astype(np.float32)

                    # Expand face colours to vertex colours
                    # Each face has 3 vertices, so we replicate each face colour 3 times
                    vertex_colours = np.repeat(face_colours, 3, axis=0)
                    geometry.colours = gfx.Buffer(vertex_colours[:, :4])
                    material = gfx.MeshPhongMaterial(color_mode='vertex')
                else:
                    material = gfx.MeshPhongMaterial(color=(0.7, 0.7, 0.7, 1.0))
            except Exception as e:
                logging.warning(f"Could not extract face colours: {e}")
                material = gfx.MeshPhongMaterial(color=(0.7, 0.7, 0.7, 1.0))

        elif hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colours'):
            # Try to use vertex colours from trimesh visual
            try:
                vertex_colours = mesh.visual.vertex_colours
                if vertex_colours is not None and len(vertex_colours) > 0:
                    # Convert from 0-255 to 0-1 range if needed
                    if vertex_colours.max() > 1.0:
                        vertex_colours = vertex_colours.astype(np.float32) / 255.0
                    else:
                        vertex_colours = vertex_colours.astype(np.float32)

                    geometry.colours = gfx.Buffer(vertex_colours[:, :4])
                    material = gfx.MeshPhongMaterial(color='vertex')
                else:
                    material = gfx.MeshPhongMaterial(color=(0.7, 0.7, 0.7, 1.0))
            except Exception as e:
                logging.warning(f"Could not extract vertex colours: {e}")
                material = gfx.MeshPhongMaterial(color=(0.7, 0.7, 0.7, 1.0))
        else:
            # No colour provided and no visual attributes - use default grey
            material = gfx.MeshPhongMaterial(color=(0.7, 0.7, 0.7, 1.0))

        # Set wireframe if requested
        if wireframe:
            material.wireframe = True

        # Create mesh object
        mesh_obj = gfx.Mesh(geometry, material)

        # Store reference (including original trimesh for slicing)
        self.meshObjects.append({
            'mesh': mesh_obj,
            'name': name or f"Mesh {len(self.meshObjects)}",
            'trimesh': mesh  # Store original trimesh for CPU-side slicing
        })

        # Add to scene
        self.scene.add(mesh_obj)

        logging.info(f"Added mesh '{name or 'unnamed'}' with {len(vertices)} vertices and {len(faces)} faces")

        return mesh_obj

    def addPart(self, part: Part, name: Optional[str] = None,
                colour: Optional[np.ndarray] = None,
                wireframe: bool = False) -> Optional[gfx.Mesh]:
        """
        Add a Part object to the viewer.

        Extracts the trimesh geometry from the Part and visualizes it with any
        transformations applied.

        :param part: The Part object to visualize
        :param name: Optional name for the part (uses part.name if not provided)
        :param colour: Optional override colour as RGBA array (0-1 range) or RGB array
        :param wireframe: If True, render as wireframe instead of solid
        :return: The pygfx Mesh object added to the scene, or None if part has no geometry
        """
        if not isinstance(part, Part):
            raise ValueError("Invalid Part object provided")

        # Get the geometry (with transformations applied)
        mesh = part.geometry

        if mesh is None:
            logging.warning(f"Part '{part.name}' has no geometry to visualize")
            return None

        # Use part name if no name provided
        display_name = name or part.name

        # Add the mesh
        return self.addMesh(mesh, name=display_name, colour=colour, wireframe=wireframe)

    def add_parts(self, parts: List[Part],
                  colours: Optional[List[np.ndarray]] = None,
                  wireframe: bool = False) -> List[gfx.Mesh]:
        """
        Add multiple :class:`Part` objects to the viewer.

        :param parts: List of Part objects to visualsze
        :param colours: Optional list of colours (one per part)
        :param wireframe: If `True`, render as wireframe instead of solid
        :return: List of pygfx Mesh objects added to the scene
        """
        mesh_objs = []

        for i, part in enumerate(parts):
            colour = colours[i] if colours and i < len(colours) else None
            mesh_obj = self.addPart(part, colour=colour, wireframe=wireframe)
            if mesh_obj is not None:
                mesh_objs.append(mesh_obj)

        return mesh_objs

    def addMeshes(self, meshes: List[trimesh.Trimesh],
                      names: Optional[List[str]] = None,
                      colours: Optional[List[np.ndarray]] = None,
                      wireframe: bool = False) -> List[gfx.Mesh]:
        """
        Add multiple trimesh meshes to the viewer.

        :param meshes: List of trimesh meshes to visualize
        :param names: Optional list of names (one per mesh)
        :param colours: Optional list of colours (one per mesh)
        :param wireframe: If `True`, render as wireframe instead of solid
        :return: List of pygfx Mesh objects added to the scene
        """
        mesh_objs = []

        for i, mesh in enumerate(meshes):
            name = names[i] if names and i < len(names) else None
            colour = colours[i] if colours and i < len(colours) else None
            mesh_obj = self.addMesh(mesh, name=name, colour=colour, wireframe=wireframe)
            mesh_objs.append(mesh_obj)

        return mesh_objs

    def clear(self):
        """
        Clear all meshes from the scene.
        """
        for mesh_info in self.meshObjects:
            self.scene.remove(mesh_info['mesh'])

        self.meshObjects.clear()
        logging.info("Cleared all meshes from viewer")

    def fitCameraToScene(self):
        """
        Adjust camera to fit all objects in the scene.
        """
        self.camera.show_object(self.scene)

    def printControls(self):
        """
        Print keyboard controls to the console.
        """
        print("\n" + "=" * 60)
        print("VIEWER KEYBOARD CONTROLS")
        print("=" * 60)
        print("Navigation:")
        print("  Mouse Drag    - Rotate camera (orbit)")
        print("  Mouse Wheel   - Zoom in/out")
        print("  Right Drag    - Pan camera")
        print("\nView Controls:")
        print("  'w' - Toggle wireframe mode")
        print("  'g' - Toggle grid visibility")
        print("  's' - Toggle SSAO (Screen Space Ambient Occlusion)")
        print("  'o' - Toggle outline rendering")
        print("  'f' - Fit camera to scene")
        print("\nSlicing Controls:")
        print("  'c'         - Toggle clipping/slicing mode")
        print("  'Arrow Up'  - Increase slice height (when clipping enabled)")
        print("  'Arrow Down'- Decrease slice height (when clipping enabled)")
        print("\nOther:")
        print("  'q' - Quit viewer")
        print("=" * 60 + "\n")

    def show(self) -> None:
        """
        Display the viewer window and start the event loop.

        This is a blocking call that will run until the window is closed.
        """
        # Fit camera to show all objects
        self.fitCameraToScene()

        # Print controls for user reference
        self.printControls()

        # Run the application
        loop.run()

    def setBackgroundColour(self, colour: tuple) -> None:
        """
        Set the background colour of the viewer.

        :param colour: colour as RGB(A) tuple (0-1 range) or colour name string
        """

        # Ensure RGBA format
        if len(colour) == 3:
            colour = tuple(colour) + (1.0,)

        background = gfx.Background.from_color(colour)
        self.scene.add(background)


def viewMesh(mesh: Union[trimesh.Trimesh, List[trimesh.Trimesh]],
                 colour: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
                 wireframe: bool = False,
                 title: str = "Trimesh Viewer") -> MeshViewer:
    """
    Convenience function to quickly visualize one or more trimesh meshes.

    :param mesh: A single trimesh mesh or list of meshes
    :param colour: Optional colour or list of colours
    :param wireframe: If True, render as wireframe
    :param title: Window title
    :return: The MeshViewer instance
    """
    viewer = MeshViewer(title=title)

    if isinstance(mesh, list):
        colours = colour if isinstance(colour, list) else [colour] * len(mesh)
        viewer.addMeshes(mesh, colours=colours, wireframe=wireframe)
    else:
        viewer.addMesh(mesh, colour=colour, wireframe=wireframe)

    viewer.show()
    return viewer


def viewPart(part: Union[Part, List[Part]],
              colour: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
              wireframe: bool = False,
              title: str = "Part Viewer") -> MeshViewer:
    """
    Convenience function to quickly visualize one or more Part objects.

    :param part: A single Part or list of Parts
    :param colour: Optional colour or list of colours
    :param wireframe: If True, render as wireframe
    :param title: Window title
    :return: The`MeshViewer` instance
    """
    viewer = MeshViewer(title=title)

    if isinstance(part, list):
        colours = colour if isinstance(colour, list) else [colour] * len(part)
        viewer.add_parts(part, colours=colours, wireframe=wireframe)
    else:
        viewer.addPart(part, colour=colour, wireframe=wireframe)

    viewer.show()
    return viewer

