
from vtk import vtkInteractorStyleTrackballActor, vtkRenderWindowInteractor, vtkMatrix4x4
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import cv2
import sys

# scikit-surgery imports to set up the camera from OpenCV calibration
from sksurgeryvtk.camera.vtk_camera_model import set_camera_pose, set_camera_intrinsics

import numpy as np

## Interactor that moves in Z rather than scaling - uses MouseWheel
## Needs to have RenWin1, RenWin2 and poly_actor assigned
class StereoInteractorStyle(vtkInteractorStyleTrackballActor):

    def __init__(self,parent=None):
        self.parent = vtkRenderWindowInteractor()
        if(parent is not None):
            self.parent = parent
        self.parent.RemoveObservers("KeyPressEvent")
        self.AddObserver("KeyPressEvent",self.key_down)
        self.AddObserver("KeyReleaseEvent",self.key_up)
        self.AddObserver("MouseWheelForwardEvent",
                         self.mouse_wheel_forward_event)
        self.AddObserver("MouseWheelBackwardEvent",
                         self.mouse_wheel_backward_event)
        # Disable VTK rendering
        self.AddObserver("LeftButtonPressEvent", self.doNothing)
        self.AddObserver("LeftButtonReleaseEvent", self.doNothing)
        self.AddObserver("MiddleButtonPressEvent", self.doNothing)
        self.AddObserver("MiddleButtonReleaseEvent", self.doNothing)
        self.AddObserver("RightButtonPressEvent", self.doNothing)
        self.AddObserver("RightButtonReleaseEvent", self.doNothing)
        # self.AddObserver("MouseMoveEvent", self.render)
        self.RenWins = []
        self.actors = []
        self.delta = 1.0
        self.leftRenderer = None
        self.rightRenderer = None
        self.w = 720
        self.h = 576
        self.near_z = 5.0
        self.far_z = 300.0
        self.l_fx = 1.0
        self.l_fy = 1.0
        self.l_cx = self.w / 2
        self.l_cy = self.h / 2
        self.r_fx = 1.0
        self.r_fy = 1.0
        self.r_cx = self.w / 2
        self.r_cy = self.h / 2
        self.t_x = 6.0
        self.left_pose = vtkMatrix4x4()
        self.right_pose = vtkMatrix4x4()
        self.right_pose.SetElement(0, 3, self.t_x)
        self.left_image = None
        self.right_image = None
        self.left_image_despeckle = None
        self.right_image_despeckle = None
        self.show_occlusion = False
        self.show_colour_diff = False
        self.show_edge_normals = False

    def doNothing(self, obj=None, ev=None):
        # A non event to stop mouse interactions
        print("VTK mouse interaction disabled")

    def addRenwin(self, renwin):
        self.RenWins.append(renwin)

    def addActor(self, actor):
        self.actors.append(actor)

    def setLeftRenderer(self, ren):
        self.leftRenderer = ren
        #self.SetCurrentRenderer(ren)

    def setRightRenderer(self, ren):
        self.rightRenderer = ren

    def setLeftProjection(self):
        print('Left fx={}, fy={}, cx={}, cy={}', self.l_fx, self.l_fy, self.l_cx, self.l_cy)
        print('Right Tx = {}, fx={}, fy={}, cx={}, cy={}', self.t_x, self.r_fx, self.r_fy, self.r_cx, self.r_cy)
        set_camera_intrinsics(self.leftRenderer, self.leftRenderer.GetActiveCamera(), self.w, self.h, self.l_fx, self.l_fy,
                              self.l_cx, self.l_cy, self.near_z, self.far_z)
        # print('Left Pose = {}'.format(self.left_pose))
        set_camera_pose(self.leftRenderer.GetActiveCamera(), self.left_pose)
        self.render()

    def setRightProjection(self):
        print('Left fx={}, fy={}, cx={}, cy={}', self.l_fx, self.l_fy, self.l_cx, self.l_cy)
        print('Right Tx = {}, fx={}, fy={}, cx={}, cy={}', self.t_x, self.r_fx, self.r_fy, self.r_cx, self.r_cy)
        set_camera_intrinsics(self.rightRenderer, self.rightRenderer.GetActiveCamera(), self.w, self.h, self.r_fx, self.r_fy,
                              self.r_cx, self.r_cy, self.near_z, self.far_z)
        self.right_pose.SetElement(0, 3, self.t_x)
        # print('Right Pose = {}'.format(self.right_pose))
        set_camera_pose(self.rightRenderer.GetActiveCamera(), self.right_pose)
        self.render()

    def key_down(self, obj, ev):
        key = self.parent.GetKeySym()
        print("Key down = {}".format(key))
        if key.lower() == 't':
            self.parent.TerminateApp()
        if key == 'Escape':
            sys.exit()
        if key == 'Control_L':
            print("fine Z motion".format(key))
            self.delta = 0.1
        elif key == "Left":
            for poly_actor in self.actors:
                poly_transform = poly_actor.GetUserTransform()
                poly_transform.PostMultiply()
                poly_transform.RotateY(-self.delta)
                poly_actor.SetUserTransform(poly_transform)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key == "Right":
            for poly_actor in self.actors:
                poly_transform = poly_actor.GetUserTransform()
                poly_transform.PostMultiply()
                poly_transform.RotateY(self.delta)
                poly_actor.SetUserTransform(poly_transform)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key == "Up":
            for poly_actor in self.actors:
                poly_transform = poly_actor.GetUserTransform()
                poly_transform.PostMultiply()
                poly_transform.RotateX(self.delta)
                poly_actor.SetUserTransform(poly_transform)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key == "Down":
            for poly_actor in self.actors:
                poly_transform = poly_actor.GetUserTransform()
                poly_transform.PostMultiply()
                poly_transform.RotateX(-self.delta)
                poly_actor.SetUserTransform(poly_transform)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key.lower() == "a":
            for poly_actor in self.actors:
                poly_transform = poly_actor.GetUserTransform()
                poly_transform.PostMultiply()
                poly_transform.RotateZ(-self.delta)
                poly_actor.SetUserTransform(poly_transform)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key.lower() == "s":
            for poly_actor in self.actors:
                poly_transform = poly_actor.GetUserTransform()
                poly_transform.PostMultiply()
                poly_transform.RotateZ(self.delta)
                poly_actor.SetUserTransform(poly_transform)
                poly_actor.Modified()
            self.render(obj, ev)
        # elif key.lower() == "x":
        #     self.l_fx = self.l_fx * (1.0 - self.delta / 100.0)
        #     self.setLeftProjection()
        #     self.r_fx = self.r_fx * (1.0 - self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "d":
        #     self.l_fx = self.l_fx * (1.0 + self.delta / 100.0)
        #     self.setLeftProjection()
        #     self.r_fx = self.r_fx * (1.0 + self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "v":
        #     self.r_fx = self.r_fx * (1.0 - self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "g":
        #     self.r_fx = self.r_fx * (1.0 + self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "y":
        #     self.l_fy = self.l_fy * (1.0 - self.delta / 100.0)
        #     self.setLeftProjection()
        #     self.r_fy = self.r_fy * (1.0 - self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "h":
        #     self.l_fy = self.l_fy * (1.0 + self.delta / 100.0)
        #     self.setLeftProjection()
        #     self.r_fy = self.r_fy * (1.0 + self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "u":
        #     self.r_fy = self.r_fy * (1.0 - self.delta / 100.0)
        #     self.setRightProjection()
        # elif key.lower() == "j":
        #     self.r_fy = self.r_fy * (1.0 + self.delta / 100.0)
        #     self.setRightProjection()
        # elif key == "comma" or key == "less":
        #     self.t_x = self.t_x * (1.0 + self.delta / 100.0)
        #     self.setRightProjection()
        # elif key == "period" or key == "more":
        #     self.t_x = self.t_x * (1.0 - self.delta / 100.0)
        #     self.setRightProjection()
        elif key.lower() == "p":
            for poly_actor in self.actors:
                poly_actor.GetProperty().SetRepresentationToPoints()
                poly_actor.Modified()
            self.render(obj, ev)
        elif key.lower() == "i":
            for poly_actor in self.actors:
                evis = poly_actor.GetProperty().GetEdgeVisibility()
                poly_actor.GetProperty().SetEdgeVisibility(not evis)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key.lower() == "o":
            for poly_actor in self.actors:
                poly_actor.SetVisibility(not poly_actor.GetVisibility())
                poly_actor.Modified()
            self.render(obj, ev)
        elif key.lower() == "k":
            for poly_actor in self.actors:
                poly_actor.GetProperty().SetRepresentationToSurface()
                poly_actor.Modified()
            self.render(obj, ev)
        elif key.lower() == "l":
            for poly_actor in self.actors:
                poly_actor.GetProperty().SetRepresentationToWireframe()
                poly_actor.Modified()
            self.render(obj, ev)
        elif key == "equal":
            for poly_actor in self.actors:
                opacity = poly_actor.GetProperty().GetOpacity()
                opacity = opacity + self.delta / 10.0
                if opacity > 1.0:
                    opacity = 1.0
                print("Setting opacity to:{}".format(opacity))
                poly_actor.GetProperty().SetOpacity(opacity)
                poly_actor.Modified()
            self.render(obj, ev)
        elif key == "minus":
            for poly_actor in self.actors:
                opacity = poly_actor.GetProperty().GetOpacity()
                opacity = opacity - self.delta / 10.0
                if opacity < 0.0:
                    opacity = 0.0
                print("Setting opacity to:{}".format(opacity))
                poly_actor.GetProperty().SetOpacity(opacity)
                poly_actor.Modified()
            self.render(obj, ev)

    def key_up(self, obj, ev):
        key = self.parent.GetKeySym()
        print("Key up = {}".format(key))
        if key == 'Shift_L':
            print("coarse Z motion".format(key))
            self.delta = 3.0

    def mouse_wheel_forward_event(self, obj, ev):
        for poly_actor in self.actors:
            poly_transform = poly_actor.GetUserTransform()
            poly_transform.PostMultiply()
            poly_transform.Translate(0.0, 0.0, self.delta)
            poly_actor.SetUserTransform(poly_transform)
            poly_actor.Modified()
        self.render()

    def mouse_wheel_backward_event(self, obj, ev):
        for poly_actor in self.actors:
            poly_transform = poly_actor.GetUserTransform()
            poly_transform.PostMultiply()
            poly_transform.Translate(0.0, 0.0, -self.delta)
            poly_actor.SetUserTransform(poly_transform)
            poly_actor.Modified()
        self.render(obj, ev)

    def render(self, obj=None, ev=None):
        for renwin in self.RenWins:
            renwin.Render()
        if self.show_occlusion:
            self.display_occlusion()
        if self.show_colour_diff:
            self.display_colour_diff()
        if self.show_edge_normals:
            self.display_edge_normals()

    def display_occlusion(self):
        print("Making ZBuffer Images")
        # Get the z-buffer image
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.RenWins[0])
        #ifilter.ReadFrontBufferOff()
        # Trying ZBuffer output from https://vtk.org/Wiki/VTK/Examples/Cxx/Utilities/ZBuffer
        ifilter.SetScale(1, 1)
        ifilter.SetInputBufferTypeToZBuffer()
        ifilter.Modified()
        ifilter.Update()
        zbuffer = ifilter.GetOutput()

        # Extract the left image only (actually placed on the right)
        # Could do this with numpy instead of VTK??
        left_depth = vtk.vtkExtractVOI()
        left_depth.SetVOI(self.w, 2*self.w-1, 0, self.h-1, 0, 0)
        left_depth.SetInputData(zbuffer)
        left_depth.SetSampleRate(1, 1, 1)
        left_depth.Update()
        left_zbuffer = left_depth.GetOutput()
        left_zbuffer_range = left_zbuffer.GetScalarRange()
        print("Right depthmap scalar range = {}", left_zbuffer_range)

        # Extract the right image only (actually placed on the right)
        # Could do this with numpy instead of VTK??
        right_depth = vtk.vtkExtractVOI()
        right_depth.SetVOI(0, self.w - 1, 0, self.h - 1, 0, 0)
        right_depth.SetInputData(zbuffer)
        right_depth.SetSampleRate(1, 1, 1)
        right_depth.Update()
        right_zbuffer = right_depth.GetOutput()
        right_zbuffer_range = right_zbuffer.GetScalarRange()
        print("Right depthmap scalar range = {}", right_zbuffer_range)

        ################# Convert to numpy array for manipulation
        #spacing = left_zbuffer.GetSpacing()

        left_vtk_data = left_zbuffer.GetPointData().GetScalars()
        left_numpy_data = vtk_to_numpy(left_vtk_data)
        # dims = numpy_data.shape
        left_numpy_data = left_numpy_data.reshape(self.h, self.w)
        left_depth_image = np.flip(left_numpy_data, 0)
        print("left_depth_image.shape= {}".format(left_depth_image.shape))
        # Convert z-buffer values to depth from camera
        near_left, far_left = self.leftRenderer.GetActiveCamera().GetClippingRange()
        # Pythonic method - perspective float depth to actual depth
        left_depth_image = -2.0 * near_left * far_left / ((left_depth_image-0.5) * 2.0 * (far_left - near_left) - near_left - far_left)
        left_min, left_max = np.min(left_depth_image), np.max(left_depth_image)
        left_threshold = 0.99 * (left_max - left_min) + left_min
        ret, left_background_image = cv2.threshold(left_depth_image, left_threshold, 1.0, cv2.THRESH_BINARY)
        kernel = np.ones((5, 5), np.uint8)
        # dilate the background to avoid edge pixels
        left_background_image = cv2.dilate(left_background_image, kernel, iterations=2)
        left_background = left_background_image == 1.0
        left_foreground = left_background_image == 0.0
        # left_depth_image[left_background] = 0.0
        left_depth_image_range = (np.min(left_depth_image[left_foreground]), np.max(left_depth_image[left_foreground]))
        print("Left depthmap scalar range = {}", left_depth_image_range)

        ################# Convert to numpy array for manipulation
        #spacing = left_zbuffer.GetSpacing()

        right_vtk_data = right_zbuffer.GetPointData().GetScalars()
        right_numpy_data = vtk_to_numpy(right_vtk_data)
        # dims = right_numpy_data.shape
        right_numpy_data = right_numpy_data.reshape(self.h, self.w)
        right_depth_image = np.flip(right_numpy_data, 0)
        print("numpy_data.shape= {}".format(right_numpy_data.shape))

        # Convert z-buffer values to depth from camera
        near_right, far_right = self.rightRenderer.GetActiveCamera().GetClippingRange()
        # Pythonic method - perpective float depth to actual depth
        right_depth_image = -2.0 * near_right * far_right / ((right_depth_image-0.5) * 2.0 * (far_right - near_right) - near_right - far_right)
        right_min, right_max = np.min(right_depth_image), np.max(right_depth_image)
        right_threshold = 0.99 * (right_max - right_min) + right_min
        ret, right_background_image = cv2.threshold(right_depth_image, right_threshold, 1.0, cv2.THRESH_BINARY)
        #cv2.imshow("Right background", right_background_image)
        #cv2.waitKey(10000)
        # dilate the background to avoid edge pixels
        right_background_image = cv2.dilate(right_background_image, kernel, iterations=2)
        right_background = right_background_image == 1.0
        right_foreground = right_background_image == 0.0
        right_depth_image_range = (np.min(right_depth_image[right_foreground]), np.max(right_depth_image[right_foreground]))
        print("right depth_image shape= {}".format(right_depth_image.shape))

        print("Making Disparity Image")

        # Try the Q matrix method -using calculations rather than matrix maths
        # cx1 = P1[0, 2]
        # cx2 = P2[0, 2]
        # tx = 1.0 / Q[3, 2]
        # f = Q[2, 3]
        # Try the Q matrix method -using calculations rather than matrix maths
        cx1 = self.l_cx
        cx2 = self.r_cx
        tx = self.t_x
        f = self.l_fx
        disparity_left = (tx * f / left_depth_image) - (cx1 - cx2)
        disparity_left_range = (np.min(disparity_left[left_foreground]), np.max(disparity_left[left_foreground]))
        print("disparity_left_range = {}".format(disparity_left_range))
        disparity_left[left_background] = 0
        disp2 = cv2.normalize(disparity_left, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imshow("Left disparity 2", disp2)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        maps = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        print(maps.shape)
        mapx_init = maps[1]
        mapy = maps[0]
        mapx = mapx_init + disparity_left
        #map_background = mapx >= self.w
        #mapx[mapx >= self.w] = self.w-1
        #mapx[disparity_left == 0] = 0
        # remapped_colour = cv2.remap(self.left_image, mapx, mapy, cv2.INTER_LINEAR)
        remapped_depth = cv2.remap(left_depth_image, mapx, mapy, cv2.INTER_LINEAR)
        remapped_background_image = cv2.remap(left_background_image, mapx, mapy, cv2.INTER_NEAREST)
        remapped_background = remapped_background_image == 1
        depth_difference = np.abs(remapped_depth - right_depth_image)
        disp2 = cv2.normalize(depth_difference, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imshow("remapped_depth 1", disp2)
        # depth_difference[right_background] = 0
        # depth_difference[remapped_background] = 0
        # disp2 = cv2.normalize(depth_difference, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # cv2.imshow("remapped_depth 2", disp2)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        # remapped_depth_diff[right_background] = 0
        # Non background min/max
        right_depth_image_range = np.min(right_depth_image[right_foreground]), np.max(right_depth_image[right_foreground])
        print("Right depthmap scalar range = {}", right_depth_image_range)
        depth_difference_range = np.min(depth_difference[right_foreground]), np.max(depth_difference[right_foreground])
        print("depth difference range = {}", depth_difference_range)
        depth_difference_threshold = 20.0
        # cv2.imshow("Resampled diff depth image", depth_difference/depth_difference_threshold)
        # cv2.waitKey(10000)
        # cv2.destroyAllWindows()
        # cv2.imshow("Resampled diff depth image", (left_depth_image - right_depth_image_range[0])/right_depth_image_range[1])
        # cv2.imshow("Resampled diff depth image", (right_depth_image - remapped_depth_diff_range[0])/remapped_depth_diff_range[1])
        right_occlusion_overlay = np.copy(self.right_image)
        right_occluded = depth_difference > depth_difference_threshold
        right_occlusion_overlay[right_occluded] = (255, 255, 0)
        cv2.imshow("Occlusion image", right_occlusion_overlay)

    def display_colour_diff(self):
        print("Making ZBuffer Images")
        # Get the z-buffer image
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.RenWins[0])
        # ifilter.ReadFrontBufferOff()
        # Trying ZBuffer output from https://vtk.org/Wiki/VTK/Examples/Cxx/Utilities/ZBuffer
        ifilter.SetScale(1, 1)
        ifilter.SetInputBufferTypeToZBuffer()
        ifilter.Modified()
        ifilter.Update()
        zbuffer = ifilter.GetOutput()

        # Extract the left image only (actually placed on the right)
        # Could do this with numpy instead of VTK??
        left_depth = vtk.vtkExtractVOI()
        left_depth.SetVOI(self.w, 2 * self.w - 1, 0, self.h - 1, 0, 0)
        left_depth.SetInputData(zbuffer)
        left_depth.SetSampleRate(1, 1, 1)
        left_depth.Update()
        left_zbuffer = left_depth.GetOutput()
        left_zbuffer_range = left_zbuffer.GetScalarRange()
        print("Left ZBuffer scalar range = {}", left_zbuffer_range)

        left_vtk_data = left_zbuffer.GetPointData().GetScalars()
        left_numpy_data = vtk_to_numpy(left_vtk_data)
        # dims = numpy_data.shape
        left_numpy_data = left_numpy_data.reshape(self.h, self.w)
        left_depth_image = np.flip(left_numpy_data, 0)
        print("left_depth_image.shape= {}".format(left_depth_image.shape))
        # Convert z-buffer values to depth from camera
        near_left, far_left = self.leftRenderer.GetActiveCamera().GetClippingRange()
        # Pythonic method - perspective float depth to actual depth
        left_depth_image = -2.0 * near_left * far_left / ((left_depth_image-0.5) * 2.0 * (far_left - near_left) - near_left - far_left)
        #cv2.imshow("Left Depth Image", left_depth_image)

        cx1 = self.l_cx
        cx2 = self.r_cx
        tx = self.t_x
        f = self.l_fx
        disparity_left = (tx * f / left_depth_image) - (cx1 - cx2)
        # disparity_left_range = (np.min(disparity_left[left_foreground]), np.max(disparity_left[left_foreground]))
        # print("disparity_left_range = {}".format(disparity_left_range))

        width = self.left_image.shape[1]
        height = self.left_image.shape[0]
        maps = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        print(maps.shape)
        mapx_init = maps[1]
        mapy = maps[0]
        mapx = mapx_init - disparity_left
        #map_background = mapx >= self.w
        #mapx[mapx >= self.w] = self.w-1
        mapx[disparity_left <= 0] = 0
        remapped_colour = cv2.remap(self.right_image_despeckle, mapx, mapy, cv2.INTER_LINEAR)
        diff_image = np.abs(self.left_image_despeckle.astype(int) - remapped_colour.astype(int)).astype(np.uint8)
        diff_image = 10.0 * diff_image
        diff_image = diff_image.clip(0, 255).astype(np.uint8)
        # cv2.imshow("Remapped colour", remapped_colour)
        cv2.imshow("Difference Image", diff_image)

    def display_edge_normals(self):
        print("Making ZBuffer Images")
        # Get the z-buffer image
        ifilter = vtk.vtkWindowToImageFilter()
        ifilter.SetInput(self.RenWins[0])
        # ifilter.ReadFrontBufferOff()
        # Trying ZBuffer output from https://vtk.org/Wiki/VTK/Examples/Cxx/Utilities/ZBuffer
        ifilter.SetScale(1, 1)
        ifilter.SetInputBufferTypeToZBuffer()
        ifilter.Modified()
        ifilter.Update()
        zbuffer = ifilter.GetOutput()

        # Extract the left image only (actually placed on the right)
        # Could do this with numpy instead of VTK??
        left_depth = vtk.vtkExtractVOI()
        left_depth.SetVOI(self.w, 2 * self.w - 1, 0, self.h - 1, 0, 0)
        left_depth.SetInputData(zbuffer)
        left_depth.SetSampleRate(1, 1, 1)
        left_depth.Update()
        left_zbuffer = left_depth.GetOutput()
        left_zbuffer_range = left_zbuffer.GetScalarRange()
        print("Left ZBuffer scalar range = {}", left_zbuffer_range)

        left_vtk_data = left_zbuffer.GetPointData().GetScalars()
        left_numpy_data = vtk_to_numpy(left_vtk_data)
        # dims = numpy_data.shape
        left_numpy_data = left_numpy_data.reshape(self.h, self.w)
        left_depth_image = np.flip(left_numpy_data, 0)
        print("left_depth_image.shape= {}".format(left_depth_image.shape))
        # Convert z-buffer values to depth from camera
        near_left, far_left = self.leftRenderer.GetActiveCamera().GetClippingRange()
        # Pythonic method - perspective float depth to actual depth
        left_depth_image = -2.0 * near_left * far_left / ((left_depth_image-0.5) * 2.0 * (far_left - near_left) - near_left - far_left)
        # scale to 0-255
        left_depth_image_uint = (255 * (left_depth_image - np.min(left_depth_image)) /  (np.max(left_depth_image) - np.min(left_depth_image))).astype(np.uint8)
        # cv2.imshow("Left Depth Image", left_depth_image_uint)
        # [laplacian]
        # Apply Laplace function
        dst = cv2.Laplacian(left_depth_image_uint.astype(np.uint8), cv2.CV_16S, ksize=5)
        # [laplacian]
        # [convert]
        # converting back to uint8
        abs_dst = cv2.convertScaleAbs(dst)
        # [convert]
        # [display]
        # Overlay on the rendering or on the image
        overlay = self.left_image.copy()
        # copy Laplacian into the green channel
        g = overlay[:, :, 1]
        laplace_threshold = 20
        g[abs_dst > laplace_threshold] = abs_dst[abs_dst > laplace_threshold]
        cv2.imshow("Depth Laplacian", overlay)



