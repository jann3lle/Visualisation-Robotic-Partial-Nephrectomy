# imports
# pyright: reportMissingImports=false
import cv2
import numpy as np
from os import path, makedirs
import errno
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
import math
import sys

# ----> CHANGED TO BELOW: from vtk.vtkIOKitPython import vtkPNGReader
from vtkmodules.vtkIOImage import vtkPNGReader
# scikit-surgery imports to set up the camera from OpenCV calibration
from sksurgeryvtk.camera.vtk_camera_model import set_camera_pose, set_camera_intrinsics
# import sksurgeryopencvpython as skscv

from vtkStereoInteractorStyle import StereoInteractorStyle
import vtkCameraModels
from StereoCalibrationObjects import Mono_calibration, Stereo_calibration, Stereo_rectification


# safe makedir - create directory unless it exists
def safe_makedir(directory):
    try:
        makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

# Read csv 4x4 matrix data into a vtkMatrix4x4 object
def read_vtk_matrix(in_mat_file):
    in_matrix = np.loadtxt(in_mat_file, delimiter=',')
    vtk_mat = vtk.vtkMatrix4x4()
    for ix, iy in np.ndindex(in_matrix.shape):
        vtk_mat.SetElement(ix, iy, in_matrix[ix,iy])
    print("vtkMatrix={}".format(vtk_mat))
    return vtk_mat


# Read csv 4x4 matrix data into a vtkMatrix4x4 object
def write_vtk_matrix(out_mat_file, vtk_mat):
    row = 0
    with open(out_mat_file, 'w') as matfile:
        for row in range(4):
            matfile.write("{}, {}, {}, {}\n".format(
                vtk_mat.GetElement(row, 0), vtk_mat.GetElement(row, 1),
                vtk_mat.GetElement(row, 2), vtk_mat.GetElement(row, 3)))
    print("vtkMatrix={}".format(vtk_mat))

def read_VTKPoly(filename):
    extension = path.splitext(filename)[1]
    if extension.lower() == ".ply":
        reader = vtk.vtkPLYReader()
    elif extension.lower() == ".obj":
        reader = vtk.vtkObjReader()
    elif extension.lower() == ".stl":
        reader = vtk.vtkSTLReader()
    else:
        reader = vtk.vtkPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def render_points_VTK(ren, points, col, opacity=1.0):
    vtk_points = vtk.vtkPoints()
    print("Point 0: ({},{},{})".format(points[0, 0], points[0, 1], points[0, 2]))
    print("len(points[0])={}".format(len(points[0])))
    print("len(points)={}".format(len(points)))
    for i in range(len(points)):
        print("Point: ({},{},{})".format(points[i, 0], points[i, 1], points[i, 2]))
        vtk_points.InsertNextPoint(points[i, 0], points[i, 1], points[i, 2])
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)

    glyph = vtk.vtkGlyph3D()
    try:
        glyph.SetInput(polydata)
    except AttributeError:
        glyph.SetInputData(polydata)
    source = vtk.vtkSphereSource()
    glyph.SetSourceConnection(source.GetOutputPort())
    glyph.Update()

    return render_VTK(ren, glyph.GetOutput(), col, opacity)

def render_VTK(ren, polydata, col, opacity=1.0, texture_image=None):

    polyMapper = vtk.vtkPolyDataMapper()
    polyMapper.SetInputData(polydata)
    polyActor = vtk.vtkActor()
    polyActor.SetMapper(polyMapper)

    polyActor.GetProperty().SetColor(col[0], col[1], col[2])
    polyActor.GetProperty().SetOpacity(opacity)

    if texture_image is not None:
        # Check if texture coordinates are there
        if polydata.GetPointData().GetTCoords() is None:
            tcoord_filter = vtk.vtkTextureMapToPlane()
            tcoord_filter.SetInputData(polydata)
            tcoord_transf = vtk.vtkTransformTextureCoords()
            tcoord_transf.SetScale(10.0,10.0,10.0)
            tcoord_transf.SetInputConnection(tcoord_filter.GetOutputPort())
            tcoord_transf.Update()
            polyMapper.SetInputData(tcoord_transf.GetOutput())
        texture = vtk.vtkTexture()
        texture.SetInputData(texture_image)
        polyActor.SetTexture(texture)

    ren.AddActor(polyActor)
    return polyActor

def get_vtk_image_from_numpy(image_in):
    # Copied from this post: http://vtk.1045678.n5.nabble.com/Image-from-OpenCV-as-texture-td5748969.html
    # Thanks to Addison Elliott
    #
    # Use VTK support function to convert Numpy to VTK array
    # The data is reshaped to one long 2D array where the first dimension is the data and the second dimension is
    # the color channels (1, 3 or 4 typically)
    # Note: Fortran memory ordering is required to import the data correctly
    # Type is not specified, default is to use type of Numpy array
    image_rgb = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)
    imge_flip = np.transpose(image_rgb, (1, 0, 2))
    imge_flip = np.flip(imge_flip, 1)
    dims = imge_flip.shape
    # print("dims = {}".format(dims))
    size = dims[:2]
    channels = dims[-1]
    vtkArray = numpy_to_vtk(imge_flip.reshape((-1, channels), order='F'), deep=False)

    # Create image, set parameters and import data
    vtk_image = vtk.vtkImageData()

    # For vtkImage, dimensions, spacing and origin is assumed to be 3D. VTK images cannot be larger than 3D and if
    # they are less than 3D, the remaining dimensions should be set to default values. The function padRightMinimum
    # adds default values to make the list 3D.
    vtk_image.SetDimensions(size[0], size[1], 1)
    #vtk_image.SetSpacing(padRightMinimum(self.spacing, 3, 1))
    vtk_image.SetOrigin(0, 0, 0)

    # Import the data (vtkArray) into the vtkImage
    vtk_image.GetPointData().SetScalars(vtkArray)
    return vtk_image


def setup_background_image(image_data, background_renderer):
    # Set up the background camera to fill the renderer with the image
    # Create an image actor to display the image
    image_actor = vtk.vtkImageActor()
    image_actor.SetInputData(image_data)
    background_renderer.AddActor(image_actor)

    origin = image_data.GetOrigin()
    spacing = image_data.GetSpacing()
    extent = image_data.GetExtent()

    camera = background_renderer.GetActiveCamera()
    camera.ParallelProjectionOn()

    xc = origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0]
    yc = origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1]
    # xd = (extent[1] - extent[0] + 1) * spacing[0]
    yd = (extent[3] - extent[2] + 1) * spacing[1]
    d = camera.GetDistance()
    camera.SetParallelScale(0.5 * yd)
    camera.SetFocalPoint(xc, yc, 0.0)
    camera.SetPosition(xc, yc, d)

# For ChAruco chessboard calibration images with Oarm - undistorted

# Read in the image number as arg instead?

if len(sys.argv) > 1:
    im_num = sys.argv[1]
else:
    im_num = '001'

# This is for UCL_SERV_CT database
basedir = r"C:\Users\Jann3\OneDrive\Documents\GitHub\Visualisation-Robotic-Partial-Nephrectomy\serv-ct"
# Input directories and files
# rectified_dir = path.join(basedir, 'Rectified')
rectified_dir = path.join(basedir, 'Experiment_1')
# left_image_file_anat = path.join(rectified_dir, 'Left_rectified\\' + im_num + '.png')
# right_image_file_anat = path.join(rectified_dir, 'Right_rectified_colour_corrected\\' + im_num + '.png')
left_image_file_anat = path.join(rectified_dir, 'Left_rectified', im_num + '.png')
# ----> CHANGED right_image_file_anat = path.join(rectified_dir, 'Right_rectified_colour_corrected', im_num + '.png')
right_image_file_anat = path.join(rectified_dir, 'Right_rectified', im_num + '.png')
ct_dir = path.join(basedir, 'CT')
ct_data_dir = path.join(ct_dir, im_num)
CT_endoscope_file = path.join(ct_data_dir, 'Endoscope.vtk')
CT_anatomy_file = path.join(ct_data_dir, 'Anatomy.vtk')
script_dir = path.join(basedir, 'scripts')
CT_anatomy_texture_file = path.join(script_dir, 'LinearTextureBig.png')
# Use Creaform surface and texture
# CT_anatomy_file = path.join(ct_data_dir, 'Creaform_to_CT' + im_num + '.vtk')
# creaform_dir = path.join(basedir, 'Creaform')
# CT_anatomy_texture_file = path.join(creaform_dir, 'PigLiverKidney.png')
CT_to_camera_matrix_file = path.join(ct_data_dir, 'CT_to_Cam_EndoPts.csv')
rectified_calib_dir = path.join(rectified_dir, 'Rectified_calibration')
camera_calib_file = path.join(rectified_calib_dir, im_num + '.json')

# Output directories and files
output_dir = path.join(basedir, r'Experiment_1')
safe_makedir(output_dir)
rectified_output_dir = path.join(output_dir, 'Reference_CT')
safe_makedir(rectified_output_dir)
other_data_output_dir = path.join(rectified_output_dir, "Other")
image_data_output_dir = path.join(other_data_output_dir, im_num)
safe_makedir(image_data_output_dir)
CT_to_camera_from_vtk = path.join(image_data_output_dir, 'CT_to_Cam_Manual.csv')
# Check for existing result from last run
if path.isfile(CT_to_camera_from_vtk):
    CT_to_camera_matrix_file = CT_to_camera_from_vtk
left_depth_dir = path.join(rectified_output_dir, 'DepthL')
safe_makedir(left_depth_dir)
left_depth_image_out = path.join(left_depth_dir, im_num + '.png')
right_depth_dir = path.join(rectified_output_dir, 'DepthR')
safe_makedir(right_depth_dir)
right_depth_image_out = path.join(right_depth_dir, im_num + '.png')
disparity_dir = path.join(rectified_output_dir, 'Disparity')
safe_makedir(disparity_dir)
disparity_image_out = path.join(disparity_dir, im_num + '.png')
left_occlusion_dir = path.join(rectified_output_dir, 'OcclusionL')
safe_makedir(left_occlusion_dir)
left_occlusion_image_out = path.join(left_occlusion_dir, im_num + '.png')
right_occlusion_dir = path.join(rectified_output_dir, 'OcclusionR')
safe_makedir(right_occlusion_dir)
right_occlusion_image_out = path.join(right_occlusion_dir, im_num + '.png')
# Load calibration
stereo_calib = Stereo_rectification()
stereo_calib.load_from_file(camera_calib_file)

L_image_anat = cv2.imread(left_image_file_anat)
R_image_anat = cv2.imread(right_image_file_anat)
w = L_image_anat.shape[1]
h = L_image_anat.shape[0]
# Start with some preprocessing
# Tried CLAHE to brighten the darker areas of the image
# Appeared to introduce artifacts - tried gridsize 8-100
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 100))
# # Also tried histogram equalisation, still not great result
# L_image_LAB = cv2.cvtColor(L_image_anat, cv2.COLOR_BGR2YUV)
# R_image_LAB = cv2.cvtColor(R_image_anat, cv2.COLOR_BGR2YUV)
# L_image_LAB[..., 0] = cv2.equalizeHist(L_image_LAB[..., 0])
# R_image_LAB[..., 0] = cv2.equalizeHist(R_image_LAB[..., 0])
# L_image_anat = cv2.cvtColor(L_image_LAB, cv2.COLOR_YUV2BGR)
# R_image_anat = cv2.cvtColor(R_image_LAB, cv2.COLOR_YUV2BGR)
# Calculate colour equalisation
# Calculate mask for specularities
upper_threshold = 160
lower_threshold = 20
L_image_HSL = cv2.cvtColor(L_image_anat, cv2.COLOR_BGR2HLS)
# Whiteness of specular highlights from L channel
# Use lower threshold to remove very dark regions
# Mask is of the highlights so they can be dilated
mask1 = np.logical_or((L_image_HSL[:, :, 1] > upper_threshold), (L_image_HSL[:, :, 1] < lower_threshold))
kernel = np.array([[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]], np.uint8)
mask1_dilated = cv2.dilate(np.asmatrix(mask1).astype(np.uint8), kernel)
mask1_dilated = cv2.dilate(np.asmatrix(mask1_dilated).astype(np.uint8), kernel)
# Back to boolean
mask1 = mask1_dilated == 0
# Same for the right image
R_image_HSL = cv2.cvtColor(R_image_anat, cv2.COLOR_BGR2HLS)
mask2 = np.logical_or((R_image_HSL[:, :, 1] > upper_threshold), (R_image_HSL[:, :, 1] < lower_threshold))
mask2_dilated = cv2.dilate(np.asmatrix(mask2).astype(np.uint8), kernel)
mask2_dilated = cv2.dilate(np.asmatrix(mask2_dilated).astype(np.uint8), kernel)
# Back to boolean
mask2 = mask2_dilated == 0
# # Calculate simple colour correction based on variances for each image
# meanL = np.mean(L_image_anat[mask1], axis=0)
# stdevL = np.std(L_image_anat[mask1], axis=0)
# meanR = np.mean(R_image_anat[mask2], axis=0)
# stdevR = np.std(R_image_anat[mask2], axis=0)
# # Scale right colour values to match left
# R_image_anat = (R_image_anat - meanR) * stdevL / stdevR + meanL
# R_image_anat = R_image_anat.clip(0, 255).astype(np.uint8)

P1 = stereo_calib.P1
P2 = stereo_calib.P2
Q = stereo_calib.Q

print("P1={}".format(P1))
print("P2={}".format(P2))
print("Q={}".format(Q))


# # Write out the configuration file as yaml
# image_overlay_config_out = path.join(out_dir_anat, "image_overlay_config.yml")
# image_overlay_config_data = cv2.FileStorage(image_overlay_config_out, cv2.FILE_STORAGE_WRITE)
# image_overlay_config_data.write('left_image_file_anat', left_image_file_anat)
# image_overlay_config_data.write('right_image_file_anat', right_image_file_anat)
# image_overlay_config_data.write('CT_anatomy_file', CT_anatomy_file)
# image_overlay_config_data.write('CT_endoscope_file', CT_endoscope_file)
# image_overlay_config_data.write('CT_to_camera_matrix_file', CT_to_camera_matrix_file)
# image_overlay_config_data.write('out_dir_anat', out_dir_anat)
# image_overlay_config_data.write('CT_to_camera_from_vtk', CT_to_camera_from_vtk)
# image_overlay_config_data.release()
#
# Read in the VTK data
CT_anatomy = read_VTKPoly(CT_anatomy_file)
CT_endoscope = read_VTKPoly(CT_endoscope_file)
CT_to_camera_coords = read_vtk_matrix(CT_to_camera_matrix_file)

render_window = vtk.vtkRenderWindow()
render_window.SetSize(w*2, h)

# Create a renderer to display the image in the background
left_background_renderer = vtk.vtkRenderer()
left_background_renderer.SetLayer(0)
left_background_renderer.InteractiveOff()
right_background_renderer = vtk.vtkRenderer()
right_background_renderer.SetLayer(0)
right_background_renderer.InteractiveOff()
render_window.SetNumberOfLayers(2)
render_window.AddRenderer(left_background_renderer)
render_window.AddRenderer(right_background_renderer)
left_background_renderer.SetViewport(0.5, 0.0, 1.0, 1.0)
right_background_renderer.SetViewport(0, 0, 0.5, 1.0)

ren1 = vtk.vtkRenderer()
ren1.SetLayer(1)
render_window.AddRenderer(ren1)

ren2 = vtk.vtkRenderer()
ren2.SetLayer(1)
render_window.AddRenderer(ren2)
ren1.SetViewport(0.5, 0.0, 1.0, 1.0)
ren2.SetViewport(0, 0, 0.5, 1.0)

iren = vtk.vtkRenderWindowInteractor()

iren.SetRenderWindow(render_window)

style = StereoInteractorStyle(iren)
iren.SetInteractorStyle(style)
style.addRenwin(render_window)
style.setLeftRenderer(ren1)
style.setRightRenderer(ren2)
style.left_image = L_image_anat
style.left_image_despeckle = L_image_anat.copy()
style.left_image_despeckle[mask1==0] = 0
style.right_image = R_image_anat
style.right_image_despeckle = R_image_anat.copy()
style.right_image_despeckle[mask2==0] = 0

image_reader_texture = vtk.vtkPNGReader()
image_reader_texture.SetFileName(CT_anatomy_texture_file)
image_reader_texture.Update()
CT_anatomy_actor = render_VTK(ren1, CT_anatomy, (1.0, 1.0, 1.0), 1.0, image_reader_texture.GetOutput())
CT_to_camera = vtk.vtkTransform()
CT_to_camera.SetMatrix(CT_to_camera_coords)
CT_anatomy_actor.SetUserTransform(CT_to_camera)
ren2.AddActor(CT_anatomy_actor)

style.addActor(CT_anatomy_actor)

#surf_pts = []
#with open(surf_pts_filename) as points_file:
#    for line in points_file:
#        row_data = [float(i) for i in line.split(',')]
#        surf_pts.append(row_data)
#surf_pts = np.asmatrix(surf_pts)
#pointsActor = render_points_VTK(ren1, surf_pts, (0, 255, 0), 0.5)
#ren2.AddActor(pointsActor)
#pointsActor.SetUserTransform(CT_to_camera)

print("Setting left camera params ({}=={}), {}, {}", P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2])
style.near_z = 10.0
style.far_z = 400.0
style.w = w
style.h = h
style.l_fx = P1[0, 0]
style.l_fy = P1[1, 1]
style.l_cx = P1[0, 2]
style.l_cy = P1[1, 2]
# Using numpy_to_vtk
left_vtk_image = get_vtk_image_from_numpy(L_image_anat)
setup_background_image(left_vtk_image, left_background_renderer)
style.setLeftProjection()

print("Setting right camera params ({}=={}), {}, {}", P2[0, 0], P2[1, 1], P2[0, 2], P2[1, 2])
style.r_fx = P2[0, 0]
style.r_fy = P2[1, 1]
style.r_cx = P2[0, 2]
style.r_cy = P2[1, 2]

# R2_vtk = vtk.vtkMatrix4x4()
#for i in range(3):
#    for j in range(3):
#        R2_vtk.SetElement(i, j, R1[i, j])
print("Left to right translation: {}".format(P2[0, 3] / P2[0, 0]))
# print("R2_vtk: {}".format(R2_vtk))
# Using numpy_to_vtk
right_vtk_image = get_vtk_image_from_numpy(R_image_anat)
setup_background_image(right_vtk_image, right_background_renderer)
# R2_vtk.SetElement(0, 3, -P2[0, 3] / P2[0, 0])
# print("R2_vtk: {}".format(R2_vtk))
style.t_x = -P2[0, 3] / P2[0, 0]
style.setRightProjection()

iren.Initialize()

render_window.Render()
# Redo camera stuff in case it makes a difference to Z-buffer resolution
style.setLeftProjection()
style.setRightProjection()

#surf_pts = np.loadtxt(fiducial_pts_filename, delimiter=',')
#surf_pts = np.asmatrix(surf_pts)
#pointsActor = render_points_VTK(ren1, surf_pts, (0, 255, 0), 0.5)
#ren2.AddActor(pointsActor)
#pointsActor.SetUserTransform(CT_to_camera)

# Add the occlusion image?
# style.show_occlusion = True
# Diff image is way too slow - needs looking at
style.show_colour_diff = True
# Depth Edge gradient (Laplacian) can be useful
style.show_edge_normals = True
# Move around first then do the calculations
style.render()
iren.Start()

# Save the manual alignment
write_vtk_matrix(CT_to_camera_from_vtk, CT_anatomy_actor.GetUserTransform().GetMatrix())

# Ensure solid rendering to enable depth map
for poly_actor in style.actors:
    poly_actor.SetVisibility(True)
    poly_actor.GetProperty().SetRepresentationToSurface()
    poly_actor.GetProperty().SetOpacity(1.0)
    poly_actor.GetProperty().SetEdgeVisibility(False)
    poly_actor.Modified()
style.render()

print("Making ZBuffer Images")
# Get the z-buffer image
ifilter = vtk.vtkWindowToImageFilter()
ifilter.SetInput(render_window)
#ifilter.ReadFrontBufferOff()
# Trying ZBuffer output from https://vtk.org/Wiki/VTK/Examples/Cxx/Utilities/ZBuffer
ifilter.SetScale(1, 1)
ifilter.SetInputBufferTypeToZBuffer()
ifilter.Modified()
ifilter.Update()
zbuffer = ifilter.GetOutput()

# Extract the left image only (actually placed on the right)
left_depth = vtk.vtkExtractVOI()
left_depth.SetVOI(w, 2*w-1, 0, h-1, 0, 0)
left_depth.SetInputData(zbuffer)
left_depth.SetSampleRate(1, 1, 1)
left_depth.Update()
left_zbuffer = left_depth.GetOutput()

# Extract the left image only (actually placed on the right)
right_depth = vtk.vtkExtractVOI()
right_depth.SetVOI(0, w-1, 0, h-1, 0, 0)
right_depth.SetInputData(zbuffer)
right_depth.SetSampleRate(1, 1, 1)
right_depth.Update()
right_zbuffer = right_depth.GetOutput()

zbuffer_range = left_zbuffer.GetScalarRange()
print("Left depthmap scalar range = {}", zbuffer_range)
right_zbuffer_range = right_zbuffer.GetScalarRange()
print("Right depthmap scalar range = {}", right_zbuffer_range)

################# Convert to numpy array for manipulation
#spacing = left_zbuffer.GetSpacing()

vtk_data = left_zbuffer.GetPointData().GetScalars()
numpy_data = vtk_to_numpy(vtk_data)
dims = numpy_data.shape
numpy_data = numpy_data.reshape(h, w)
depth_image = np.flip(numpy_data,0)

print("numpy_data.shape= {}".format(numpy_data.shape))

# Convert z-buffer values to depth from camera
near_left, far_left = ren1.GetActiveCamera().GetClippingRange()

# Pythonic method - perpective float depth to actual depth
depth_image_d = -2.0 * near_left * far_left / ((depth_image-0.5) * 2.0 * (far_left - near_left) - near_left - far_left)
print("depth_image distance shape= {}".format(depth_image.shape))
depth_image_scaled = 256.0*depth_image_d
cv2.imwrite(left_depth_image_out, depth_image_scaled.astype(np.ushort))

vtk_data_right = right_zbuffer.GetPointData().GetScalars()
numpy_data_right = vtk_to_numpy(vtk_data_right)
dims_right = numpy_data_right.shape
numpy_data_right = numpy_data_right.reshape(h, w)
depth_image_right = np.flip(numpy_data_right, 0)

depth_image_range = (np.amin(depth_image), np.amax(depth_image))
print("Left depth_image_range = {}", depth_image_range)
right_depth_image_range = (np.amin(depth_image_right), np.amax(depth_image_right))
print("Right depth_image_range = {}", right_depth_image_range)

near_right, far_right = ren2.GetActiveCamera().GetClippingRange()
# Pythonic method - perspective OpenGL float depth to actual depth
depth_image_right_d = -2.0 * near_right * far_right / ((depth_image_right-0.5) * 2.0 * (far_right - near_right) - near_right - far_right)
depth_image_scaled_R = 256.0*depth_image_right_d
cv2.imwrite(right_depth_image_out, depth_image_scaled_R.astype(np.ushort))


print("Making Disparity Image")

# Try the Q matrix method -using calculations rather than matrix maths
# cx1 = P1[0, 2]
# cx2 = P2[0, 2]
# tx = 1.0 / Q[3, 2]
# f = Q[2, 3]
# Try the Q matrix method -using calculations rather than matrix maths
cx1 = style.l_cx
cx2 = style.r_cx
tx = style.t_x
f = style.l_fx
print("cx1={}, cx2={}, tx={}, f={}".format(cx1, cx2, tx, f))
print("stereo_calib.Q={}".format(stereo_calib.Q))

disp_image = (tx * f / depth_image_d) - (cx1 - cx2)
# Also calculate disparity from right to left
disp_image_right = -(tx * f / depth_image_right_d) - (cx1 - cx2)
disp_image_scaled = 256.0*disp_image
cv2.imwrite(disparity_image_out, disp_image_scaled.astype(np.ushort))

# Calculate regions of occlusion (left and right)
resample_image = L_image_anat.copy()
resample_image_diff = L_image_anat.copy()
occlusion_image_L = L_image_anat.copy()
occlusion_image_R = L_image_anat.copy()
right_to_left_depth_image = depth_image_d.copy()
right_to_left_depth_diff_image = depth_image_d.copy()
# Define lower and uppper limits ofspecular
spec_lo = np.array([180, 180, 180])
spec_hi = np.array([255, 255, 255])

# Mask image to only select browns
#l_mask = cv2.inRange(L_rect_image_anat, spec_lo, spec_hi)
#r_mask = cv2.inRange(R_rect_image_anat, spec_lo, spec_hi)

#l_max =
# Change image to red where we found brown
#image[mask>0]=(0,0,255)
#mask = L_rect_image_anat > disp.min()
# Small error in depth is ok
epsilon = 3.0
# Should look to see if some of this can be written in a more pythonic way
# Lose the for loop, perhaps?
for x in range(w):
    for y in range(h):
        # Start with the left image
        x_resample = x - int(disp_image[y, x] + 0.5)
        #x_resample = x + int(disp_image[y, x])
        # Check if we are at the far clipping plane
        if depth_image[y, x] > 0.98:
            # Point not visible in the CT - Blue
            colL = (255, 0, 0)
            right_to_left_depth_image[y, x] = 0
            right_to_left_depth_diff_image[y, x] = 0
            resample_image[y, x] = colL
            resample_image_diff[y, x] = colL
        elif x_resample < 0:
                # Point outside image - Yellow
                colL = (0, 255, 255)
                resample_image[y, x] = colL
                resample_image_diff[y, x] = colL
                right_to_left_depth_image[y, x] = 0
                right_to_left_depth_diff_image[y, x] = 0
        else:
            left_z = depth_image_d[y, x]
            right_z = depth_image_right_d[y, x_resample]
            depth_diff = np.abs(right_z - left_z)
            # Z coords left and right should be the same
            if depth_diff < epsilon:
                # Good image point - keep the colour
                colL = L_image_anat[y, x]
                right_to_left_depth_image[y, x] = right_z
                right_to_left_depth_diff_image[y, x] = depth_diff
                colR = R_image_anat[y, x_resample]
                resample_image[y, x] = colR
                # Scale by 10 to make differences visible
                resample_image_diff[y, x] = (10 * np.abs(colR.astype(int) - colL.astype(int))).clip(0, 255)
            else:
                # Left point not seen in right - Red
                colL = (0, 0, 255)
                right_to_left_depth_image[y, x] = 0
                right_to_left_depth_diff_image[y, x] = 0
                resample_image[y, x] = colL
                resample_image_diff[y, x] = colL

        # Now do the right image
        x_resample = x - int(disp_image_right[y, x] + 0.5)
        #x_resample = x + int(disp_image_right[y, x])
        # Check if we are at the far clipping plane
        if depth_image_right[y, x] > 0.98:
            # Point not visible in the CT - Blue
            colR = (255, 0, 0)
        elif x_resample >= w:
            # Point outside image - Yellow
            colR = (0, 255, 255)
        else:
            left_z = depth_image_d[y, x_resample]
            right_z = depth_image_right_d[y, x]
            # Z coords left and right should be the same
            if np.abs(right_z - left_z) < epsilon:
                # Good image point - keep the colour
                colR = R_image_anat[y, x]
            else:
                # Left point not seen in right - Green
                colR = (0, 255, 0)
        occlusion_image_L[y, x] = colL
        occlusion_image_R[y, x] = colR

cv2.imwrite(left_occlusion_image_out, occlusion_image_L)
cv2.imwrite(right_occlusion_image_out, occlusion_image_R)
right_to_left_depth_image_filename = path.join(image_data_output_dir, im_num + '_depth_right_to_left.png')
right_to_left_depth_image_scaled = 256.0 * right_to_left_depth_image
cv2.imwrite(right_to_left_depth_image_filename, right_to_left_depth_image_scaled.astype(np.ushort))
right_to_left_depth_diff_image_filename = path.join(image_data_output_dir, im_num + '_depth_right_to_left_diff.png')
right_to_left_depth_diff_image_scaled = 256.0 * right_to_left_depth_diff_image
cv2.imwrite(right_to_left_depth_diff_image_filename, right_to_left_depth_diff_image_scaled.astype(np.ushort))

# Sanity check resample left to right
resampled_right_filename = path.join(image_data_output_dir, im_num + '_right_to_left.png')
cv2.imwrite(resampled_right_filename, resample_image)

resampled_right_diff_filename = path.join(image_data_output_dir, im_num + '_right_to_left_diff.png')
cv2.imwrite(resampled_right_diff_filename, resample_image_diff)

colors = cv2.cvtColor(L_image_anat, cv2.COLOR_BGR2RGB)

# Display cameras, CT, recon from disp
#window_size = 3
#min_disp = 16
#num_disp = 96
#stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
#                              numDisparities=num_disp,
#                              blockSize=16,
#                              P1=8 * 3 * window_size ** 2,
#                              P2=32 * 3 * window_size ** 2,
#                              disp12MaxDiff=1,
#                              uniquenessRatio=10,
#                              speckleWindowSize=100,
#                              speckleRange=32
#                              )

#print('computing disparity...')
#disp = stereo.compute(L_rect_image_anat, R_rect_image_anat).astype(np.float32) / 16.0
#print('disparity done')
#disp_imagefile = path.join(out_dir_anat, path.splitext(path.basename(left_image_file_anat))[0]+'_disparity_SGBM.png')
#cv2.imwrite(disp_imagefile, disp)

# To show OpenGL recon instead
disp = disp_image
points = cv2.reprojectImageTo3D(disp, Q)
mask = disp > disp.min()
points = points[mask]
colors = colors[mask]



#colors = [val for sublist in colors for val in sublist]
colors_vtk = vtk.vtkUnsignedCharArray()
colors_vtk.SetNumberOfComponents(3)
colors_vtk.SetNumberOfTuples(len(points))
colors_vtk.SetName("Colors")
print("colors[0]={}".format(colors[0]))
print("colors[1]={}".format(colors[1]))
print("colors[1][0]={}".format(colors[1][0]))
for i in range(len(points)):
    colors_vtk.InsertTuple3(i, colors[i][0], colors[i][1], colors[i][2])

vtk_points = vtk.vtkPoints()
for i in range(len(points)):
    vtk_points.InsertNextPoint(points[i, 0], points[i, 1], points[i, 2])


# Code here is to use Dan reconstruction
# For rectified images no need for R,T
Camera1_rect = P1[0:3, 0:3]
#Camera1_rect = Camera1_rect * R1
Camera2_rect = P2[0:3, 0:3]
#Camera2_rect = Camera2_rect * R2
#R_rect = R2 * R1.T
R_rect = np.eye(3)
T_rect = np.matrix([[P2[0, 3] / P2[0, 0]], [0], [0]])
#points = skscv.reconstruct_points_using_stoyanov(L_rect_image_anat, Camera1_rect, R_rect_image_anat, Camera2_rect, R_rect, T_rect, False)
#points_3d = points[:, 0:3]
#projected_points = Camera1_rect * np.asmatrix(points_3d).T
#print('Projected points: {}'.format(projected_points))
#projected_points = projected_points / projected_points[2, None, :]
#print('Projected points norm: {}'.format(projected_points))
#projected_points = projected_points.T

#height, width = L_rect_image_anat.shape[:2]
#projected_image= np.zeros((height,width,3), np.uint8)
# Look up colour from L_image_undistorted
#print('L_rect_image_anat.shape: {}'.format(L_rect_image_anat.shape))
#print('projected_image.shape: {}'.format(projected_image.shape))
#print('points_3d.shape: {}'.format(points_3d.shape))
#print('projected_points.shape: {}'.format(projected_points.shape))
#print('len(projected_points): {}'.format(len(projected_points)))

#vtk_points = vtk.vtkPoints()
#for i in range(len(points_3d)):
#    vtk_points.InsertNextPoint(points_3d[i, 0], points_3d[i, 1], points_3d[i, 2])
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)

#colors = vtk.vtkUnsignedCharArray()
#colors.SetNumberOfComponents(3)
#colors.SetNumberOfTuples(polydata.GetNumberOfPoints())

#out_colours = np.zeros((len(points_3d),3), np.uint8)
#print('out_colours.shape: {}'.format(out_colours.shape))
#print('out_colours: {}'.format(out_colours))
#for i in range(len(projected_points)):
    #print('Projected points[{}] norm: {}'.format(i, projected_points[i]))
    #print('Projected points[i][0] norm: {}'.format(projected_points.item(i, 0)))
#    x = int(round(projected_points.item(i, 0)))
#    y = int(round(projected_points.item(i, 1)))
#    if 0 <= x < L_rect_image_anat.shape[1] and 0 <= y < L_rect_image_anat.shape[0]:
#        out_colours[i] = L_rect_image_anat[y, x]
#        colors.InsertTuple3(i, L_rect_image_anat[y, x][2], L_rect_image_anat[y, x][1], L_rect_image_anat[y, x][0])
#        projected_image[y, x] = out_colours[i]
#    else:
#        out_colours[i] = (0,0,0)
#        colors.InsertTuple3(i, 0.0, 0.0, 0.0)


#vtk_points = vtk.vtkPoints()
#for i in range(len(points)):
#    print("Inserting Point: ({},{},{})".format(points[i, 0], points[i, 1], points[i, 2]))
#    vtk_points.InsertNextPoint(points[i, 0], points[i, 1], points[i, 2])
polydata = vtk.vtkPolyData()
polydata.SetPoints(vtk_points)

polydata.GetPointData().SetScalars(colors_vtk)
recon_centre = np.mean(points[:1000, :], 0)
print("Centroid of points: {}".format(recon_centre))
pointNormalsArray = vtk.vtkDoubleArray()
pointNormalsArray.SetNumberOfComponents(3)
pointNormalsArray.SetNumberOfTuples(polydata.GetNumberOfPoints())

for i in range(len(points)):
    norm = -points[i] / np.linalg.norm(points[i])
    pointNormalsArray.SetTuple(i, norm)

polydata.GetPointData().SetNormals(pointNormalsArray)
glyph = vtk.vtkGlyph3D()
try:
    glyph.SetInput(polydata)
except AttributeError:
    glyph.SetInputData(polydata)

source = vtk.vtkRegularPolygonSource()
source.SetRadius(0.2)
source.SetNumberOfSides(4)
source.SetNormal(1.0, 0.0, 0.0)


glyph.SetSourceConnection(source.GetOutputPort())
glyph.Update()

glyph.SetColorModeToColorByScalar()
glyph.SetVectorModeToUseNormal()
glyph.ScalingOff()
glyph.Update()
pointMapper1 = vtk.vtkPolyDataMapper()
pointMapper1.SetInputConnection(glyph.GetOutputPort())
pointActor1 = vtk.vtkActor()
pointActor1.SetMapper(pointMapper1)
ren = vtk.vtkRenderer()

render_window = vtk.vtkRenderWindow()
render_window.SetSize(500, 1000)

render_window.AddRenderer(ren)
# Place camera half way up looking at the cameras and reconstruction
# Placed away in the Y direction since you expect left/right cameras to be separated in X
sep = np.linalg.norm(recon_centre)
ren.GetActiveCamera().SetPosition(recon_centre[0]/2.0, recon_centre[1]/2.0+3.0*sep, recon_centre[2]/2.0)
ren.GetActiveCamera().SetFocalPoint(recon_centre[0]/2.0, recon_centre[1]/2.0, recon_centre[2]/2.0)
up_v = -recon_centre / sep
ren.GetActiveCamera().SetViewUp(up_v)
ren.AddActor(pointActor1)
iren = vtk.vtkRenderWindowInteractor()

iren.SetRenderWindow(render_window)

style = vtk.vtkInteractorStyleTrackballCamera()
iren.SetInteractorStyle(style)

# Stereo cameras only have translation
R_rect = np.eye(3)
T_rect = [P2[0, 3] / P2[0, 0], 0, 0]

cameras = vtkCameraModels.StereoCameraModel(R_rect, T_rect)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(cameras.get_model())
mapper.SetScalarModeToUseCellData()
mapper.ScalarVisibilityOn();

cameras_actor = vtk.vtkActor()
cameras_actor.SetMapper(mapper)
ren.AddActor(cameras_actor)

# include the CT surface
ren.AddActor(CT_anatomy_actor)

# Add the ensdoscope too, but transparent

CT_endoscope_actor = render_VTK(ren, CT_endoscope, (0.0, 0.0, 1.0), 0.3)
CT_to_camera = vtk.vtkTransform()
CT_endoscope_actor.SetUserTransform(CT_anatomy_actor.GetUserTransform())

ren.SetBackground(0.1, 0.1, 0.1)

# enable user interface interactor

iren.Initialize()

render_window.Render()

iren.Start()

print("Saving rendered image")
# Get the z-buffer image
ifilter = vtk.vtkWindowToImageFilter()
ifilter.SetInput(render_window)
#ifilter.ReadFrontBufferOff()
ifilter.SetScale(1, 1)
ifilter.Modified()
ifilter.Update()
writer = vtk.vtkPNGWriter()
writer.SetInputConnection(ifilter.GetOutputPort())
rendered_image_filename = path.join(image_data_output_dir, im_num + '_rendering.png')
writer.SetFileName(rendered_image_filename)
writer.Write()
sys.exit(0)


