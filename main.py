import cv2
import vtk
import json
import numpy as np
from pathlib import Path
import sys
from PySide6.QtWidgets import QApplication
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import sksurgeryvtk.widgets.vtk_base_calibrated_window as bcw


def load_images(left_img_dir):
    """ Loads all images from left stereo camera"""
    # Note: First loaded image = Frame_1017 (not Frame_9 as 9 > 1)
    left_img_paths = list(left_img_dir.glob("*.png"))
    if not left_img_paths:
        print("No images found in directory")
    return left_img_paths

def load_specific_image(left_img_paths, img_path_no):
    """ Loads specific image for chessboard overlay"""
    if img_path_no < 0 or img_path_no >= len(left_img_paths):
        print("Invalid image index")
        return None
    
    img_path = left_img_paths[img_path_no]

    image = cv2.imread(str(img_path))
    if image is not None:
        print(f"Image {img_path} successfully loaed!")
        return image
    else:
        print(f"Unable to load image: {img_path}")
        return None

def load_calibration_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert lists to np arrays
    calib_params = {
        "left_camera_matrix": np.array(data["left_camera_matrix"]),
        "left_dist_coeffs": np.array(data["left_dist_coeffs"]),
        "right_camera_matrix": np.array(data["right_camera_matrix"]),
        "right_dist_coeffs": np.array(data["right_dist_coeffs"]),
        "left_rvecs": [np.array(r) for r in data["left_rvecs"]],
        "left_tvecs": [np.array(t) for t in data["left_tvecs"]],
        "right_rvecs": [np.array(r) for r in data["right_rvecs"]],
        "right_tvecs": [np.array(t) for t in data["right_tvecs"]],
        "valid_frame_indices": data["valid_frame_indices"]
    }
    return calib_params

def create_frame_mapping(valid_frame_indices):
    """
    Create a dictionary mapping original frame numbers
    to calibration data indices.

    Parameters:
        valid_frame_indices (list): List of frame numbers used in calibration.

    Returns:
        dict: Mapping {original_frame_number: calibration_index}
    """
    mapping = {frame_num: idx for idx, frame_num in enumerate(valid_frame_indices)}
    return mapping

def get_calibration_for_frame(requested_frame, frame_mapping, calib_params):
    """
    Safely retrieves the rvec and tvec for a valid image frame.

    Parameters:
        requested_frame (int): Original image frame number (e.g. 11)
        frame_mapping (dict): Maps original frame numbers to calibration indices
        calib_params (dict): Loaded calibration data (from your JSON)

    Returns:
        rvec (np.ndarray): Rotation vector for the frame
        tvec (np.ndarray): Translation vector for the frame

    Raises:
        ValueError: If the requested frame is not valid (not in the mapping)
    """
    if requested_frame not in frame_mapping:
        raise ValueError(f"Frame {requested_frame} is invalid (not used in calibration).")

    calib_index = frame_mapping[requested_frame]
    print(f"Current Frame Index: {requested_frame}, New Frame Index: {calib_index}")
    rvecs = calib_params["left_rvecs"]
    tvecs = calib_params["left_tvecs"]
    K = calib_params["left_camera_matrix"]


    try:
        rvec = np.array(rvecs[calib_index])
        tvec = np.array(tvecs[calib_index])
        K = np.array(K).reshape((3, 3)) 
        return rvec, tvec, K
    except IndexError:
        raise IndexError(f"Calibration index {calib_index} out of bounds for rvecs/tvecs.")


def create_charuco_board_vtk_model(num_x=19, num_y=26, square_size=5.0):
    """
    Create a flat chessboard/Charuco board model in VTK.
    num_x, num_y: number of squares in x and y (like your charuco)
    square_size: size of each square in mm or your unit
    """
    append_filter = vtk.vtkAppendPolyData()

    for i in range(num_x):
        for j in range(num_y):
            # Create black/white squares in a checker pattern for visualisation
            if (i + j) % 2 == 0:
                square = vtk.vtkCubeSource()
                square.SetXLength(square_size)
                square.SetYLength(square_size)
                square.SetZLength(0.5)  # thin board
                square.SetCenter(i * square_size + square_size / 2, j * square_size + square_size / 2, 0)
                square.Update()
                append_filter.AddInputData(square.GetOutput())

    append_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(append_filter.GetOutput())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 1, 1)  # white board

    return actor

# def create_charuco_board_vtk_model(num_x=19, num_y=26, square_size=5.0):
#     """
#     Create a flat Charuco board in VTK with correct dimensions.
#     The origin is at the top-left corner (like OpenCV Charuco).
#     """
#     append_filter = vtk.vtkAppendPolyData()

#     for i in range(num_x):
#         for j in range(num_y):
#             square = vtk.vtkCubeSource()
#             square.SetXLength(square_size)
#             square.SetYLength(square_size)
#             square.SetZLength(0.1)  # thinner board
#             square.SetCenter(
#                 i * square_size + square_size / 2,
#                 j * square_size + square_size / 2,
#                 0
#             )
#             square.Update()
#             append_filter.AddInputData(square.GetOutput())

#     append_filter.Update()

#     mapper = vtk.vtkPolyDataMapper()
#     mapper.SetInputData(append_filter.GetOutput())

#     actor = vtk.vtkActor()
#     actor.SetMapper(mapper)
#     actor.GetProperty().SetColor(1, 1, 1)  # default white, can texture later

#     return actor

def rvec_tvec_to_vtk_transform(rvec, tvec):
    # Convert OpenCV rvec/tvec to REUSABLE vtkTransform
    R, _ = cv2.Rodrigues(rvec)
    transform = vtk.vtkTransform()
    mat4 = vtk.vtkMatrix4x4()

    # OpenCV rotation matrix and translation vector to 4x4 matrix
    for i in range(3):
        for j in range(3):
            mat4.SetElement(i, j, R[i, j])
        mat4.SetElement(i, 3, tvec[i])

    # Bottom row
    mat4.SetElement(3, 0, 0)
    mat4.SetElement(3, 1, 0)
    mat4.SetElement(3, 2, 0)
    mat4.SetElement(3, 3, 1)

    transform.SetMatrix(mat4)
    return transform

def show_chessboard_in_vtk(rvec, tvec, K, image):
    """ 
    Show the Charuco board in VTKOverlayWindow with correct intrinsics, pose and background.
    rvec, tvec: OpenCV extrinsics (object to camera)
    K: 3x3 LEFT camera intrinsic matrix
    image: LEFT OpenCV BGR image for background
    """

    # ---- Create ChAruco board actor ----
    board_actor = create_charuco_board_vtk_model()
    # board_actor.SetScale(3, 3, 1) 
    # bounds = board_actor.GetBounds()
    # print("Board bounds:", bounds)

    transform = rvec_tvec_to_vtk_transform(rvec, tvec)
    #board_actor.SetUserTransform(transform)

    # ---- Create overlay window ----
    window = VTKOverlayWindow(camera_matrix = K, init_pose = False)
    window.GetRenderWindow().SetNumberOfLayers(5)

   # ---- Set the background image frame to layer 0 ----
    # vtk_img = numpy_to_vtk_image_data(image)
    # bg_actor = vtk.vtkImageActor()
    # bg_actor.SetInputData(vtk_img)
    # window.add_vtk_actor(bg_actor, layer=0)
    window.set_video_image(image)


    # ---- Get renderer and camera for layer 1
    renderer = window.get_renderer(layer = 1)
    vtk_cam = renderer.GetActiveCamera()

    # ---- Extract intrinsics ----
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # ---- Set intrinsics ----
    cm.set_camera_intrinsics(renderer, vtk_cam, width=image.shape[1], height=image.shape[0],
                             f_x=fx, f_y=fy, c_x=cx, c_y=cy, near=1, far=1000)

    # ---- Convert rvec/tvec (oc - object to camera) into camera to world (co) ----
    R, _ = cv2.Rodrigues(rvec)
    T_oc = np.eye(4)
    T_oc[:3, :3] = R
    T_oc[:3, 3] = tvec.flatten()
    T_co = np.linalg.inv(T_oc) # camera-to-object

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, T_co[i,j])


    cm.set_camera_pose(vtk_cam, vtk_matrix, opencv_style=True)

    # ---- Add actors to scene ----
    window.add_vtk_actor(board_actor, layer=1)
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(30,30,30)
    window.add_vtk_actor(axes, layer=1)

    # ---- Start rendering ----
    window.GetRenderWindow().Render()
    # window.GetRenderWindow().SetSize(image.shape[1], image.shape[0])  
    window.show()
    # window.Start()
    return window


def main():
    left_img_paths = load_images(left_img_dir)

    image = load_specific_image(left_img_paths, img_path_no)
    calib_params = load_calibration_json(calibration_json_path)

    frame_mapping = create_frame_mapping(valid_frame_indices)
    print(frame_mapping)

    try:
        rvec, tvec, K = get_calibration_for_frame(img_path_no, frame_mapping, calib_params)
        print(f"Retrieved rvec: {rvec}")
        print(f"Retrieved tvec: {tvec}")
        print(f"Retrieved cam mtx: {K}")
    except ValueError as e:
        print(e)
        return  # Exit early if frame is invalid

    # Visualise in vtk
    window = show_chessboard_in_vtk(rvec, tvec, K, image)
    print("VTK window should be visible now")

    app.exec()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import sksurgeryvtk.camera.vtk_camera_model as cm
    from sksurgeryvtk.widgets.vtk_overlay_window import VTKOverlayWindow
    
    # -- -- Config ----
    img_path_no = 9
    valid_frame_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 47, 48, 49]
    
    try:
        base_dir = Path(__file__).resolve().parent
        print(f"Base directory is: {base_dir}")
    except NameError:
        base_dir = Path.cwd()
    
    left_img_dir = base_dir / 'data' / 'img_dir' / 'Frames_Left'
    results_folder = base_dir / 'data' / 'results' / 'calibrator_new_params'
    calibration_json_path = results_folder / "stereo_calibration_params_new.json"

    
    main()
