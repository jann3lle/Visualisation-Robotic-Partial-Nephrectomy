import cv2
import vtk
import json
import numpy as np
from pathlib import Path
from sksurgeryvtk.widgets.vtk_overlay_window import VTKOverlayWindow
import sksurgeryvtk.camera.vtk_camera_model as cam_utils  # for set_camera_pose/intrinsics


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

    try:
        rvec = np.array(rvecs[calib_index])
        tvec = np.array(tvecs[calib_index])
        return rvec, tvec
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

# --- keep your helper functions: load_images, load_specific_image, load_calibration_json, etc. ---

def rvec_tvec_to_camera_to_world(rvec, tvec):
    """Convert OpenCV rvec/tvec to 4x4 camera-to-world matrix."""
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    # OpenCV gives object-to-camera, so invert to get camera-to-world
    return np.linalg.inv(T)

def main():
    # Load image paths and image
    left_img_paths = load_images(left_img_dir)
    image = load_specific_image(left_img_paths, img_path_no)
    if image is None:
        return

    # Load calibration JSON
    calib_params = load_calibration_json(calibration_json_path)

    # Create mapping and get extrinsics
    frame_mapping = create_frame_mapping(valid_frame_indices)
    rvec, tvec = get_calibration_for_frame(img_path_no, frame_mapping, calib_params)

    # Get camera intrinsics
    K = calib_params["left_camera_matrix"]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Convert rvec/tvec to camera-to-world
    cam_to_world = rvec_tvec_to_camera_to_world(rvec, tvec)

    # Create VTK overlay window
    window = VTKOverlayWindow(camera_matrix=K, init_pose=False)
    window.set_video_image(image)  # Set background image (BGR)

    # Set calibrated intrinsics
    renderer = window.get_renderer(layer=1)
    vtk_cam = renderer.GetActiveCamera()
    cam_utils.set_camera_intrinsics(
        renderer, vtk_cam,
        width=image.shape[1], height=image.shape[0],
        f_x=fx, f_y=fy, c_x=cx, c_y=cy,
        near=1, far=2000
    )

    # Set calibrated pose
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, cam_to_world[i, j])
    cam_utils.set_camera_pose(vtk_cam, vtk_matrix, opencv_style=True)

    # Create Charuco board actor and add to scene
    board_actor = create_charuco_board_vtk_model()
    # transform = rvec_tvec_to_vtk_transform(rvec, tvec)
    transform = rvec_tvec_to_camera_to_world(rvec, tvec)
    board_actor.SetUserTransform(transform)
    window.add_vtk_actor(board_actor)

    # Start rendering
    window.render()
    window.Start()

if __name__ == "__main__":
    img_path_no = 0
    valid_frame_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 47, 48, 49]
    
    try:
        base_dir = Path(__file__).resolve().parent
    except NameError:
        base_dir = Path.cwd()
    
    left_img_dir = base_dir / 'data' / 'img_dir' / 'Frames_Left'
    results_folder = base_dir / 'data' / 'results' / 'calibrator_new_params'
    calibration_json_path = results_folder / "stereo_calibration_params_new.json"

    main()
