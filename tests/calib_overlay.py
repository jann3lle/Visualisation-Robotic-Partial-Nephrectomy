from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from mpl_toolkits.mplot3d import Axes3D

from sksurgeryimage.calibration.charuco_plus_chessboard_point_detector import CharucoPlusChessboardPointDetector

def load_images(left_img_dir):
    """ Loads all images from left stereo camera"""
        # Note: First loaded image = Frame_1017 (not Frame_9 as 9 > 1)
    left_img_paths = list(left_img_dir.glob("*.png"))
    if not left_img_paths:
         print("No images found in directory")
    return left_img_paths

def load_specific_image(left_img_paths, img_path_no):
    """ Loads specifc image for chessboard overlay """
    if img_path_no < 0 or img_path_no >= len(left_img_paths):
        print("Invalid image index")
        return None

    img_path = left_img_paths[img_path_no]
    
    image = cv2.imread(str(img_path))
    if image is not None:
        print(f"Image {img_path} successfully loaded!")
        return image
    else:
        print(f"Unable to load image: {img_path}")    
        return None

def load_calibration_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Convert lists to np arrays for matrices, rvecs, tvecs
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

def overlay_virtual_charuco(image, rvec, tvec, camera_matrix, dist_coeffs, 
                            charuco_board_cols=19,
                            charuco_board_rows=26,
                            square_length=5.0,
                            marker_length=4.0):
    """
    Overlays a virtual Charuco board on the image, highlighting the true top-left 3D corner in green.
    """
    # Create the Charuco board
    charuco_board = cv2.aruco.CharucoBoard_create(
        squaresX=charuco_board_cols,
        squaresY=charuco_board_rows,
        squareLength=square_length,
        markerLength=marker_length,
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )

    # Get 3D object points
    obj_points = charuco_board.chessboardCorners  # shape (N, 3)

    # Find the "true top-left" corner: min y, then min x
    top_left_index = np.lexsort((obj_points[:, 0], obj_points[:, 1]))[0]
    top_left_corner_3d = obj_points[top_left_index]

    # Project all 3D corners
    img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, camera_matrix, dist_coeffs)
    img_points = img_points.reshape(-1, 2).astype(int)

    # Draw all corners as red dots
    for pt in img_points:
        cv2.circle(image, tuple(pt), 3, (0, 0, 255), -1)  # red

    # Highlight true top-left corner in green
    # Find visually top-left corner (min Y, then min X in image space)
    # Index 0 in obj_points corresponds to the 0,0 corner in board space
    top_left_corner_2d = img_points[0]
    cv2.circle(image, tuple(top_left_corner_2d), 10, (0, 255, 0), -1)  # Green
    cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec, tvec, 30)  # Axis length = 30 mm


    return image, obj_points


def main(left_img_dir):
     print(f"left image directory is: {left_img_dir}")
     left_img_paths = load_images(left_img_dir)
     
     if left_img_paths:
          print("Images found in chosen directory!")
     else:
          print("No images found in directory")
          return
     
     image = load_specific_image(left_img_paths, img_path_no)

     if not calibration_json_path.exists():
         print(f"Calibration file not found: {calibration_json_path}")
     else:
         print(f"Calibration file found: {calibration_json_path}")
     
     calib_params = load_calibration_json(calibration_json_path)
     frame_mapping = create_frame_mapping(valid_frame_indices)
     print(frame_mapping)

     try:
        rvec, tvec = get_calibration_for_frame(img_path_no, frame_mapping, calib_params)
        print("Retrieved rvec:", rvec)
        print("Retrieved tvec:", tvec)
     except ValueError as e:
        print(e)
        return  # Exit early if frame is invalid
     
     reference_image = cv2.imread(str(left_img_paths[img_path_no]), cv2.IMREAD_GRAYSCALE)

     camera_matrix = calib_params["left_camera_matrix"]
     dist_coeffs = calib_params["left_dist_coeffs"]

     # After overlaying your virtual charuco board:
     overlayed_image, obj_points = overlay_virtual_charuco(
        image.copy(),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
        charuco_board_cols=19,
        charuco_board_rows=26,
        square_length=5.0,
        marker_length=4.0
     )

    # Convert BGR to RGB for matplotlib
     overlayed_image_rgb = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
     original_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
     
     # Prepare figure
     fig, axes = plt.subplots(1, 2, figsize=(16,8))
     fig.suptitle(f"Image path: {left_img_paths[img_path_no]}", fontsize = 12)
     
     # Left subplot: Original Image
     axes[0].imshow(original_image_rgb)
     axes[0].set_title("Original Image")
     axes[0].axis('off')

     # Right subplot: Overlayed Image
     axes[1].imshow(overlayed_image_rgb)
     axes[1].set_title("Overlayed Image")
     axes[1].axis('off')

     plt.tight_layout(rect = [0, 0.03, 1, 0.95])
     plt.show()

     print("First few 3D corners of Charuco board:")
     for i, pt in enumerate(obj_points[:5]):
        print(f"Point {i}: {pt}")

     cv2.drawFrameAxes(overlayed_image_rgb, camera_matrix, dist_coeffs, rvec, tvec, 30)  # length=30mm



if __name__ == "__main__":
    # ---- Configuration ----
    img_path_no = 9

    chessboard_rows = 12 # y-axis ; corresponding to the chessboard corners
    chessboard_columns = 17 # x-axis
    square_size = 3 #mm

    valid_frame_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 47, 48, 49]

    # ---- Image directory ----
    try:
        base_dir = Path(__file__).resolve().parent.parent
        print(f"base directory is: {base_dir}")
    except NameError:
        base_dir = Path.cwd()
    
    #left_img_dir = base_dir/ 'data' / 'img_dir' / 'Frames_Left'
    left_img_dir = base_dir/ 'data' / 'results' / 'rectified_imgs'/ 'Left'
    # ---- Save parameters ----
    results_folder = base_dir / 'data' / 'results' / 'calibrator_params'
    results_folder.mkdir(parents=True, exist_ok=True)

    # Define the JSON filename (change to your actual file name)
    calibration_json_filename = "stereo_calibration_params_new.json"

    # Combine path + filename
    calibration_json_path = results_folder / calibration_json_filename
    
    # ---- Functions ----
    main(left_img_dir)