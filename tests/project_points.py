import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_images(img_dir):
    left_img_paths = list(img_dir.glob("*.png"))
    if not left_img_paths:
        print("No images found in directory")
    return left_img_paths

def load_specific_image(left_imgs, img_path_no):
    if img_path_no < 0 or img_path_no > len(left_imgs):
        print("Invalid image index")
        return None
    image_path = left_imgs[img_path_no]
    image = cv2.imread(str(image_path))
    return image

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


def project_points(rvec, tvec, K, image):
    # Suppose you have the same Charuco board definition you used for calibration:
    # (num_x, num_y, square_size must match calibration EXACTLY)
    board = cv2.aruco.CharucoBoard_create(
        squaresX=19,
        squaresY=26,
        squareLength=5.0,   
        markerLength=3.0,   
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    )

    # Get the 3D object points of the Charuco corners (in board coords)
    obj_points = board.chessboardCorners   # shape (N, 3)

    # Project them into your camera
    img_points, _ = cv2.projectPoints(obj_points, rvec, tvec, K, None)

    # Draw them on your background image
    vis = image.copy()
    for p in img_points:
        x, y = int(p[0][0]), int(p[0][1])
        cv2.circle(vis, (x, y), 4, (0, 0, 255), -1)

    # Convert BGR -> RGB for matplotlib
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # Show with matplotlib
    plt.figure(figsize=(10, 8))
    plt.imshow(vis_rgb)
    plt.title("Reprojected Charuco")
    plt.axis("off")
    plt.show()

def main():
    # config
    base_dir = Path(__file__).resolve().parent.parent
    img_dir = base_dir / 'data' / 'results' / 'rectified_imgs' / 'Left'
    json_path = base_dir / 'data' / 'results' / 'calibrator_params' / 'stereo_calibration_params_new.json'
    valid_frame_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 47, 48, 49]
    
    img_path_no = 9

    left_imgs = load_images(img_dir)
    image = load_specific_image(left_imgs, img_path_no)
    calib_params = load_calibration_json(json_path)
    frame_mapping = create_frame_mapping(valid_frame_indices)
    rvec, tvec, K = get_calibration_for_frame(img_path_no, frame_mapping, calib_params)
    project_points(rvec, tvec, K, image)

if __name__ == "__main__":
    main()