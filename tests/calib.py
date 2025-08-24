from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import json
import numpy as np


import sksurgerycalibration.video.video_calibration_driver_stereo as sc
import sksurgeryimage.calibration.charuco as ch
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector as pd


def load_left_images(left_img_dir):
    left_img_paths = sorted(left_img_dir.glob("*.png"))
    if not left_img_paths:
        print(f"No images found in {left_img_dir}")
        return []
    left_images = []
    for left_img_path in left_img_paths:
        left_img = cv2.imread(str(left_img_path))
        if left_img is None:
            print(f"Failed to load Left image {left_img_path}")
        else:
            left_images.append(left_img)
            print(f"Loaded Left: {left_img_path.name}")
    return left_images

def load_right_images(right_img_dir):
    right_img_paths = sorted(right_img_dir.glob("*.png"))
    if not right_img_paths:
        print(f"No images found in {right_img_dir}")
        return []
    
    right_images = []
    for right_img_path in right_img_paths:
        right_image = cv2.imread(str(right_img_path))
        if right_image is None:
            print(f"Failed to load Right image {right_img_path.name}")
        else:
            right_images.append(right_image)
            print(f"Loaded Right: {right_img_path.name}")
    return right_images

def setup_calibrator(ref_img_path, min_points=50):
    # Setup detector and a temporary calibrator (not used for final data storage)
    ref_img = cv2.imread(str(ref_img_path))
    # ChAruco detection
    detector = pd.CharucoPlusChessboardPointDetector(
        reference_image=ref_img,
        error_if_no_chessboard=False,
        error_if_no_charuco=False,  # optionally True if you want it strict
        number_of_charuco_squares=(19, 26),  # adjust to your board
        size_of_charuco_squares=(5, 4),      # (outer square, inner square) in mm - NEED TO CONFIRM 
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250),
        use_chessboard_inset=True,           # use inset chessboard if you have one
        number_of_chessboard_squares=(11, 16),
        chessboard_square_size=3,
        )

    calibrator = sc.StereoVideoCalibrationDriver(detector, detector, min_points) # 2 detectors for left and right cameras
    
    return detector, calibrator

def detect_points_only(calibrator, left_image, right_image):
    left_ids, _, _ = calibrator.left_point_detector.get_points(left_image)
    right_ids, _, _ =calibrator.right_point_detector.get_points(right_image)

    left_pts = left_ids.shape[0] if left_ids is not None else 0
    right_pts = right_ids.shape[0] if right_ids is not None else 0
    return left_pts, right_pts


def extract_valid_frame_indices(calibrator,left_images, right_images, min_points=50):
    valid_frame_indices = []
    for idx, (l_img, r_img) in enumerate(zip(left_images, right_images)):
        left_pts, right_pts = detect_points_only(calibrator, l_img, r_img)
        if left_pts >= min_points and right_pts >= min_points:
            valid_frame_indices.append(idx)
            print(f"Frame {idx}: Left points = {left_pts}, Right points = {right_pts}")
        else:
            print(f"Frame {idx} skipped - Not enough points")
    return valid_frame_indices

def run_calibration_and_save(detector, valid_idxs, left_images, right_images, min_points=50):
    calibrator = sc.StereoVideoCalibrationDriver(detector, detector, min_points)

    for idx in valid_idxs:
        l_img, r_img = left_images[idx], right_images[idx]
        success = calibrator.grab_data(l_img, r_img)
        if success:
            print(f"✅ Frame {idx} added to calibrator")
        else:
            print(f"❌ grab_data failed for frame {idx}")

    num_left = len(getattr(calibrator.video_data.left_data, "images_array", []))
    num_right = len(getattr(calibrator.video_data.right_data, "images_array", []))
    print(f"📊 Calibrator holds {num_left} left and {num_right} right images")

    if num_left == 0 or num_right == 0:
        print("❌ No valid frames to calibrate")
        return None

    reproj_err, recon_err, calib_params = calibrator.calibrate()
    print(f"✅ Calibration complete: Reprojection error = {reproj_err}, Reconstruction error = {recon_err}")

    if calib_params is None:
        print("Calibration failed or no valid frames found. Exiting.")
    
    print("Attributes of calib_params:")
    print(dir(calib_params))

    return calib_params

    
def visualize_valid_pairs_matplotlib(left_images, right_images, valid_idxs):
    for idx in valid_idxs:
        left_img = left_images[idx]
        right_img = right_images[idx]

        # Convert BGR (OpenCV) to RGB for correct color display in matplotlib
        left_img_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(left_img_rgb)
        axes[0].set_title(f"Left Image - Frame {idx}")
        axes[0].axis('off')

        axes[1].imshow(right_img_rgb)
        axes[1].set_title(f"Right Image - Frame {idx}")
        axes[1].axis('off')

        plt.tight_layout()
        fig.canvas.manager.set_window_title('Press ESC to exit, or any other key to continue')

        exit_requested = False

        def on_key(event):
            nonlocal exit_requested
            if event.key == 'escape':
                exit_requested = True
            plt.close()

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()
        
        if exit_requested:
            print("ESC pressed - exiting visualisation")
            break

def save_calib_params_json(calib_params, filename, valid_frame_indices):
    # Extract parameters
    data = {
        'left_camera_matrix': calib_params.left_params.camera_matrix.tolist(),
        'left_dist_coeffs': calib_params.left_params.dist_coeffs.tolist(),
        'right_camera_matrix': calib_params.right_params.camera_matrix.tolist(),
        'right_dist_coeffs': calib_params.right_params.dist_coeffs.tolist(),
        'left_rvecs': [rv.tolist() for rv in calib_params.left_params.rvecs],
        'left_tvecs': [tv.tolist() for tv in calib_params.left_params.tvecs],
        'right_rvecs': [rv.tolist() for rv in calib_params.right_params.rvecs],
        'right_tvecs': [tv.tolist() for tv in calib_params.right_params.tvecs],
        'left_to_right_r': calib_params.l2r_rmat.tolist(),
        'left_to_right_t': calib_params.l2r_tvec.tolist(),
        'essential_matrix': calib_params.essential.tolist(),
        'fundamental_matrix': calib_params.fundamental.tolist()
    }
    
    if valid_frame_indices is not None:
        data['valid_frame_indices'] = valid_frame_indices
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Calibration parameters saved to {filename}")
    

def main():
    left_images = load_left_images(left_img_dir)
    right_images = load_right_images(right_img_dir)

    pattern_path = base_dir/ 'scikit-surgerycalibration' / 'tests' / 'data' / '2020_01_20_storz' / 'pattern_4x4_19x26_5_4_with_inset_9x14.png'
    if not pattern_path.exists():
        raise FileNotFoundError(f"Pattern image not found: {pattern_path}")
    
    detector, calibrator = setup_calibrator(pattern_path)
    print(f"detector:{detector}")
    print(f"calibrator: {calibrator}")

    valid_idxs = extract_valid_frame_indices(calibrator, left_images, right_images, min_points=50)
    print(f"Valid frame indices = {valid_idxs}")

    calib_params = run_calibration_and_save(detector, valid_idxs, left_images, right_images, min_points=50)

    visualize_valid_pairs_matplotlib(left_images, right_images, valid_idxs)

    if calib_params is not None:
        results_folder.mkdir(parents=True, exist_ok=True)
        save_calib_params_json(calib_params, str(filename), valid_idxs)
    else:
        print("Calibration parameters not saved because calibration failed.")


if __name__ == "__main__":
    # ----- Config -----
    try:
        base_dir = Path(__file__).resolve().parent.parent
        print(f"Base directory is: {base_dir}")
    except NameError:
        base_dir = Path.cwd()
    
    left_img_dir = base_dir / 'data' / 'img_dir' / 'Frames_Left'
    right_img_dir = base_dir / 'data' / 'img_dir' / 'Frames_Right'
    
    results_folder = base_dir / 'data' / 'results' / 'calibrator_new_params'
    filename = results_folder / "stereo_calibration_params_new.json"

    main()
