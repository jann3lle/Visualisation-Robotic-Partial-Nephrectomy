import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import sksurgeryimage.calibration.charuco as ch
import sksurgeryimage.calibration.charuco_plus_chessboard_point_detector as pd
import sksurgerycalibration.video.video_calibration_driver_stereo as sc

# --------- STEP 1: Load Images ---------
def load_images(folder):
    files = sorted(glob.glob(f'{folder}/*.jpg'))
    images = [cv2.imread(f) for f in files]
    for f in files:
        print(f"Loaded: {f}")

    return images

# --------- STEP 2: Create Detector & Calibrator ---------
def setup_calibrator(reference_image_path, min_points=50):
    ref_img = cv2.imread(reference_image_path)
    detector = pd.CharucoPlusChessboardPointDetector(ref_img, error_if_no_chessboard=False)
    calibrator = sc.StereoVideoCalibrationDriver(detector, detector, min_points)
    return detector, calibrator

# # --------- STEP 3: Extract Calibration Points from Stereo Image Pairs ---------

def extract_points(calibrator, left_images, right_images, min_points=50):
    valid_pairs = []
    for i, (left, right) in enumerate(zip(left_images, right_images)):
        try:
            n_left, n_right = calibrator.grab_data(left, right)
            print(f"Pair {i}: Left={n_left}, Right={n_right}")

            if n_left >= min_points and n_right >= min_points:
                valid_pairs.append(i)
            else:
                print(f"Pair {i} skipped: not enough points (Left={n_left}, Right={n_right})")

        except Exception as e:
            print(f"Pair {i}: FAILED - {e}")

    return valid_pairs


# --------- STEP 4: Visualize Detected Points ---------

def visualize_detected_points(video_data, delay=0.5):

    left_images = video_data.left_data.images_array
    right_images = video_data.right_data.images_array
    left_points = video_data.left_data.image_points_arrays
    right_points = video_data.right_data.image_points_arrays

    for i, (img_l, img_r, pts_l, pts_r) in enumerate(zip(left_images, right_images, left_points, right_points)):
        img_l_copy = img_l.copy()
        img_r_copy = img_r.copy()

        # Draw left points in red
        if pts_l is not None:
            for pt in pts_l:
                cv2.circle(img_l_copy, tuple(int(v) for v in pt[0]), 4, (0, 0, 255), -1)

        # Draw right points in green
        if pts_r is not None:
            for pt in pts_r:
                cv2.circle(img_r_copy, tuple(int(v) for v in pt[0]), 4, (0, 255, 0), -1)

        # Convert BGR to RGB for matplotlib
        img_l_rgb = cv2.cvtColor(img_l_copy, cv2.COLOR_BGR2RGB)
        img_r_rgb = cv2.cvtColor(img_r_copy, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_l_rgb)
        axes[0].set_title(f'Left {i}')
        axes[0].axis('off')

        axes[1].imshow(img_r_rgb)
        axes[1].set_title(f'Right {i}')
        axes[1].axis('off')

        plt.tight_layout()
        fig.canvas.manager.set_window_title('Press ESC to exit, any other key to continue')

        # Store whether ESC was pressed
        exit_requested = False

        def on_key(event):
            nonlocal exit_requested
            if event.key == 'escape':
                exit_requested = True
            plt.close()  # Close after any keypress

        fig.canvas.mpl_connect('key_press_event', on_key)
        plt.show()

        if exit_requested:
            print("ESC pressed — exiting visualization.")
            break

# --------- STEP 5: Run Calibration ---------

def run_calibration(detector, calibrator, valid_pairs):
    # Debugging
    max_index = len(calibrator.video_data.left_data.images_array) - 1
    valid_pairs = [i for i in valid_pairs if i <= max_index]
    print(f"Filtered valid pairs: {valid_pairs}")

    # Filter out invalid images (images with not enough detected points)
    calibrator.video_data.left_data.images_array = [calibrator.video_data.left_data.images_array[i] for i in valid_pairs]
    calibrator.video_data.right_data.images_array = [calibrator.video_data.right_data.images_array[i] for i in valid_pairs]

    calibrator.video_data.left_data.image_points_arrays = [calibrator.video_data.left_data.image_points_arrays[i] for i in valid_pairs]
    calibrator.video_data.right_data.image_points_arrays = [calibrator.video_data.right_data.image_points_arrays[i] for i in valid_pairs]

    calibrator.video_data.left_data.ids_arrays = [calibrator.video_data.left_data.ids_arrays[i] for i in valid_pairs]
    calibrator.video_data.left_data.object_points_arrays = [calibrator.video_data.left_data.object_points_arrays[i] for i in valid_pairs]

    calibrator.video_data.right_data.ids_arrays = [calibrator.video_data.right_data.ids_arrays[i] for i in valid_pairs]
    calibrator.video_data.right_data.object_points_arrays = [calibrator.video_data.right_data.object_points_arrays[i] for i in valid_pairs]

    # Initial calibration
    reproj_err, recon_err, _ = calibrator.calibrate()
    print(f"Initial Reprojection Error: {reproj_err}")
    print(f"Initial Reconstruction Error: {recon_err}")

    # Run iterative refinement
    ref_img = ch.make_charuco_with_chessboard()
    reference_ids, _, reference_image_points = detector.get_points(ref_img)

    reproj_err, recon_err, _ = calibrator.iterative_calibration(
        number_of_iterations=2,
        reference_ids=reference_ids,
        reference_image_points=reference_image_points,
        reference_image_size=(ref_img.shape[1], ref_img.shape[0])
    )

    print(f"Final Reprojection Error (iterative): {reproj_err}")
    print(f"Final Reconstruction Error (iterative): {recon_err}")

    return reproj_err, recon_err


# --------- STEP 6: Save Results ---------
def save_results(calibrator, output_dir, basename="stereo_calib"):
    os.makedirs(output_dir, exist_ok=True)
    calibrator.save_params(output_dir, basename)
    calibrator.save_data(output_dir, basename)
    print(f"Saved results to: {output_dir}")

# --------- STEP 7: Visualize Reference Pattern Detection ---------
def visualize_reference_pattern(detector):
    ref_img = ch.make_charuco_with_chessboard()
    _, _, ref_points = detector.get_points(ref_img)

    plt.imshow(cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB))
    plt.title("Detected Pattern (Charuco + Chessboard)")
    plt.scatter([p[0] for p in ref_points], [p[1] for p in ref_points], c='red', s=10)
    plt.show()

# --------- MAIN WORKFLOW ---------
def main():
    # Step 1: Load stereo image sequences
    left_images = load_images('tests/data/ChAruco_LR_frames_Steve_Axis_Tests/ExtractedFrames_L')
    right_images = load_images('tests/data/ChAruco_LR_frames_Steve_Axis_Tests/ExtractedFrames_R')
   
    # Step 2: Set up point detector and stereo calibrator
    detector, calibrator = setup_calibrator(
        'tests/data/2020_01_20_storz/pattern_4x4_19x26_5_4_with_inset_9x14.png'
    )

    # Step 3: Detect and collect valid calibration points
    valid_pairs = extract_points(calibrator, left_images, right_images)

    # Step 4: Visual feedback
    visualize_detected_points(calibrator.video_data)

    # Step 5: Calibrate stereo system
    run_calibration(detector, calibrator, valid_pairs)

    # Step 6: Save calibration results
    save_results(calibrator, {results_folder})

    # Step 7: Optional pattern detection visualization
    visualize_reference_pattern(detector)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    print(f"Base directory is: {base_dir}")
    results_folder = base_dir / 'data'/'results'/'calibration_results'
    results_folder.mkdir(parents=True, exist_ok=True)
    main()
