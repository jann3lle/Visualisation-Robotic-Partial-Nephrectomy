import os
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
CALIB_DIR = "tests/jannelle/calibration_results"
LEFT_IMAGE_DIR = "tests/data/ChAruco_LR_frames_Steve_Axis_Tests/ExtractedFrames_L"
BOARD_SQUARE_SIZE = 3 # Calibration built assuming mm

# Choose the exact image filename you want to process:
SPECIFIC_IMAGE_NAME = "Frame_011.jpg"  # Change to the exact filename

def load_images(folder):
    paths = sorted(glob.glob(os.path.join(folder, '*.jpg')))
    images = {os.path.basename(p): cv2.imread(p) for p in paths}
    return images, paths

def find_image_for_extrinsics(images, frame_num):
    for name in images.keys():
        if f"_{frame_num:03d}." in name:
            return name
    return None

def load_intrinsics(calib_dir, basename='stereo_calib'):
    K = np.loadtxt(os.path.join(calib_dir, f"{basename}.left.intrinsics.txt"))
    dist = np.loadtxt(os.path.join(calib_dir, f"{basename}.left.distortion.txt"))  # If distortion file exists

    # ----- Debugging -----
    print("K:\n", K)
    print("Distortion coefficients:\n", dist)

    return K, dist

def load_extrinsics_for_frame(calib_dir, basename, frame_num):
    # File example: 'stereo_calib.left.extrinsics.17.txt'
    extrinsics_path = os.path.join(calib_dir, f"{basename}.left.extrinsics.{frame_num}.txt")
    extrinsics = np.loadtxt(extrinsics_path)  # 4x4 matrix

    R = extrinsics[:3, :3]  # rotation matrix
    tvec = extrinsics[:3, 3].reshape(3, 1)  # translation vector


    rvec, _ = cv2.Rodrigues(R)  # convert rotation matrix to rvec

    # ----- Debugging ----
    print("Extrinsics matrix:\n", extrinsics)
    print("R:\n", R)
    print("Tvec:\n,", tvec)
    
    return rvec, tvec

def generate_offset_board_3d_points(start_row, start_col, rows, cols, square_size):
    objp = np.zeros((rows * cols, 3), np.float32)
    #objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)

    # Offset origin to the (start_row, start_col) position within the larger board
    objp[:, 0] += start_col
    objp[:, 1] += start_row

    objp *= square_size
    return objp

# def generate_board_3d_points(rows, cols, square_size):
#     objp = np.zeros((rows * cols, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
#     objp *= square_size
#     return objp

def visualize_3d_board(obj_points, rows, cols):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    xs = obj_points[:, 0]
    ys = obj_points[:, 1]
    zs = obj_points[:, 2]
    
    ax.scatter(xs, ys, zs, c='r', s=30)
    
    #Draw lines along the rows and columns (optional)
    # for r in range(rows):
    #     idx_start = r * cols
    #     idx_end = idx_start + cols
    #     ax.plot(xs[idx_start:idx_end], ys[idx_start:idx_end], zs[idx_start:idx_end], c='k')
    
    # for c in range(cols):
    #     col_x = [xs[r * cols + c] for r in range(rows)]
    #     col_y = [ys[r * cols + c] for r in range(rows)]
    #     col_z = [zs[r * cols + c] for r in range(rows)]
    #     ax.plot(col_x, col_y, col_z, c='k')


    ax.set_title('3D Virtual Chessboard')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def overlay_virtual_board_on_single_image(img, rvec, tvec, K, dist, obj_points, image_name, rows=9, cols=14):
    if img is None:
        print(f"Image {image_name} not loaded.")
        return

    # Reshape rvec and tvec in case they're flat arrays
    rvec = rvec.reshape(3, 1)
    tvec = tvec.reshape(3, 1)


    print("Image shape:", img.shape)  # (h, w, 3)
    print("rvec:", rvec.ravel())
    print("tvec:", tvec.ravel())


    # Project 3D object points
    projected, _ = cv2.projectPoints(obj_points, rvec, tvec, K, None)
    projected = projected.reshape(-1, 2)


    # Check if projected points are inside image bounds
    h, w = img.shape[:2]
    out_of_bounds = 0
    for i, pt in enumerate(projected):
        x, y = pt
        print(f"Projected point {i}: ({x:.2f}, {y:.2f})")
        if not (0 <= x < w and 0 <= y < h):
            print(f"⚠️ Point {i} is out of bounds!")
            out_of_bounds += 1

    if out_of_bounds == len(projected):
        print("🚫 All projected points are outside the image — something is wrong with calibration or extrinsics.")


    img_copy = img.copy()

    # --- Draw red dots ---
    for pt in projected:
        pt_int = tuple(int(x) for x in pt)
        cv2.circle(img_copy, pt_int, 6, (0, 0, 255), -1)  # Red in BGR, size 6

    # --- Optional: draw grid lines between points to show board layout ---
    # for r in range(rows):
    #     for c in range(cols - 1):
    #         idx1 = r * cols + c
    #         idx2 = r * cols + (c + 1)
    #         pt1 = tuple(int(x) for x in projected[idx1])
    #         pt2 = tuple(int(x) for x in projected[idx2])
    #         cv2.line(img_copy, pt1, pt2, (0, 0, 255), 1)

    # for c in range(cols):
    #     for r in range(rows - 1):
    #         idx1 = r * cols + c
    #         idx2 = (r + 1) * cols + c
    #         pt1 = tuple(int(x) for x in projected[idx1])
    #         pt2 = tuple(int(x) for x in projected[idx2])
    #         cv2.line(img_copy, pt1, pt2, (0, 0, 255), 1)

    for c in range(cols):
        for r in range(rows - 1):
            idx1 = c * rows + r
            idx2 = c * rows + (r + 1)
            pt1 = tuple(int(x) for x in projected[idx1])
            pt2 = tuple(int(x) for x in projected[idx2])
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 1)

    for r in range(rows):
        for c in range(cols - 1):
            idx1 = c * rows + r
            idx2 = (c + 1) * rows + r
            pt1 = tuple(int(x) for x in projected[idx1])
            pt2 = tuple(int(x) for x in projected[idx2])
            cv2.line(img_copy, pt1, pt2, (0, 0, 255), 1)

    # Display with matplotlib
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(f"Overlay on: {image_name}")
    plt.axis('off')
    plt.show()


def main():
    images, _ = load_images(LEFT_IMAGE_DIR)
    if SPECIFIC_IMAGE_NAME not in images:
        print(f"Image {SPECIFIC_IMAGE_NAME} not found in {LEFT_IMAGE_DIR}")
        return
    
    frame_num = int(SPECIFIC_IMAGE_NAME.split('_')[1].split('.')[0])  # Extract frame number from filename

    image_name = find_image_for_extrinsics(images, frame_num)
    print("Matching image:", image_name)
    print("Extrinsics file used: extrinsics.{}.txt".format(frame_num))

    if image_name is None:
        print(f"No matching image found for frame {frame_num}")
        return


    K, dist = load_intrinsics(CALIB_DIR)
    rvec, tvec  = load_extrinsics_for_frame(CALIB_DIR, 'stereo_calib', frame_num)

    #obj_points = generate_board_3d_points(rows=9, cols=14, square_size=BOARD_SQUARE_SIZE)


   # Assuming central chessboard starts at row 5, col 6 inside a 19x26 board
    obj_points = generate_offset_board_3d_points(
        start_row=15,
        start_col=12,
        rows=9,
        cols=14,
        square_size=BOARD_SQUARE_SIZE
    )

    visualize_3d_board(obj_points, rows=9, cols=14)  

    overlay_virtual_board_on_single_image(images[SPECIFIC_IMAGE_NAME], rvec, tvec, K, dist, obj_points, SPECIFIC_IMAGE_NAME)

if __name__ == '__main__':
    main()
