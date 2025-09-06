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


def show_charuco_overlay_vtk(rvec, tvec, K, image,
                             num_x=19, num_y=26, square_size=5.0,
                             marker_size=4.0,
                             debug_points=True):
    """
    Overlay a Charuco board on a background image using VTKOverlayWindow.
    The background plane is scaled according to camera intrinsics so that
    object points (mm) align with image pixels.
    """

    height, width = image.shape[:2]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # --- Create VTK overlay window ---
    window = VTKOverlayWindow(camera_matrix=K, init_pose=False)
    window.GetRenderWindow().SetNumberOfLayers(5)

    # --- Set video in Layer 0 ---
    window.set_video_image(image)  # video will fill layer 0

    # --- Renderer & camera for Layer 1 ---
    renderer = window.get_renderer(layer=1)
    vtk_cam = renderer.GetActiveCamera()

    # --- Set camera intrinsics ---
    cm.set_camera_intrinsics(renderer, vtk_cam,
                             width=width, height=height,
                             f_x=fx, f_y=fy, c_x=cx, c_y=cy,
                             near=1, far=1000)

    # --- Compute video plane size in mm at Z-plane = 100 mm ---
    z_plane = 100.0  # mm
    plane_width = width / fx * z_plane
    plane_height = height / fy * z_plane

    # --- Video plane as a quad in Layer 0 coordinates ---
    plane_actor = vtk.vtkActor()
    plane_mapper = vtk.vtkPolyDataMapper()
    plane_source = vtk.vtkPlaneSource()
    plane_source.SetOrigin(-cx/fx*plane_width, -cy/fy*plane_height, z_plane)
    plane_source.SetPoint1((width-cx)/fx*plane_width, -cy/fy*plane_height, z_plane)
    plane_source.SetPoint2(-cx/fx*plane_width, (height-cy)/fy*plane_height, z_plane)
    plane_source.Update()
    plane_mapper.SetInputData(plane_source.GetOutput())
    plane_actor.SetMapper(plane_mapper)
    # Texture the plane with the image
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vtk_image = vtk.vtkImageImport()
    vtk_image.CopyImportVoidPointer(img_rgb.tobytes(), len(img_rgb.tobytes()))
    vtk_image.SetDataScalarTypeToUnsignedChar()
    vtk_image.SetNumberOfScalarComponents(3)
    vtk_image.SetWholeExtent(0,width-1,0,height-1,0,0)
    vtk_image.SetDataExtentToWholeExtent()
    vtk_image.Update()
    texture = vtk.vtkTexture()
    texture.SetInputData(vtk_image.GetOutput())
    texture.InterpolateOn()
    plane_actor.SetTexture(texture)
    renderer.AddActor(plane_actor)

    # --- Create Charuco board actor ---
    append_filter = vtk.vtkAppendPolyData()
    for i in range(num_x):
        for j in range(num_y):
            if (i+j) % 2 == 0:
                square = vtk.vtkPlaneSource()
                square.SetOrigin(i*square_size, j*square_size, 0)
                square.SetPoint1((i+1)*square_size, j*square_size, 0)
                square.SetPoint2(i*square_size, (j+1)*square_size, 0)
                square.Update()
                append_filter.AddInputData(square.GetOutput())
    append_filter.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(append_filter.GetOutput())
    board_actor = vtk.vtkActor()
    board_actor.SetMapper(mapper)
    board_actor.GetProperty().SetColor(1,1,1)
    window.add_vtk_actor(board_actor, layer=1)

    # --- Apply rvec/tvec ---
    R, _ = cv2.Rodrigues(rvec)
    T_oc = np.eye(4)
    T_oc[:3,:3] = R
    T_oc[:3,3] = tvec.flatten()
    T_co = np.linalg.inv(T_oc)
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i,j,T_co[i,j])
    board_actor.SetUserMatrix(vtk_matrix)

    # --- Debug: project points from OpenCV ---
    if debug_points:
        import cv2.aruco as aruco
        board = aruco.CharucoBoard_create(
            squaresX=num_x, squaresY=num_y,
            squareLength=square_size,
            markerLength=marker_size,
            dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        )
        obj_pts = board.chessboardCorners
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        img_pts = img_pts.reshape(-1,2)
        for pt in img_pts:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(square_size*0.05)
            sphere.SetCenter(pt[0], pt[1], z_plane)
            sphere.Update()
            mapper_s = vtk.vtkPolyDataMapper()
            mapper_s.SetInputData(sphere.GetOutput())
            actor_s = vtk.vtkActor()
            actor_s.SetMapper(mapper_s)
            actor_s.GetProperty().SetColor(1,0,0)
            window.add_vtk_actor(actor_s, layer=1)

    # --- Axes for reference ---
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(30,30,30)
    window.add_vtk_actor(axes, layer=1)

    window.GetRenderWindow().Render()
    window.show()
    return window

def show_chessboard_in_vtk(rvec, tvec, K, image,
                           num_x=19, num_y=26, square_size=5.0,
                           marker_size=4.0,
                           debug_points=True):
    """
    Show a Charuco board in VTKOverlayWindow with correct intrinsics, pose, and background.
    Optionally overlays OpenCV projected points as red spheres for alignment debugging.
    """

    # --- Create VTKOverlayWindow ---
    #from your_overlay_module import VTKOverlayWindow  # replace with actual import
    window = VTKOverlayWindow(camera_matrix=K, init_pose=False)
    window.GetRenderWindow().SetNumberOfLayers(5)

    # --- Set video background in Layer 0 ---
    window.set_video_image(image)

    # --- Get renderer and camera for models in Layer 1 ---
    renderer = window.get_renderer(layer=1)
    vtk_cam = renderer.GetActiveCamera()

    # --- Extract intrinsics ---
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]

    # --- Set camera intrinsics in VTK ---
    cm.set_camera_intrinsics(renderer, vtk_cam,
                             width=image.shape[1],
                             height=image.shape[0],
                             f_x=fx, f_y=fy, c_x=cx, c_y=cy,
                             near=1, far=1000)

    # --- Create Charuco board actor ---
    append_filter = vtk.vtkAppendPolyData()
    for i in range(num_x):
        for j in range(num_y):
            if (i+j) % 2 == 0:
                square = vtk.vtkPlaneSource()
                square.SetOrigin(i*square_size, j*square_size, 0)
                square.SetPoint1((i+1)*square_size, j*square_size, 0)
                square.SetPoint2(i*square_size, (j+1)*square_size, 0)
                square.Update()
                append_filter.AddInputData(square.GetOutput())
    append_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(append_filter.GetOutput())
    board_actor = vtk.vtkActor()
    board_actor.SetMapper(mapper)
    board_actor.GetProperty().SetColor(1,1,1)
    window.add_vtk_actor(board_actor, layer=1)

    # --- Apply rvec/tvec transform ---
    R, _ = cv2.Rodrigues(rvec)
    T_oc = np.eye(4)
    T_oc[:3,:3] = R
    T_oc[:3,3] = tvec.flatten()
    T_co = np.linalg.inv(T_oc)

    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i,j,T_co[i,j])
    board_actor.SetUserMatrix(vtk_matrix)

    # --- Optional debug: add OpenCV projected points as spheres ---
    if debug_points:
        # Compute object points in board coordinates
        import cv2.aruco as aruco
        board = aruco.CharucoBoard_create(
            squaresX=num_x, squaresY=num_y,
            squareLength=square_size,
            markerLength=marker_size,
            dictionary=aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        )
        obj_pts = board.chessboardCorners  # (N,3)
        img_pts, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, None)
        img_pts = img_pts.reshape(-1,2)

        for pt in img_pts:
            sphere = vtk.vtkSphereSource()
            sphere.SetRadius(square_size*0.05)  # small sphere
            sphere.SetCenter(pt[0], pt[1], 0)
            sphere.Update()

            mapper_s = vtk.vtkPolyDataMapper()
            mapper_s.SetInputData(sphere.GetOutput())
            actor_s = vtk.vtkActor()
            actor_s.SetMapper(mapper_s)
            actor_s.GetProperty().SetColor(1,0,0)  # red
            window.add_vtk_actor(actor_s, layer=1)

    # --- Axes for reference ---
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(30,30,30)
    window.add_vtk_actor(axes, layer=1)

    # --- Render and show ---
    window.GetRenderWindow().Render()
    window.show()
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
    # Show Charuco in VTK
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
        base_dir = Path(__file__).resolve().parent.parent
        print(f"Base directory is: {base_dir}")
    except NameError:
        base_dir = Path.cwd()
    
    #left_img_dir = base_dir / 'data' / 'img_dir' / 'Frames_Left'
    left_img_dir = base_dir / 'data' / 'results' / 'rectified_imgs' / 'Left'
    results_folder = base_dir / 'data' / 'results' / 'calibrator_params'
    calibration_json_path = results_folder / "stereo_calibration_params_new.json"

    
    main()