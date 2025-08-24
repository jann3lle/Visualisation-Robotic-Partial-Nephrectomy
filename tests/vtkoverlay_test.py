# pyright: reportMissingImports=false
import vtk
from vtk.util import numpy_support
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import json

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

def rvec_tvec_to_vtk_transform(rvec, tvec):
    # Convert OpenCV rvec/tvec to vtkTransform
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

def show_chessboard_in_vtk(rvec, tvec):
    # Create charuco board model
    board_actor = create_charuco_board_vtk_model()

    # Transform the model using rvec/tvec 
    transform = rvec_tvec_to_vtk_transform(rvec, tvec)
    board_actor.SetUserTransform(transform)

    # Create renderer (sets the scene)
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0.2, 0.3, 0.4)
    renderer.AddActor(board_actor)

    # Coordinate axis for visual debugging
    axes = vtk.vtkAxesActor()
    axes.SetTotalLength(30, 30, 30)
    renderer.AddActor(axes)

    # Create render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)

    # Interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()



# def overlay_vtk_model_on_image(image, rvec, tvec, camera_matrix, dist_coeffs):
#     """
#     Render the Charuco board model with VTK from the camera pose, then overlay onto image.
#     """

#     # Create VTK renderer and window
#     renderer = vtk.vtkRenderer()
#     render_window = vtk.vtkRenderWindow()
#     render_window.SetOffScreenRendering(1)
#     render_window.AddRenderer(renderer)
#     render_window.SetSize(image.shape[1], image.shape[0])

#     # Create Charuco board actor
#     board_actor = create_charuco_board_vtk_model()

#     # Apply pose transform to board actor
#     transform = rvec_tvec_to_vtk_transform(rvec, tvec)
#     board_actor.SetUserTransform(transform)

#     renderer.AddActor(board_actor)
#     renderer.SetBackground(0, 0, 0)  # black background

#     # Setup vtk camera using intrinsic matrix
#     vtk_cam = vtk.vtkCamera()

#     fx = camera_matrix[0,0]
#     fy = camera_matrix[1,1]
#     cx = camera_matrix[0,2]
#     cy = camera_matrix[1,2]
#     width = image.shape[1]
#     height = image.shape[0]

#     # Approximate view angle from focal length
#     view_angle = 2 * np.degrees(np.arctan(height / (2 * fy)))
#     vtk_cam.SetViewAngle(view_angle)
#     vtk_cam.SetPosition(0, 0, 0)
#     vtk_cam.SetFocalPoint(0, 0, 1)
#     vtk_cam.SetViewUp(0, -1, 0)
#     renderer.SetActiveCamera(vtk_cam)

#     render_window.Render()

#     # Grab VTK render window as numpy array
#     w2if = vtk.vtkWindowToImageFilter()
#     w2if.SetInput(render_window)
#     w2if.Update()

#     vtk_image = w2if.GetOutput()
#     vtk_array = vtk_image.GetPointData().GetScalars()
#     components = vtk_array.GetNumberOfComponents()

#     # Fix dimension order
#     width, height, _ = vtk_image.GetDimensions()
#     # vtk_to_numpy returns a flat array, reshape it as (height, width, channels)
#     arr = numpy_support.vtk_to_numpy(vtk_array)
#     arr = arr.reshape(height, width, components)

#     # VTK outputs RGB, convert to BGR for OpenCV
#     arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

#     # Convert BGR to RGB for matplotlib display
#     overlay_img_rgb = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2RGB)

#     # Display with matplotlib
#     plt.figure(figsize=(10, 8))
#     plt.imshow(overlay_img_rgb)
#     plt.title("Overlay of Charuco VTK model on image")
#     plt.axis('off')
#     plt.show()


def main():
    left_img_paths = load_images(left_img_dir)

    if left_img_paths:
        print("Images found in chosen directory")
    else:
        print("No images found in directory")
        return
    
    image = load_specific_image(left_img_paths, img_path_no)

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
     
    #reference_image = cv2.imread(str(left_img_paths[img_path_no]), cv2.IMREAD_GRAYSCALE)

    camera_matrix = calib_params["left_camera_matrix"]
    dist_coeffs = calib_params["left_dist_coeffs"]
    
    #overlay_vtk_model_on_image(image, rvec, tvec, camera_matrix, dist_coeffs)

    # Visualise in vtk
    show_chessboard_in_vtk(rvec, tvec)

if __name__ == "__main__":
    # --- Config ---
    img_path_no = 0

    try:
        base_dir = Path(__file__).resolve().parent.parent
        print(f"Base directory is: {base_dir}")
    except NameError:
        base_dir = Path.cwd()
    
    valid_frame_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 42, 43, 44, 45, 47, 48, 49]

    left_img_dir = base_dir / 'data' / 'img_dir' / 'Frames_Left'
    results_folder = base_dir / 'data' / 'results' / 'calibrator_new_params'
    calibration_json_filename = results_folder / "stereo_calibration_params_new.json"
    calibration_json_path = results_folder / calibration_json_filename
    
    main()


