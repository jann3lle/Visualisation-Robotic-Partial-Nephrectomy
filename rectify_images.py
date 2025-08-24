# imports
import cv2
import numpy as np
from os import path, mkdir, listdir
import glob
import StereoCalibrationObjects 
#import Mono_calibration, Stereo_calibration, Stereo_rectification
import argparse

# Colours
colors = np.array([[255, 100, 100], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]])
def makedir(dir):
    if (path.exists(dir)):
        print("Path {} already exists".format(dir))
    else:
        print("Creating path {}".format(dir))
        mkdir(dir)

parser = argparse.ArgumentParser()
parser.add_argument("left_dir", help="left calibration images directory")
parser.add_argument("right_dir", help="right calibration images directory")
parser.add_argument("calib", help="stereo calibration directory or file")
parser.add_argument("out_dir", help="output rectified images directory")
parser.add_argument("--diag_out", help="output diagnostic images and info directory",)
args = parser.parse_args()

left_filenames = sorted(listdir(args.left_dir))
right_filenames = sorted(listdir(args.right_dir))

R1 = np.zeros(shape=(3, 3))
R2 = np.zeros(shape=(3, 3))
P1 = np.zeros(shape=(3, 4))
P2 = np.zeros(shape=(3, 4))
Q = np.zeros(shape=(4, 4))
stereo_calib = StereoCalibrationObjects.Stereo_calibration()
stereo_calib.load_from_file(args.calib)
# Check the image dimensions
left_imfile = path.join(args.left_dir, left_filenames[0])
L_image = cv2.imread(left_imfile)
w = L_image.shape[1]
h = L_image.shape[0]
print("width = "+str(w)+",  height = " + str(h))
cv2.stereoRectify(stereo_calib.camera1.camera_matrix.astype(np.float64),
                  stereo_calib.camera1.dist_coeffs.astype(np.float64),
                  stereo_calib.camera2.camera_matrix.astype(np.float64),
                  stereo_calib.camera2.dist_coeffs.astype(np.float64), (w, h),
                  stereo_calib.R.astype(np.float64), stereo_calib.T.astype(np.float64), R1, R2, P1, P2, Q,
                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0)
print("P1={}".format(P1))
print("P2={}".format(P2))
stereo_rect = StereoCalibrationObjects.Stereo_rectification(P1, P2, Q)
# Write out the new calibration
makedir(args.out_dir)
stereo_calib_file_out = path.join(args.out_dir, 'RectifiedCalibration.json')
stereo_rect.save_to_file(stereo_calib_file_out)
undistort_rectify_map1_x, undistort_rectify_map1_y = cv2.initUndistortRectifyMap(stereo_calib.camera1.camera_matrix,
                                                                                 stereo_calib.camera1.dist_coeffs,
                                                                                 R1, P1, (w, h), cv2.CV_32FC1)
undistort_rectify_map2_x, undistort_rectify_map2_y = cv2.initUndistortRectifyMap(stereo_calib.camera2.camera_matrix,
                                                                                 stereo_calib.camera2.dist_coeffs,
                                                                                 R2, P2, (w, h), cv2.CV_32FC1)
left_out_dir = path.join(args.out_dir, "Left")
makedir(left_out_dir)
right_out_dir = path.join(args.out_dir, "Right")
makedir(right_out_dir)
if args.diag_out is not None:
    print(f"Outputting rectified images with lines to {args.diag_out}\n")
    makedir(args.diag_out)
for file_num, (left_file, right_file) in enumerate(zip(left_filenames, right_filenames)):
    left_imfile = path.join(args.left_dir, left_file)
    right_imfile = path.join(args.right_dir, right_file)
    L_image = cv2.imread(left_imfile)
    R_image = cv2.imread(right_imfile)
    print("Starting rectification of image {}\n".format(file_num))
    L_rect_image = cv2.remap(L_image, undistort_rectify_map1_x, undistort_rectify_map1_y, cv2.INTER_LINEAR)
    R_rect_image = cv2.remap(R_image, undistort_rectify_map2_x, undistort_rectify_map2_y, cv2.INTER_LINEAR)
    left_im_outfile = path.join(left_out_dir, left_file)
    cv2.imwrite(left_im_outfile, L_rect_image)
    right_im_outfile = path.join(right_out_dir, right_file)
    cv2.imwrite(right_im_outfile, R_rect_image)
    print("Done rectifying images {}\n".format(file_num))
    if args.diag_out is not None:
        LR_image = np.concatenate((L_rect_image, R_rect_image), axis=1)
        # recolour every 50 rows
        row = 50
        colour = 0
        while row < h:
            LR_image[row, 0:w*2] = colors[colour]
            row += 50
            colour = (colour+1) % len(colors)
        im_outfile = path.join(args.diag_out, left_file)
        cv2.imwrite(im_outfile,LR_image)
