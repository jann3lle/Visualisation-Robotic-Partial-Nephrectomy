# imports
import cv2
import numpy as np
from os import path, makedirs
import sys
### Classes to store and retrieve stereo calibrations

class Mono_calibration(object):
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs


class Stereo_calibration(object):
    def __init__(self, camera1: Mono_calibration=None, camera2: Mono_calibration=None, R=None, T=None, E=None, F=None):
        self.camera1 = camera1
        self.camera2 = camera2
        self.R = R
        self.T = T
        self.E = E
        self.F = F

    def load_from_file(self, filename):
        cal_ext = path.splitext(path.basename(filename))[1]
        print("Calibration extension = {}".format(cal_ext))
        self.camera1 = Mono_calibration()
        self.camera2 = Mono_calibration()
        R = None
        T = None
        E = None
        F = None
        if cal_ext == '.yaml' or cal_ext == '.yml' or cal_ext == '.json':
            # read in from a single file - reads various old format yaml files
            camera_calib_data = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            self.camera1.camera_matrix = camera_calib_data.getNode('K1').mat()
            if self.camera1.camera_matrix is None:
                self.camera1.camera_matrix = camera_calib_data.getNode('cameraMatrix1').mat()
            self.camera1.dist_coeffs = camera_calib_data.getNode('D1').mat()
            if self.camera1.dist_coeffs is None:
                self.camera1.dist_coeffs = camera_calib_data.getNode('distCoeffs1').mat()
            self.camera2.camera_matrix = camera_calib_data.getNode('K2').mat()
            if self.camera2.camera_matrix is None:
                self.camera2.camera_matrix = camera_calib_data.getNode('cameraMatrix2').mat()
            self.camera2.dist_coeffs = camera_calib_data.getNode('D2').mat()
            if self.camera2.dist_coeffs is None:
                self.camera2.dist_coeffs = camera_calib_data.getNode('distCoeffs2').mat()
            self.R = camera_calib_data.getNode('R').mat()
            self.T = camera_calib_data.getNode('T').mat()
            self.E = camera_calib_data.getNode('E').mat()
            self.F = camera_calib_data.getNode('F').mat()
        else:
            camera_param_file = path.join(filename, 'params')
            mat1_file = path.join(camera_param_file, 'calib.left.intrinsics.txt')
            self.camera1.camera_matrix = np.asmatrix(np.loadtxt(mat1_file))
            dist1_file = path.join(camera_param_file, 'calib.left.distortion.txt')
            self.camera1.dist_coeffs = np.asmatrix(np.loadtxt(dist1_file))
            mat2_file = path.join(camera_param_file, 'calib.right.intrinsics.txt')
            self.camera2.camera_matrix = np.asmatrix(np.loadtxt(mat2_file))
            dist2_file = path.join(camera_param_file, 'calib.right.distortion.txt')
            self.camera2.dist_coeffs = np.asmatrix(np.loadtxt(dist2_file))
            left_to_right_mat_file = path.join(camera_param_file, 'calib.l2r.txt')
            left_to_right_mat = np.asmatrix(np.loadtxt(left_to_right_mat_file))
            self.R = left_to_right_mat[:3, :3]
            self.T = left_to_right_mat[:3, 3]
            # Do we need these? Calculate anyway
            essential_file = path.join(camera_param_file, 'calib.essential.txt')
            if path.isfile(essential_file):
                self.E = np.asmatrix(np.loadtxt(essential_file))
            else:
                S = [[0,-self.T[2], self.T[1]],
                    [self.T[2], 0, -self.T[0]],
                    [-self.T[1], self.T[0], 0]]
                self.E = S * self.R
            fundamental_file = path.join(camera_param_file, 'calib.fundamental.txt')
            if path.isfile(fundamental_file):
                self.F = np.asmatrix(np.loadtxt(fundamental_file))
            else:
                self.F = np.linalg.inv(self.camera2.camera_matrix.T) * self.E * np.linalg.inv(self.camera1.camera_matrix)

    def save_to_file(self, filename):
        # Write out the new calibration
        camera_calib_data_out = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        camera_calib_data_out.write('K1', self.camera1.camera_matrix)
        camera_calib_data_out.write('D1', self.camera1.dist_coeffs)
        camera_calib_data_out.write('K2', self.camera2.camera_matrix)
        camera_calib_data_out.write('D2', self.camera2.dist_coeffs)
        camera_calib_data_out.write('R', self.R)
        camera_calib_data_out.write('T', self.T)
        camera_calib_data_out.write('E', self.E)
        camera_calib_data_out.write('F', self.F)
        camera_calib_data_out.release()


class Stereo_rectification(object):
    def __init__(self, P1=None, P2=None, Q=None):
        self.P1 = P1
        self.P2 = P2
        self.Q = Q

    def load_from_file(self, filename):
        cal_ext = path.splitext(path.basename(filename))[1]
        print("Calibration extension = {}".format(cal_ext))
        if cal_ext == '.yaml' or cal_ext == '.yml' or cal_ext == '.json':
            # read in from a single file - reads various old format yaml files
            camera_calib_data = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            self.P1 = camera_calib_data.getNode('P1').mat()
            self.P2 = camera_calib_data.getNode('P2').mat()
            self.Q = camera_calib_data.getNode('Q').mat()
        else:
            print("Extension {} - filetype not supported\n")
            sys.exit(-1)

    def save_to_file(self, filename):
        # Write out the new calibration
        camera_calib_data_out = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        camera_calib_data_out.write('P1', self.P1)
        camera_calib_data_out.write('P2', self.P2)
        camera_calib_data_out.write('Q', self.Q)
        camera_calib_data_out.release()
