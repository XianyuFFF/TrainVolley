import numpy as np
import cv2
import json
import glob
from utils.path_parser import get_camera_info_dir
import os


class CameraMatrix:
    def __init__(self, K, r_vec, t_vec, dist_coefs):
        self.K = np.asarray(K, dtype=np.float32)
        self.r_vec = np.asarray(r_vec, dtype=np.float32)
        self.t_vec = np.asarray(t_vec, dtype=np.float32)
        self.dist_coefs = np.asarray(dist_coefs, dtype=np.float32)
        self.proj_mat = self.init_project_mat()

    def init_project_mat(self):
        R, _ = cv2.Rodrigues(self.r_vec)
        m = np.dot(self.K, R)
        t = np.dot(self.K, self.t_vec)
        rt_mat = np.matrix([
            [m[0, 0], m[0, 1], m[0, 2], t[0, 0]],
            [m[1, 0], m[1, 1], m[1, 2], t[1, 0]],
            [m[2, 0], m[2, 1], m[2, 2], t[2, 0]]
        ], dtype=np.float32)
        return rt_mat


class Camera:
    def __init__(self, id, name, camera_matrix, calibration_video, land_marker):
        self.id = id
        self.name = name
        self.camera_matrix = camera_matrix
        self.camera_info_json_dir = get_camera_info_dir(self.id, self.name)
        self.calibration_video = calibration_video
        self.land_marker = land_marker

        if os.path.exists(self.camera_info_json_dir):
            self.pack_json_camera()
        else:
            if self.calibration_video is None:
                print("{} calibration video needed".format(self.name))
                exit(0)
            try:
                self.calibrate(self.calibration_video, self.camera_info_json_dir)
                self.pack_json_camera()
            except FileNotFoundError as e:
                print("{}: {} calibration video {} not exist".format(e, self.name, self.calibration_video))

    def pack_json_camera(self):
        with open(self.camera_info_json_dir, 'r') as f:
            camera_info = json.load(f)
        camera_matrix = camera_info['camera_matrix']
        dist_coefs = camera_info['dist_coefs']

        if not camera_info.get('r_vec') or not camera_info.get('t_vec'):
            try:
                r_vec, t_vec = self.camera_world(camera_matrix, dist_coefs)
                camera_info['r_vec'] = r_vec.tolist()
                camera_info['t_vec'] = t_vec.tolist()
                R, _ = cv2.Rodrigues(r_vec)
            except FileNotFoundError as e:
                print('{}: marker file {} has some problem'.format(e, self.land_marker))
        else:
            r_vec = np.array(camera_info['r_vec'], dtype=np.float32)
            R, _ = cv2.Rodrigues(r_vec)
            t_vec = np.array(camera_info['t_vec'], dtype=np.float32)

        # dist_coefs = np.zeros((1, 5))
        # if projection wrong,try switch dist_coefs to zeros
        setattr(self, 'camera_matrix', CameraMatrix(camera_matrix, r_vec, t_vec, dist_coefs))
        setattr(self.camera_matrix, 'R', np.asarray(R, dtype=np.float32))

        with open(self.camera_info_json_dir, 'w') as f:
            json.dump(camera_info, f)

    def camera_world(self, camera_matrix, dist_coefs):
        image_floor_points, object_floor_points = self.load_land_marker()
        _, r_vet, t_vet = cv2.solvePnP(object_floor_points, image_floor_points,
                                       np.asarray(camera_matrix, dtype=np.float32),
                                       np.asarray(dist_coefs, dtype=np.float32)
                                       )
        return r_vet, t_vet

    def load_land_marker(self):
        with open(self.land_marker, 'r') as f:
            content = json.load(f)
        p2ds = np.asarray(content[self.name]["p_2ds"], dtype=np.float32).reshape((4, 1, 2))
        p3ds = np.asarray(content[self.name]["p_3ds"], dtype=np.float32).reshape((4, 1, 3))
        return p2ds, p3ds


    def transto2d(self, points):
        points = np.asarray(points, dtype=np.float32)

        return cv2.projectPoints(points, self.camera_matrix.r_vec, self.camera_matrix.t_vec.reshape((1,3)), self.camera_matrix.K,
                                 self.camera_matrix.dist_coefs)

    def find_homograph_matrix(self, other_camera, plane_norm):
        K1 = np.asmatrix(self.camera_matrix.K)
        K2 = np.asmatrix(other_camera.camera_matrix.K)

        R1 = np.asmatrix(self.camera_matrix.R)
        R2 = np.asmatrix(other_camera.camera_matrix.R)

        T1 = np.asmatrix(self.camera_matrix.t_vec)
        T2 = np.asmatrix(other_camera.camera_matrix.t_vec)

        R_1to2 = R2 * R1.T
        tvec_1to2 = R2 * (-R1.T * T1) + T2

        normal = R1 * np.asmatrix(plane_norm).T
        origin = T1

        normal = np.asarray(normal).reshape(1, 3)
        origin = np.asarray(origin).reshape(1, 3)

        d_inv = 1.0 / np.dot(normal[0], origin[0])

        homography_elu = R_1to2 + d_inv * tvec_1to2 * normal
        homography = K2 * homography_elu * K1.I

        return homography

    def find_estimate_matrix(self, other_cam):
        R1 = self.camera_matrix.R
        R2 = other_cam.camera_matrix.R

        T1 = self.camera_matrix.T
        T2 = other_cam.camera_matrix.T

        R = np.matrix(R2) * np.matrix(R1).T
        t = -T1 + T2

        a1, a2, a3 = t
        _t = np.matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

        E = _t * R

        return E

    @staticmethod
    def calibrate(input_arg, out, debug_dir=None, frame_step=20):
        if '*' in input_arg:
            source = glob(input_arg)
        else:
            source = cv2.VideoCapture(input_arg)
        pattern_size = (9, 6)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        # pattern_points *= square_size

        obj_points = []
        img_points = []
        h, w = 0, 0
        i = -1
        while True:
            i += 1
            if isinstance(source, list):
                if i == len(source):
                    break
                img = cv2.imread(source[i])
            else:
                retval, img = source.read()
                if not retval:
                    break
                if i % frame_step != 0:
                    continue

            print('Searching for chessboard in frame ' + str(i) + '...'),
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = img.shape[:2]
            found, corners = cv2.findChessboardCorners(img, pattern_size, cv2.CALIB_CB_FILTER_QUADS)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(img, corners, (5, 5), (-1, -1), term)
            if debug_dir:
                img_chess = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(img_chess, pattern_size, corners, found)
                cv2.imwrite(os.path.join(debug_dir, '%04d.png' % i), img_chess)
            if not found:
                print('not found')
                continue
            img_points.append(corners.reshape(1, -1, 2))
            obj_points.append(pattern_points.reshape(1, -1, 3))
            print('ok')

        # if corners!=None:
        #     with open(corners, 'wb') as fw:
        #         pickle.dump(img_points, fw)
        #         pickle.dump(obj_points, fw)
        #         pickle.dump((w, h), fw)

        print('\nPerforming calibration...')
        rms, camera_matrix, dist_coefs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, (w, h), None, None)
        print("RMS:", rms)
        print("camera matrix:\n", camera_matrix)
        print("distortion coefficients: ", dist_coefs.ravel())
        calibration = {'rms': rms, 'camera_matrix': camera_matrix.tolist(), 'dist_coefs': dist_coefs.tolist()}
        with open(out, 'w') as fw:
            json.dump(calibration, fw)
