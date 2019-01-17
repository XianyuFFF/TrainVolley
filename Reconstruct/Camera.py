import numpy as np
import cv2
import json


class CameraMatrix:
    def __init__(self, K, r_vec, t_vec, dist_coefs):
        self.K = np.array(K, dtype=np.float32)
        self.r_vec = np.array(r_vec, dtype=np.float32)
        self.t_vec = np.array(t_vec, dtype=np.float32)
        self.dist_coefs = np.array([], dtype=np.float32)
        self.proj_mat = self.init_project_mat()

    def init_project_mat(self):
        R, _ = cv2.Rodrigues(self.r_vec)
        m = np.dot(self.K, R)
        t = np.dot(self.K, self.t_vec)
        rt_mat = np.matrix([
            [m[0, 0], m[0, 1], m[0, 2], t[0, 0]],
            [m[1, 0], m[1, 1], m[1, 2], t[1, 0]],
            [m[2, 0], m[2, 1], m[2, 2], t[2, 0]]
        ])
        return rt_mat


class Camera:
    def __init__(self, id, name, camera_matrix):
        self.id = id
        self.name = name
        self.camera_matrix = camera_matrix


    def pack_json_camera(self, camera_info_json_dir):
        with open(camera_info_json_dir, 'r') as f:
            camera_info = json.load(f)
        camera_matrix = camera_info['camera_matrix']
        r_vec = camera_info['r_vet']
        R, _ = cv2.Rodrigues(r_vec)
        t_vec = camera_info['t_vet']
        dist_coefs = camera_info['dist_coefs']
        # dist_coefs = np.zeros((1, 5))
        # if projection wrong,try switch dist_coefs to zeros
        setattr(self, 'camera_matrix', CameraMatrix(camera_matrix, r_vec,t_vec, dist_coefs))


    def transto2d(self, points):
        return cv2.projectPoints(points, self.camera_matrix.R, self.camera_matrix.T, self.camera_matrix.K,
                                 self.camera_matrix.dist_coefs)

    def find_homograph_matrix(self, other_camera, plane_norm):
        K1 = np.asmatrix(self.camera_matrix.K)
        K2 = np.asmatrix(other_camera.camera_matrix.K)

        R1 = np.asmatrix(self.camera_matrix.R)
        R2 = np.asmatrix(other_camera.camera_matrix.R)

        T1 = np.asmatrix(self.camera_matrix.T)
        T2 = np.asmatrix(other_camera.camera_matrix.T)

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
