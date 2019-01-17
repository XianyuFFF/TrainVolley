import numpy as np
import cv2


class CameraMatrix:
    def __init__(self, K, R, T, dist_coefs):
        self.K = np.array(K, dtype=np.float32)
        self.R = np.array(R, dtype=np.float32)
        self.T = np.array(T, dtype=np.float32)
        self.dist_coefs = np.array([], dtype=np.float32)
        self.proj_mat = self.init_project_mat()

    def init_project_mat(self):
        m = np.dot(self.K, self.R)
        t = np.dot(self.K, self.T)
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

    def transto2d(self, points):
        K = self.camera_matrix.K
        r_vec, _ = cv2.Rodrigues(self.camera_matrix.R)
        T = self.camera_matrix.T
        dist_coefs = self.camera_matrix.dist_coefs
        return cv2.projectPoints(points, r_vec, T, K, dist_coefs)

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
