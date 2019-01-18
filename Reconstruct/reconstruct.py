import cv2
import numpy as np


def transto3d(image1_point, image2_point, camera1, camera2, funamental_matrix):
    p1 = np.array(image1_point, dtype=np.float32).reshape((1,1, 2))
    p2 = np.array(image2_point, dtype=np.float32).reshape((1,1, 2))
    # print(funamental_matrix)

    # print("before correctMatches: p1:{} p2:{}".format(p1, p2))

    p1, p2 = cv2.correctMatches(funamental_matrix, p1, p2)
    # print("after correctMatches: p1:{} p2:{}".format(p1, p2))

    X = cv2.triangulatePoints(camera1.camera_matrix.proj_mat, camera2.camera_matrix.proj_mat, p1, p2)
    X /= X[3]
    X = X[:3].T.reshape((1, 3)).tolist()
    return X


def find_fundamental_matrix(camera1, camera2):
    K1 = camera1.camera_matrix.K
    K2 = camera2.camera_matrix.K

    R1 = camera1.camera_matrix.R
    R2 = camera2.camera_matrix.R

    T1 = camera1.camera_matrix.t_vec
    T2 = camera2.camera_matrix.t_vec

    R = np.matrix(R1) * np.matrix(R2).I
    t = -T2 + T1

    a1, a2, a3 = t
    _t = np.matrix([[0, -a3, a2], [a3, 0, -a1], [-a2, a1, 0]])

    E = _t * R

    F = np.matrix(K2).T.I * E * np.matrix(K1).I

    newF = []
    for f in F.tolist():
        for ff in f:
            newF.append(float(ff))

    newF = np.matrix(newF).reshape((3, 3))

    return newF


def get_homography_point(G, p):
    p = np.matrix([p[0], p[1], 1]).T
    r = G * p
    r /= r[2]
    return r[:2].reshape(1, 2)
