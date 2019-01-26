import cv2
import numpy as np


def transto3d(image1_point, image2_point, camera1, camera2, fundamental_matrix=None):
    p1 = np.array(image1_point, dtype=np.float32).reshape((1, 1, 2))
    p2 = np.array(image2_point, dtype=np.float32).reshape((1, 1, 2))

    # print(fundamental_matrix)
    if fundamental_matrix is not None:
        # print("before correctMatches: p1:{} p2:{}".format(p1, p2))
        p1, p2 = cv2.correctMatches(fundamental_matrix, p1, p2)
        # print("after correctMatches: p1:{} p2:{}".format(p1, p2))

    X = cv2.triangulatePoints(camera1.camera_matrix.proj_mat, camera2.camera_matrix.proj_mat, p1, p2)
    X /= X[3]
    X = X[:3].T.reshape((1, 3)).tolist()
    return X


def find_fundamental_matrix(camera0, camera1):
    p3ds = [
        [-3000, 0, 0],
        [-3000, 9000, 0],
        [0, 0, 0],
        [0, 9000, 0],
        [-3000, 0, 3000],
        [-3000, 9000, 3000],
        [0, 0, 3000],
        [0, 9000, 3000],
    ]

    c0_ps = camera0.transto2d(p3ds)[0]
    c1_ps = camera1.transto2d(p3ds)[0]

    F, _ = cv2.findFundamentalMat(c0_ps, c1_ps)

    return F


def get_homography_point(G, p):
    p = np.matrix([p[0], p[1], 1]).T
    r = G * p
    r /= r[2]
    return r[:2].reshape(1, 2)
