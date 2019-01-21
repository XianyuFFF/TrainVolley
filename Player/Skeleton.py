import numpy as np

from Reconstruct.reconstruct import transto3d
from utils.calculate import vector_angle


openpose_pose_keys = {0: "nose",
                      1: "neck",
                      2: "right_shoulder",
                      3: "right_elbow",
                      4: "right_wrist",
                      5: "left_shoulder",
                      6: "left_elbow",
                      7: "left_wrist",
                      8: "mind_hip",
                      9: "right_hip",
                      10: "right_knee",
                      11: "right_ankle",
                      12: "left_hip",
                      13: "left_knee",
                      14: "left_ankle",
                      15: "right_eye",
                      16: "left_eye",
                      17: "right_ear",
                      18: "left_ear",
                      19: "left_big_toe",
                      20: "left_small_toe",
                      21: "left_heel",
                      22: "right_big_toe",
                      23: "right_small_toe",
                      24: "right_heel",
                      25: "back_ground"}

class PlayerSkeleton:
    def __init__(self):
        self.skeleton2ds = {}

    def reconstruct_3d(self, cams, fundamental_matrix):
        self.pose_3d = Pose3d()
        for skeleton_key in openpose_pose_keys.values():

            if sum(getattr(self.skeleton2ds[0], skeleton_key)) < 0 or sum(
                    getattr(self.skeleton2ds[1], skeleton_key)) < 0:
                continue
            else:
                setattr(self.pose_3d, skeleton_key, transto3d(getattr(self.skeleton2ds[cams[0].id], skeleton_key),
                                                              getattr(self.skeleton2ds[cams[1].id], skeleton_key),
                                                              cams[0], cams[1], fundamental_matrix
                                                              )
                        )

    def left_shoulder_angle(self):
        forearm = np.asarray(getattr(self.pose_3d, 'left_elbow')) - np.asarray(getattr(self.pose_3d, 'left_shoulder'))
        hindarm = np.asarray(getattr(self.pose_3d, 'left_wrist')) - np.asarray(getattr(self.pose_3d, 'left_elbow'))
        return vector_angle(forearm, hindarm)

    def right_shoulder_angle(self):
        forearm = np.asarray(getattr(self.pose_3d, 'right_elbow')) - np.asarray(getattr(self.pose_3d, 'left_shoulder'))
        hindarm = np.asarray(getattr(self.pose_3d, 'right_wrist')) - np.asarray(getattr(self.pose_3d, 'left_elbow'))
        return vector_angle(forearm, hindarm)

    def body_ground_angle(self):
        neck = np.asarray(getattr(self.pose_3d, 'neck'))
        mind_hip = np.asarray(getattr(self.pose_3d, 'mind_hip'))
        ground = np.array([0, 0, 1])
        return 90 - vector_angle(neck - mind_hip, ground)

    def json_format(self):
        return vars(self.pose_3d)

    def __str__(self):
        if hasattr(self, 'skeleton3d'):
            return " ".join(["{} : {}\n".format(name, value) for name, value in vars(self.skeleton3d).items()])
        else:
            return "need to be reconstructed"

# TODO class Hand2d


class Pose2d:
    def __init__(self, keypoints):
        for i in range(0, len(keypoints), 3):
            key = openpose_pose_keys[i]
            setattr(self, key, [keypoints[i], keypoints[i + 1]])
            setattr(self, '{}_score', keypoints[i + 2])

    def __str__(self):
        return " ".join(["{} : {}\n".format(name, value) for name, value in vars(self).items()])


class Pose3d:
    def __init__(self):
        for value in openpose_pose_keys.values():
            setattr(self, value, (-1, -1, -1))

    def __str__(self):
        return " ".join(["{} : {}\n".format(name, value) for name, value in vars(self).items()])







