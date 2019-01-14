from reconstruct import transto3d


openpose_skeleton_keys = {0: "nose",
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
    def __init__(self, cam_num):
        self.skeleton2ds = [Skeleton2d() for i in range(cam_num)]

    def reconstruct_3d(self, cams, fundamental_matrix):
        self.skeleton3d = Skeleton3d()
        for skeleton_key in openpose_skeleton_keys.values():

            if sum(getattr(self.skeleton2ds[0], skeleton_key)) < 0 or sum(
                    getattr(self.skeleton2ds[1], skeleton_key)) < 0:
                continue
            else:
                setattr(self.skeleton3d, skeleton_key, transto3d(getattr(self.skeleton2ds[0], skeleton_key),
                                                                 getattr(self.skeleton2ds[1], skeleton_key),
                                                                 cams[0], cams[1], fundamental_matrix
                                                                 ))

    def json_format(self):
        return vars(self.skeleton3d)

    def __str__(self):
        if hasattr(self, 'skeleton3d'):
            return " ".join(["{} : {}\n".format(name, value) for name, value in vars(self.skeleton3d).items()])
        else:
            return "need to be reconstructed"


class Skeleton2d:
    def __init__(self):
        for value in openpose_skeleton_keys.values():
            setattr(self, value, (-1, -1))

    def __str__(self):
        return " ".join(["{} : {}\n".format(name, value) for name, value in vars(self).items()])


class Skeleton3d:
    def __init__(self):
        for value in openpose_skeleton_keys.values():
            setattr(self, value, (-1, -1, -1))

    def __str__(self):
        return " ".join(["{} : {}\n".format(name, value) for name, value in vars(self).items()])
