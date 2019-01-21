import numpy as np
from scipy.spatial.distance import euclidean


def deter_throw_ball_hand(skeletons, ball_locs):
    early_pose_3ds = [skeleton.pose3d for skeleton in skeletons]
    early_ball_locs = [ball_loc.position3d for ball_loc in ball_locs]

    early_left_hand_poses = [getattr(pose_3d, 'left_wrist') for pose_3d in early_pose_3ds]
    early_right_hand_poses = [getattr(pose_3d, 'right_wrist') for pose_3d in early_pose_3ds]

    average_left_hand_loc = np.average(np.asarray(early_left_hand_poses), axis=0)
    average_right_hand_loc = np.average(np.asarray(early_right_hand_poses), axis=0)

    if euclidean(average_left_hand_loc, early_ball_locs) < euclidean(average_right_hand_loc, early_ball_locs):
        return 'left'
    else:
        return 'right'
