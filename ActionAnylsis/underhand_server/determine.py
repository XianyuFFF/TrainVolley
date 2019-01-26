import numpy as np
from scipy.spatial.distance import euclidean


def deter_throw_ball_hand(skeletons, ball_locs):

    early_pose_3ds = [skeleton.pose_3d for skeleton in skeletons[:5]]

    early_ball_locs = [ball_loc.position_3d.center for ball_loc in ball_locs[:5]]

    early_ball_average_loc = np.average(early_ball_locs, axis=0)

    # print(early_ball_average_loc)

    early_left_hand_poses = [getattr(pose_3d, 'left_wrist') for pose_3d in early_pose_3ds]
    early_right_hand_poses = [getattr(pose_3d, 'right_wrist') for pose_3d in early_pose_3ds]


    average_left_hand_loc = np.average(np.asarray(early_left_hand_poses), axis=0)
    average_right_hand_loc = np.average(np.asarray(early_right_hand_poses), axis=0)

    # print(average_left_hand_loc)
    # print(average_right_hand_loc)

    if euclidean(average_left_hand_loc, early_ball_average_loc) < euclidean(average_right_hand_loc, early_ball_average_loc):
        return 'left'
    else:
        return 'right'
