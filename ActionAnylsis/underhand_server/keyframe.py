import numpy as np
from utils.peak import peak_index
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean


# maybe not a good idea
def deter_begin_frame(skeleton3ds, throw_ball_hand):
    if throw_ball_hand == 'left':
        wrist_Z = [skeleton3d.pose_3d.left_wrist[2] for skeleton3d in skeleton3ds]
    else:
        wrist_Z = [skeleton3d.pose_3d.right_wrist[2] for skeleton3d in skeleton3ds]
    return peak_index(wrist_Z)[0]


def deter_beat_frame(skeleton3ds, throw_ball_hand, ball_sequence, throw_frame):
    if throw_ball_hand == 'left':
        wrist_locs = [skeleton3d.pose_3d.right_wrist for skeleton3d in skeleton3ds[throw_frame + 1:]]
    else:
        wrist_locs = [skeleton3d.pose_3d.left_wrist for skeleton3d in skeleton3ds[throw_frame + 1:]]

    after_throw_ball_locs = [
        ball_pose3d.position_3d.center for ball_pose3d in ball_sequence[throw_frame + 1:]]

    dists = [euclidean(wrist_locs[i], after_throw_ball_locs[i]) for i in range(len(after_throw_ball_locs))]

    print(wrist_locs[int(np.argmin(dists))])

    return np.argmin(dists) + throw_frame + 1


def deter_recover_frame(skeleton3ds, throw_ball_hand):
    if throw_ball_hand == 'left':
        wrist_Z = [skeleton3d.pose_3d.rigth_wrist[2] for skeleton3d in skeleton3ds]
    else:
        wrist_Z = [skeleton3d.pose_3d.left_wrist[2] for skeleton3d in skeleton3ds]
    return peak_index(wrist_Z)[-1]


def deter_ball_land_frame(ball_sequence, beat_time):
    for i in range(beat_time, len(ball_sequence)):
        ball_cordiante_3d = ball_sequence[i].position_3d.center
        if ball_cordiante_3d[2] < 200:
            return i
    return -1
