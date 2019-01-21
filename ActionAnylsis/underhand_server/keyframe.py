import numpy as np
from utils.peak import peak_index


def deter_begin_frame(skeleton3ds, throw_ball_hand):
    if throw_ball_hand == 'left':
        wrist_Z = [skeleton3d.pose3d.left_wrist[2] for skeleton3d in skeleton3ds]
    else:
        wrist_Z = [skeleton3d.pose3d.right_wrist[2] for skeleton3d in skeleton3ds]
    return peak_index(wrist_Z)[0]


def deter_beat_frame(skeleton3ds, throw_ball_hand):
    if throw_ball_hand == 'left':
        wrist_Z = [skeleton3d.pose3d.right_wrist[2] for skeleton3d in skeleton3ds]
    else:
        wrist_Z = [skeleton3d.pose3d.left_wrist[2] for skeleton3d in skeleton3ds]
    wrist_delta_z = abs(np.array(wrist_Z[1:]) - np.array(wrist_Z[:-1]))
    return peak_index(-wrist_delta_z)[0]


def deter_recover_frame(skeleton3ds, throw_ball_hand):
    if throw_ball_hand == 'left':
        wrist_Z = [skeleton3d.pose3d.rigth_wrist[2] for skeleton3d in skeleton3ds]
    else:
        wrist_Z = [skeleton3d.pose3d.left_wrist[2] for skeleton3d in skeleton3ds]
    return peak_index(wrist_Z)[-1]


def deter_ball_land_frame(ball_sequence, beat_time):
    for i in range(beat_time, len(ball_sequence)):
        ball_cordiante_3d = ball_sequence[i].position_3d.center
        if ball_cordiante_3d[2] < 120:
            return i
    return -1
