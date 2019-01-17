import functools

import cv2
import numpy as np
import os


class Trajectory:
    def __init__(self, start_frame, traj, length, z1, z2):
        self.start_frame = start_frame
        self.traj = traj
        self.length = length
        self.z1 = z1
        self.z2 = z2


def ball_size_filter(max_size, min_size, w, h):
    if max_size > w > min_size and max_size > h > min_size:
        return True
    else:
        return False


def init_traject(frame_idx, locs, max_ball_size, min_ball_size):
    flag = True
    init_locs = []
    for i, p in enumerate(locs[frame_idx:frame_idx + 3]):
        x, y, w, h = p
        if not ball_size_filter(max_ball_size, min_ball_size, w, h):
            flag = False
        init_locs.append((x, y))
    return flag, init_locs


def linear_check(traject):
    t1, t2, t3 = traject
    x1, _ = t1
    x2, _ = t2
    x3, _ = t3
    if abs((x2 - x1) - (x3 - x2)) < 2:
        return True
    else:
        return False


def deter_ball_pos(traject, traject_frames, current_frame):
    x = [t[0] for t in traject]
    y = [t[1] for t in traject]
    z1 = np.polyfit(traject_frames, x, 1)
    new_x = np.poly1d(z1)(current_frame)
    z2 = np.polyfit(traject_frames, y, 2)
    new_y = np.poly1d(z2)(current_frame)
    return z1, z2, int(new_x), int(new_y)


def match_new_p(new_x, new_y, loc, max_ball_size):
    assert isinstance(new_x, int), "new_x should be int"
    assert isinstance(new_y, int), "new_y should be int"

    ca_x, ca_y, _, _ = loc
    if abs(new_x - ca_x) < 0.5 * max_ball_size and abs(new_y - ca_y) < 0.5 * max_ball_size:
        return True, (ca_x, ca_y)
    return False, (new_x, new_y)


def trajectory(locs, min_ball_size, max_ball_size):
    trajects = []
    i = 0
    while i < len(locs):
        loc = locs[i]
        start_frame = i
        if start_frame + 3 >= len(locs):
            i+=1
            continue
        _, _, w, h = loc
        if not ball_size_filter(max_ball_size, min_ball_size, w, h):
            i+=1
            continue
        flag, traject = init_traject(start_frame, locs, max_ball_size, min_ball_size)
        if not flag:
            i+=1
            continue
        missing_frame = 0
        if not linear_check(traject):
            i += 1
            continue
        current_frame = i + 3
        z1, z2 = None, None
        current_traject_frames = list(range(start_frame, current_frame))
        while current_frame < len(locs):
            z1, z2, new_x, new_y = deter_ball_pos(traject, current_traject_frames, current_frame)
            matched, new_p = match_new_p(new_x, new_y, locs[current_frame], max_ball_size)
            if not matched:
                missing_frame += 1
            if missing_frame > 3:
                break
            traject.append(new_p)
            current_traject_frames.append(current_frame)
            current_frame += 1
        print("find traj: start from {}, length:{}".format(start_frame, len(traject)))
        trajects.append(Trajectory(start_frame, traject, len(traject), z1, z2))
        i = start_frame + len(traject)
    return trajects


def not_conflict(a_traj, b_traj):
    if a_traj.start_frame > b_traj.start_frame + b_traj.length or b_traj.start_frame > a_traj.start_frame + \
            a_traj.length:
        return True
    else:
        return False


def filter_trajectories(trajectories):
    wanted = []
    while trajectories:
        best_one = max(trajectories, key=lambda x: x.length)
        confict_to_best = functools.partial(not_conflict, best_one)
        wanted.append(best_one)
        trajectories.remove(best_one)
        trajectories = list(filter(confict_to_best, trajectories))
    return wanted


def merge_trajectories(wanted, total_loc):
    final_trajectories = {}
    new_added = []
    wanted = sorted(wanted, key=lambda x: x.start_frame)
    for i, w in enumerate(wanted):
        if i == 0:
            new_added.append(extend_trajectories(None, w, total_loc))
            new_added.append(extend_trajectories(w, wanted[1], total_loc))
        elif i == len(wanted) - 1:
            new_added.append(extend_trajectories(w, None, total_loc))
        else:
            new_added.append(extend_trajectories(w, wanted[i + 1], total_loc))
    wanted.extend(new_added)
    for i, w in enumerate(wanted):
        if w is not None:
            for j, tar in enumerate(w.traj):
                final_trajectories[w.start_frame + j] = tar
    return final_trajectories


def dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return (x1 - x2) ** 2 + (y1 - y2) ** 2


def extend_trajectories(pre_trajectory, behind_trajectory, total_loc):
    if pre_trajectory is None:
        start_frame = 0
        end_frame = behind_trajectory.start_frame
        if start_frame>=end_frame:
            return None
        behind_z1, behind_z2 = behind_trajectory.z1, behind_trajectory.z2
        behind_xs = [int(np.poly1d(behind_z1)(frame)) for frame in range(start_frame, end_frame)]
        behind_ys = [int(np.poly1d(behind_z2)(frame)) for frame in range(start_frame, end_frame)]
        behind_ps = list(zip(behind_xs, behind_ys))
        return Trajectory(start_frame, behind_ps, len(behind_ps), None, None)

    if behind_trajectory is None:
        start_frame = pre_trajectory.start_frame + pre_trajectory.length
        end_frame = total_loc
        if start_frame>=end_frame:
            return None
        pre_z1, pre_z2 = pre_trajectory.z1, pre_trajectory.z2
        pre_xs = [int(np.poly1d(pre_z1)(frame)) for frame in range(start_frame, end_frame)]
        pre_ys = [int(np.poly1d(pre_z2)(frame)) for frame in range(start_frame, end_frame)]
        pre_ps = list(zip(pre_xs, pre_ys))
        return Trajectory(start_frame, pre_ps, len(pre_ps), None, None)

    start_frame = pre_trajectory.start_frame + pre_trajectory.length
    end_frame = behind_trajectory.start_frame
    if start_frame >= end_frame:
        return None
    pre_z1, pre_z2 = pre_trajectory.z1, pre_trajectory.z2
    pre_xs = [int(np.poly1d(pre_z1)(frame)) for frame in range(start_frame, end_frame)]
    pre_ys = [int(np.poly1d(pre_z2)(frame)) for frame in range(start_frame, end_frame)]
    pre_ps = list(zip(pre_xs, pre_ys))

    behind_z1, behind_z2 = behind_trajectory.z1, behind_trajectory.z2
    behind_xs = [int(np.poly1d(behind_z1)(frame)) for frame in range(start_frame, end_frame)]
    behind_ys = [int(np.poly1d(behind_z2)(frame)) for frame in range(start_frame, end_frame)]
    behind_ps = list(zip(behind_xs, behind_ys))

    dists = np.array([dist(p1, p2) for (p1, p2) in zip(pre_ps, behind_ps)])
    cross_frame = np.argmin(dists)

    res_ps = pre_ps[:cross_frame]
    res_ps.extend(behind_ps[cross_frame:])
    return Trajectory(start_frame, res_ps, len(res_ps), None, None)


def filter_locs(locs, min_ball_size, max_ball_size):
    trajectorys = trajectory(locs, min_ball_size, max_ball_size)
    # final_res = merge_trajectories(filter_trajectories(trajectorys), len(locs))
    final_res = merge_trajectories(trajectorys, len(locs))
    return final_res

