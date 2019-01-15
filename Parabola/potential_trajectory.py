import scipy.spatial.distance as sd
import numpy as np
import math
import functools


class Candidate:
    def __init__(self, center, size, score_):
        self.center = center
        self.size = size
        self.score = score_
        self.link_number = 0
        self.previous = None


class Trajectory:
    def __init__(self, ball_candidates, start_frame):
        self.ball_positions = ball_candidates
        self.missed = 0
        self.matched = False
        self.start_frame = start_frame
        self.growing = True

    def update_function(self):
        X = [ball_candidate.center[0] for ball_candidate in self.ball_positions]
        Y = [ball_candidate.center[1] for ball_candidate in self.ball_positions]
        T = list(range(self.start_frame, self.start_frame+len(self.ball_positions)))
        self.x_params = np.polyfit(X, T, 1)
        self.y_params = np.polyfit(Y, T, 2)

    def predict_new_position(self, t):
        x = np.poly1d(self.x_params)(t)
        y = np.poly1d(self.y_params)(t)
        return [x, y]

    def predict_new_node(self):
        center = self.predict_new_position(len(self.ball_positions)+1)
        size = self.ball_positions[-1].size
        self.add_new_node(Candidate(center, size, 0.5))

    def add_new_node(self, candidate):
        self.ball_positions.append(candidate)

    def cal_value(self):
        value = 0
        for i, ball_position in self.ball_positions:
            value += math.log( ball_position.score / (1 - ball_position.score))
        self.value = value


def check_match(position, center, D_frm):
    if sd.euclidean(position, center) < D_frm:
        return True
    else:
        return False


def line_check(ball_candidates):
    c1, c2, c3 = ball_candidates
    x1, x2, x3 = c1.center[0], c2.center[0], c3.center[0]
    if abs(abs(x1-x2) - abs(x2-x3)) < 3:
        return True
    else:
        return False


def candidate_dist(candidate1, candidate2):
    return sd.euclidean(candidate1.center, candidate2.center)


def nearest_candidate(src_candidate, candidates):
    if len(candidates) == 0:
        return None
    return min(candidates, key = functools.partial(candidate_dist, src_candidate))


def potential_trajectory(frame_candidates, D_frm):
    growing_trajectorys = []
    for frame_num, candidates in enumerate(frame_candidates):
        for i, candidate in enumerate(candidates):
            added = False
            for j, growing_trajectory in enumerate(growing_trajectorys):
                if not growing_trajectory.growing:
                    continue

                if check_match(growing_trajectory.predict_new_position, candidate.center, D_frm):
                    growing_trajectory.add_new_node(candidate)
                    growing_trajectory.matched = True
                    growing_trajectory.missed = 0
                    growing_trajectory.update_function()
                    added = True

            if not added and frame_num != 0:
                p_nearest_candidate = nearest_candidate(candidate, frame_candidates[frame_num-1])
                if p_nearest_candidate:
                    if candidate_dist(candidate, p_nearest_candidate) < D_frm:
                        candidate.link_number = p_nearest_candidate.link_number + 1
                        candidate.previous = p_nearest_candidate
                    if candidate.link_number == 3:
                        ball_candiadates = [candidate]
                        candidate_ = candidate
                        while candidate_.previous:
                            ball_candiadates.append(candidate_.previous)
                            candidate_ = candidate_.previous
                        ball_candiadates = ball_candiadates.reverse()[-3:]
                        if line_check(ball_candiadates):
                            new_trajectory = Trajectory(ball_candiadates, i-2)
                            new_trajectory.update_function()
                            new_trajectory.matched = True
                            growing_trajectorys.append(new_trajectory)

        for j, growing_trajectory in enumerate(growing_trajectorys):
            if not growing_trajectory.matched:
                growing_trajectory.missed += 1
                growing_trajectory.predict_new_node()

                if growing_trajectory.missed >= len(growing_trajectory.ball_positions):
                    growing_trajectory.growing = False

    return growing_trajectorys


def not_conflict(best_trajectory, other_trajectory):
    best_start = best_trajectory.start_frame
    best_end = best_start + len(best_trajectory.ball_positions)

    other_start = other_trajectory.start_frame
    other_end = other_start + len(other_trajectory.ball_positions)

    if max(best_start, other_start) <= min(best_end, other_end):
        return True
    else:
        return False


def filter_conflict_trajectorys(trajectorys):
    for i, trajectory in enumerate(trajectorys):
        trajectory.cal_value()

    wanted = []
    while trajectorys:
        best_one = max(trajectorys, key=lambda x: x.value)
        wanted.append(best_one)
        trajectorys.remove(best_one)
        trajectorys = list(filter(functools.partial(not_conflict, best_one), trajectorys))
    return wanted



#
# def add_straight_line_trajectory(start_time, duration, start_point, end_point, average_size):
#     start_x, start_y = start_point
#     end_x, end_y = end_point
#
#     delta_x, delta_y = (end_x - start_x) / (duration + 1), (end_y - start_y) / (duration + 1)
#     xs = [start_x + i * delta_x for i in range(1, duration + 1)]
#     ys = [start_y + i * delta_y for i in range(1, duration + 1)]
#     trajs = []
#     for i in range(duration):
#         trajs.append(
#             TraNode((xs[i], ys[i]), (average_size, average_size), by_method="merge_traj", frame_no=start_time + i))
#
#     generated_stline_traj = Trajectory(start_time)
#     generated_stline_traj.average_size = average_size
#     generated_stline_traj.tra_nodes = trajs
#     generated_stline_traj.tra_xs = xs
#     generated_stline_traj.tra_ys = ys
#     return generated_stline_traj
#
#
# def res_num(res1, res2, res_range):
#     min_x, max_x = res_range
#     res_s = []
#     if max_x >= res1 >= min_x:
#         res_s.append(res1)
#
#     if max_x >= res1 >= min_x:
#         res_s.append(res2)
#
#     return res_s
#
#
# def cal_duration(pre_len, rear_len, duration):
#     if pre_len + rear_len == duration - 1:
#         return pre_len, rear_len
#     if pre_len == 0:
#         return 0, duration - 1
#     elif rear_len == 0:
#         return duration - 1, 0
#     else:
#         pre_len_ratio = pre_len / (pre_len + rear_len)
#         rear_len_ratio = rear_len / (pre_len + rear_len)
#
#         new_pre_len = int(round(duration * pre_len_ratio))
#         new_rear_len = int(round(duration * rear_len_ratio))
#         return new_pre_len, new_rear_len
#
#
# def interpolate_two_trajectory(precede_trajectory, rear_trajectory):
#     precede_X = precede_trajectory.tra_xs
#     precede_Y = precede_trajectory.tra_ys
#
#     rear_X = rear_trajectory.tra_xs
#     rear_Y = rear_trajectory.tra_ys
#
#     pre_len, rear_len = len(precede_X), len(rear_X)
#
#     p_last_x = precede_X[-1]
#     r_first_x = rear_X[0]
#
#     min_x = min(p_last_x, r_first_x)
#     max_x = max(p_last_x, r_first_x)
#
#     start_time = precede_trajectory.start_frame + len(precede_trajectory.tra_nodes)
#     duration = rear_trajectory.start_frame - start_time
#     average_size = precede_trajectory.average_size / 2 + rear_trajectory.average_size / 2
#
#     if pre_len < 3 or rear_len < 3:
#         generate_trajectory = add_straight_line_trajectory(start_time, duration,
#                                                            (precede_X[-1], precede_Y[-1]),
#                                                            (rear_X[0], rear_Y[0]), average_size)
#
#         return True, generate_trajectory
#
#     popt, pcov = curve_fit(func, precede_X, precede_Y)
#     a1, b1, c1 = popt
#
#     popt, pcov = curve_fit(func, rear_X, rear_Y)
#     a2, b2, c2 = popt
#
#     has_sovle, res1, res2 = root((a1 - a2), (b1 - b2), (c1 - c2))
#
#     if has_sovle is False or (has_sovle is True and not res_num(res1, res2, (min_x, max_x))):
#         generate_trajectory = add_straight_line_trajectory(start_time, duration,
#                                                            (precede_X[-1], precede_Y[-1]),
#                                                            (rear_X[0], rear_Y[0]), average_size)
#
#         return True, generate_trajectory
#
#     else:
#         # TODO when root_res # > 2, should do something different
#         root_res_x = res_num(res1, res2, (min_x, max_x))
#         root_res_x = root_res_x[0]
#
#         pre_z1 = precede_trajectory.z1
#         pre_t = (root_res_x - pre_z1[1]) / pre_z1[0]
#         pre_len = pre_t - len(precede_trajectory.tra_xs)
#
#         rear_z1 = rear_trajectory.z1
#         rear_t = (root_res_x - rear_z1[1]) / (rear_z1[0])
#         rear_len = abs(rear_t)
#
#         pre_duration, rear_duration = cal_duration(pre_len, rear_len, duration)
#
#         print(pre_duration)
#         pre_delta_x = (root_res_x - precede_X[-1]) / (pre_duration + 1)
#         pre_xs = [precede_X[-1] + (i * pre_delta_x) for i in range(1, 1 + pre_duration)]
#         pre_ys = [func(x, a1, b1, c1) for x in pre_xs]
#
#         pre_new_nodes = [
#             TraNode((x, y), (average_size, average_size), by_method="merge_traj", frame_no=start_time + i)
#             for i, (x, y) in enumerate(zip(pre_xs, pre_ys))]
#
#         for new_node in pre_new_nodes:
#             precede_trajectory.extend(new_node, direction="forward")
#
#         rear_delta_x = (rear_X[0] - root_res_x) / (rear_duration + 1)
#         rear_xs = [root_res_x + (i * rear_delta_x) for i in range(1, 1 + rear_duration)]
#         rear_ys = [func(x, a1, b1, c1) for x in rear_xs]
#         rear_new_nodes = [TraNode((x, y), (average_size, average_size), by_method="merge_traj",
#                                   frame_no=rear_trajectory.start_frame - (i + 1))
#                           for i, (x, y) in enumerate(zip(rear_xs, rear_ys))
#                           ]
#         for new_node in rear_new_nodes:
#             rear_trajectory.extend(new_node, direction="backward")
#
#         root_res_y = func(root_res_x, a1, b1, c1)
#         root_res_node = TraNode((int(root_res_x), int(root_res_y)), (int(average_size), int(average_size)),
#                                 by_method="root_node",
#                                 frame_no=start_time + len(pre_new_nodes))
#
#         precede_trajectory.extend(root_res_node, direction="forward")
#         return False, None





















