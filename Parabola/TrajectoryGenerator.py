import copy
from scipy.optimize import curve_fit
import json
import cv2

from Parabola.Candidate import Candidate
from Parabola.Trajectory import Trajectory


class TraGenerator:
    def __init__(self, start_frame, duration, trajectorys):
        self.start_frame = start_frame
        self.duration = duration
        self.trajectorys = trajectorys

    # def check_trajectories_conflict(self):
    #     for i, trajectory in enumerate(self.trajectorys[:-1]):
    #         print(trajectory.start_frame)
    #         if trajectory.start_frame + len(trajectory.ball_positions) > self.trajectorys[i + 1].start_frame:
    #             return False
    #         if trajectory.start_frame > self.trajectorys[i + 1].start_frame:
    #             return False
    #     return True

    def drop_useless_trajectory(self):
        remaind_trajectories = []
        for trajectory in self.trajectorys:
            if trajectory.start_frame < self.start_frame + self.duration and trajectory.start_frame + len(
                    trajectory.ball_positions) > self.start_frame:
                remaind_trajectories.append(trajectory)
        self.trajectories = remaind_trajectories

    def head_tail_extend(self):
        head_trajectory = self.trajectorys[0]
        tail_trajectory = self.trajectorys[-1]

        # print(head_trajectory.start_frame)
        # print(self.start_frame)
        # assert head_trajectory.start_frame >= self.start_frame, "head_trajectory time wrong"
        if head_trajectory.start_frame > self.start_frame:
            head_trajectory.generate(head_trajectory.start_frame - self.start_frame, direction="backward")

        if tail_trajectory.start_frame + len(tail_trajectory.ball_positions) < self.start_frame + self.duration:
            tail_trajectory.generate(
                self.start_frame + self.duration - (tail_trajectory.start_frame + len(tail_trajectory.ball_positions)),
                direction="forward")

    def interpolate(self):
        generated_line_trajectories = []
        for pre_trajectory, rear_trajectory in zip(self.trajectorys[:-1], self.trajectorys[1:]):
            if pre_trajectory.start_frame + len(pre_trajectory.ball_positions) < rear_trajectory.start_frame:
                is_line_trajectory, line_trajectory = interpolate_two_trajectory(pre_trajectory, rear_trajectory)
                if is_line_trajectory:
                    generated_line_trajectories.append(line_trajectory)

        total_trajectories = copy.copy(self.trajectorys)
        total_trajectories.extend(generated_line_trajectories)
        final_trajectories = sorted(total_trajectories, key=lambda x: x.start_frame)
        self.trajectories = final_trajectories


    def show_result(self, video_name, save_result=False, res_video_name=None):
        cap = cv2.VideoCapture(video_name)
        if save_result is True:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            res_video = cv2.VideoWriter(res_video_name, fourcc, float(fps), (int(width), int(height)), True)

        node_pools = []
        for trajectory in self.trajectories:
            for node in trajectory.ball_positions:
               node_pools.append(node)
        print("final node num: ", len(node_pools))

        num = 0
        i = 0

        while True:
            ret_, frame_ = cap.read()
            if not ret_:
                break
            if self.start_frame + self.duration > num >= self.start_frame:
                showed_node = node_pools[i]
                res_frame = showed_node.show(frame_)
                if save_result is True:
                    res_video.write(res_frame)
                # cv2.imshow("result", res_frame)
                # cv2.waitKey(0)
                i += 1
            num += 1

    def result_json_format(self):
        start_time = self.start_frame
        duration = self.duration
        centers = []
        sizes = []
        scores = []

        for i, trajectory in enumerate(self.trajectories):
            for j, ball_position in enumerate(trajectory.ball_positions):
                centers.append(ball_position.center)
                sizes.append(ball_position.size)
                scores.append(ball_position.score)
        result = {"start_time": start_time, "duration":duration,
                  "centers":centers, "sizes":sizes, "scores":scores
                  }
        return result

    def save_result_in_json(self, json_file_dir):
        with open(json_file_dir, 'w') as f:
            json.dump(self.result_json_format(), f)


def add_straight_line_trajectory(start_time, duration, start_point, end_point, average_size):
    start_x, start_y = start_point
    end_x, end_y = end_point
    delta_x, delta_y = (end_x - start_x) / (duration + 1), (end_y - start_y) / (duration + 1)
    xs = [start_x + i * delta_x for i in range(1, duration + 1)]
    ys = [start_y + i * delta_y for i in range(1, duration + 1)]
    candidates = [Candidate([xs[i], ys[i]], [average_size, average_size]) for i in range(duration)]
    new_trajectory = Trajectory(candidates, start_time)
    return new_trajectory


def res_num(res1, res2, res_range):
    min_x, max_x = res_range
    res_s = []
    if max_x >= res1 >= min_x:
        res_s.append(res1)

    if max_x >= res1 >= min_x:
        res_s.append(res2)

    return res_s


def cal_duration(pre_len, rear_len, duration):
    if pre_len + rear_len == duration - 1:
        return pre_len, rear_len
    if pre_len == 0:
        return 0, duration - 1
    elif rear_len == 0:
        return duration - 1, 0
    else:
        pre_len_ratio = pre_len / (pre_len + rear_len)
        rear_len_ratio = rear_len / (pre_len + rear_len)

        new_pre_len = int(round(duration * pre_len_ratio))
        new_rear_len = int(round(duration * rear_len_ratio))
        return new_pre_len, new_rear_len


def func(x, a, b, c):
    return a * x * x + b * x + c


def root(a, b, c):
    if b ** 2 < 4 * a * c:
        return False, None, None
    else:
        tmp = (b ** 2 - 4 * a * c) ** 0.5
        res1 = (-b + tmp) / (2 * a)
        res2 = (-b + tmp) / (2 * a)
        return True, res1, res2


def interpolate_two_trajectory(precede_trajectory, rear_trajectory):
    precede_X = [ball_position.center[0] for ball_position in precede_trajectory.positions]
    precede_Y = [ball_position.center[1] for ball_position in precede_trajectory.positions]

    rear_X = [ball_position.center[0] for ball_position in rear_trajectory.positions]
    rear_Y = [ball_position.center[1] for ball_position in rear_trajectory.positions]

    pre_len, rear_len = len(precede_X), len(rear_X)

    p_last_x = precede_X[-1]
    r_first_x = rear_X[0]

    min_x = min(p_last_x, r_first_x)
    max_x = max(p_last_x, r_first_x)

    start_time = precede_trajectory.start_frame + len(precede_trajectory.ball_positions)
    duration = rear_trajectory.start_frame - start_time
    average_size = (sum(precede_trajectory.ball_positions[-1].size) + sum(rear_trajectory.ball_positions[-1].size)) / 2

    if pre_len < 3 or rear_len < 3:
        generate_trajectory = add_straight_line_trajectory(start_time, duration,
                                                           (precede_X[-1], precede_Y[-1]),
                                                           (rear_X[0], rear_Y[0]), average_size)

        return True, generate_trajectory

    popt, pcov = curve_fit(func, precede_X, precede_Y)
    a1, b1, c1 = popt

    popt, pcov = curve_fit(func, rear_X, rear_Y)
    a2, b2, c2 = popt

    has_sovle, res1, res2 = root((a1 - a2), (b1 - b2), (c1 - c2))

    if has_sovle is False or (has_sovle is True and not res_num(res1, res2, (min_x, max_x))):
        generate_trajectory = add_straight_line_trajectory(start_time, duration,
                                                           (precede_X[-1], precede_Y[-1]),
                                                           (rear_X[0], rear_Y[0]), average_size)

        return True, generate_trajectory

    else:
        # TODO when root_res # > 2, should do something different
        root_res_x = res_num(res1, res2, (min_x, max_x))
        root_res_x = root_res_x[0]

        pre_z1 = precede_trajectory.z1
        pre_t = (root_res_x - pre_z1[1]) / pre_z1[0]
        pre_len = pre_t - len(precede_trajectory.tra_xs)

        rear_z1 = rear_trajectory.z1
        rear_t = (root_res_x - rear_z1[1]) / (rear_z1[0])
        rear_len = abs(rear_t)

        pre_duration, rear_duration = cal_duration(pre_len, rear_len, duration)

        print(pre_duration)
        pre_delta_x = (root_res_x - precede_X[-1]) / (pre_duration + 1)
        pre_xs = [precede_X[-1] + (i * pre_delta_x) for i in range(1, 1 + pre_duration)]
        pre_ys = [func(x, a1, b1, c1) for x in pre_xs]

        pre_new_nodes = [Candidate([x, y], [average_size, average_size], 0.5) for i, (x, y) in enumerate(zip(pre_xs, pre_ys))]

        for new_node in pre_new_nodes:
            precede_trajectory.add_new_node(new_node)

        rear_delta_x = (rear_X[0] - root_res_x) / (rear_duration + 1)
        rear_xs = [root_res_x + (i * rear_delta_x) for i in range(1, 1 + rear_duration)]
        rear_ys = [func(x, a1, b1, c1) for x in rear_xs]
        rear_new_nodes = [Candidate([x, y], [average_size, average_size], 0.5)
                          for i, (x, y) in enumerate(zip(rear_xs, rear_ys))]

        for new_node in rear_new_nodes:
            rear_trajectory.front_add_new_node(new_node)

        root_res_y = func(root_res_x, a1, b1, c1)
        root_res_node = Candidate((int(root_res_x), int(root_res_y)), (int(average_size), int(average_size)), 0.5)

        precede_trajectory.add_new_node(root_res_node)
        return False, None
