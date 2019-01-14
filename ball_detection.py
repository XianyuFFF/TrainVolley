import cv2
import math
import numpy as np
# from moments import describe_shape, template_feature_match
# import libbgs
import copy
from scipy.optimize import curve_fit, fsolve
from functools import partial
from MASK_RCNN.mrcnn import utils
import MASK_RCNN.mrcnn.model as modellib
import os
from MASK_RCNN.coco import coco
from MASK_RCNN.volleyball import VolleyballConfig
from Moments import Moments

color_map = {
    'mask_rcnn': (0, 0, 139),
    'merge_traj': (28, 28, 28),
    'predict': (0, 0, 255),
    'root_node': (0, 255, 0)
}


class BallFinder:
    def __init__(self, size_range=(8, 50), shape_thresh_ratio=1.6,
                 delta_ball_loc=80):
        self.min_size, self.max_size = size_range
        self.shape_thresh_ratio = shape_thresh_ratio
        self.last_pos = None
        self.last_size = None
        self.delta_ball_loc = delta_ball_loc

    def suitable_size(self, r_s):
        roi = r_s[0]
        y1, x1, y2, x2 = roi
        w, h = x2 - x1, y2 - y1
        if self.max_size >= w >= self.min_size and self.max_size >= h >= self.min_size:
            return True
        else:
            return False

    def near_enough(self, r_s):
        roi = r_s[0]
        y1, x1, y2, x2 = roi
        x, y = (x1 + x2) // 2, (y1 + y2) // 2
        last_x, last_y = self.last_pos
        if abs(last_x - x) < self.delta_ball_loc and abs(last_y - y) < self.delta_ball_loc:
            return True
        else:
            return False

    def dist_to_last_pos(self, pos):
        last_x, last_y = self.last_pos
        x, y = pos
        return math.sqrt((last_x - x) ** 2 + (last_y - y) ** 2)

    def locate(self, rois, scores, frame_no):
        r_s_es = zip(rois, scores)
        if self.last_pos:
            r_s_es = list(filter(self.near_enough, r_s_es))

        r_s_es = list(filter(self.suitable_size, r_s_es))

        if len(r_s_es) == 1:
            y1, x1, y2, x2 = r_s_es[0][0]
            x, y = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            # current pos should be center point
            self.last_pos = (x, y)
            self.last_size = (w, h)
            find_node = TraNode(copy.copy(self.last_pos), (w, h), by_method='mask_rcnn', frame_no=frame_no)
            return True, find_node
        elif len(r_s_es) > 1:
            best_r_s = max(r_s_es, key=lambda x: x[1])
            y1, x1, y2, x2 = best_r_s[0]
            x, y = (x1 + x2) // 2, (y1 + y2) // 2
            w, h = x2 - x1, y2 - y1
            # current pos should be center point
            self.last_pos = (x, y)
            self.last_size = (w, h)
            find_node = TraNode(copy.copy(self.last_pos), (w, h), by_method='mask_rcnn', frame_no=frame_no)
            return True, find_node
        else:
            self.last_pos = None
            return False, "nothing to match"


class TraGenerator:
    def __init__(self, start_frame, duration, trajectories_):
        self.start_frame = start_frame
        self.duration = duration
        self.trajectories = trajectories_

    def check_trajectories_conflict(self):
        for i, trajectory in enumerate(self.trajectories[:-1]):
            print(trajectory.start_frame)
            if trajectory.start_frame + len(trajectory.tra_nodes) > trajectories[i + 1].start_frame:
                return False

            if trajectory.start_frame > trajectories[i + 1].start_frame:
                return False
        return True

    def drop_useless_trajectory(self):
        remaind_trajectories = []
        for trajectory in self.trajectories:
            if trajectory.start_frame < self.start_frame + self.duration and trajectory.start_frame + len(
                    trajectory.tra_nodes) > self.start_frame:
                remaind_trajectories.append(trajectory)
        self.trajectories = remaind_trajectories

    def head_tail_extend(self):
        head_trajectory = self.trajectories[0]
        tail_trajectory = self.trajectories[-1]

        assert head_trajectory.start_frame >= self.start_frame, "head_trajectory time wrong"
        if head_trajectory.start_frame > self.start_frame:
            head_trajectory.generate(head_trajectory.start_frame - self.start_frame, direction="backward")

        if tail_trajectory.start_frame + len(tail_trajectory.tra_nodes) < self.start_frame + self.duration:
            tail_trajectory.generate(
                self.start_frame + self.duration - (tail_trajectory.start_frame + len(tail_trajectory.tra_nodes)),
                direction="forward")

    def interpolate(self):
        generated_line_trajectories = []
        for pre_trajectory, rear_trajectory in zip(self.trajectories[:-1], self.trajectories[1:]):
            if pre_trajectory.start_frame + len(pre_trajectory.tra_nodes) < rear_trajectory.start_frame:
                is_line_trajectory, line_trajectory = interpolate_two_trajectory(pre_trajectory, rear_trajectory)
                if is_line_trajectory:
                    generated_line_trajectories.append(line_trajectory)

        total_trajectories = copy.copy(self.trajectories)
        total_trajectories.extend(generated_line_trajectories)
        final_trajectories = sorted(total_trajectories, key=lambda x: x.start_frame)
        self.trajectories = final_trajectories

    def show_result(self, video_name, save_result=False, res_video_name=None):
        cap_ = cv2.VideoCapture(video_name)
        if save_result is True:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = int(cap.get(cv2.CAP_PROP_FPS))

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            res_video = cv2.VideoWriter(res_video_name, fourcc, float(fps), (int(width), int(height)), True)

        node_pools = []
        for trajectory in self.trajectories:
            for node in trajectory.tra_nodes:
                if self.start_frame + self.duration > node.frame_no >= self.start_frame:
                    node_pools.append(node)
        print("final node num: ", len(node_pools))

        num = 0
        i = 0

        while True:
            ret_, frame_ = cap_.read()
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


class Trajectory:
    def __init__(self, start_frame):
        self.tra_nodes = []
        self.tra_xs = []
        self.tra_ys = []
        self.z1 = 0
        self.z2 = 0
        self.average_size = 0
        self.miss_detected = 0
        self.start_frame = start_frame

    def check_and_update(self, tra_node):
        if tra_node.by_method == "predict":
            self.tra_nodes.append(tra_node)
            self.tra_xs.append(tra_node.x)
            self.tra_ys.append(tra_node.y)
            self.z1 = np.polyfit(list(range(len(self.tra_xs))), self.tra_xs, 1)
            self.z2 = np.polyfit(list(range(len(self.tra_ys))), self.tra_ys, 2)
            self.miss_detected += 1

        if len(self.tra_nodes) == 2:
            check_x = copy.copy(self.tra_xs)
            check_x.append(tra_node.x)
            if abs((check_x[0] - check_x[1]) - (check_x[2] - check_x[1])) < self.average_size / 2:
                self.average_size = (len(self.tra_nodes) * self.average_size + (tra_node.w + tra_node.h) / 2) / (
                        len(self.tra_nodes) + 1)
                self.tra_nodes.append(tra_node)
                self.tra_xs.append(tra_node.x)
                self.tra_ys.append(tra_node.y)
                self.z1 = np.polyfit(list(range(len(self.tra_xs))), self.tra_xs, 1)
                self.z2 = np.polyfit(list(range(len(self.tra_ys))), self.tra_ys, 2)
                return True
            else:
                return False

        if len(self.tra_nodes) < 2:
            self.average_size = (len(self.tra_nodes) * self.average_size + (tra_node.w + tra_node.h) / 2) / (
                    len(self.tra_nodes) + 1)
            self.tra_nodes.append(tra_node)
            self.tra_xs.append(tra_node.x)
            self.tra_ys.append(tra_node.y)
            return True
        else:
            pred_x = np.poly1d(self.z1)(len(self.tra_xs) + 1)
            pred_y = np.poly1d(self.z2)(len(self.tra_ys) + 1)
            if abs(tra_node.x - pred_x) <= self.average_size / 2 and abs(tra_node.y - pred_y) <= self.average_size / 2:
                self.average_size = (len(self.tra_nodes) * self.average_size + (tra_node.w + tra_node.h) / 2) / (
                        len(self.tra_nodes) + 1)
                self.tra_nodes.append(tra_node)
                self.tra_xs.append(tra_node.x)
                self.tra_ys.append(tra_node.y)
                self.z1 = np.polyfit(list(range(len(self.tra_xs))), self.tra_xs, 1)
                self.z2 = np.polyfit(list(range(len(self.tra_ys))), self.tra_ys, 2)
                self.miss_detected = 0
                return True
            else:
                return False

    def predict(self, frame_no):
        if len(self.tra_nodes) >= 3:
            pred_x = np.poly1d(self.z1)(len(self.tra_xs) + 1)
            pred_y = np.poly1d(self.z2)(len(self.tra_ys) + 1)
            tra_node = TraNode((pred_x, pred_y), (self.average_size, self.average_size), by_method="predict",
                               frame_no=frame_no)
            return True, tra_node
        else:
            return False, None

    def generate(self, duration, direction):
        if direction == "forward":
            for i in range(1 + self.start_frame, 1 + duration + self.start_frame):
                can_predict, predict_node = self.predict(i)
                if not can_predict:
                    break
                self.extend(predict_node, direction=direction)
        elif direction == "backward":
            for i in range(1, 1 + duration):
                can_predict, predict_node = self.predict(self.start_frame - i)
                if not can_predict:
                    break
                self.extend(predict_node, direction=direction)

    def extend(self, tra_node, direction):
        if direction == "forward":
            self.average_size = (len(self.tra_nodes) * self.average_size + (tra_node.w + tra_node.h) / 2) / (
                    len(self.tra_nodes) + 1)
            self.tra_nodes.append(tra_node)
            self.tra_xs.append(tra_node.x)
            self.tra_ys.append(tra_node.y)
            self.z1 = np.polyfit(list(range(len(self.tra_xs))), self.tra_xs, 1)
            self.z2 = np.polyfit(list(range(len(self.tra_ys))), self.tra_ys, 2)
        elif direction == "backward":
            self.average_size = (len(self.tra_nodes) * self.average_size + (tra_node.w + tra_node.h) / 2) / (
                    len(self.tra_nodes) + 1)
            self.tra_nodes.insert(0, tra_node)
            self.tra_xs.insert(0, tra_node.x)
            self.tra_ys.insert(0, tra_node.y)
            self.z1 = np.polyfit(list(range(len(self.tra_xs))), self.tra_xs, 1)
            self.z2 = np.polyfit(list(range(len(self.tra_ys))), self.tra_ys, 2)
            self.start_frame -= 1


class TraNode:
    def __init__(self, loc, size, by_method, frame_no):
        self.x = loc[0]
        self.y = loc[1]
        self.w = size[0]
        self.h = size[1]
        self.by_method = by_method
        self.frame_no = frame_no

    def show(self, frame_):
        color = color_map[self.by_method]

        print(self.w, self.h)
        frame_ = cv2.rectangle(frame_, (int(self.x - self.w // 2), int(self.y - self.w // 2)),
                               (int(self.x + self.w // 2), int(self.y + self.w // 2)),
                               color)
        return frame_

    def __str__(self):
        return "loc:{} {} size:{} {}".format(self.x, self.y, self.w, self.h)


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


def add_straight_line_trajectory(start_time, duration, start_point, end_point, average_size):
    start_x, start_y = start_point
    end_x, end_y = end_point

    delta_x, delta_y = (end_x - start_x) / (duration + 1), (end_y - start_y) / (duration + 1)
    xs = [start_x + i * delta_x for i in range(1, duration + 1)]
    ys = [start_y + i * delta_y for i in range(1, duration + 1)]
    trajs = []
    for i in range(duration):
        trajs.append(
            TraNode((xs[i], ys[i]), (average_size, average_size), by_method="merge_traj", frame_no=start_time + i))

    generated_stline_traj = Trajectory(start_time)
    generated_stline_traj.average_size = average_size
    generated_stline_traj.tra_nodes = trajs
    generated_stline_traj.tra_xs = xs
    generated_stline_traj.tra_ys = ys
    return generated_stline_traj


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


def interpolate_two_trajectory(precede_trajectory, rear_trajectory):
    precede_X = precede_trajectory.tra_xs
    precede_Y = precede_trajectory.tra_ys

    rear_X = rear_trajectory.tra_xs
    rear_Y = rear_trajectory.tra_ys

    pre_len, rear_len = len(precede_X), len(rear_X)

    p_last_x = precede_X[-1]
    r_first_x = rear_X[0]

    min_x = min(p_last_x, r_first_x)
    max_x = max(p_last_x, r_first_x)

    start_time = precede_trajectory.start_frame + len(precede_trajectory.tra_nodes)
    duration = rear_trajectory.start_frame - start_time
    average_size = precede_trajectory.average_size / 2 + rear_trajectory.average_size / 2

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

        pre_new_nodes = [
            TraNode((x, y), (average_size, average_size), by_method="merge_traj", frame_no=start_time + i)
            for i, (x, y) in enumerate(zip(pre_xs, pre_ys))]

        for new_node in pre_new_nodes:
            precede_trajectory.extend(new_node, direction="forward")

        rear_delta_x = (rear_X[0] - root_res_x) / (rear_duration + 1)
        rear_xs = [root_res_x + (i * rear_delta_x) for i in range(1, 1 + rear_duration)]
        rear_ys = [func(x, a1, b1, c1) for x in rear_xs]
        rear_new_nodes = [TraNode((x, y), (average_size, average_size), by_method="merge_traj",
                                  frame_no=rear_trajectory.start_frame - (i + 1))
                          for i, (x, y) in enumerate(zip(rear_xs, rear_ys))
                          ]
        for new_node in rear_new_nodes:
            rear_trajectory.extend(new_node, direction="backward")

        root_res_y = func(root_res_x, a1, b1, c1)
        root_res_node = TraNode((int(root_res_x), int(root_res_y)), (int(average_size), int(average_size)),
                                by_method="root_node",
                                frame_no=start_time + len(pre_new_nodes))

        precede_trajectory.extend(root_res_node, direction="forward")
        return False, None


def load_mask_rnn():
    MODEL_DIR = os.path.join("MASK_RCNN", "logs")
    VOLLEYBALL_MODEL_PATH = os.path.join("MASK_RCNN", "mask_rcnn_volleyball.h5")

    class InferenceConfig(VolleyballConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    model.load_weights(VOLLEYBALL_MODEL_PATH, by_name=True)
    return model


class Candidate:
    def __init__(self, region, score_):
        self.region = region
        self.score = score_


if __name__ == '__main__':

    Asset_path = "asset"

    learning_stage = 1
    duration = 400

    mask_network = load_mask_rnn()

    video_name = 'cam1.mp4'
    res_video_name = "res_" + video_name
    cap = cv2.VideoCapture(video_name)

    ball_finder = BallFinder()

    trajectories = []

    num = 0
    current_trajectory = Trajectory(learning_stage)

    circle_file_dir = os.path.join(Asset_path, "circle.jpg")
    ball_moments = Moments(circle_file_dir)

    while True:
        ret, frame = cap.read()
        if not ret or num > learning_stage + duration:
            break

        if num == learning_stage:
            print("learning stage is gone")

        if num >= learning_stage:

            frame = frame[..., ::-1]

            r = mask_network.detect([frame], verbose=0)[0]

            rois = r['rois']
            scores = r['scores']

            candidates = []

            for roi, score in zip(rois, scores):
                y1, x1, y2, x2 = roi
                w, h = x2 - x1, y2 - y1
                candidate = Candidate((x1+w/2, y1+h/2, w, h), score)
                candidates.append(candidate)

            frame_different_region = ball_moments.find_loc(frame)

            candidates.append(Candidate(frame_different_region, 0.7))

