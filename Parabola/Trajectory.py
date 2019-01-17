import math

import numpy as np

from Parabola.Candidate import Candidate


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

    def front_add_new_node(self, candidate):
        self.ball_positions.insert(0, candidate)
        self.start_frame -= 1

    def cal_value(self):
        value = 0
        for i, ball_position in enumerate(self.ball_positions):
            value += math.log(ball_position.score / (1 - ball_position.score))
        self.value = value

    def generate(self, duration, direction):
        if direction == "forward":
            for i in range(1 + self.start_frame, 1 + duration + self.start_frame):
                center = self.predict_new_position(i)
                size =  self.ball_positions[-1].size
                self.add_new_node(Candidate(center, size, 0.5))

        elif direction == "backward":
            for i in range(1, 1 + duration):
                center = self.predict_new_position(i)
                size = self.ball_positions[-1].size
                self.ball_positions.insert(0, (Candidate(center, size, 0.5)))
                self.start_frame -= 1


def not_conflict(best_trajectory, other_trajectory):
    best_start = best_trajectory.start_frame
    best_end = best_start + len(best_trajectory.ball_positions)

    other_start = other_trajectory.start_frame
    other_end = other_start + len(other_trajectory.ball_positions)

    if max(best_start, other_start) <= min(best_end, other_end):
        return True
    else:
        return False