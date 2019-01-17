import functools

from scipy.spatial import distance as sd

from Parabola.Candidate import nearest_candidate, candidate_dist, line_check
from Parabola.Trajectory import Trajectory, not_conflict


class TrajectoryFilter:
    def __init__(self, D_frm, frame_candidates):
        self.frame_candidates = frame_candidates
        self.D_frm = D_frm

    def check_match(self, position, center):
        if sd.euclidean(position, center) < self.D_frm:
            return True
        else:
            return False

    def potential_trajectorys(self):
        growing_trajectorys = []
        for frame_num, candidates in enumerate(self.frame_candidates):
            for i, candidate in enumerate(candidates):
                added = False
                for j, growing_trajectory in enumerate(growing_trajectorys):
                    if not growing_trajectory.growing:
                        continue

                    if self.check_match(growing_trajectory.predict_new_position(frame_num), candidate.center):
                        growing_trajectory.add_new_node(candidate)
                        growing_trajectory.matched = True
                        growing_trajectory.missed = 0
                        growing_trajectory.update_function()
                        added = True

                if not added and frame_num != 0:
                    p_nearest_candidate = nearest_candidate(candidate, self.frame_candidates[frame_num - 1])
                    if p_nearest_candidate:
                        if candidate_dist(candidate, p_nearest_candidate) < self.D_frm:
                            candidate.link_number = p_nearest_candidate.link_number + 1
                            candidate.previous = p_nearest_candidate
                        if candidate.link_number >= 3:
                            ball_candiadates = [candidate]
                            candidate_ = candidate
                            while candidate_.previous:
                                ball_candiadates.append(candidate_.previous)
                                candidate_ = candidate_.previous
                            ball_candiadates.reverse()
                            if line_check(ball_candiadates[-3:]):
                                new_trajectory = Trajectory(ball_candiadates[-3:], i - 2)
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

    @staticmethod
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