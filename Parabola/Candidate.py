import functools
import cv2

from scipy.spatial import distance as sd


class Candidate:
    def __init__(self, center, size, score_):
        self.center = center
        self.size = size
        self.score = score_
        self.link_number = 0
        self.previous = None

    def show(self, image):
        return cv2.rectangle(image,
                             (
                                 (int(self.center[0] - self.size[0] / 2)),
                                 (int(self.center[1] - self.size[1] / 2))
                             ),
                             (
                                 (int(self.center[0] + self.size[0] / 2)),
                                 (int(self.center[1] + self.size[1] / 2))
                             ),
                             (0, 255, 0), 2)


def candidate_dist(candidate1, candidate2):
    return sd.euclidean(candidate1.center, candidate2.center)


def nearest_candidate(src_candidate, candidates):
    if len(candidates) == 0:
        return None
    return min(candidates, key=functools.partial(candidate_dist, src_candidate))


def line_check(ball_candidates):
    c1, c2, c3 = ball_candidates
    x1, x2, x3 = c1.center[0], c2.center[0], c3.center[0]
    if abs(abs(x1 - x2) - abs(x2 - x3)) < 3:
        return True
    else:
        return False
