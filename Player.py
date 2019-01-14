import random
from Skeleton import PlayerSkeleton


class Player:
    def __init__(self, camera_num, delta_time):
        self.player_skeleton = PlayerSkeleton(camera_num)
        self.delta_time = delta_time

    def position_in_given_time(self, t):
        return

    # if player's feet stay in court, it will be foul.
    def foul_detection(self, start_time, beat_time, court):
        detected_skeleton_sequence = self.player_skeleton.sequence_based_time(start_time, beat_time)
        foot_keys = ["left_big_toe", "right_big_toe", "left_small_toe",
                     "right_small_toe", "left_big_heel", "right_big_heel"]
        for i, skeleton3d in detected_skeleton_sequence:
            for j, foot_key in enumerate(foot_keys):
                if hasattr(skeleton3d, foot_key) and court.in_court(getattr(skeleton3d, foot_key)):
                    return False
                elif not hasattr(skeleton3d, foot_key):
                    continue
        return True


