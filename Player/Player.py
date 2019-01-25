from Player.Skeleton import PlayerSkeleton
import os
from pathlib import Path
import json
from Player.Skeleton import Pose2d
from utils.helper import get_file_name


class PlayerSkeletonSequence:
    def __init__(self):
        self.skeletons = {}

    def sequence_based_time(self, begin_frame, end_frame):
        skeleton_sequence = []
        for i in range(begin_frame, end_frame+1):
            skeleton_sequence.append(self.skeletons[i])
        return skeleton_sequence

    def reconstruct(self, cams, fundamental_matrix):
        for skeleton in self.skeletons.values():
            skeleton.reconstruct_3d(cams, fundamental_matrix)

    # it can only solve the problem when # of people is 1
    def pack_json_skeleton(self, videos, output_snippets_path, cams):
        current_skeleton = PlayerSkeleton()

        for i, video_dir in enumerate(videos):
            video_name = get_file_name(video_dir)
            snippets_dir = os.path.join(output_snippets_path, 'snippets', video_name)

            p = Path(snippets_dir)
            for path in p.glob(video_name + '*.json'):
                json_path = str(path)
                print(path)
                frame_id = int(path.stem.split('_')[-2])

                data = json.load(open(json_path))
                person = data['people'][0]

                keypoints = person['pose_keypoints_2d']

                pose2d = Pose2d(keypoints)
                current_skeleton.skeleton2ds[cams[i].id] = pose2d
                self.skeletons[frame_id] = current_skeleton


class Player:
    def __init__(self):
        self.player_skeletons = PlayerSkeletonSequence()

    def position_in_given_time(self, t):
        return

    # if player's feet stay in court, it will be foul.
    def foul_detection(self, start_time, beat_time, court):
        detected_skeleton_sequence = self.player_skeletons.sequence_based_time(start_time, beat_time)
        foot_keys = ["left_big_toe", "right_big_toe", "left_small_toe",
                     "right_small_toe", "left_big_heel", "right_big_heel"]
        for i, skeleton3d in enumerate(detected_skeleton_sequence):
            for j, foot_key in enumerate(foot_keys):
                if hasattr(skeleton3d, foot_key) and court.in_court(getattr(skeleton3d, foot_key)):
                    return False
                elif not hasattr(skeleton3d, foot_key):
                    continue
        return True


