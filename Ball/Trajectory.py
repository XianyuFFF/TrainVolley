from reconstruct import transto3d
from pathlib import Path
import json
import os
from utils import path_parser

# Need to split ball's state into "possessed", "fly" and "beat"
# if ball's movement is same as player's hand's movement, ball's state will be "possessed"
# if ball's fly change, there may be "beat" or "ground" or "net"
Ball_State = {"possessed", "fly", "beat", "ground", "net", "roll", "toss"}


class Trajectory:
    def __init__(self):
        self.ball_position_sequence = BallPositionSequence()
        self.paths = []


class BallPositionSequence:
    def __init__(self):
        self.ball_position_sequence = {}

    def pack_json_ball(self, video_dir, output_trajectory_path, cam_id):
        json_path = path_parser.detected_ball_json_dir(video_dir, output_trajectory_path, cam_id)

        data = json.load(open(json_path))

        for frame_id, ball_info in data.items():
            center = ball_info['center']
            size = ball_info['size']
            # score = value['score']
            current_ball_position = BallPosition()
            current_ball_position.ball_cordiante_2ds[cam_id] = center
            self.ball_position_sequence[frame_id] = current_ball_position

    # TODO def optimize(self):


class BallPosition:
    def __init__(self):
        self.ball_cordiante_2ds = {}

    def reconstruct_3d(self, cams, fundamental_matrix):
        self.position_3d = Position3d(
            transto3d(self.ball_cordiante_2ds[cams[0].id],
                      self.ball_cordiante_2ds[cams[0].id],
                      cams[0], cams[1], fundamental_matrix
            )
        )


class Cordinate2d:
    def __init__(self, center):
        self.center = center


class Position3d:
    def __init__(self, center):
        self.center = center
