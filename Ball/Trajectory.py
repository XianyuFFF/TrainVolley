from Reconstruct.reconstruct import transto3d
import json
from utils import path_parser

# Need to split ball's state into "possessed", "fly" and "beat"
# if ball's movement is same as player's hand's movement, ball's state will be "possessed"
# if ball's fly change, there may be "beat" or "ground" or "net"
Ball_State = {"possessed", "fly", "beat", "ground", "net", "roll", "toss"}


class Trajectory:
    def __init__(self):
        self.ball_position_sequence = {}
        self.paths = []

    def pack_json_ball(self, video_dir, output_trajectory_path, cam_id):
        json_path = path_parser.detected_ball_json_dir(video_dir, output_trajectory_path, cam_id)

        data = json.load(open(json_path))

        for frame_id, ball_center in data.items():
            current_ball_position = BallPosition()
            current_ball_position.ball_cordiante_2ds[cam_id] = ball_center
            self.ball_position_sequence[frame_id] = current_ball_position

    def reconstruct(self, cams, fundamental_matrix):
        for ball_position in self.ball_position_sequence.values():
            ball_position.reconstruct3d(cams, fundamental_matrix)


    #def optimize(self):
# TODO



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
