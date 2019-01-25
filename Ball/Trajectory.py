from Reconstruct.reconstruct import transto3d
import json
from utils import path_parser
from scipy.signal import savgol_filter
import numpy as np

# Need to split ball's state into "possessed", "fly" and "beat"
# if ball's movement is same as player's hand's movement, ball's state will be "possessed"
# if ball's fly change, there may be "beat" or "ground" or "net"
Ball_State = {"possessed", "fly", "beat", "ground", "net", "roll", "toss"}


class Trajectory:
    def __init__(self):
        self.ball_position_sequence = {}
        self.paths = []

    def pack_json_ball(self, videos, output_trajectory_path):
        for i, video_dir in enumerate(videos):
            json_path = path_parser.detected_ball_json_dir(video_dir, output_trajectory_path, i)
            print(json_path)
            data = json.load(open(json_path))
            for frame_id, bbox in data.items():
                x, y, w, h = bbox
                ball_center = (int(x+w/2), int(y+h/2))

                if self.ball_position_sequence.get(frame_id):
                    self.ball_position_sequence[frame_id].ball_cordinate_2ds[i] = Cordinate2d(ball_center)
                else:
                    current_ball_position = BallPosition()
                    current_ball_position.ball_cordinate_2ds[i] = Cordinate2d(ball_center)
                    self.ball_position_sequence[frame_id] = current_ball_position


    def reconstruct(self, cams, fundamental_matrix):
        for ball_position in self.ball_position_sequence.values():
            ball_position.reconstruct_3d(cams, fundamental_matrix)

    def optimize(self):
        X, Y, Z = [], [], []
        for k in sorted(self.ball_position_sequence.keys()):
            X.append(self.ball_position_sequence[k].position_3d.X)
            Y.append(self.ball_position_sequence[k].position_3d.Y)
            Z.append(self.ball_position_sequence[k].position_3d.Z)

        # TODO adjust window length
        filter_x = savgol_filter(np.array(X), window_length=61, polyorder=1, mode='interp')
        filter_y = savgol_filter(np.array(Y), window_length=61, polyorder=1, mode='interp')
        filter_z = savgol_filter(np.array(Z), window_length=61, polyorder=2, mode='interp')

        centers = list(zip(zip(filter_x,filter_y), filter_z))

        for i, k in enumerate(self.ball_position_sequence.keys()):
            self.ball_position_sequence[k].position_3d.center = centers[i]

    def get_ball_loc_sequence(self, begin_frame, end_frame):
        ball_loc_sequence = []
        for i in range(begin_frame, end_frame+1):
            ball_loc_sequence.append(self.ball_position_sequence[i])
        return ball_loc_sequence


class BallPosition:
    def __init__(self):
        self.ball_cordinate_2ds = {}

    def reconstruct_3d(self, cams, fundamental_matrix):

        self.position_3d = Position3d(
            transto3d(self.ball_cordinate_2ds[cams[0].id],
                      self.ball_cordinate_2ds[cams[1].id],
                      cams[0], cams[1], fundamental_matrix
                      )
        )


class Cordinate2d:
    def __init__(self, center):
        self.x = center[0]
        self.y = center[1]

    def __str__(self):
        return "({} {})".format(self.x, self.y)


class Position3d:
    def __init__(self, center):
        self.X = center[0]
        self.Y = center[1]
        self.Z = center[2]

    def __str__(self):
        return "({} {} {})".format(self.X, self.Y, self.Z)

