from Player import Player
from Ball import Ball
from Detect import get_ball_loc_data, get_openpose_data
from Detect import Moments
from Reconstruct.reconstruct import find_fundamental_matrix
import os


Asset_path = 'asset'


class World:
    def __init__(self, cams, video_dirs, output_snippets_path, ball_work_path, fps):
        super(World, self).__init__()
        self.cams = cams
        self.fundamental_matrix = find_fundamental_matrix(*self.cams)
        self.videos = video_dirs
        self.output_snippets_path = output_snippets_path
        self.player = Player()
        self.ball = Ball()
        self.ball_work_path = ball_work_path

    def detection_and_reconstruct(self):

        circle_file_dir = os.path.join(Asset_path, "circle.jpg")
        ball_moments = Moments(circle_file_dir)

        for i, video_dir in enumerate(self.videos):
            # TODO if detect file exist check, pass the stage

            # detect stage
            get_openpose_data(video_dir, self.output_snippets_path)
            get_ball_loc_data(video_dir, self.ball_work_path, ball_moments, i)

            # data loading
            self.player.player_skeletons.pack_json_skeleton(video_dir, self.output_snippets_path, i)
            self.ball.trajectory.pack_json_ball(video_dir, self.ball_work_path, i)

        # 3d reconstruct
        self.player.player_skeletons.reconstruct(self.cams, self.fundamental_matrix)
        self.ball.trajectory.reconstruct(self.cams, self.fundamental_matrix)

    def analyse(self):
        return {}

    def show_result(self):
        pass